"""Global hotkey mappings and the Windows hotkey service."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
WM_HOTKEY = 0x0312


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_ssize_t),
        ("time", ctypes.c_uint32),
        ("pt", _POINT),
        ("lPrivate", ctypes.c_uint32),
    ]


@dataclass
class HotkeyMap:
    """Describe the default global shortcuts."""

    create_selection: str = "Ctrl+Shift+Z"
    toggle_edit_mode: str = "Ctrl+Shift+X"


@dataclass(frozen=True)
class HotkeyBinding:
    """Store a parsed global shortcut and its Windows registration id."""

    identifier: int
    sequence: str
    modifiers: int
    virtual_key: int


class _WindowsHotkeyEventFilter(QAbstractNativeEventFilter):
    """Route WM_HOTKEY messages back into the service."""

    def __init__(self, handler) -> None:
        super().__init__()
        self._handler = handler

    def nativeEventFilter(self, event_type, message):
        event_name = bytes(event_type).decode(errors="ignore")
        if event_name not in {"windows_generic_MSG", "windows_dispatcher_MSG"}:
            return False, 0

        native_message = _MSG.from_address(int(message))
        if native_message.message == WM_HOTKEY:
            self._handler(int(native_message.wParam))
            return True, 0

        return False, 0


class GlobalHotkeyService(QObject):
    """Register and dispatch application-wide Windows shortcuts."""

    create_selection_triggered = Signal()
    toggle_edit_mode_triggered = Signal()

    def __init__(self, app, hotkeys: HotkeyMap, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._hotkeys = hotkeys
        self._bindings = self._build_bindings(hotkeys)
        self._event_filter = _WindowsHotkeyEventFilter(self._dispatch_hotkey)
        self._running = False

    @staticmethod
    def parse_shortcut(shortcut: str) -> tuple[int, int]:
        """Translate a shortcut string such as Ctrl+Shift+Z into Win32 parts."""

        modifier_tokens = {
            "CTRL": MOD_CONTROL,
            "CONTROL": MOD_CONTROL,
            "SHIFT": MOD_SHIFT,
            "ALT": MOD_ALT,
            "WIN": MOD_WIN,
            "WINDOWS": MOD_WIN,
            "META": MOD_WIN,
        }
        special_keys = {
            "SPACE": 0x20,
            "ENTER": 0x0D,
            "ESC": 0x1B,
            "ESCAPE": 0x1B,
            "TAB": 0x09,
        }

        modifiers = 0
        key_token: str | None = None
        for raw_token in shortcut.split("+"):
            token = raw_token.strip().upper()
            if not token:
                continue
            if token in modifier_tokens:
                modifiers |= modifier_tokens[token]
                continue
            if key_token is not None:
                raise ValueError(f"Shortcut '{shortcut}' contains multiple primary keys.")
            key_token = token

        if key_token is None:
            raise ValueError(f"Shortcut '{shortcut}' does not contain a primary key.")

        if key_token in special_keys:
            return modifiers, special_keys[key_token]
        if len(key_token) == 1 and key_token.isalnum():
            return modifiers, ord(key_token)
        if key_token.startswith("F") and key_token[1:].isdigit():
            index = int(key_token[1:])
            if 1 <= index <= 24:
                return modifiers, 0x70 + index - 1

        raise ValueError(f"Shortcut '{shortcut}' uses an unsupported key token '{key_token}'.")

    def start(self) -> None:
        """Register all configured global shortcuts."""

        if self._running:
            return
        if os.name != "nt":
            raise RuntimeError("Global hotkeys are only implemented for Windows.")

        from app.logger import get_logger, get_debug_logger

        user32 = ctypes.windll.user32
        self._app.installNativeEventFilter(self._event_filter)
        try:
            for binding in self._bindings:
                success = user32.RegisterHotKey(None, binding.identifier, binding.modifiers, binding.virtual_key)
                if not success:
                    raise RuntimeError(f"无法注册全局快捷键 {binding.sequence}")
                get_debug_logger().debug("已注册快捷键 %s (id=%d)", binding.sequence, binding.identifier)
        except Exception:
            get_logger().error("快捷键注册失败，已回滚")
            self.stop()
            raise

        self._running = True

    def stop(self) -> None:
        """Unregister all global shortcuts."""

        if os.name == "nt":
            user32 = ctypes.windll.user32
            for binding in self._bindings:
                user32.UnregisterHotKey(None, binding.identifier)

        self._app.removeNativeEventFilter(self._event_filter)
        self._running = False

    def re_register(self, new_hotkeys: HotkeyMap) -> tuple[bool, str]:
        """Re-register with new hotkey mappings. Rolls back on failure.

        Returns (True, "") on success, (False, error_message) on failure.
        On failure the previous hotkeys are restored and re-registered.
        """

        was_running = self._running
        old_hotkeys = self._hotkeys
        old_bindings = self._bindings

        if was_running:
            self.stop()

        self._hotkeys = new_hotkeys
        try:
            self._bindings = self._build_bindings(new_hotkeys)
        except ValueError as exc:
            self._hotkeys = old_hotkeys
            self._bindings = old_bindings
            if was_running:
                try:
                    self.start()
                except RuntimeError:
                    pass
            return False, str(exc)

        try:
            self.start()
            return True, ""
        except RuntimeError as exc:
            self._hotkeys = old_hotkeys
            self._bindings = old_bindings
            if was_running:
                try:
                    self.start()
                except RuntimeError:
                    pass
            return False, str(exc)

    def _dispatch_hotkey(self, identifier: int) -> None:
        from app.logger import get_debug_logger

        if identifier == 1:
            get_debug_logger().debug("快捷键触发: create_selection")
            self.create_selection_triggered.emit()
        elif identifier == 2:
            get_debug_logger().debug("快捷键触发: toggle_edit_mode")
            self.toggle_edit_mode_triggered.emit()

    @classmethod
    def _build_bindings(cls, hotkeys: HotkeyMap) -> tuple[HotkeyBinding, HotkeyBinding]:
        create_modifiers, create_key = cls.parse_shortcut(hotkeys.create_selection)
        edit_modifiers, edit_key = cls.parse_shortcut(hotkeys.toggle_edit_mode)
        return (
            HotkeyBinding(1, hotkeys.create_selection, create_modifiers, create_key),
            HotkeyBinding(2, hotkeys.toggle_edit_mode, edit_modifiers, edit_key),
        )
