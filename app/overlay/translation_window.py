"""Translation-window overlay bound to one selection group."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.overlay.window_interaction import set_window_click_through, set_window_excluded_from_capture
from app.state.group_state import ScreenRegion

WINDOW_MARGIN = 12
SCREEN_MARGIN = 8
DEFAULT_HEIGHT = 72
MIN_WINDOW_WIDTH = 160
MIN_WINDOW_HEIGHT = 60
# Product caps in *screen* (device) pixels. Converted to Qt logical size via DPR
# so a 125% display does not show a ~1188px-wide card when "950" was requested.
MAX_DYNAMIC_WINDOW_WIDTH = 950
MAX_DYNAMIC_WINDOW_HEIGHT = 165
# Layout margins (12+12) + small slack so QLabel word-wrap does not orphan one glyph.
TEXT_WIDTH_PADDING = 36
TEXT_HEIGHT_PADDING = 44
BODY_FONT_PX = 14


def _device_pixel_ratio(screen: object | None = None) -> float:
    """Return DPR for a QScreen, or primary / default 1.0."""

    target = screen
    if target is None:
        target = QGuiApplication.primaryScreen()
    if target is None:
        return 1.0
    ratio = getattr(target, "devicePixelRatio", None)
    if callable(ratio):
        return max(1.0, float(ratio()))
    if isinstance(screen, (int, float)):
        return max(1.0, float(screen))
    return 1.0


def max_window_width(
    *,
    dpr: float | None = None,
    screen: object | None = None,
    available_width: int | None = None,
) -> int:
    """Max card width in Qt *logical* pixels for this display.

    Product cap is 950 *device* (physical) pixels. Logical cap = 950 / DPR so
    100% / 125% / 150% screens all land near the same on-screen size. Also
    never wider than the usable area of that screen.
    """

    ratio = dpr if dpr is not None else _device_pixel_ratio(screen)
    cap = max(MIN_WINDOW_WIDTH, int(MAX_DYNAMIC_WINDOW_WIDTH / max(1.0, ratio)))
    if available_width is not None:
        cap = min(cap, max(MIN_WINDOW_WIDTH, int(available_width)))
    return cap


def max_window_height(
    *,
    dpr: float | None = None,
    screen: object | None = None,
    available_height: int | None = None,
) -> int:
    """Max card height in Qt logical pixels (165 device px / DPR, screen-clamped)."""

    ratio = dpr if dpr is not None else _device_pixel_ratio(screen)
    cap = max(MIN_WINDOW_HEIGHT, int(MAX_DYNAMIC_WINDOW_HEIGHT / max(1.0, ratio)))
    if available_height is not None:
        cap = min(cap, max(MIN_WINDOW_HEIGHT, int(available_height)))
    return cap


@dataclass
class TranslationWindowModel:
    """Describe one translation overlay window."""

    group_id: int
    x: int
    y: int
    width: int
    height: int
    source_language: str
    target_language: str
    text: str
    preferred_dock: str = "bottom"
    visible: bool = True
    accent_color: str = "#2F80ED"


class TranslationWindowWidget(QWidget):
    """Render one low-distraction translation overlay."""

    moved = Signal(int, int, int)
    dock_changed = Signal(int, str)
    feedback_requested = Signal(int)
    body_immersive_toggled = Signal(int, bool)

    def __init__(self, model: TranslationWindowModel, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self.model = model
        self._editable = False
        self._body_immersive = False
        self._group_immersive = False
        self._corner_collapsed = False
        self._dragging = False
        self._drag_origin_global = QPoint()
        self._drag_origin_top_left = QPoint()
        self._input_passthrough_enabled = False
        # Base geometry ignores edit-mode clearance; shift is applied once on screen.
        self._base_x = model.x
        self._base_y = model.y
        self._edit_shift_y = 0

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        set_window_excluded_from_capture(self, True)
        self.setMouseTracking(True)

        self.group_badge = QLabel(self)
        self.group_badge.setObjectName("translationGroupBadge")

        self.language_pair_label = QLabel(self)
        self.language_pair_label.setObjectName("translationLanguagePair")

        # Corner mini-bar (mock .corner-bar): ↑↓←→ | copy | report | body ◎
        # Floating Tool so it sits outside the card without shifting body text.
        self.dock_controls_widget = QWidget(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.dock_controls_widget.setObjectName("translationCornerBar")
        self.dock_controls_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.dock_controls_widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        set_window_excluded_from_capture(self.dock_controls_widget, True)
        dock_layout = QHBoxLayout(self.dock_controls_widget)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(3)

        self.corner_tools_widget = QWidget(self.dock_controls_widget)
        self.corner_tools_widget.setObjectName("translationCornerTools")
        tools_layout = QHBoxLayout(self.corner_tools_widget)
        tools_layout.setContentsMargins(2, 2, 2, 2)
        tools_layout.setSpacing(0)
        self.dock_up_button = QPushButton("↑")
        self.dock_down_button = QPushButton("↓")
        self.dock_left_button = QPushButton("←")
        self.dock_right_button = QPushButton("→")
        self.copy_button = QPushButton("⧉")
        self.feedback_button = QPushButton("⚑")
        self.body_immersive_button = QPushButton("⊘")
        self.body_immersive_button.setCheckable(True)
        self.body_immersive_button.setProperty("visibilityToggle", True)
        self.body_immersive_button.setAccessibleName("切换译文框体可见性")
        self.collapse_button = QPushButton("›")
        self.collapse_button.setObjectName("cornerCollapseButton")
        self.collapse_button.setToolTip("收缩工具")
        self.dock_up_button.setToolTip("停靠上")
        self.dock_down_button.setToolTip("停靠下")
        self.dock_left_button.setToolTip("停靠左")
        self.dock_right_button.setToolTip("停靠右")
        self.copy_button.setToolTip("复制译文")
        self.feedback_button.setToolTip("翻译有误")
        self.body_immersive_button.setToolTip("隐藏译文框体，仅保留文字")
        for button in (
            self.dock_up_button,
            self.dock_down_button,
            self.dock_left_button,
            self.dock_right_button,
            self.copy_button,
            self.feedback_button,
            self.body_immersive_button,
        ):
            button.setObjectName("dockButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedSize(28, 28)
            tools_layout.addWidget(button)
        for button in (
            self.dock_up_button,
            self.dock_down_button,
            self.dock_left_button,
            self.dock_right_button,
        ):
            button.setCheckable(True)
            button.setAutoExclusive(True)
        self.collapse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_button.setFixedSize(28, 28)
        dock_layout.addWidget(self.corner_tools_widget)
        dock_layout.addWidget(self.collapse_button)
        self.feedback_button.setText("⚑")
        self.feedback_button.setAccessibleName("翻译有误")

        self.dock_up_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "top"))
        self.dock_down_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "bottom"))
        self.dock_left_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "left"))
        self.dock_right_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "right"))
        self.copy_button.clicked.connect(self.copy_text)
        self.feedback_button.clicked.connect(lambda: self.feedback_requested.emit(self.model.group_id))
        self.body_immersive_button.toggled.connect(self._on_body_immersive_toggled)
        self.collapse_button.clicked.connect(self.toggle_corner_collapsed)

        self.translation_label = QLabel(self)
        self.translation_label.setObjectName("translationBody")
        self.translation_label.setWordWrap(True)
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.translation_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        # Header stays in layout always so immersive chrome-off cannot collapse
        # body text upward (implementer §8.2 / mock visibility:hidden).
        self.header_widget = QWidget(self)
        self.header_widget.setObjectName("translationHeader")
        self.header_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        header_row = QHBoxLayout(self.header_widget)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self.group_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        header_row.addWidget(self.language_pair_label, stretch=1)
        self.language_pair_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetNoConstraint)
        layout.addWidget(self.header_widget)
        layout.addWidget(self.translation_label, stretch=1)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.setMaximumSize(max_window_width(), max_window_height())
        self.apply_model(model)
        self.apply_edit_mode(False)

    @staticmethod
    def compute_geometry(
        region: ScreenRegion,
        screen: QRect,
        preferred_dock: str,
        text: str = "",
        *,
        device_pixel_ratio: float | None = None,
    ) -> QRect:
        """Return a translation-window geometry sized to text near one region.

        ``device_pixel_ratio`` should be the DPR of the monitor that contains
        ``region`` (not always the primary screen).
        """

        available_width = max(80, screen.width() - SCREEN_MARGIN * 2)
        available_height = max(80, screen.height() - SCREEN_MARGIN * 2)
        dpr = device_pixel_ratio if device_pixel_ratio is not None else _device_pixel_ratio()
        max_w = max_window_width(dpr=dpr, available_width=available_width)
        max_h = max_window_height(dpr=dpr, available_height=available_height)
        width = TranslationWindowWidget._preferred_width(max_w, text, dpr=dpr)
        height = TranslationWindowWidget._preferred_height(width, max_h, text, dpr=dpr)
        width = min(max(MIN_WINDOW_WIDTH, width), max_w)
        height = min(max(MIN_WINDOW_HEIGHT, height), max_h)

        left_bound = screen.x() + SCREEN_MARGIN
        right_bound = screen.x() + screen.width() - width - SCREEN_MARGIN
        top_bound = screen.y() + SCREEN_MARGIN
        bottom_bound = screen.y() + screen.height() - height - SCREEN_MARGIN

        def clamp_x(value: int) -> int:
            return min(max(value, left_bound), right_bound)

        def clamp_y(value: int) -> int:
            return min(max(value, top_bound), bottom_bound)

        if preferred_dock == "top":
            x = clamp_x(region.x)
            y = region.y - height - WINDOW_MARGIN
            if y < top_bound:
                y = clamp_y(region.y + region.height + WINDOW_MARGIN)
            return QRect(x, y, width, height)

        if preferred_dock == "left":
            x = region.x - width - WINDOW_MARGIN
            if x < left_bound:
                x = clamp_x(region.x + region.width + WINDOW_MARGIN)
            y = clamp_y(region.y)
            return QRect(x, y, width, height)

        if preferred_dock == "right":
            x = region.x + region.width + WINDOW_MARGIN
            if x > right_bound:
                x = clamp_x(region.x - width - WINDOW_MARGIN)
            y = clamp_y(region.y)
            return QRect(x, y, width, height)

        x = clamp_x(region.x)
        y = region.y + region.height + WINDOW_MARGIN
        if y > bottom_bound:
            y = clamp_y(region.y - height - WINDOW_MARGIN)
        return QRect(x, y, width, height)

    @staticmethod
    def _body_font() -> QFont:
        font = QFont("Segoe UI")
        font.setFamilies(["Segoe UI", "Microsoft YaHei", "sans-serif"])
        font.setPixelSize(BODY_FONT_PX)
        return font

    @staticmethod
    def _body_metrics() -> QFontMetrics:
        return QFontMetrics(TranslationWindowWidget._body_font())

    @staticmethod
    def _preferred_width(
        available_width: int,
        text: str,
        *,
        dpr: float | None = None,
    ) -> int:
        """Fit width to content; never stretch to selection-box width."""

        cap = max_window_width(dpr=dpr, available_width=available_width)
        floor = min(MIN_WINDOW_WIDTH, cap)
        if not text.strip():
            return floor
        estimated_width = TranslationWindowWidget._estimate_text_width(text)
        return min(max(floor, estimated_width), cap)

    @staticmethod
    def _estimate_text_width(text: str) -> int:
        metrics = TranslationWindowWidget._body_metrics()
        max_line_width = 0
        for line in text.splitlines() or [text]:
            max_line_width = max(max_line_width, metrics.horizontalAdvance(line))
        # +1px slack avoids last-glyph wrap from fractional layout.
        return max_line_width + TEXT_WIDTH_PADDING + 1

    @staticmethod
    def _preferred_height(
        width: int,
        available_height: int,
        text: str,
        *,
        dpr: float | None = None,
    ) -> int:
        cap = max_window_height(dpr=dpr, available_height=available_height)
        floor = min(MIN_WINDOW_HEIGHT, cap)
        if not text.strip():
            return min(DEFAULT_HEIGHT, cap)

        metrics = TranslationWindowWidget._body_metrics()
        body_width = max(1, width - TEXT_WIDTH_PADDING)
        bounds = metrics.boundingRect(
            QRect(0, 0, body_width, 10_000),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
            text,
        )
        estimated_height = TEXT_HEIGHT_PADDING + bounds.height()
        return min(max(floor, estimated_height), cap)

    @property
    def dock_controls_visible(self) -> bool:
        """Expose dock controls visibility for tests and diagnostics."""

        return self.dock_controls_widget.isVisible()

    def apply_model(self, model: TranslationWindowModel) -> None:
        """Update geometry and labels from the latest model state."""

        max_w, max_h = max_window_width(), max_window_height()
        model.width = min(max(MIN_WINDOW_WIDTH, int(model.width)), max_w)
        model.height = min(max(MIN_WINDOW_HEIGHT, int(model.height)), max_h)
        self.model = model
        self.setWindowTitle(f"Translation {model.group_id}")
        self._base_x = model.x
        self._base_y = model.y
        self.translation_label.setWordWrap(True)
        self.translation_label.setMinimumWidth(0)
        # Keep body inside card; use content box not full window width.
        self.translation_label.setMaximumWidth(max(1, model.width - TEXT_WIDTH_PADDING))
        self._apply_screen_geometry()
        self.group_badge.setText(str(model.group_id))
        self.language_pair_label.setText(f"{model.source_language} → {model.target_language}")
        self.translation_label.setText(model.text)
        for button, dock in (
            (self.dock_up_button, "top"),
            (self.dock_down_button, "bottom"),
            (self.dock_left_button, "left"),
            (self.dock_right_button, "right"),
        ):
            button.setChecked(model.preferred_dock == dock)
        self._apply_styles()
        self._update_corner_bar_position()
        self.update()

    def apply_edit_mode(self, enabled: bool, shift_y: int | None = None) -> None:
        """Switch between minimal display and edit display.

        ``shift_y`` is an absolute screen offset from the base position (not
        cumulative). Repeated enable with the same value must not stack.
        """

        self._editable = enabled
        if enabled:
            if shift_y is not None:
                self._edit_shift_y = int(shift_y)
            # else keep previous absolute shift when chrome-only refresh
        else:
            self._edit_shift_y = 0
        self._apply_screen_geometry()
        self.language_pair_label.setHidden(not enabled or self._body_immersive or self._group_immersive)
        # Corner bar stays available in per-card immersive mode so ◎ can be undone.
        if enabled and not self._group_immersive:
            self.dock_controls_widget.show()
            self._update_corner_bar_position()
        else:
            self.dock_controls_widget.hide()
        set_window_click_through(self, not enabled)
        self._apply_styles()
        self.update()

    def corner_toolbar_height(self) -> int:
        """Measured corner-bar height for edit-mode clearance."""

        self.dock_controls_widget.adjustSize()
        return max(1, self.dock_controls_widget.height())

    def base_position(self) -> tuple[int, int]:
        """Logical top-left without edit-mode clearance."""

        return self._base_x, self._base_y

    def set_base_position(self, x: int, y: int) -> None:
        self._base_x = int(x)
        self._base_y = int(y)
        self._apply_screen_geometry()
        self._update_corner_bar_position()

    def _screen_for_caps(self):
        """Prefer the monitor that currently hosts this window."""

        handle = self.screen()
        if handle is not None:
            return handle
        at = QGuiApplication.screenAt(QPoint(self._base_x, self._base_y))
        if at is not None:
            return at
        return QGuiApplication.primaryScreen()

    def _clamped_size(self, width: int, height: int) -> tuple[int, int]:
        host = self._screen_for_caps()
        dpr = _device_pixel_ratio(host)
        avail_w = avail_h = None
        if host is not None:
            geo = host.availableGeometry()
            avail_w = max(80, geo.width() - SCREEN_MARGIN * 2)
            avail_h = max(80, geo.height() - SCREEN_MARGIN * 2)
        max_w = max_window_width(dpr=dpr, available_width=avail_w)
        max_h = max_window_height(dpr=dpr, available_height=avail_h)
        w = min(max(MIN_WINDOW_WIDTH, int(width)), max_w)
        h = min(max(MIN_WINDOW_HEIGHT, int(height)), max_h)
        return w, h

    def _apply_screen_geometry(self) -> None:
        width, height = self._clamped_size(self.model.width, self.model.height)
        self.model.width = width
        self.model.height = height
        self.translation_label.setMaximumWidth(max(1, width - TEXT_WIDTH_PADDING))
        # Lock both min and max to the same size (Tool + translucent can ignore one).
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)
        self.resize(width, height)
        self.move(self._base_x, self._base_y + self._edit_shift_y)

    def sizeHint(self) -> QSize:
        w, h = self._clamped_size(self.model.width, self.model.height)
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def maximumSize(self) -> QSize:
        host = self._screen_for_caps()
        return QSize(
            max_window_width(screen=host),
            max_window_height(screen=host),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        width, height = self._clamped_size(self.width(), self.height())
        if self.width() != width or self.height() != height:
            self.model.width = width
            self.model.height = height
            self.setMinimumSize(width, height)
            self.setMaximumSize(width, height)
            self.resize(width, height)
            self.translation_label.setMaximumWidth(max(1, width - TEXT_WIDTH_PADDING))

    def copy_text(self) -> None:
        QApplication.clipboard().setText(self.model.text)

    def set_body_immersive(self, on: bool) -> None:
        """Chrome off / text on without moving body text (mock is-body-hidden)."""

        self._body_immersive = bool(on)
        self.body_immersive_button.blockSignals(True)
        self.body_immersive_button.setChecked(on)
        self.body_immersive_button.setText("◉" if on else "⊘")
        self.body_immersive_button.setToolTip(
            "显示译文框体" if on else "隐藏译文框体，仅保留文字"
        )
        self.body_immersive_button.blockSignals(False)
        # Keep header in layout; only stop painting chrome labels.
        self.group_badge.setVisible(True)
        self.language_pair_label.setVisible(True)
        transparent = "color: transparent; background: transparent; border: none;"
        if self._body_immersive or self._group_immersive:
            self.group_badge.setStyleSheet(transparent)
            self.language_pair_label.setStyleSheet(transparent)
            self.header_widget.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
            if self._group_immersive:
                self.dock_controls_widget.hide()
            elif self._editable:
                self.dock_controls_widget.show()
                self._update_corner_bar_position()
        else:
            self.group_badge.setStyleSheet("")
            self.language_pair_label.setStyleSheet("")
            self.header_widget.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
            )
            self.language_pair_label.setHidden(not self._editable)
            if self._editable and not self._group_immersive:
                self.dock_controls_widget.show()
                self._update_corner_bar_position()
            self._apply_styles()
        self.update()

    def set_group_immersive(self, on: bool) -> None:
        """Hide group chrome while preserving the card's own body preference."""

        self._group_immersive = bool(on)
        if self._group_immersive:
            self.dock_controls_widget.hide()
        elif self._editable:
            self.dock_controls_widget.show()
            self._update_corner_bar_position()
        self.set_body_immersive(self._body_immersive)

    def set_corner_collapsed(self, collapsed: bool) -> None:
        self._corner_collapsed = bool(collapsed)
        self.corner_tools_widget.setVisible(not self._corner_collapsed)
        self.collapse_button.setText("‹" if self._corner_collapsed else "›")
        self.collapse_button.setToolTip("展开工具" if self._corner_collapsed else "收缩工具")
        self.dock_controls_widget.adjustSize()
        self._update_corner_bar_position()

    def toggle_corner_collapsed(self) -> None:
        self.set_corner_collapsed(not self._corner_collapsed)

    @property
    def corner_collapsed(self) -> bool:
        return self._corner_collapsed

    def _on_body_immersive_toggled(self, on: bool) -> None:
        self.set_body_immersive(on)
        self.body_immersive_toggled.emit(self.model.group_id, on)

    def _update_corner_bar_position(self) -> None:
        if not self.dock_controls_widget.isVisible() and not self._editable:
            return
        self.dock_controls_widget.adjustSize()
        # Top-right outside card (mock .corner-bar top:-28 right:0).
        geo = self.frameGeometry()
        x = geo.right() - self.dock_controls_widget.width() + 1
        y = geo.top() - self.dock_controls_widget.height() - 4
        self.dock_controls_widget.move(x, y)

    @property
    def body_immersive(self) -> bool:
        return self._body_immersive

    def body_text_global_top_left(self) -> QPoint:
        """Screen position of translation body — used to assert no jump on ◎."""

        return self.translation_label.mapToGlobal(QPoint(0, 0))

    @property
    def input_passthrough_enabled(self) -> bool:
        """Expose click-through state for tests and diagnostics."""

        return bool(getattr(self, "_input_passthrough_enabled", False))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._editable:
            self._dragging = True
            self._drag_origin_global = event.globalPosition().toPoint()
            self._drag_origin_top_left = self.frameGeometry().topLeft()
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._editable:
            delta = event.globalPosition().toPoint() - self._drag_origin_global
            self.move(self._drag_origin_top_left + delta)
            self._update_corner_bar_position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.releaseMouse()
            # Persist base (without edit clearance); display keeps current shift.
            self._base_x = self.x()
            self._base_y = self.y() - self._edit_shift_y
            self.model.x = self._base_x
            self.model.y = self._base_y
            self._update_corner_bar_position()
            self.moved.emit(self.model.group_id, self._base_x, self._base_y)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Re-assert size after Win32/DWM maps the Tool window (DPR-aware caps).
        self._apply_screen_geometry()
        set_window_excluded_from_capture(self, True)
        set_window_excluded_from_capture(self.dock_controls_widget, True)
        set_window_click_through(self, not self._editable)
        if self._editable and not self._group_immersive:
            self.dock_controls_widget.show()
            self._update_corner_bar_position()

    def hideEvent(self, event) -> None:
        self.dock_controls_widget.hide()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self.dock_controls_widget.close()
        super().closeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        if self._body_immersive or self._group_immersive:
            # Chrome off: no fill / border / shadow; padding + header still occupy space.
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        fill_alpha = 235 if self._editable else 225
        painter.setBrush(QColor(255, 255, 255, fill_alpha))
        border_color = QColor(self.model.accent_color if self._editable else "#94A3B8")
        border_color.setAlpha(245 if self._editable else 180)
        painter.setPen(QPen(border_color, 1.5 if self._editable else 1))
        painter.drawRoundedRect(QRectF(rect), 10, 10)
        painter.end()

    def _apply_styles(self) -> None:
        if self._body_immersive or self._group_immersive:
            # Keep body text styles only; chrome labels already transparent.
            self.setStyleSheet(
                """
                QLabel#translationBody {
                    color: #e2e8f0;
                    border: none;
                    font-size: 14px;
                    font-weight: 600;
                    line-height: 1.5;
                    background: transparent;
                }
                QWidget#translationHeader {
                    background: transparent;
                    border: none;
                }
                """
            )
            self._apply_corner_styles()
            return
        pair_color = "#475569"
        badge_background = "rgba(71, 85, 105, 0.85)"
        self.setStyleSheet(
            f"""
            QLabel#translationGroupBadge {{
                background: {badge_background};
                color: white;
                border: none;
                padding: 1px 6px;
                border-radius: 5px;
                font-size: 10px;
                font-weight: 700;
                max-width: 22px;
            }}
            QLabel#translationLanguagePair {{
                color: {pair_color};
                border: none;
                font-size: 13px;
                font-weight: 600;
                background: transparent;
            }}
            QLabel#translationBody {{
                color: #1e293b;
                border: none;
                font-size: 14px;
                line-height: 1.5;
                background: transparent;
            }}
            QWidget#translationHeader {{
                background: transparent;
                border: none;
            }}
            """
        )
        self._apply_corner_styles()

    def _apply_corner_styles(self) -> None:
        """Style the detached corner bar like the mock's tools + collapse pills."""

        self.dock_controls_widget.setStyleSheet(
            """
            QWidget#translationCornerBar {
                background: transparent;
                border: none;
            }
            QWidget#translationCornerTools {
                background: rgba(8, 12, 22, 0.82);
                border: 1px solid rgba(51, 65, 85, 0.65);
                border-radius: 999px;
            }
            QPushButton#dockButton {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 6px;
                padding: 0 4px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#dockButton:hover {
                background: rgba(51, 65, 85, 0.75);
                color: #f1f5f9;
            }
            QPushButton#dockButton:checked {
                background: rgba(37, 99, 235, 0.35);
                color: #bfdbfe;
                border: 1px solid rgba(147, 197, 253, 0.45);
            }
            QPushButton#dockButton[visibilityToggle="true"] {
                color: #cbd5e1;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton#cornerCollapseButton {
                background: rgba(8, 12, 22, 0.82);
                color: #64748b;
                border: 1px solid rgba(51, 65, 85, 0.65);
                border-radius: 14px;
                padding: 0;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#cornerCollapseButton:hover {
                color: #e2e8f0;
            }
            """
        )
