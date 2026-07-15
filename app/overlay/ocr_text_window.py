"""OCR text viewer overlay bound to one selection group."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
)
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
MAX_DYNAMIC_WINDOW_WIDTH = 950
MAX_DYNAMIC_WINDOW_HEIGHT = 165
# Layout margins (12+12) + slack so QLabel word-wrap does not orphan one glyph.
TEXT_WIDTH_PADDING = 36
TEXT_HEIGHT_PADDING = 44
BODY_FONT_PX = 14


def _device_pixel_ratio(screen: object | None = None) -> float:
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
    ratio = dpr if dpr is not None else _device_pixel_ratio(screen)
    cap = max(MIN_WINDOW_HEIGHT, int(MAX_DYNAMIC_WINDOW_HEIGHT / max(1.0, ratio)))
    if available_height is not None:
        cap = min(cap, max(MIN_WINDOW_HEIGHT, int(available_height)))
    return cap


def _media_control_icon(paused: bool) -> QIcon:
    """Return a crisp pause/play icon without relying on font glyph metrics."""

    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#cbd5e1"))
    if paused:
        path = QPainterPath()
        path.moveTo(5.0, 3.0)
        path.lineTo(14.0, 9.0)
        path.lineTo(5.0, 15.0)
        path.closeSubpath()
        painter.drawPath(path)
    else:
        painter.drawRoundedRect(QRectF(4.0, 3.0, 3.5, 12.0), 1.5, 1.5)
        painter.drawRoundedRect(QRectF(10.5, 3.0, 3.5, 12.0), 1.5, 1.5)
    painter.end()
    return QIcon(pixmap)


@dataclass
class OcrTextWindowModel:
    """Describe one OCR text viewer window."""

    group_id: int
    x: int
    y: int
    width: int
    height: int
    text: str
    accent_color: str = "#2F80ED"
    paused: bool = False


class OcrTextWindowWidget(QWidget):
    """Render the latest cleaned OCR text for one selection group."""

    refresh_clicked = Signal(int)
    pause_toggled = Signal(int)
    body_immersive_toggled = Signal(int, bool)

    def __init__(self, model: OcrTextWindowModel, parent: QWidget | None = None) -> None:
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
        self._base_x = model.x
        self._base_y = model.y
        self._edit_shift_y = 0

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        set_window_excluded_from_capture(self, True)
        self.setMouseTracking(True)

        self.group_badge = QLabel(self)
        self.group_badge.setObjectName("ocrGroupBadge")

        self.title_label = QLabel("OCR")
        self.title_label.setObjectName("ocrTitle")

        # Detached mock-style corner bar: pause | refresh | copy | ◎ | collapse.
        self.corner_bar = QWidget(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.corner_bar.setObjectName("ocrCornerBar")
        self.corner_bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.corner_bar.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        set_window_excluded_from_capture(self.corner_bar, True)
        corner_layout = QHBoxLayout(self.corner_bar)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(3)

        self.corner_tools_widget = QWidget(self.corner_bar)
        self.corner_tools_widget.setObjectName("ocrCornerTools")
        tools_layout = QHBoxLayout(self.corner_tools_widget)
        tools_layout.setContentsMargins(2, 2, 2, 2)
        tools_layout.setSpacing(0)

        self.pause_button = QPushButton("")
        self.pause_button.setToolTip("暂停")
        self.pause_button.setProperty("pauseControl", True)
        self.pause_button.setIconSize(QSize(18, 18))
        self.refresh_button = QPushButton("↻")
        self.refresh_button.setToolTip("刷新 OCR")
        self.copy_button = QPushButton("⧉")
        self.copy_button.setToolTip("复制 OCR")
        self.body_immersive_button = QPushButton("⊘")
        self.body_immersive_button.setToolTip("隐藏 OCR 框体，仅保留文字")
        self.body_immersive_button.setCheckable(True)
        self.body_immersive_button.setProperty("visibilityToggle", True)
        self.body_immersive_button.setAccessibleName("切换 OCR 框体可见性")
        for button in (
            self.pause_button,
            self.refresh_button,
            self.copy_button,
            self.body_immersive_button,
        ):
            button.setObjectName("ocrCornerButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedSize(28, 28)
            tools_layout.addWidget(button)

        self.collapse_button = QPushButton("›")
        self.collapse_button.setObjectName("ocrCornerCollapseButton")
        self.collapse_button.setToolTip("收缩工具")
        self.collapse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_button.setFixedSize(28, 28)
        corner_layout.addWidget(self.corner_tools_widget)
        corner_layout.addWidget(self.collapse_button)

        self.pause_button.clicked.connect(lambda: self.pause_toggled.emit(self.model.group_id))
        self.refresh_button.clicked.connect(lambda: self.refresh_clicked.emit(self.model.group_id))
        self.copy_button.clicked.connect(self.copy_text)
        self.body_immersive_button.toggled.connect(self._on_body_immersive_toggled)
        self.collapse_button.clicked.connect(self.toggle_corner_collapsed)

        self.ocr_label = QLabel(self)
        self.ocr_label.setObjectName("ocrBody")
        self.ocr_label.setWordWrap(True)
        self.ocr_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.ocr_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        # Header remains laid out so immersive mode cannot shift body text up.
        self.header_widget = QWidget(self)
        self.header_widget.setObjectName("ocrHeader")
        self.header_widget.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        header_row = QHBoxLayout(self.header_widget)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self.group_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        header_row.addWidget(self.title_label, stretch=1)
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetNoConstraint)
        layout.addWidget(self.header_widget)
        layout.addWidget(self.ocr_label, stretch=1)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.setMaximumSize(max_window_width(), max_window_height())
        self.apply_model(model)
        self.apply_edit_mode(False)

    @staticmethod
    def compute_geometry(
        region: ScreenRegion,
        screen: QRect,
        text: str = "",
        *,
        device_pixel_ratio: float | None = None,
    ) -> QRect:
        """Return an OCR-viewer geometry sized to text near one selection region."""

        available_width = max(80, screen.width() - SCREEN_MARGIN * 2)
        available_height = max(80, screen.height() - SCREEN_MARGIN * 2)
        dpr = device_pixel_ratio if device_pixel_ratio is not None else _device_pixel_ratio()
        max_w = max_window_width(dpr=dpr, available_width=available_width)
        max_h = max_window_height(dpr=dpr, available_height=available_height)
        width = OcrTextWindowWidget._preferred_width(max_w, text, dpr=dpr)
        height = OcrTextWindowWidget._preferred_height(width, max_h, text, dpr=dpr)
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

        x = clamp_x(region.x)
        y = region.y - height - WINDOW_MARGIN
        if y < top_bound:
            y = clamp_y(region.y + region.height + WINDOW_MARGIN)
        return QRect(x, y, width, height)

    @staticmethod
    def _body_font() -> QFont:
        font = QFont("Segoe UI")
        font.setFamilies(["Segoe UI", "Microsoft YaHei", "sans-serif"])
        font.setPixelSize(BODY_FONT_PX)
        return font

    @staticmethod
    def _body_metrics() -> QFontMetrics:
        return QFontMetrics(OcrTextWindowWidget._body_font())

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
        estimated_width = OcrTextWindowWidget._estimate_text_width(text)
        return min(max(floor, estimated_width), cap)

    @staticmethod
    def _estimate_text_width(text: str) -> int:
        metrics = OcrTextWindowWidget._body_metrics()
        max_line_width = 0
        for line in text.splitlines() or [text]:
            max_line_width = max(max_line_width, metrics.horizontalAdvance(line))
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

        metrics = OcrTextWindowWidget._body_metrics()
        body_width = max(1, width - TEXT_WIDTH_PADDING)
        bounds = metrics.boundingRect(
            QRect(0, 0, body_width, 10_000),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
            text,
        )
        estimated_height = TEXT_HEIGHT_PADDING + bounds.height()
        return min(max(floor, estimated_height), cap)

    @property
    def input_passthrough_enabled(self) -> bool:
        """Expose click-through state for tests and diagnostics."""

        return bool(getattr(self, "_input_passthrough_enabled", False))

    def apply_model(self, model: OcrTextWindowModel) -> None:
        """Update geometry and text from the latest OCR state."""

        max_w, max_h = max_window_width(), max_window_height()
        model.width = min(max(MIN_WINDOW_WIDTH, int(model.width)), max_w)
        model.height = min(max(MIN_WINDOW_HEIGHT, int(model.height)), max_h)
        self.model = model
        self.setWindowTitle(f"OCR {model.group_id}")
        self._base_x = model.x
        self._base_y = model.y
        self.ocr_label.setWordWrap(True)
        self.ocr_label.setMinimumWidth(0)
        self.ocr_label.setMaximumWidth(max(1, model.width - TEXT_WIDTH_PADDING))
        self._apply_screen_geometry()
        self.group_badge.setText(str(model.group_id))
        self.ocr_label.setText(model.text)
        self.set_paused(model.paused)
        self._apply_styles()
        self._update_corner_bar_position()
        self.update()

    def apply_edit_mode(self, enabled: bool, shift_y: int | None = None) -> None:
        """Show detached controls only while overlays are editable.

        ``shift_y`` is absolute (not cumulative) clearance from base position.
        """

        self._editable = bool(enabled)
        if enabled:
            if shift_y is not None:
                self._edit_shift_y = int(shift_y)
        else:
            self._edit_shift_y = 0
        self._apply_screen_geometry()
        if self._editable and not self._group_immersive:
            self.corner_bar.show()
            self._update_corner_bar_position()
        else:
            self.corner_bar.hide()
        set_window_click_through(self, not self._editable)

    def main_toolbar_clearance_source_height(self) -> int:
        """Placeholder; manager supplies main-toolbar height as shift_y."""

        return max(1, self.corner_bar.height() if self.corner_bar.height() else 28)

    def base_position(self) -> tuple[int, int]:
        return self._base_x, self._base_y

    def set_base_position(self, x: int, y: int) -> None:
        self._base_x = int(x)
        self._base_y = int(y)
        self._apply_screen_geometry()
        self._update_corner_bar_position()

    def _screen_for_caps(self):
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
        self.ocr_label.setMaximumWidth(max(1, width - TEXT_WIDTH_PADDING))
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)
        self.resize(width, height)
        self.move(self._base_x, self._base_y + self._edit_shift_y)

    def sizeHint(self) -> QSize:
        w, h = self._clamped_size(self.model.width, self.model.height)
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        width, height = self._clamped_size(self.width(), self.height())
        if self.width() != width or self.height() != height:
            self.model.width = width
            self.model.height = height
            self.setMinimumSize(width, height)
            self.setMaximumSize(width, height)
            self.resize(width, height)
            self.ocr_label.setMaximumWidth(max(1, width - TEXT_WIDTH_PADDING))

    def set_paused(self, paused: bool) -> None:
        self.pause_button.setText("")
        self.pause_button.setIcon(_media_control_icon(bool(paused)))
        self.pause_button.setToolTip("继续" if paused else "暂停")
        self.pause_button.setAccessibleName("继续 OCR" if paused else "暂停 OCR")
        self.pause_button.setProperty("isOn", bool(paused))
        self.pause_button.style().unpolish(self.pause_button)
        self.pause_button.style().polish(self.pause_button)

    def copy_text(self) -> None:
        """Copy the current OCR text to the clipboard."""

        QApplication.clipboard().setText(self.model.text)

    def set_body_immersive(self, on: bool) -> None:
        """Chrome off / text on without moving OCR body (mock is-body-hidden)."""

        self._body_immersive = bool(on)
        self.body_immersive_button.blockSignals(True)
        self.body_immersive_button.setChecked(on)
        self.body_immersive_button.setText("◉" if on else "⊘")
        self.body_immersive_button.setToolTip(
            "显示 OCR 框体" if on else "隐藏 OCR 框体，仅保留文字"
        )
        self.body_immersive_button.blockSignals(False)
        self.group_badge.setVisible(True)
        self.title_label.setVisible(True)
        transparent = "color: transparent; background: transparent; border: none;"
        if self._body_immersive or self._group_immersive:
            for widget in (
                self.group_badge,
                self.title_label,
            ):
                widget.setStyleSheet(transparent)
            self.header_widget.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
        else:
            self.group_badge.setStyleSheet("")
            self.title_label.setStyleSheet("")
            self.header_widget.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
            )
            self._apply_styles()
        if self._group_immersive:
            self.corner_bar.hide()
        elif self._editable:
            self.corner_bar.show()
            self._update_corner_bar_position()
        self.update()

    def _on_body_immersive_toggled(self, on: bool) -> None:
        self.set_body_immersive(on)
        self.body_immersive_toggled.emit(self.model.group_id, on)

    def set_group_immersive(self, on: bool) -> None:
        self._group_immersive = bool(on)
        if self._group_immersive:
            self.corner_bar.hide()
        elif self._editable:
            self.corner_bar.show()
            self._update_corner_bar_position()
        self.set_body_immersive(self._body_immersive)

    def set_corner_collapsed(self, collapsed: bool) -> None:
        self._corner_collapsed = bool(collapsed)
        self.corner_tools_widget.setVisible(not self._corner_collapsed)
        self.collapse_button.setText("‹" if self._corner_collapsed else "›")
        self.collapse_button.setToolTip("展开工具" if self._corner_collapsed else "收缩工具")
        self.corner_bar.adjustSize()
        self._update_corner_bar_position()

    def toggle_corner_collapsed(self) -> None:
        self.set_corner_collapsed(not self._corner_collapsed)

    @property
    def corner_collapsed(self) -> bool:
        return self._corner_collapsed

    @property
    def corner_controls_visible(self) -> bool:
        return self.corner_bar.isVisible()

    def _update_corner_bar_position(self) -> None:
        self.corner_bar.adjustSize()
        geo = self.frameGeometry()
        x = geo.right() - self.corner_bar.width() + 1
        y = geo.top() - self.corner_bar.height() - 4
        self.corner_bar.move(x, y)

    @property
    def body_immersive(self) -> bool:
        return self._body_immersive

    def body_text_global_top_left(self) -> QPoint:
        return self.ocr_label.mapToGlobal(QPoint(0, 0))

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
            self._base_x = self.x()
            self._base_y = self.y() - self._edit_shift_y
            self.model.x = self._base_x
            self.model.y = self._base_y
            self._update_corner_bar_position()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_screen_geometry()
        set_window_excluded_from_capture(self, True)
        set_window_excluded_from_capture(self.corner_bar, True)
        set_window_click_through(self, not self._editable)
        if self._editable and not self._group_immersive:
            self.corner_bar.show()
            self._update_corner_bar_position()

    def hideEvent(self, event) -> None:
        self.corner_bar.hide()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self.corner_bar.close()
        super().closeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        if self._body_immersive or self._group_immersive:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        # OCR is the quiet dark companion card in the mock.  The selection and
        # translation outlines already carry the group accent, so repeating a
        # bright blue outline here makes the overlay look fragmented.
        painter.setBrush(QColor(15, 23, 42, 204))
        border_color = QColor("#334155")
        border_color.setAlpha(160)
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(QRectF(rect), 10, 10)
        painter.end()

    def _apply_styles(self) -> None:
        if self._body_immersive or self._group_immersive:
            self.setStyleSheet(
                """
                QLabel#ocrBody {
                    color: #f8fafc;
                    border: none;
                    font-size: 14px;
                    line-height: 1.5;
                    background: transparent;
                }
                QWidget#ocrHeader {
                    background: transparent;
                    border: none;
                }
                """
            )
            self._apply_corner_styles()
            return
        self.setStyleSheet(
            """
            QLabel#ocrGroupBadge {
                background: rgba(255, 255, 255, 0.14);
                color: white;
                border: none;
                padding: 1px 6px;
                border-radius: 5px;
                font-size: 10px;
                font-weight: 700;
                max-width: 22px;
            }
            QLabel#ocrTitle {
                color: #64748b;
                border: none;
                font-size: 13px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#ocrBody {
                color: #f8fafc;
                border: none;
                font-size: 14px;
                line-height: 1.5;
                background: transparent;
            }
            QWidget#ocrHeader {
                background: transparent;
                border: none;
            }
            """
        )
        self._apply_corner_styles()

    def _apply_corner_styles(self) -> None:
        self.corner_bar.setStyleSheet(
            """
            QWidget#ocrCornerBar {
                background: transparent;
                border: none;
            }
            QWidget#ocrCornerTools {
                background: rgba(8, 12, 22, 0.82);
                border: 1px solid rgba(51, 65, 85, 0.65);
                border-radius: 999px;
            }
            QPushButton#ocrCornerButton {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 6px;
                padding: 0;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#ocrCornerButton:hover {
                background: rgba(51, 65, 85, 0.65);
                color: #f8fafc;
            }
            QPushButton#ocrCornerButton:checked,
            QPushButton#ocrCornerButton[isOn="true"] {
                background: rgba(37, 99, 235, 0.38);
                color: #bfdbfe;
                border: 1px solid rgba(147, 197, 253, 0.45);
            }
            QPushButton#ocrCornerButton[visibilityToggle="true"] {
                color: #cbd5e1;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton#ocrCornerButton[pauseControl="true"] {
                padding: 0;
            }
            QPushButton#ocrCornerCollapseButton {
                background: rgba(8, 12, 22, 0.82);
                color: #64748b;
                border: 1px solid rgba(51, 65, 85, 0.65);
                border-radius: 14px;
                padding: 0;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#ocrCornerCollapseButton:hover {
                color: #e2e8f0;
            }
            """
        )
