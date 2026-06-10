"""OCR text viewer overlay bound to one selection group."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.overlay.window_interaction import set_window_click_through
from app.state.group_state import ScreenRegion

WINDOW_MARGIN = 12
SCREEN_MARGIN = 8
DEFAULT_HEIGHT = 116


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


class OcrTextWindowWidget(QWidget):
    """Render the latest cleaned OCR text for one selection group."""

    def __init__(self, model: OcrTextWindowModel, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self.model = model
        self._dragging = False
        self._drag_origin_global = QPoint()
        self._drag_origin_top_left = QPoint()
        self._input_passthrough_enabled = False

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)

        self.group_badge = QLabel(self)
        self.group_badge.setObjectName("ocrGroupBadge")

        self.title_label = QLabel("OCR")
        self.title_label.setObjectName("ocrTitle")

        self.copy_button = QPushButton("\u590d\u5236")
        self.copy_button.setObjectName("ocrCopyButton")
        self.copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_text)

        self.ocr_label = QLabel(self)
        self.ocr_label.setObjectName("ocrBody")
        self.ocr_label.setWordWrap(True)
        self.ocr_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self.group_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        header_row.addWidget(self.title_label, stretch=1)
        header_row.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignRight)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addLayout(header_row)
        layout.addWidget(self.ocr_label, stretch=1)

        self.apply_model(model)
        self.setMinimumSize(160, 72)
        set_window_click_through(self, False)

    @staticmethod
    def compute_geometry(region: ScreenRegion, screen: QRect) -> QRect:
        """Return an OCR-viewer geometry near one selection region."""

        width = min(region.width, max(80, screen.width() - SCREEN_MARGIN * 2))
        height = DEFAULT_HEIGHT

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

    @property
    def input_passthrough_enabled(self) -> bool:
        """Expose click-through state for tests and diagnostics."""

        return bool(getattr(self, "_input_passthrough_enabled", False))

    def apply_model(self, model: OcrTextWindowModel) -> None:
        """Update geometry and text from the latest OCR state."""

        self.model = model
        self.setWindowTitle(f"OCR {model.group_id}")
        self.setGeometry(model.x, model.y, model.width, model.height)
        self.group_badge.setText(str(model.group_id))
        self.ocr_label.setText(model.text)
        self._apply_styles()
        self.update()

    def copy_text(self) -> None:
        """Copy the current OCR text to the clipboard."""

        QApplication.clipboard().setText(self.model.text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self.copy_button.geometry().contains(event.position().toPoint())
        ):
            self._dragging = True
            self._drag_origin_global = event.globalPosition().toPoint()
            self._drag_origin_top_left = self.frameGeometry().topLeft()
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            delta = event.globalPosition().toPoint() - self._drag_origin_global
            self.move(self._drag_origin_top_left + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.releaseMouse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        set_window_click_through(self, False)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(QColor(15, 23, 42, 224))
        border_color = QColor(self.model.accent_color)
        border_color.setAlpha(220)
        painter.setPen(QPen(border_color, 1.5))
        painter.drawRoundedRect(QRectF(rect), 12, 12)
        painter.end()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QLabel#ocrGroupBadge {
                background: rgba(255, 255, 255, 0.14);
                color: white;
                border: none;
                padding: 2px 8px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
                max-width: 24px;
            }
            QLabel#ocrTitle {
                color: #cbd5e1;
                border: none;
                font-size: 11px;
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
            QPushButton#ocrCopyButton {
                background: rgba(255, 255, 255, 0.12);
                color: white;
                border: 1px solid rgba(226, 232, 240, 0.28);
                border-radius: 7px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton#ocrCopyButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton#ocrCopyButton:pressed {
                background: rgba(255, 255, 255, 0.1);
            }
            """
        )
