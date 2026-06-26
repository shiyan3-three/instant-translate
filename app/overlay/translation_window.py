"""Translation-window overlay bound to one selection group."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.overlay.window_interaction import set_window_click_through, set_window_excluded_from_capture
from app.state.group_state import ScreenRegion

WINDOW_MARGIN = 12
SCREEN_MARGIN = 8
DEFAULT_HEIGHT = 132
MIN_WINDOW_WIDTH = 260


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

    def __init__(self, model: TranslationWindowModel, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self.model = model
        self._editable = False
        self._dragging = False
        self._drag_origin_global = QPoint()
        self._drag_origin_top_left = QPoint()
        self._input_passthrough_enabled = False

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        set_window_excluded_from_capture(self, True)
        self.setMouseTracking(True)

        self.group_badge = QLabel(self)
        self.group_badge.setObjectName("translationGroupBadge")

        self.language_pair_label = QLabel(self)
        self.language_pair_label.setObjectName("translationLanguagePair")

        self.dock_controls_widget = QWidget(self)
        dock_layout = QHBoxLayout(self.dock_controls_widget)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(4)
        self.dock_up_button = QPushButton("\u4e0a")
        self.dock_down_button = QPushButton("\u4e0b")
        self.dock_left_button = QPushButton("\u5de6")
        self.dock_right_button = QPushButton("\u53f3")
        self.feedback_button = QPushButton("翻译有误")
        for button in (
            self.dock_up_button,
            self.dock_down_button,
            self.dock_left_button,
            self.dock_right_button,
            self.feedback_button,
        ):
            button.setObjectName("dockButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumSize(34, 28)
            dock_layout.addWidget(button)
        self.feedback_button.setMinimumWidth(56)
        self.dock_controls_widget.setMinimumWidth(216)

        self.dock_up_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "top"))
        self.dock_down_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "bottom"))
        self.dock_left_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "left"))
        self.dock_right_button.clicked.connect(lambda: self.dock_changed.emit(self.model.group_id, "right"))
        self.feedback_button.clicked.connect(lambda: self.feedback_requested.emit(self.model.group_id))

        self.translation_label = QLabel(self)
        self.translation_label.setObjectName("translationBody")
        self.translation_label.setWordWrap(True)
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        header_row.addWidget(self.group_badge, alignment=Qt.AlignmentFlag.AlignLeft)
        header_row.addWidget(self.language_pair_label, stretch=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addLayout(header_row)
        layout.addWidget(self.translation_label, stretch=1)

        self.apply_model(model)
        self.apply_edit_mode(False)
        self.setMinimumSize(MIN_WINDOW_WIDTH, 60)

    @staticmethod
    def compute_geometry(region: ScreenRegion, screen: QRect, preferred_dock: str) -> QRect:
        """Return a translation-window geometry matched to one region."""

        available_width = max(80, screen.width() - SCREEN_MARGIN * 2)
        width = min(max(region.width, MIN_WINDOW_WIDTH), available_width)
        height = DEFAULT_HEIGHT

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

    @property
    def dock_controls_visible(self) -> bool:
        """Expose dock controls visibility for tests and diagnostics."""

        return not self.dock_controls_widget.isHidden()

    def apply_model(self, model: TranslationWindowModel) -> None:
        """Update geometry and labels from the latest model state."""

        self.model = model
        self.setWindowTitle(f"Translation {model.group_id}")
        self.setGeometry(model.x, model.y, model.width, model.height)
        self.group_badge.setText(str(model.group_id))
        self.language_pair_label.setText(f"{model.source_language} => {model.target_language}")
        self.translation_label.setText(model.text)
        self._apply_styles()
        self.update()

    def apply_edit_mode(self, enabled: bool) -> None:
        """Switch between minimal display and edit display."""

        self._editable = enabled
        self.language_pair_label.setHidden(not enabled)
        self.dock_controls_widget.setHidden(True)
        set_window_click_through(self, not enabled)
        self._apply_styles()
        self.update()

    @property
    def input_passthrough_enabled(self) -> bool:
        """Expose click-through state for tests and diagnostics."""

        return bool(getattr(self, "_input_passthrough_enabled", False))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._editable
            and (
                self.dock_controls_widget.isHidden()
                or not self.dock_controls_widget.geometry().contains(event.position().toPoint())
            )
        ):
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
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.releaseMouse()
            self.moved.emit(self.model.group_id, self.x(), self.y())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        set_window_excluded_from_capture(self, True)
        set_window_click_through(self, not self._editable)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        fill_alpha = 235 if self._editable else 225
        painter.setBrush(QColor(255, 255, 255, fill_alpha))
        border_color = QColor(self.model.accent_color if self._editable else "#94A3B8")
        border_color.setAlpha(245 if self._editable else 180)
        painter.setPen(QPen(border_color, 2 if self._editable else 1))
        painter.drawRoundedRect(QRectF(rect), 12, 12)
        painter.end()

    def _apply_styles(self) -> None:
        pair_color = "#475569"
        badge_background = "rgba(71, 85, 105, 0.85)"
        self.setStyleSheet(
            f"""
            QLabel#translationGroupBadge {{
                background: {badge_background};
                color: white;
                border: none;
                padding: 2px 8px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 700;
                max-width: 24px;
            }}
            QLabel#translationLanguagePair {{
                color: {pair_color};
                border: none;
                font-size: 11px;
                font-weight: 600;
                background: transparent;
            }}
            QLabel#translationBody {{
                color: #1e293b;
                border: none;
                font-size: 18px;
                line-height: 1.5;
                background: transparent;
            }}
            QPushButton#dockButton {{
                background: rgba(15, 23, 42, 0.96);
                color: white;
                border: 1px solid rgba(148, 163, 184, 0.45);
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton#dockButton:hover {{
                background: rgba(30, 41, 59, 0.98);
            }}
            QPushButton#dockButton:pressed {{
                background: rgba(2, 6, 23, 0.98);
            }}
            """
        )
