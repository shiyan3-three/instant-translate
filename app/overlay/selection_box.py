"""Visible fixed-region selection boxes."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QWidget

TOOLBAR_MARGIN = 10


@dataclass
class SelectionBoxModel:
    """Describe one fixed screen region."""

    group_id: int
    x: int
    y: int
    width: int
    height: int
    accent_color: str = "#2F80ED"
    paused: bool = False


class SelectionToolbarWidget(QWidget):
    """Floating toolbar shown while one selection box is in edit mode."""

    reselect_requested = Signal()
    delete_requested = Signal()
    pause_toggled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self.setObjectName("selectionToolbarPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.reselect_button = QPushButton("重选")
        self.pause_button = QPushButton("暂停")
        self.delete_button = QPushButton("删除")
        for button in (self.reselect_button, self.pause_button, self.delete_button):
            button.setObjectName("toolbarButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)

        self.reselect_button.clicked.connect(self.reselect_requested.emit)
        self.pause_button.clicked.connect(self.pause_toggled.emit)
        self.delete_button.clicked.connect(self.delete_requested.emit)
        self._apply_styles()

    def set_paused(self, paused: bool) -> None:
        """Update button text to match pause state."""

        self.pause_button.setText("继续" if paused else "暂停")

    def _apply_styles(self) -> None:
        normal_button_style = """
            QPushButton {
                background: rgba(51, 65, 85, 0.96);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(71, 85, 105, 0.98);
            }
            QPushButton:pressed {
                background: rgba(30, 41, 59, 0.98);
            }
        """
        danger_button_style = """
            QPushButton {
                background: rgba(194, 65, 12, 0.92);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(219, 88, 30, 0.96);
            }
            QPushButton:pressed {
                background: rgba(170, 55, 10, 0.96);
            }
        """
        self.reselect_button.setStyleSheet(normal_button_style)
        self.pause_button.setStyleSheet(normal_button_style)
        self.delete_button.setStyleSheet(danger_button_style)
        self.setStyleSheet(
            """
            QWidget#selectionToolbarPanel {{
                background: rgba(15, 23, 42, 0.9);
                border: none;
                border-radius: 10px;
            }}
            """
        )


class SelectionBoxWidget(QWidget):
    """Render a low-distraction always-on-top selection outline."""

    reselect_requested = Signal(int)
    delete_requested = Signal(int)
    pause_toggled = Signal(int)
    moved = Signal(int, int, int)

    def __init__(self, model: SelectionBoxModel, parent: QWidget | None = None) -> None:
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

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)

        self.group_badge = QLabel(str(model.group_id), self)
        self.group_badge.setObjectName("groupBadge")

        self.size_badge = QLabel("", self)
        self.size_badge.setObjectName("sizeBadge")

        self.toolbar_panel = SelectionToolbarWidget()
        self.toolbar_panel.reselect_requested.connect(lambda: self.reselect_requested.emit(self.model.group_id))
        self.toolbar_panel.pause_toggled.connect(lambda: self.pause_toggled.emit(self.model.group_id))
        self.toolbar_panel.delete_requested.connect(lambda: self.delete_requested.emit(self.model.group_id))

        self.apply_model(model)
        self.apply_edit_mode(False)

    @staticmethod
    def format_dimensions(width: int, height: int) -> str:
        """Return a compact region size label."""

        return f"{width} x {height}"

    @property
    def outline_width(self) -> int:
        """Return the current border width used for painting."""

        return 4 if self._editable else 2

    @property
    def toolbar_visible(self) -> bool:
        """Expose toolbar visibility for tests and diagnostics."""

        return self.toolbar_panel.isVisible()

    @property
    def size_badge_visible(self) -> bool:
        """Expose size-label visibility for tests and diagnostics."""

        return not self.size_badge.isHidden()

    def apply_model(self, model: SelectionBoxModel) -> None:
        """Refresh geometry and color from the latest model state."""

        self.model = model
        self.setGeometry(model.x, model.y, model.width, model.height)
        self.group_badge.setText(str(model.group_id))
        self.size_badge.setText(self.format_dimensions(model.width, model.height))
        self.toolbar_panel.set_paused(model.paused)
        self._update_overlay_geometry()
        self._apply_child_styles()
        self.update()

    def apply_edit_mode(self, enabled: bool) -> None:
        """Update visual emphasis for edit mode."""

        self._editable = enabled
        self.size_badge.setHidden(not enabled)
        if enabled and self.isVisible():
            self.toolbar_panel.show()
            self._update_toolbar_position()
        else:
            self.toolbar_panel.hide()
        self._apply_child_styles()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(
            self.outline_width / 2,
            self.outline_width / 2,
            -self.outline_width / 2,
            -self.outline_width / 2,
        )
        pen = QPen(QColor(self.model.accent_color), self.outline_width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(rect), 10, 10)
        painter.end()

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

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._update_toolbar_position()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_overlay_geometry()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._editable:
            self.toolbar_panel.show()
            self._update_toolbar_position()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.toolbar_panel.hide()

    def closeEvent(self, event) -> None:
        self.toolbar_panel.close()
        super().closeEvent(event)

    def _update_overlay_geometry(self) -> None:
        self.group_badge.adjustSize()
        self.group_badge.move(10, 8)

        self.size_badge.adjustSize()
        preferred_x = self.width() - self.size_badge.width() - 10
        if preferred_x <= self.group_badge.x() + self.group_badge.width() + 8:
            self.size_badge.move(10, 36)
        else:
            self.size_badge.move(max(10, preferred_x), 8)

        self._update_toolbar_position()

    def _update_toolbar_position(self) -> None:
        if not self.toolbar_panel.isVisible() and not self._editable:
            return

        self.toolbar_panel.adjustSize()
        screen = QApplication.primaryScreen().virtualGeometry()
        frame = self.frameGeometry()
        toolbar_width = self.toolbar_panel.sizeHint().width()
        toolbar_height = self.toolbar_panel.sizeHint().height()

        x = frame.x()
        max_x = screen.x() + screen.width() - toolbar_width - 8
        x = min(max(x, screen.x() + 8), max_x)

        above_y = frame.y() - toolbar_height - TOOLBAR_MARGIN
        below_y = frame.y() + frame.height() + TOOLBAR_MARGIN
        if above_y >= screen.y() + 8:
            y = above_y
        else:
            y = min(screen.y() + screen.height() - toolbar_height - 8, below_y)

        self.toolbar_panel.move(x, y)

    def _apply_child_styles(self) -> None:
        badge_background = "rgba(15, 23, 42, 0.68)"
        size_background = "rgba(15, 23, 42, 0.84)"
        self.setStyleSheet(
            f"""
            QLabel#groupBadge {{
                background: {badge_background};
                color: white;
                border: none;
                padding: 2px 8px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#sizeBadge {{
                background: {size_background};
                color: white;
                border: none;
                padding: 2px 8px;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            """
        )
