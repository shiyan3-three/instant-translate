"""Visible fixed-region selection boxes."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QRegion
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from app.overlay.window_interaction import set_window_click_through

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
    source_language: str = "English"
    target_language: str = "中文"


class SelectionToolbarWidget(QWidget):
    """Floating toolbar shown while one selection box is in edit mode."""

    reselect_requested = Signal()
    delete_requested = Signal()
    pause_toggled = Signal()
    source_language_changed = Signal(str)
    target_language_changed = Signal(str)

    LANGUAGES = ["English", "中文", "日本語"]

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

        self.source_combo = QComboBox()
        self.source_combo.addItems(self.LANGUAGES)
        self.source_combo.setCurrentText("English")
        self.source_combo.setObjectName("toolbarCombo")

        self.target_combo = QComboBox()
        self.target_combo.addItems(self.LANGUAGES)
        self.target_combo.setCurrentText("中文")
        self.target_combo.setObjectName("toolbarCombo")

        self.reselect_button = QPushButton("重选")
        self.pause_button = QPushButton("暂停")
        self.delete_button = QPushButton("删除")

        layout.addWidget(self.source_combo)
        layout.addWidget(self.target_combo)

        for button in (self.reselect_button, self.pause_button, self.delete_button):
            button.setObjectName("toolbarButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)

        self.reselect_button.clicked.connect(self.reselect_requested.emit)
        self.pause_button.clicked.connect(self.pause_toggled.emit)
        self.delete_button.clicked.connect(self.delete_requested.emit)
        self.source_combo.currentTextChanged.connect(self.source_language_changed.emit)
        self.target_combo.currentTextChanged.connect(self.target_language_changed.emit)
        self._apply_styles()

    def set_paused(self, paused: bool) -> None:
        """Update button text to match pause state."""

        self.pause_button.setText("继续" if paused else "暂停")

    def set_languages(self, source: str, target: str) -> None:
        """Update combo boxes without re-emitting signals."""

        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentText(source)
        self.source_combo.blockSignals(False)
        self.target_combo.blockSignals(True)
        self.target_combo.setCurrentText(target)
        self.target_combo.blockSignals(False)

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
            QComboBox#toolbarCombo {{
                background: rgba(51, 65, 85, 0.96);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 4px 26px 4px 10px;
                font-size: 12px;
                font-weight: 600;
            }}
            QComboBox#toolbarCombo:hover {{
                background: rgba(71, 85, 105, 0.98);
            }}
            QComboBox#toolbarCombo QAbstractItemView {{
                background: rgba(30, 41, 59, 0.98);
                color: white;
                border: none;
                border-radius: 6px;
                selection-background-color: rgba(71, 85, 105, 0.98);
            }}
            QComboBox#toolbarCombo::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border: none;
            }}
            """
        )


class SelectionBoxWidget(QWidget):
    """Render a low-distraction always-on-top selection outline."""

    reselect_requested = Signal(int)
    delete_requested = Signal(int)
    pause_toggled = Signal(int)
    source_language_changed = Signal(int, str)
    target_language_changed = Signal(int, str)
    moved = Signal(int, int, int)
    resized = Signal(int, int, int, int, int)  # group_id, x, y, w, h

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
        self._resizing = False
        self._resize_edge: str = ""
        self._drag_origin_global = QPoint()
        self._drag_origin_top_left = QPoint()
        self._drag_origin_geometry = QRect()
        self._input_passthrough_enabled = False

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
        self.toolbar_panel.source_language_changed.connect(lambda lang: self.source_language_changed.emit(self.model.group_id, lang))
        self.toolbar_panel.target_language_changed.connect(lambda lang: self.target_language_changed.emit(self.model.group_id, lang))

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
        self.toolbar_panel.set_languages(model.source_language, model.target_language)
        self._update_overlay_geometry()
        self._apply_child_styles()
        self.update()

    def apply_edit_mode(self, enabled: bool) -> None:
        """Update visual emphasis and interaction mode for edit mode."""

        self._editable = enabled
        self.size_badge.setHidden(not enabled)

        # Win32 WS_EX_TRANSPARENT toggle — reliable at runtime unlike setWindowFlags
        set_window_click_through(self, not enabled)
        self._update_input_mask()

        if enabled:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()

        if enabled:
            self.toolbar_panel.show()
            self._update_toolbar_position()
        else:
            self.toolbar_panel.hide()

        self._apply_child_styles()
        self.update()

    @property
    def input_passthrough_enabled(self) -> bool:
        """Expose click-through state for tests and diagnostics."""

        return bool(getattr(self, "_input_passthrough_enabled", False))

    @property
    def body_hit_area_enabled(self) -> bool:
        """Return whether the transparent body is painted for mouse hit testing."""

        return self._editable and not self.input_passthrough_enabled

    def capture_with_chrome_hidden(self, callback):
        """Run a capture callback without badges or toolbar polluting OCR."""

        group_badge_hidden = self.group_badge.isHidden()
        size_badge_hidden = self.size_badge.isHidden()
        toolbar_visible = self.toolbar_panel.isVisible()

        self.group_badge.hide()
        self.size_badge.hide()
        self.toolbar_panel.hide()
        QApplication.processEvents()
        try:
            return callback()
        finally:
            self.group_badge.setHidden(group_badge_hidden)
            self.size_badge.setHidden(size_badge_hidden)
            self.toolbar_panel.setVisible(toolbar_visible)
            if toolbar_visible:
                self._update_toolbar_position()
            QApplication.processEvents()

    def _update_input_mask(self) -> None:
        """Keep edit-mode hit testing active across the whole selection body."""

        if self.body_hit_area_enabled:
            self.setMask(QRegion(self.rect()))
        else:
            self.clearMask()

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
        if self.body_hit_area_enabled:
            # A near-transparent fill keeps the body hittable without visually covering content.
            painter.setBrush(QColor(255, 255, 255, 1))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(rect), 10, 10)
        painter.end()

    _RESIZE_MARGIN = 8
    _MIN_SIZE = 24

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._editable:
            super().mousePressEvent(event)
            return

        pos = event.position().toPoint()
        edge = self._hit_test_edge(pos)

        if edge:
            self._resizing = True
            self._resize_edge = edge
            self._drag_origin_global = event.globalPosition().toPoint()
            self._drag_origin_geometry = self.geometry()
        else:
            self._dragging = True
            self._drag_origin_global = event.globalPosition().toPoint()
            self._drag_origin_top_left = self.frameGeometry().topLeft()

        self.grabMouse()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._editable:
            super().mouseMoveEvent(event)
            return

        if self._resizing:
            delta = event.globalPosition().toPoint() - self._drag_origin_global
            geo = QRect(self._drag_origin_geometry)
            edge = self._resize_edge

            if "left" in edge:
                geo.setLeft(min(geo.left() + delta.x(), geo.right() - self._MIN_SIZE))
            elif "right" in edge:
                geo.setRight(max(geo.right() + delta.x(), geo.left() + self._MIN_SIZE))
            if "top" in edge:
                geo.setTop(min(geo.top() + delta.y(), geo.bottom() - self._MIN_SIZE))
            elif "bottom" in edge:
                geo.setBottom(max(geo.bottom() + delta.y(), geo.top() + self._MIN_SIZE))

            self.setGeometry(geo)
            event.accept()
            return

        if self._dragging:
            delta = event.globalPosition().toPoint() - self._drag_origin_global
            self.move(self._drag_origin_top_left + delta)
            event.accept()
            return

        # Update cursor hint when hovering edges
        pos = event.position().toPoint()
        edge = self._hit_test_edge(pos)
        cursors = {
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "topleft": Qt.CursorShape.SizeFDiagCursor,
            "bottomright": Qt.CursorShape.SizeFDiagCursor,
            "topright": Qt.CursorShape.SizeBDiagCursor,
            "bottomleft": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(edge, Qt.CursorShape.SizeAllCursor))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._resizing:
            self._resizing = False
            self._resize_edge = ""
            self.releaseMouse()
            g = self.geometry()
            self.resized.emit(self.model.group_id, g.x(), g.y(), g.width(), g.height())
            event.accept()
            return

        if self._dragging:
            self._dragging = False
            self.releaseMouse()
            self.moved.emit(self.model.group_id, self.x(), self.y())
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _hit_test_edge(self, pos) -> str:
        """Return which edge the point is near, or empty string for interior."""

        m = self._RESIZE_MARGIN
        w, h = self.width(), self.height()
        left = pos.x() <= m
        right = pos.x() >= w - m
        top = pos.y() <= m
        bottom = pos.y() >= h - m

        if top and left:
            return "topleft"
        if top and right:
            return "topright"
        if bottom and left:
            return "bottomleft"
        if bottom and right:
            return "bottomright"
        if top:
            return "top"
        if bottom:
            return "bottom"
        if left:
            return "left"
        if right:
            return "right"
        return ""

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._update_toolbar_position()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_overlay_geometry()
        self._update_input_mask()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        set_window_click_through(self, not self._editable)
        self._update_input_mask()
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
