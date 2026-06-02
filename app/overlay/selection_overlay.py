"""Interactive full-screen overlay used to select a fixed screen region."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QApplication, QWidget

from app.overlay.selection_box import SelectionBoxWidget
from app.state.group_state import ScreenRegion


class RegionSelectionOverlay(QWidget):
    """Provide a temporary full-screen drag-selection overlay."""

    selection_completed = Signal(object)
    selection_cancelled = Signal()

    def __init__(self, app: QApplication, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self._app = app
        self._origin = QPoint()
        self._current = QPoint()
        self._dragging = False
        self._group_id: int | None = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    @staticmethod
    def build_region(start: QPoint, end: QPoint) -> ScreenRegion:
        """Normalize two points into a positive-width screen region."""

        x1 = min(start.x(), end.x())
        y1 = min(start.y(), end.y())
        x2 = max(start.x(), end.x())
        y2 = max(start.y(), end.y())
        return ScreenRegion(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @staticmethod
    def format_region_label(region: ScreenRegion) -> str:
        """Return the live size label shown during drag selection."""

        return SelectionBoxWidget.format_dimensions(region.width, region.height)

    def begin(self, group_id: int) -> None:
        """Show the overlay and prepare to capture one group region."""

        self._group_id = group_id
        self._dragging = False
        geometry = self._app.primaryScreen().virtualGeometry()
        self.setGeometry(geometry)
        self.show()
        self.raise_()
        self.activateWindow()
        self.repaint()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        point = event.globalPosition().toPoint()
        self._origin = point
        self._current = point
        self._dragging = True
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            return
        self._current = event.globalPosition().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return

        self._current = event.globalPosition().toPoint()
        self._dragging = False
        region = self.build_region(self._origin, self._current)
        self.hide()
        self.selection_completed.emit(region)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.selection_cancelled.emit()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 65))

        if self._dragging:
            rect = QRect(self.mapFromGlobal(self._origin), self.mapFromGlobal(self._current)).normalized()
            painter.setPen(QPen(QColor("#F2C94C"), 2, Qt.PenStyle.SolidLine))
            painter.fillRect(rect, QColor(242, 201, 76, 45))
            painter.drawRect(rect)

            label_text = self.format_region_label(
                ScreenRegion(x=rect.x(), y=rect.y(), width=rect.width(), height=rect.height())
            )
            label_rect = QRect(rect.x(), max(10, rect.y() - 34), 96, 24)
            painter.fillRect(label_rect, QColor(15, 23, 42, 220))
            painter.setPen(QColor("white"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)

        painter.end()
