import numpy as np
from typing import List, Dict

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QGraphicsPathItem
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QBrush, QPen, QPainterPath, QCursor, QPixmap, QImage, QPainter

from app.ui.commands.brush import BrushStrokeCommand, ClearBrushStrokesCommand, \
                            SegmentBoxesCommand, EraseUndoCommand
from app.ui.commands.base import PathCommandBase as pcb
import imkit as imk
from modules.utils.inpaint_strokes import (
    MANUAL_STROKE_ROLES,
    STORABLE_STROKE_ROLES,
    STROKE_ROLE_ADD,
    STROKE_ROLE_EXCLUDE,
    STROKE_ROLE_GENERATED,
    STROKE_ROLE_RESTORE_PREVIEW,
    normalize_stroke_role,
)


class DrawingManager:
    """Manages all drawing-related tools and state."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._scene = viewer._scene

        self.add_stroke_color = QColor(255, 0, 0, 100)
        self.exclude_stroke_color = QColor(0, 140, 255, 110)
        self.restore_preview_color = QColor(24, 168, 82, 120)
        self.brush_size = 25
        self.eraser_size = 25
        
        self.brush_cursor = self.create_inpaint_cursor('brush', self.brush_size)
        self.eraser_cursor = self.create_inpaint_cursor('eraser', self.eraser_size)
        self.exclude_cursor = self.create_inpaint_cursor('exclude', self.brush_size)
        self.restore_cursor = self.create_inpaint_cursor('restore', self.brush_size)

        self.current_path = None
        self.current_path_item = None
        self.current_restore_page_index = None
        
        self.before_erase_state = []
        self.after_erase_state = []

    def _iter_path_items(self):
        photo_item = getattr(self.viewer, 'photo', None)
        for item in self._scene.items():
            if isinstance(item, QGraphicsPathItem) and item != photo_item and hasattr(item, 'path'):
                role = normalize_stroke_role(
                    item.data(pcb.ROLE_KEY),
                    brush=item.brush().color().name(QColor.HexArgb),
                )
                yield item, role

    def _make_path_item(self, role: str) -> QGraphicsPathItem:
        if role == STROKE_ROLE_EXCLUDE:
            color = self.exclude_stroke_color
        elif role == STROKE_ROLE_RESTORE_PREVIEW:
            color = self.restore_preview_color
        else:
            color = self.add_stroke_color

        pen = QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        item = self._scene.addPath(self.current_path, pen)
        item.setZValue(0.8)
        item.setData(pcb.ROLE_KEY, role)
        return item

    def start_stroke(self, scene_pos: QPointF):
        """Starts a new drawing or erasing stroke."""
        self.viewer.drawing_path = QPainterPath() # drawing_path is on viewer in original
        self.viewer.drawing_path.moveTo(scene_pos)
        
        self.current_path = QPainterPath()
        self.current_path.moveTo(scene_pos)

        if self.viewer.current_tool == 'brush':
            self.current_path_item = self._make_path_item(STROKE_ROLE_ADD)
        elif self.viewer.current_tool == 'exclude':
            self.current_path_item = self._make_path_item(STROKE_ROLE_EXCLUDE)
        elif self.viewer.current_tool == 'restore':
            self.current_path_item = self._make_path_item(STROKE_ROLE_RESTORE_PREVIEW)
            self.current_restore_page_index = None
            if self.viewer.webtoon_mode:
                page_index, _ = self.viewer.scene_to_page_coordinates(scene_pos)
                self.current_restore_page_index = page_index
        elif self.viewer.current_tool == 'eraser':
            # Capture the current state before starting erase operation
            self.before_erase_state = []
            try:
                for item, role in self._iter_path_items():
                    if role not in STORABLE_STROKE_ROLES:
                        continue
                    props = pcb.save_path_properties(item)
                    if props:
                        self.before_erase_state.append(props)
            except Exception as e:
                print(f"Warning: Error capturing before_erase_state: {e}")
                import traceback
                traceback.print_exc()
                self.before_erase_state = []

    def continue_stroke(self, scene_pos: QPointF):
        """Continues an existing drawing or erasing stroke."""
        if not self.current_path:
            return

        self.current_path.lineTo(scene_pos)
        if self.viewer.current_tool in {'brush', 'exclude', 'restore'} and self.current_path_item:
            self.current_path_item.setPath(self.current_path)
        elif self.viewer.current_tool == 'eraser':
            self.erase_at(scene_pos)

    def end_stroke(self):
        """Finalizes the current stroke and creates an undo command."""
        if self.current_path_item:
            if self.viewer.current_tool in {'brush', 'exclude'}:
                command = BrushStrokeCommand(self.viewer, self.current_path_item)
                self.viewer.command_emitted.emit(command)
            elif self.viewer.current_tool == 'restore':
                payload = {
                    'path': QPainterPath(self.current_path),
                    'width': int(self.brush_size),
                }
                if self.current_restore_page_index is not None:
                    payload['page_index'] = int(self.current_restore_page_index)
                self._scene.removeItem(self.current_path_item)
                self.viewer.restore_stroke_requested.emit(payload)

        if self.viewer.current_tool == 'eraser':
            # Capture the current state after erase operation
            self.after_erase_state = []
            try:
                for item, role in self._iter_path_items():
                    if role not in STORABLE_STROKE_ROLES:
                        continue
                    props = pcb.save_path_properties(item)
                    if props:
                        self.after_erase_state.append(props)
            except Exception as e:
                print(f"Warning: Error capturing after_erase_state: {e}")
                import traceback
                traceback.print_exc()
                self.after_erase_state = []
            
            # Only create undo command if we have valid before/after states
            if hasattr(self, 'before_erase_state'):
                # Create copies of the lists before passing them to avoid clearing issues
                before_copy = list(self.before_erase_state)
                after_copy = list(self.after_erase_state)
                command = EraseUndoCommand(self.viewer, before_copy, after_copy)
                self.viewer.command_emitted.emit(command)
                self.before_erase_state.clear()
                self.after_erase_state.clear()
            else:
                print("Warning: No before_erase_state found, skipping undo command creation")
        
        self.current_path = None
        self.current_path_item = None
        self.current_restore_page_index = None
        self.viewer.drawing_path = None

    def erase_at(self, pos: QPointF):
        erase_path = QPainterPath()
        erase_path.addEllipse(pos, self.eraser_size, self.eraser_size)

        for item in self._scene.items(erase_path):
            if isinstance(item, QGraphicsPathItem) and item != self.viewer.photo:
                self._erase_item_path(item, erase_path, pos)

    def _erase_item_path(self, item, erase_path, pos):
        path = item.path()
        new_path = QPainterPath()
        
        role = normalize_stroke_role(
            item.data(pcb.ROLE_KEY),
            brush=item.brush().color().name(QColor.HexArgb),
        )
        if role == STROKE_ROLE_RESTORE_PREVIEW:
            return

        if role == STROKE_ROLE_GENERATED:
            # Map erase shape into item's local coordinates to ensure robust boolean ops
            try:
                local_erase_path = item.mapFromScene(erase_path)
            except Exception:
                # Fallback: translate by item position if mapping isn't available
                local_erase_path = QPainterPath(erase_path)
                local_erase_path.translate(-item.pos().x(), -item.pos().y())

            # Ensure consistent fill rule for robust subtraction of filled polygons
            path.setFillRule(Qt.FillRule.WindingFill)
            local_erase_path.setFillRule(Qt.FillRule.WindingFill)

            result = path.subtracted(local_erase_path)
            if not result.isEmpty():
                new_path = result
        else: # Human-drawn stroke
            element_count = path.elementCount()
            i = 0
            while i < element_count:
                e = path.elementAt(i)
                point = QPointF(e.x, e.y)
                if not erase_path.contains(point):
                    if e.type == QPainterPath.ElementType.MoveToElement: new_path.moveTo(point)
                    elif e.type == QPainterPath.ElementType.LineToElement: new_path.lineTo(point)
                    elif e.type == QPainterPath.ElementType.CurveToElement:
                        if i + 2 < element_count:
                            c1, c2 = path.elementAt(i + 1), path.elementAt(i + 2)
                            c1_p, c2_p = QPointF(c1.x, c1.y), QPointF(c2.x, c2.y)
                            if not (erase_path.contains(c1_p) or erase_path.contains(c2_p)):
                                new_path.cubicTo(point, c1_p, c2_p)
                        i += 2
                else:
                    if (i + 1) < element_count:
                        next_e = path.elementAt(i + 1)
                        next_p = QPointF(next_e.x, next_e.y)
                        if not erase_path.contains(next_p):
                            new_path.moveTo(next_p)
                            if next_e.type == QPainterPath.ElementType.CurveToDataElement:
                                i += 2
                i += 1

        if new_path.isEmpty():
            self._scene.removeItem(item)
        else:
            item.setPath(new_path)

    def set_brush_size(self, size, scaled_size):
        self.brush_size = size
        self.brush_cursor = self.create_inpaint_cursor("brush", scaled_size)
        self.exclude_cursor = self.create_inpaint_cursor("exclude", scaled_size)
        self.restore_cursor = self.create_inpaint_cursor("restore", scaled_size)

    def set_eraser_size(self, size, scaled_size):
        self.eraser_size = size
        self.eraser_cursor = self.create_inpaint_cursor("eraser", scaled_size)

    def create_inpaint_cursor(self, cursor_type, size):
        size = max(1, size)
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        if cursor_type == "brush":
            painter.setBrush(QBrush(QColor(255, 0, 0, 127)))
            painter.setPen(Qt.PenStyle.NoPen)
        elif cursor_type == "exclude":
            painter.setBrush(QBrush(QColor(0, 140, 255, 127)))
            painter.setPen(Qt.PenStyle.NoPen)
        elif cursor_type == "restore":
            painter.setBrush(QBrush(QColor(24, 168, 82, 127)))
            painter.setPen(Qt.PenStyle.NoPen)
        elif cursor_type == "eraser":
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
            painter.setPen(QColor(0, 0, 0, 127))
        else:
            painter.setBrush(QBrush(QColor(0, 0, 0, 127)))
            painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, (size - 1), (size - 1))
        painter.end()
        return QCursor(pixmap, size // 2, size // 2)
    
    def save_brush_strokes(self) -> List[Dict]:
        strokes = []
        
        # Also collect any currently visible strokes
        for item, role in self._iter_path_items():
            if role not in STORABLE_STROKE_ROLES:
                continue
            strokes.append({
                'path': item.path(),
                'pen': item.pen().color().name(QColor.HexArgb),
                'brush': item.brush().color().name(QColor.HexArgb),
                'width': item.pen().width(),
                'role': role,
            })
        return strokes

    def load_brush_strokes(self, strokes: List[Dict]):
        self.clear_brush_strokes(page_switch=True)
        for stroke in reversed(strokes):
            role = normalize_stroke_role(stroke.get('role'), brush=stroke.get('brush'))
            if role not in STORABLE_STROKE_ROLES:
                continue
            pen = QPen()
            pen.setColor(QColor(stroke['pen']))
            pen.setWidth(stroke['width'])
            pen.setStyle(Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            brush = QBrush(QColor(stroke['brush']))
            if role == STROKE_ROLE_GENERATED:
                item = self._scene.addPath(stroke['path'], pen, brush)
            else:
                item = self._scene.addPath(stroke['path'], pen)
            item.setData(pcb.ROLE_KEY, role)
                
    def clear_brush_strokes(self, page_switch=False):
        if page_switch:      
            items_to_remove = [
                item
                for item, role in self._iter_path_items()
                if role in STORABLE_STROKE_ROLES or role == STROKE_ROLE_RESTORE_PREVIEW
            ]
            for item in items_to_remove:
                self._scene.removeItem(item)
            self._scene.update()
        else:
            command = ClearBrushStrokesCommand(self.viewer, roles=MANUAL_STROKE_ROLES)
            self.viewer.command_emitted.emit(command)

    def has_drawn_elements(self):
        for _item, role in self._iter_path_items():
            if role in STORABLE_STROKE_ROLES:
                return True
        return False

    def generate_mask_from_strokes(self):
        if not self.viewer.hasPhoto():
            return None

        if not self.has_drawn_elements():
            return None

        is_webtoon_mode = self.viewer.webtoon_mode
        if is_webtoon_mode:
            visible_image, mappings = self.viewer.get_visible_area_image()
            if visible_image is None:
                return None
            height, width = visible_image.shape[:2]
        else:
            image_rect = self.viewer.photo.boundingRect()
            width, height = int(image_rect.width()), int(image_rect.height())

        if width <= 0 or height <= 0:
            return None

        add_qimg = QImage(width, height, QImage.Format_Grayscale8)
        generated_qimg = QImage(width, height, QImage.Format_Grayscale8)
        exclude_qimg = QImage(width, height, QImage.Format_Grayscale8)
        add_qimg.fill(0)
        generated_qimg.fill(0)
        exclude_qimg.fill(0)

        add_painter = QPainter(add_qimg)
        generated_painter = QPainter(generated_qimg)
        exclude_painter = QPainter(exclude_qimg)

        if is_webtoon_mode:
            visible_image, mappings = self.viewer.get_visible_area_image()
            if mappings:
                visible_scene_top = mappings[0]['scene_y_start']
                add_painter.translate(0, -visible_scene_top)
                generated_painter.translate(0, -visible_scene_top)
                exclude_painter.translate(0, -visible_scene_top)

        add_painter.setBrush(QBrush(QColor(255, 255, 255)))
        generated_painter.setBrush(QBrush(QColor(255, 255, 255)))
        exclude_painter.setBrush(QBrush(QColor(255, 255, 255)))
        generated_painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))

        for item, role in self._iter_path_items():
            if role not in STORABLE_STROKE_ROLES:
                continue
            if role == STROKE_ROLE_GENERATED:
                painter = generated_painter
                painter.save()
                painter.translate(item.pos())
                painter.drawPath(item.path())
                painter.restore()
                continue

            painter = exclude_painter if role == STROKE_ROLE_EXCLUDE else add_painter
            painter.setPen(
                QPen(
                    QColor(255, 255, 255),
                    max(1, item.pen().width()),
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )
            painter.save()
            painter.translate(item.pos())
            painter.drawPath(item.path())
            painter.restore()

        add_painter.end()
        generated_painter.end()
        exclude_painter.end()

        def qimage_to_np(qimg):
            if qimg.width() <= 0 or qimg.height() <= 0:
                return np.zeros((max(1, qimg.height()), max(1, qimg.width())), dtype=np.uint8)

            ptr = qimg.constBits()
            arr = np.array(ptr).reshape(qimg.height(), qimg.bytesPerLine())
            return arr[:, :qimg.width()]

        add_mask = qimage_to_np(add_qimg)
        generated_mask = qimage_to_np(generated_qimg)
        exclude_mask = qimage_to_np(exclude_qimg)

        kernel = np.ones((5, 5), np.uint8)
        add_mask = imk.dilate(add_mask, kernel, iterations=2)
        generated_mask = imk.dilate(generated_mask, kernel, iterations=3)
        exclude_mask = imk.dilate(exclude_mask, kernel, iterations=2)

        include_mask = np.where((add_mask > 0) | (generated_mask > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(include_mask) == 0:
            return None

        final_mask = np.where((include_mask > 0) & ~(exclude_mask > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(final_mask) == 0:
            return None
        return final_mask
    
    def draw_segmentation_lines(self, bboxes):
        stroke = self.make_segmentation_stroke_data(bboxes)
        if stroke is None:
            return

        # Wrap in one GraphicsPathItem & emit
        fill_color = QtGui.QColor(255, 0, 0, 128)  # Semi-transparent red
        outline_color = QtGui.QColor(255, 0, 0)    # Solid red
        item = QtWidgets.QGraphicsPathItem(stroke['path'])
        item.setPen(QtGui.QPen(outline_color, 2, QtCore.Qt.SolidLine))
        item.setBrush(QtGui.QBrush(fill_color))
        item.setData(pcb.ROLE_KEY, STROKE_ROLE_GENERATED)

        self.viewer.command_emitted.emit(SegmentBoxesCommand(self.viewer, [item]))
        
        # Ensure the rectangles are visible
        self.viewer._scene.update()

    def make_segmentation_stroke_data(self, bboxes):
        if not bboxes:
            return None

        # 1) Compute tight ROI so mask stays small
        xs = [x for x1, _, x2, _ in bboxes for x in (x1, x2)]
        ys = [y for _, y1, _, y2 in bboxes for y in (y1, y2)]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))
        w, h = max_x - min_x + 1, max_y - min_y + 1

        # 2) Down-sample factor to cap mask size
        long_edge = 2048
        ds = max(1.0, max(w, h) / long_edge)
        mw, mh = int(w / ds) + 2, int(h / ds) + 2

        # 3) Paint the true bboxes
        mask = np.zeros((mh, mw), np.uint8)
        for x1, y1, x2, y2 in bboxes:
            x1i = int((x1 - min_x) / ds)
            y1i = int((y1 - min_y) / ds)
            x2i = int((x2 - min_x) / ds)
            y2i = int((y2 - min_y) / ds)
            mask = imk.rectangle(mask, (x1i, y1i), (x2i, y2i), 255, -1)

        # 4) Morphological closing to bridge small gaps
        ksize = 15
        kernel = imk.get_structuring_element(imk.MORPH_RECT, (ksize, ksize))
        mask_closed = imk.morphology_ex(mask, imk.MORPH_CLOSE, kernel)

        # 5) Build merged contour path
        contours, _ = imk.find_contours(mask_closed)
        if not contours:
            return None

        path = QtGui.QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        for cnt in contours:
            pts = cnt.squeeze(1)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            x0, y0 = pts[0]
            path.moveTo(x0 * ds + min_x, y0 * ds + min_y)
            for x, y in pts[1:]:
                path.lineTo(x * ds + min_x, y * ds + min_y)
            path.closeSubpath()

        if path.isEmpty():
            return None

        return {
            'path': path,
            'pen': QColor(255, 0, 0).name(QColor.HexArgb),
            'brush': QColor(255, 0, 0, 128).name(QColor.HexArgb),
            'width': 2,
            'role': STROKE_ROLE_GENERATED,
        }

    # def draw_segmentation_lines(self, bboxes, layers: int = 1, scale_factor: float = 1.0):
    #     if not self.viewer.hasPhoto() or not bboxes: return
        
    #     all_points = np.array(bboxes).reshape(-1, 2)
    #     centroid = np.mean(all_points, axis=0)

    #     scaled_segments = []
    #     for x1, y1, x2, y2 in bboxes:
    #         p1 = (np.array([x1, y1]) - centroid) * scale_factor + centroid
    #         p2 = (np.array([x2, y2]) - centroid) * scale_factor + centroid
    #         scaled_segments.append((*p1, *p2))
            
    #     fill_color = QColor(255, 0, 0, 128)
    #     outline_color = QColor(255, 0, 0)
    #     pen = QPen(outline_color, 2, Qt.SolidLine)
    #     brush = QBrush(fill_color)

    #     items = []
    #     for _ in range(layers):
    #         for x1, y1, x2, y2 in scaled_segments:
    #             path = QPainterPath()
    #             path.addRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
    #             path_item = QGraphicsPathItem(path)
    #             path_item.setPen(pen)
    #             path_item.setBrush(brush)
    #             items.append(path_item)
        
    #     if items:
    #         command = SegmentBoxesCommand(self.viewer, items)
    #         self.viewer.command_emitted.emit(command)
