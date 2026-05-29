from __future__ import annotations

import os
import unittest

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import QPointF

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import TextBlockItem, TextBlockState
from app.ui.commands.box import DeleteTextBoxCommand, TextBoxChangeCommand
from modules.utils.textblock import TextBlock


class _FakeViewer:
    def __init__(self):
        self._scene = QtWidgets.QGraphicsScene()
        self.text_items: list[TextBlockItem] = []

    def add_text_item(self, properties):
        if isinstance(properties, dict):
            properties = TextItemProperties.from_dict(properties)
        item = TextBlockItem(
            text=properties.text,
            font_family=properties.font_family,
            font_size=properties.font_size,
            render_color=properties.text_color,
            alignment=properties.alignment,
            line_spacing=properties.line_spacing,
            outline_color=properties.outline_color,
            outline_width=properties.outline_width,
            bold=properties.bold,
            italic=properties.italic,
            underline=properties.underline,
            direction=properties.direction,
            vertical_alignment=properties.vertical_alignment,
            source_rect=properties.source_rect,
            block_anchor=properties.block_anchor,
            block_id=properties.block_id,
            editor_frame=properties.editor_frame,
        )
        if properties.width is not None:
            item.set_text(properties.text, properties.width)
        item.setPos(QPointF(*properties.position))
        item.setRotation(properties.rotation)
        item.setScale(properties.scale)
        if properties.transform_origin:
            item.setTransformOriginPoint(QPointF(*properties.transform_origin))
        self._scene.addItem(item)
        self.text_items.append(item)
        return item


class _FakeTextController:
    def __init__(self):
        self.clear_count = 0

    def clear_text_edits(self):
        self.clear_count += 1


class _FakeMain:
    def __init__(self):
        self.image_viewer = _FakeViewer()
        self.blk_list = []
        self.curr_tblock = None
        self.curr_tblock_item = None
        self.text_ctrl = _FakeTextController()


class TextBoxCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_text_box_change_command_restores_source_rect_and_block_id(self) -> None:
        main = _FakeMain()
        item = TextBlockItem(
            text="hello",
            font_size=12,
            source_rect=(10, 20, 80, 40),
            block_anchor=(10, 20, 80, 40),
            block_id="manual-1",
            editor_frame=True,
        )
        item.setTextWidth(80)
        item.setPos(QPointF(10, 20))
        main.image_viewer._scene.addItem(item)
        main.image_viewer.text_items.append(item)

        old_state = TextBlockState.from_item(item)
        new_state = TextBlockState(
            rect=(30, 45, 140, 105),
            rotation=12.0,
            transform_origin=QPointF(8, 9),
            block_id="manual-1",
            font_size=18.0,
            text_width=110.0,
            source_rect=(30, 45, 110, 60),
        )

        command = TextBoxChangeCommand(main, old_state, new_state)
        command.redo()

        self.assertEqual(item.block_id, "manual-1")
        self.assertEqual(tuple(item.source_rect), (30.0, 45.0, 110.0, 60.0))
        self.assertAlmostEqual(item.pos().x(), 30.0)
        self.assertAlmostEqual(item.pos().y(), 45.0)
        self.assertAlmostEqual(item.textWidth(), 110.0)
        self.assertAlmostEqual(item.font_size, 18.0)
        self.assertAlmostEqual(item.rotation(), 12.0)

        command.undo()

        self.assertEqual(item.block_id, "manual-1")
        self.assertEqual(tuple(item.source_rect), (10.0, 20.0, 80.0, 40.0))
        self.assertAlmostEqual(item.pos().x(), 10.0)
        self.assertAlmostEqual(item.pos().y(), 20.0)

    def test_delete_text_box_command_round_trips_manual_block(self) -> None:
        main = _FakeMain()
        blk = TextBlock(
            text_bbox=np.array([10, 20, 90, 60]),
            text="",
            translation="",
        )
        blk.block_id = "manual-delete"
        blk.manual_text_box = True
        main.blk_list.append(blk)
        main.curr_tblock = blk

        item = TextBlockItem(
            text="manual",
            font_size=14,
            source_rect=(10, 20, 80, 40),
            block_anchor=(10, 20, 80, 40),
            block_id="manual-delete",
            editor_frame=True,
        )
        item.setPos(QPointF(10, 20))
        main.image_viewer._scene.addItem(item)
        main.image_viewer.text_items.append(item)
        main.curr_tblock_item = item

        command = DeleteTextBoxCommand(main, item, blk, main.blk_list)
        command.redo()

        self.assertNotIn(item, main.image_viewer._scene.items())
        self.assertNotIn(item, main.image_viewer.text_items)
        self.assertEqual(main.blk_list, [])
        self.assertIsNone(main.curr_tblock_item)
        self.assertEqual(main.text_ctrl.clear_count, 1)

        command.undo()

        self.assertEqual(len(main.blk_list), 1)
        self.assertEqual(getattr(main.blk_list[0], "block_id", ""), "manual-delete")
        restored_items = [
            scene_item
            for scene_item in main.image_viewer._scene.items()
            if isinstance(scene_item, TextBlockItem)
        ]
        self.assertEqual(len(restored_items), 1)
        self.assertEqual(restored_items[0].block_id, "manual-delete")
        self.assertIn(restored_items[0], main.image_viewer.text_items)


if __name__ == "__main__":
    unittest.main()
