import os
from collections import deque
from typing import Set

import imkit as imk
from PIL import Image
from PySide6.QtCore import QTimer, QThread, QObject, Signal, QSize, QPoint, Qt, Slot
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QListWidget

from app.path_materialization import ensure_path_materialized


class ImageLoadWorker(QObject):
    """Worker thread for loading images in the background."""

    image_loaded = Signal(int, str, QImage)  # index, file_path, image

    def __init__(self):
        super().__init__()
        self.load_queue = deque()
        self._queued_indices: set[int] = set()
        self.should_stop = False
        self._processing = False

    @Slot(int, str, QSize)
    def add_to_queue(self, index: int, file_path: str, target_size: QSize):
        """Add an image to the loading queue."""
        if self.should_stop or index in self._queued_indices:
            return
        self.load_queue.append((index, file_path, target_size))
        self._queued_indices.add(index)
        if not self._processing:
            self.process_queue()

    @Slot()
    def clear_queue(self):
        """Clear the loading queue."""
        self.load_queue.clear()
        self._queued_indices.clear()

    @Slot()
    def stop(self):
        """Stop the worker."""
        self.should_stop = True
        self.clear_queue()

    @Slot()
    def process_queue(self):
        """Process the loading queue."""
        if self._processing or self.should_stop:
            return

        self._processing = True
        try:
            while self.load_queue and not self.should_stop:
                index, file_path, target_size = self.load_queue.popleft()
                self._queued_indices.discard(index)
                image = self._load_and_resize_image(file_path, target_size)
                if image is not None and not image.isNull():
                    self.image_loaded.emit(index, file_path, image)
        finally:
            self._processing = False

    def _load_and_resize_image(self, file_path: str, target_size: QSize) -> QImage | None:
        """Load and resize an image to the target size."""
        try:
            ensure_path_materialized(file_path)
            image = imk.read_image(file_path)
            if image is None:
                return None

            height, width = image.shape[:2]
            target_width, target_height = target_size.width(), target_size.height()
            scale_x = target_width / width
            scale_y = target_height / height
            scale = min(scale_x, scale_y)

            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = imk.resize(image, (new_width, new_height), mode=Image.Resampling.LANCZOS)

            h, w, ch = resized_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(resized_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return qt_image.copy()
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            return None


class ListViewImageLoader(QObject):
    """Lazy image loader for QListWidget that loads thumbnails only when visible."""

    queue_image_load_requested = Signal(int, str, QSize)
    clear_queue_requested = Signal()
    stop_worker_requested = Signal()

    def __init__(self, list_widget: QListWidget, avatar_size: tuple = (60, 80)):
        super().__init__(list_widget)
        self.list_widget = list_widget
        self.avatar_size = QSize(avatar_size[0], avatar_size[1])

        self.loaded_images: dict[int, QPixmap] = {}
        self.visible_items: Set[int] = set()
        self.file_paths: list[str] = []
        self.cards = []

        self.worker_thread = QThread(self)
        self.worker = ImageLoadWorker()
        self.worker.moveToThread(self.worker_thread)
        self.worker.image_loaded.connect(self._on_image_loaded)
        self.queue_image_load_requested.connect(self.worker.add_to_queue, Qt.ConnectionType.QueuedConnection)
        self.clear_queue_requested.connect(self.worker.clear_queue, Qt.ConnectionType.QueuedConnection)
        self.stop_worker_requested.connect(self.worker.stop, Qt.ConnectionType.QueuedConnection)
        self.worker_thread.start()

        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_visible_items)

        if hasattr(self.list_widget, 'verticalScrollBar'):
            scrollbar = self.list_widget.verticalScrollBar()
            if scrollbar:
                scrollbar.valueChanged.connect(self._on_scroll)

        self.max_loaded_images = 20
        self.preload_buffer = 2

    def set_file_paths(self, file_paths: list[str], cards: list):
        """Set the file paths and card references for lazy loading."""
        cards_copy = cards.copy() if cards else []
        self.clear()
        self.file_paths = file_paths.copy()
        self.cards = cards_copy

        if not self.worker_thread.isRunning():
            self.worker_thread.start()

        self._schedule_update()

    def clear(self):
        """Clear all loaded images and reset state."""
        self.loaded_images.clear()
        self.visible_items.clear()
        self.file_paths.clear()
        self.cards.clear()

        if self.worker:
            self.clear_queue_requested.emit()

    def _on_scroll(self):
        """Handle scroll events with debouncing."""
        self._schedule_update()

    def _schedule_update(self):
        """Schedule an update of visible items."""
        self.update_timer.start(100)

    def _update_visible_items(self):
        """Update which items are visible and manage loading/unloading."""
        if not self.list_widget or not self.file_paths:
            return

        new_visible_items = self._get_visible_item_indices()

        items_to_load = set()
        for index in new_visible_items:
            start_idx = max(0, index - self.preload_buffer)
            end_idx = min(len(self.file_paths), index + self.preload_buffer + 1)
            items_to_load.update(range(start_idx, end_idx))

        for index in items_to_load:
            if index not in self.loaded_images and 0 <= index < len(self.file_paths):
                self._queue_image_load(index)

        self._manage_memory(items_to_load)
        self.visible_items = new_visible_items

    def _get_visible_item_indices(self) -> Set[int]:
        """Get indices of currently visible items."""
        visible_indices = set()
        if not self.list_widget:
            return visible_indices

        viewport_rect = self.list_widget.viewport().rect()
        probe_x = max(1, viewport_rect.center().x())
        top_item = self._find_visible_item(probe_x, viewport_rect.top(), 1, viewport_rect.bottom())
        bottom_item = self._find_visible_item(probe_x, max(0, viewport_rect.bottom() - 1), -1, viewport_rect.top())

        if top_item is None:
            return visible_indices

        top_index = self.list_widget.row(top_item)
        if bottom_item is not None:
            bottom_index = self.list_widget.row(bottom_item)
        else:
            row_height = max(1, self.list_widget.sizeHintForRow(top_index))
            visible_rows = max(1, (viewport_rect.height() // row_height) + 1)
            bottom_index = min(self.list_widget.count() - 1, top_index + visible_rows)

        visible_indices.update(range(top_index, bottom_index + 1))
        return visible_indices

    def _find_visible_item(self, x: int, start_y: int, step: int, limit_y: int):
        """Probe vertically within the viewport until a row is found."""
        y = start_y
        while (y <= limit_y) if step > 0 else (y >= limit_y):
            item = self.list_widget.itemAt(QPoint(x, y))
            if item is not None:
                return item
            y += step
        return None

    def _queue_image_load(self, index: int):
        """Queue an image for loading."""
        if 0 <= index < len(self.file_paths):
            file_path = self.file_paths[index]
            if ensure_path_materialized(file_path) or os.path.exists(file_path):
                self.queue_image_load_requested.emit(index, file_path, self.avatar_size)
                if not self.worker_thread.isRunning():
                    self.worker_thread.start()

    def _on_image_loaded(self, index: int, file_path: str, image: QImage):
        """Handle when an image has been loaded."""
        if 0 <= index < len(self.cards) and index < len(self.file_paths) and self.file_paths[index] == file_path:
            pixmap = QPixmap.fromImage(image)
            self.loaded_images[index] = pixmap

            card = self.cards[index]
            if card and hasattr(card, '_avatar'):
                card._avatar.set_dayu_image(pixmap)
                card._avatar.setVisible(True)

                list_item = self.list_widget.item(index)
                if list_item:
                    list_item.setSizeHint(card.sizeHint())

    def _manage_memory(self, needed_items: Set[int]):
        """Manage memory by unloading images that are no longer needed."""
        if len(self.loaded_images) <= self.max_loaded_images:
            return

        items_to_unload = []
        for index in list(self.loaded_images.keys()):
            if index not in needed_items:
                items_to_unload.append(index)

        excess_count = len(self.loaded_images) - self.max_loaded_images
        items_to_unload.sort()

        for i, index in enumerate(items_to_unload):
            if i >= excess_count:
                break

            del self.loaded_images[index]

            if 0 <= index < len(self.cards):
                card = self.cards[index]
                if card and hasattr(card, '_avatar'):
                    card._avatar.setVisible(False)

    def force_load_image(self, index: int):
        """Force load an image immediately (for current selection)."""
        if 0 <= index < len(self.file_paths) and index not in self.loaded_images:
            self._queue_image_load(index)

    def shutdown(self):
        """Shutdown the loader and clean up resources."""
        if self.worker:
            self.stop_worker_requested.emit()

        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(5000)

        self.clear()
