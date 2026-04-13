from __future__ import annotations

from collections import deque

import imkit as imk
from PIL import Image
from PySide6.QtCore import QThread, QObject, Signal, QSize, QTimer, Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QListView

from app.path_materialization import ensure_path_materialized

from .list_view import PageListModel


class ImageLoadWorker(QObject):
    image_loaded = Signal(str, QImage)

    def __init__(self) -> None:
        super().__init__()
        self.load_queue = deque()
        self._queued_paths: set[str] = set()
        self.should_stop = False
        self._processing = False

    @Slot(str, QSize)
    def add_to_queue(self, file_path: str, target_size: QSize) -> None:
        if self.should_stop or not file_path or file_path in self._queued_paths:
            return
        self.load_queue.append((file_path, target_size))
        self._queued_paths.add(file_path)
        if not self._processing:
            self.process_queue()

    @Slot()
    def clear_queue(self) -> None:
        self.load_queue.clear()
        self._queued_paths.clear()

    @Slot()
    def stop(self) -> None:
        self.should_stop = True
        self.clear_queue()

    @Slot()
    def process_queue(self) -> None:
        if self._processing or self.should_stop:
            return

        self._processing = True
        try:
            while self.load_queue and not self.should_stop:
                file_path, target_size = self.load_queue.popleft()
                self._queued_paths.discard(file_path)
                image = self._load_and_resize_image(file_path, target_size)
                if image is not None and not image.isNull():
                    self.image_loaded.emit(file_path, image)
        finally:
            self._processing = False

    def _load_and_resize_image(self, file_path: str, target_size: QSize) -> QImage | None:
        try:
            ensure_path_materialized(file_path)
            image = imk.read_image(file_path)
            if image is None:
                return None

            height, width = image.shape[:2]
            target_width, target_height = target_size.width(), target_size.height()
            scale = min(target_width / max(width, 1), target_height / max(height, 1))
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            resized = imk.resize(image, (new_width, new_height), mode=Image.Resampling.LANCZOS)

            h, w, ch = resized.shape
            bytes_per_line = ch * w
            qt_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return qt_image.copy()
        except Exception as exc:
            print(f"Error processing image {file_path}: {exc}")
            return None


class ListViewImageLoader(QObject):
    queue_image_load_requested = Signal(str, QSize)
    clear_queue_requested = Signal()
    stop_worker_requested = Signal()

    def __init__(self, list_view: QListView, avatar_size: tuple = (60, 80)):
        super().__init__(list_view)
        self.list_view = list_view
        self.avatar_size = QSize(avatar_size[0], avatar_size[1])

        self.loaded_images: dict[str, QPixmap] = {}
        self.visible_paths: set[str] = set()
        self.file_paths: list[str] = []
        self.model: PageListModel | None = None

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

        scrollbar = self.list_view.verticalScrollBar()
        if scrollbar:
            scrollbar.valueChanged.connect(self._on_scroll)

        self.max_loaded_images = 24
        self.preload_buffer = 2

    def set_model(self, model: PageListModel) -> None:
        self.model = model

    def set_file_paths(self, file_paths: list[str]) -> None:
        self.clear()
        self.file_paths = file_paths.copy()

        if not self.worker_thread.isRunning():
            self.worker_thread.start()

        self._schedule_update()

    def clear(self) -> None:
        if self.model:
            for file_path in list(self.loaded_images.keys()):
                self.model.clear_thumbnail(file_path)
        self.loaded_images.clear()
        self.visible_paths.clear()
        self.file_paths.clear()

        if self.worker:
            self.clear_queue_requested.emit()

    def _on_scroll(self) -> None:
        self._schedule_update()

    def _schedule_update(self) -> None:
        self.update_timer.start(100)

    def _update_visible_items(self) -> None:
        if not self.list_view or not self.file_paths or self.model is None:
            return

        visible_rows = self._get_visible_rows()
        items_to_load: set[str] = set()
        for row in visible_rows:
            start = max(0, row - self.preload_buffer)
            end = min(len(self.file_paths), row + self.preload_buffer + 1)
            for path in self.file_paths[start:end]:
                items_to_load.add(path)

        for file_path in items_to_load:
            if file_path not in self.loaded_images:
                self._queue_image_load(file_path)

        self._manage_memory(items_to_load)
        self.visible_paths = items_to_load

    def _get_visible_rows(self) -> list[int]:
        if self.model is None:
            return []
        viewport_rect = self.list_view.viewport().rect()
        visible_rows: list[int] = []
        for row in range(self.model.rowCount()):
            index = self.model.index(row, 0)
            rect = self.list_view.visualRect(index)
            if rect.isValid() and rect.intersects(viewport_rect):
                visible_rows.append(row)
        return visible_rows

    def _queue_image_load(self, file_path: str) -> None:
        self.queue_image_load_requested.emit(file_path, self.avatar_size)
        if not self.worker_thread.isRunning():
            self.worker_thread.start()

    def _on_image_loaded(self, file_path: str, image: QImage) -> None:
        if self.model is None or file_path not in self.file_paths:
            return
        pixmap = QPixmap.fromImage(image)
        self.loaded_images[file_path] = pixmap
        self.model.set_thumbnail(file_path, pixmap)

    def _manage_memory(self, needed_paths: set[str]) -> None:
        if len(self.loaded_images) <= self.max_loaded_images or self.model is None:
            return

        unload_candidates = [path for path in self.loaded_images if path not in needed_paths]
        unload_candidates.sort(key=lambda p: self.file_paths.index(p) if p in self.file_paths else 10**9)
        excess_count = len(self.loaded_images) - self.max_loaded_images
        for file_path in unload_candidates[:excess_count]:
            self.loaded_images.pop(file_path, None)
            self.model.clear_thumbnail(file_path)

    def force_load_image(self, index: int) -> None:
        if not (0 <= index < len(self.file_paths)):
            return
        file_path = self.file_paths[index]
        if file_path not in self.loaded_images:
            self._queue_image_load(file_path)

    def shutdown(self) -> None:
        if self.worker:
            self.stop_worker_requested.emit()

        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(5000)

        self.clear()
