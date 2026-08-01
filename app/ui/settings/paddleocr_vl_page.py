from PySide6 import QtCore, QtWidgets

from modules.ocr.persistent_cache import (
    DEFAULT_OCR_RESULT_CACHE_LIMIT,
    OCRPersistentResultCache,
)
from modules.ocr.paddle_crop.transport import DEFAULT_PADDLE_DIRECT_SERVER_URL

from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.line_edit import MLineEdit
from ..dayu_widgets.spin_box import MSpinBox


class PaddleOCRVLPage(QtWidgets.QWidget):
    DEFAULT_SERVER_URL = DEFAULT_PADDLE_DIRECT_SERVER_URL
    DEFAULT_MAX_NEW_TOKENS = 1024
    DEFAULT_PARALLEL_WORKERS = 8
    DEFAULT_PERSISTENT_CACHE_ENABLED = True
    DEFAULT_PERSISTENT_CACHE_LIMIT = DEFAULT_OCR_RESULT_CACHE_LIMIT

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_label = MLabel(self.tr("PaddleOCR VL Settings")).h3()
        layout.addWidget(title_label)

        note = MLabel(
            self.tr(
                "Connect Comic Translate to your local PaddleOCR VL Docker service.\n"
                "The bundled runtime sends cropped text regions directly to llama.cpp with the official OCR: prompt.\n"
                "Keep the default localhost URL if you want Comic Translate to start the bundled Docker runtime on demand.\n"
                "Markdown and visualization options apply only to custom /layout-parsing endpoints."
            )
        ).secondary()
        note.setWordWrap(True)
        note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(note)

        server_group = QtWidgets.QVBoxLayout()
        server_label = MLabel(self.tr("Server URL")).strong()
        self.server_url_input = MLineEdit()
        self.server_url_input.setMinimumWidth(360)
        self.server_url_input.setMaximumWidth(560)
        self.server_url_input.setPlaceholderText(self.DEFAULT_SERVER_URL)
        server_group.addWidget(server_label)
        server_group.addWidget(self.server_url_input)
        layout.addLayout(server_group)

        toggles_group = QtWidgets.QVBoxLayout()
        toggles_label = MLabel(self.tr("Response Options")).h4()
        toggles_group.addWidget(toggles_label)
        self.prettify_markdown_checkbox = MCheckBox(self.tr("Prettify Markdown"))
        self.visualize_checkbox = MCheckBox(self.tr("Visualize"))
        toggles_group.addWidget(self.prettify_markdown_checkbox)
        toggles_group.addWidget(self.visualize_checkbox)
        layout.addLayout(toggles_group)

        perf_label = MLabel(self.tr("Performance")).h4()
        layout.addWidget(perf_label)

        perf_note = MLabel(
            self.tr(
                "The bundled llama.cpp runtime loads one managed model slot for the whole OCR stage.\n"
                "Parallel workers control the client request queue and do not load extra model copies.\n"
                "Keep 1024 tokens unless a validated corpus proves a lower limit never truncates text."
            )
        ).secondary()
        perf_note.setWordWrap(True)
        perf_note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(perf_note)

        max_tokens_layout = QtWidgets.QHBoxLayout()
        max_tokens_label = MLabel(self.tr("Max New Tokens"))
        self.max_new_tokens_spinbox = MSpinBox().small()
        self.max_new_tokens_spinbox.setRange(64, 2048)
        self.max_new_tokens_spinbox.setSingleStep(64)
        self.max_new_tokens_spinbox.setValue(self.DEFAULT_MAX_NEW_TOKENS)
        self.max_new_tokens_spinbox.setFixedWidth(90)
        max_tokens_layout.addWidget(max_tokens_label)
        max_tokens_layout.addWidget(self.max_new_tokens_spinbox)
        max_tokens_layout.addStretch(1)
        layout.addLayout(max_tokens_layout)

        workers_layout = QtWidgets.QHBoxLayout()
        workers_label = MLabel(self.tr("Parallel Workers"))
        self.parallel_workers_spinbox = MSpinBox().small()
        self.parallel_workers_spinbox.setRange(1, 8)
        self.parallel_workers_spinbox.setValue(self.DEFAULT_PARALLEL_WORKERS)
        self.parallel_workers_spinbox.setFixedWidth(90)
        workers_layout.addWidget(workers_label)
        workers_layout.addWidget(self.parallel_workers_spinbox)
        workers_layout.addStretch(1)
        layout.addLayout(workers_layout)

        cache_label = MLabel(self.tr("Persistent OCR Result Cache")).h4()
        layout.addWidget(cache_label)

        cache_note = MLabel(
            self.tr(
                "Reuses exact PaddleOCR VL crop results across runs. Images are not stored. "
                "The cache is available only for the bundled managed Docker endpoint; custom "
                "endpoints continue normally without persistent caching."
            )
        ).secondary()
        cache_note.setWordWrap(True)
        cache_note.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(cache_note)

        self.persistent_cache_checkbox = MCheckBox(
            self.tr("Enable persistent OCR result cache")
        )
        self.persistent_cache_checkbox.setChecked(
            self.DEFAULT_PERSISTENT_CACHE_ENABLED
        )
        layout.addWidget(self.persistent_cache_checkbox)

        cache_limit_layout = QtWidgets.QHBoxLayout()
        cache_limit_label = MLabel(self.tr("Maximum cached crops"))
        self.persistent_cache_limit_spinbox = MSpinBox().small()
        self.persistent_cache_limit_spinbox.setRange(100, 1_000_000)
        self.persistent_cache_limit_spinbox.setSingleStep(1_000)
        self.persistent_cache_limit_spinbox.setValue(
            self.DEFAULT_PERSISTENT_CACHE_LIMIT
        )
        self.persistent_cache_limit_spinbox.setFixedWidth(110)
        cache_limit_layout.addWidget(cache_limit_label)
        cache_limit_layout.addWidget(self.persistent_cache_limit_spinbox)
        cache_limit_layout.addStretch(1)
        layout.addLayout(cache_limit_layout)

        self.persistent_cache_stats_label = MLabel(
            self.tr("Hits: — · Misses: — · Items: —")
        ).secondary()
        self.persistent_cache_stats_label.setWordWrap(True)
        layout.addWidget(self.persistent_cache_stats_label)

        cache_actions = QtWidgets.QHBoxLayout()
        self.persistent_cache_refresh_button = QtWidgets.QPushButton(
            self.tr("Refresh statistics")
        )
        self.persistent_cache_export_button = QtWidgets.QPushButton(
            self.tr("Export JSONL")
        )
        self.persistent_cache_clear_button = QtWidgets.QPushButton(
            self.tr("Clear cache")
        )
        cache_actions.addWidget(self.persistent_cache_refresh_button)
        cache_actions.addWidget(self.persistent_cache_export_button)
        cache_actions.addWidget(self.persistent_cache_clear_button)
        cache_actions.addStretch(1)
        layout.addLayout(cache_actions)

        self.persistent_cache_refresh_button.clicked.connect(
            self.refresh_persistent_cache_stats
        )
        self.persistent_cache_export_button.clicked.connect(
            self.export_persistent_cache
        )
        self.persistent_cache_clear_button.clicked.connect(
            self.clear_persistent_cache
        )

        layout.addStretch(1)

    def _open_cache_store(self) -> OCRPersistentResultCache:
        return OCRPersistentResultCache(
            result_cache_limit=int(self.persistent_cache_limit_spinbox.value())
        )

    def refresh_persistent_cache_stats(self) -> None:
        with self._open_cache_store() as store:
            stats = store.stats()
        if not stats.get("enabled", False):
            self.persistent_cache_stats_label.setText(
                self.tr("Cache unavailable: {reason}").format(
                    reason=str(stats.get("disabled_reason", "") or "unknown error")
                )
            )
            return
        self.persistent_cache_stats_label.setText(
            self.tr("Hits: {hits} · Misses: {misses} · Items: {items}").format(
                hits=int(stats.get("lookup_hits", 0) or 0),
                misses=int(stats.get("lookup_misses", 0) or 0),
                items=int(stats.get("item_count", 0) or 0),
            )
        )

    def export_persistent_cache(self) -> None:
        output_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export PaddleOCR VL Cache"),
            "paddleocr-vl-cache.jsonl",
            self.tr("JSON Lines (*.jsonl)"),
        )
        if not output_path:
            return
        try:
            with self._open_cache_store() as store:
                count = store.export_jsonl(output_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("PaddleOCR VL Cache"),
                self.tr("Cache export failed. The database was left unchanged.\n{error}").format(
                    error=str(exc)
                ),
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            self.tr("PaddleOCR VL Cache"),
            self.tr("Exported {count} cached OCR results.").format(count=count),
        )

    def clear_persistent_cache(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Clear PaddleOCR VL Cache"),
            self.tr(
                "Clear all persistent PaddleOCR VL result-cache entries? "
                "No source images will be deleted."
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            with self._open_cache_store() as store:
                store.clear()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("PaddleOCR VL Cache"),
                self.tr("Cache clear failed. The database was left unchanged.\n{error}").format(
                    error=str(exc)
                ),
            )
            return
        self.refresh_persistent_cache_stats()
