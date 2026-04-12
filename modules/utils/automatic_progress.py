from __future__ import annotations

import copy
import math
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from PySide6.QtCore import QCoreApplication, QSettings


AUTOMATIC_PROGRESS_TRANSLATIONS = {
    "calculating": QCoreApplication.translate("AutomaticProgress", "Calculating"),
    "recent_history": QCoreApplication.translate("AutomaticProgress", "Recent History"),
    "live_learning": QCoreApplication.translate("AutomaticProgress", "Live Learning"),
    "live_stable": QCoreApplication.translate("AutomaticProgress", "Live Stable"),
}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]
    if seconds < 0:
        seconds = 0
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_finish_time(eta_sec: float | None) -> str:
    if eta_sec is None:
        return AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]
    finish = datetime.now().astimezone() + timedelta(seconds=max(eta_sec, 0.0))
    return finish.strftime("%H:%M")


class AutomaticProgressTracker:
    STARTUP_HISTORY_GROUP = "automatic_progress/startup_history"
    BATCH_HISTORY_GROUP = "automatic_progress/batch_history"
    MAX_HISTORY_ITEMS = 12

    def __init__(self) -> None:
        self.settings = QSettings("ComicLabs", "ComicTranslate")
        self.reset()

    def reset(self, *, page_total: int = 0, run_type: str = "batch") -> None:
        now = time.monotonic()
        self.run_started_at = now
        self.page_total = max(int(page_total or 0), 0)
        self.run_type = str(run_type or "batch")
        self.current_page_started_at: float | None = None
        self.current_page_index: int | None = None
        self.current_page_name = ""
        self.current_stage_started_at: float | None = None
        self.current_stage_name = ""
        self.completed_page_durations: deque[float] = deque(maxlen=5)
        self.completed_stage_durations: dict[str, deque[float]] = {}
        self.startup_step_started_at: dict[str, float] = {}
        self.last_event: dict[str, Any] | None = None

    def enrich(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = copy.deepcopy(payload)
        now = time.monotonic()
        event.setdefault("elapsed_sec", now - self.run_started_at)
        elapsed_sec = float(event.get("elapsed_sec") or 0.0)
        event["elapsed_sec"] = elapsed_sec

        phase = str(event.get("phase") or "")
        step_key = str(event.get("step_key") or "")
        status = str(event.get("status") or "")
        page_index = event.get("page_index")
        page_total = int(event.get("page_total") or self.page_total or 0)
        image_name = str(event.get("image_name") or self.current_page_name or "")
        stage_name = str(event.get("stage_name") or step_key or self.current_stage_name or "")

        if page_total:
            self.page_total = page_total

        if phase == "pipeline":
            if step_key == "page_start":
                self.current_page_started_at = now
                self.current_page_index = int(page_index) if page_index is not None else None
                self.current_page_name = image_name
                self.current_stage_started_at = now
                self.current_stage_name = "page_start"
            elif step_key == "page_done" and self.current_page_started_at is not None:
                self.completed_page_durations.append(now - self.current_page_started_at)
                self.current_stage_started_at = now
                self.current_stage_name = "page_done"
            elif status == "running":
                if stage_name and stage_name != self.current_stage_name:
                    if self.current_stage_name and self.current_stage_started_at is not None:
                        self.completed_stage_durations.setdefault(self.current_stage_name, deque(maxlen=10)).append(
                            now - self.current_stage_started_at
                        )
                    self.current_stage_name = stage_name
                    self.current_stage_started_at = now

        if phase in {"gemma_startup", "ocr_startup"}:
            if status in {"starting", "running", "waiting_health"} and step_key not in self.startup_step_started_at:
                self.startup_step_started_at[step_key] = now
            elif status == "completed":
                started = self.startup_step_started_at.get(step_key)
                if started is not None:
                    self._append_history(self.STARTUP_HISTORY_GROUP, step_key, now - started)

        eta_sec, eta_confidence = self._estimate_eta(event, now)
        event["eta_sec"] = eta_sec
        event["eta_confidence"] = eta_confidence
        event["eta_finish_at_local"] = format_finish_time(eta_sec)
        event["elapsed_text"] = format_duration(elapsed_sec)
        event["eta_text"] = format_duration(eta_sec)
        event["overall_progress_percent"] = self._estimate_progress(event)
        event["page_total"] = page_total
        event["image_name"] = image_name
        event["stage_name"] = stage_name
        self.last_event = event
        return event

    def record_batch_completion(self, *, success: bool, total_images: int | None = None) -> None:
        if not success:
            return
        image_count = int(total_images or self.page_total or len(self.completed_page_durations) or 0)
        elapsed = time.monotonic() - self.run_started_at
        if image_count <= 0 or elapsed <= 0:
            return
        entry = {
            "image_count": image_count,
            "elapsed_sec": elapsed,
            "per_page_sec": elapsed / image_count,
        }
        self._append_history(self.BATCH_HISTORY_GROUP, "recent", entry)

    def _estimate_eta(self, event: dict[str, Any], now: float) -> tuple[float | None, str]:
        phase = str(event.get("phase") or "")
        if phase in {"gemma_startup", "ocr_startup"}:
            return self._estimate_startup_eta(event, now)
        if phase == "pipeline":
            return self._estimate_pipeline_eta(event, now)
        return None, AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]

    def _estimate_startup_eta(self, event: dict[str, Any], now: float) -> tuple[float | None, str]:
        step_key = str(event.get("step_key") or "")
        started = self.startup_step_started_at.get(step_key)
        if started is None:
            self.startup_step_started_at[step_key] = now
            return None, AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]
        elapsed = now - started
        history = self._read_history(self.STARTUP_HISTORY_GROUP, step_key)
        if not history:
            return None, AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]
        median_value = _median_number(history)
        if median_value is None:
            return None, AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]
        return max(median_value - elapsed, 0.0), AUTOMATIC_PROGRESS_TRANSLATIONS["recent_history"]

    def _estimate_pipeline_eta(self, event: dict[str, Any], now: float) -> tuple[float | None, str]:
        if self.page_total <= 0:
            return None, AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]

        completed_pages = len(self.completed_page_durations)
        if completed_pages == 0:
            recent = self._read_history(self.BATCH_HISTORY_GROUP, "recent")
            per_page = _median_field(recent, "per_page_sec")
            if per_page is not None:
                return per_page * self.page_total, AUTOMATIC_PROGRESS_TRANSLATIONS["recent_history"]
            return None, AUTOMATIC_PROGRESS_TRANSLATIONS["calculating"]

        if completed_pages < 3:
            per_page = sum(self.completed_page_durations) / completed_pages
            return per_page * max(self.page_total - completed_pages, 0), AUTOMATIC_PROGRESS_TRANSLATIONS["live_learning"]

        recent = list(self.completed_page_durations)
        per_page = sum(recent) / len(recent)
        remaining = per_page * max(self.page_total - completed_pages, 0)

        if self.current_stage_started_at is not None and self.current_page_index is not None:
            current_stage = str(event.get("stage_name") or self.current_stage_name or "")
            current_elapsed = now - self.current_stage_started_at
            stage_history = list(self.completed_stage_durations.get(current_stage, []))
            if stage_history:
                remaining += max((sum(stage_history) / len(stage_history)) - current_elapsed, 0.0)

        return remaining, AUTOMATIC_PROGRESS_TRANSLATIONS["live_stable"]

    def _estimate_progress(self, event: dict[str, Any]) -> float:
        if str(event.get("phase") or "") != "pipeline":
            return 0.0
        page_total = int(event.get("page_total") or self.page_total or 0)
        if page_total <= 0:
            return 0.0
        page_index = int(event.get("page_index") or 0)
        stage_name = str(event.get("stage_name") or "")
        stage_order = {
            "page_start": 0,
            "detect": 1,
            "ocr": 2,
            "inpaint": 3,
            "translation": 4,
            "render": 5,
            "save": 6,
            "page_done": 7,
        }
        units = page_total * 8
        current_units = min(max(page_index * 8 + stage_order.get(stage_name, 0), 0), units)
        return round((current_units / units) * 100.0, 1) if units else 0.0

    def _append_history(self, group: str, key: str, value: Any) -> None:
        history = self._read_history(group, key)
        history.append(value)
        history = history[-self.MAX_HISTORY_ITEMS :]
        self.settings.beginGroup(group)
        self.settings.setValue(key, history)
        self.settings.endGroup()

    def _read_history(self, group: str, key: str) -> list[Any]:
        self.settings.beginGroup(group)
        value = self.settings.value(key, [], type=list)
        self.settings.endGroup()
        return list(value or [])


def _median_number(values: list[Any]) -> float | None:
    numbers: list[float] = []
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        numbers.append(number)
    if not numbers:
        return None
    numbers.sort()
    mid = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[mid]
    return (numbers[mid - 1] + numbers[mid]) / 2.0


def _median_field(values: list[Any], field: str) -> float | None:
    return _median_number([
        float(item[field])
        for item in values
        if isinstance(item, dict) and field in item
        and isinstance(item[field], (int, float))
    ])
