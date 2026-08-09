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

# Anonymized wall-clock samples: (page_count, elapsed_seconds). The tracker fits
# a tiny weighted linear model from these samples, then gives local history a
# higher weight so the equation adapts to the user's hardware and settings.
PIPELINE_ETA_SEED_SAMPLES: tuple[tuple[int, int], ...] = (
    (100, 12 * 60 + 7),
    (99, 15 * 60 + 40),
    (98, 12 * 60 + 42),
    (86, 9 * 60 + 9),
    (111, 12 * 60 + 56),
    (92, 10 * 60 + 31),
    (96, 10 * 60 + 8),
    (91, 9 * 60 + 49),
    (90, 10 * 60 + 12),
    (116, 12 * 60 + 5),
    (100, 9 * 60 + 58),
    (91, 9 * 60 + 58),
    (90, 9 * 60),
    (91, 10 * 60 + 7),
    (90, 9 * 60 + 27),
    (87, 8 * 60 + 52),
    (91, 9 * 60 + 58),
    (91, 10 * 60 + 43),
    (111, 11 * 60 + 2),
    (105, 11 * 60 + 20),
    (98, 10 * 60 + 9),
    (102, 11 * 60 + 29),
    (105, 9 * 60 + 51),
    (122, 11 * 60 + 12),
    (107, 10 * 60 + 6),
    (100, 10 * 60 + 12),
    (112, 13 * 60 + 12),
    (105, 12 * 60 + 45),
    (110, 9 * 60 + 9),
    (103, 11 * 60),
    (109, 11 * 60 + 21),
    (106, 9 * 60 + 21),
    (106, 10 * 60 + 16),
    (93, 10 * 60 + 27),
    (92, 8 * 60 + 19),
    (89, 8 * 60 + 19),
    (90, 8 * 60 + 29),
    (91, 10 * 60 + 6),
    (90, 9 * 60 + 39),
    (99, 13 * 60 + 5),
    (91, 10 * 60 + 21),
    (90, 9 * 60),
    (93, 9 * 60 + 59),
    (94, 10 * 60 + 36),
    (93, 11 * 60 + 32),
    (90, 8 * 60 + 46),
    (92, 9 * 60 + 9),
    (92, 10 * 60 + 41),
    (95, 16 * 60 + 15),
    (101, 17 * 60 + 32),
    (98, 12 * 60 + 13),
)
PIPELINE_ETA_HISTORY_WEIGHT = 8.0
PIPELINE_ETA_MIN_SEC_PER_PAGE = 1.0
PIPELINE_ETA_MAX_SEC_PER_PAGE = 30.0


def _coerce_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fit_weighted_linear_eta_model(samples: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    valid = [
        (pages, elapsed, weight)
        for pages, elapsed, weight in samples
        if pages > 0 and elapsed > 0 and weight > 0
    ]
    if not valid:
        return None

    if len(valid) == 1:
        pages, elapsed, _weight = valid[0]
        return 0.0, elapsed / pages

    weight_sum = sum(weight for _pages, _elapsed, weight in valid)
    mean_pages = sum(pages * weight for pages, _elapsed, weight in valid) / weight_sum
    mean_elapsed = sum(elapsed * weight for _pages, elapsed, weight in valid) / weight_sum
    variance = sum(weight * (pages - mean_pages) ** 2 for pages, _elapsed, weight in valid)

    if variance <= 1e-9:
        slope = sum(weight * (elapsed / pages) for pages, elapsed, weight in valid) / weight_sum
        return 0.0, _clamp_pipeline_slope(slope)

    covariance = sum(
        weight * (pages - mean_pages) * (elapsed - mean_elapsed)
        for pages, elapsed, weight in valid
    )
    slope = covariance / variance
    if slope <= 0 or not math.isfinite(slope):
        slope = sum(weight * (elapsed / pages) for pages, elapsed, weight in valid) / weight_sum
        return 0.0, _clamp_pipeline_slope(slope)

    intercept = mean_elapsed - slope * mean_pages
    return max(intercept, 0.0), _clamp_pipeline_slope(slope)


def _clamp_pipeline_slope(slope: float) -> float:
    return min(max(float(slope), PIPELINE_ETA_MIN_SEC_PER_PAGE), PIPELINE_ETA_MAX_SEC_PER_PAGE)


def _pipeline_history_samples(history: list[Any]) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        pages = _coerce_finite_float(item.get("image_count"))
        if pages is None or pages <= 0:
            continue
        elapsed = _coerce_finite_float(item.get("elapsed_sec"))
        if elapsed is None or elapsed <= 0:
            per_page = _coerce_finite_float(item.get("per_page_sec"))
            elapsed = per_page * pages if per_page is not None and per_page > 0 else None
        if elapsed is None or elapsed <= 0:
            continue
        samples.append((pages, elapsed, PIPELINE_ETA_HISTORY_WEIGHT))
    return samples


def _fit_pipeline_eta_model(history: list[Any]) -> tuple[tuple[float, float] | None, bool]:
    samples = [(float(pages), float(elapsed), 1.0) for pages, elapsed in PIPELINE_ETA_SEED_SAMPLES]
    history_samples = _pipeline_history_samples(history)
    samples.extend(history_samples)
    return _fit_weighted_linear_eta_model(samples), bool(history_samples)


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


STAGE_STATE_LABELS = {
    "done": QCoreApplication.translate("AutomaticProgress", "완료"),
    "running": QCoreApplication.translate("AutomaticProgress", "진행 중"),
    "pending": QCoreApplication.translate("AutomaticProgress", "대기"),
    "fused": QCoreApplication.translate("AutomaticProgress", "인페인팅에 포함"),
    "unknown": QCoreApplication.translate("AutomaticProgress", "추정 불가"),
}


def format_stage_breakdown(rows: Any) -> str:
    """파이프라인 전 단계를 실행 순서대로 적는다. 마우스를 올렸을 때만 보인다.

    남은 시간이 한 숫자로만 보이면 그게 번역에서 오는지 인페인팅에서 오는지 알 수
    없다. 끝난 단계까지 함께 적는 이유는, 남은 단계만 보이면 지금 파이프라인의
    어디쯤 와 있는지가 오히려 흐려지기 때문이다.

    분해가 없으면 빈 문자열이고, 그러면 UI 는 툴팁을 붙이지 않는다.
    """

    if not isinstance(rows, list) or not rows:
        return ""
    entries: list[tuple[str, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or row.get("stage") or "").strip()
        if not label:
            continue
        state = str(row.get("state") or "")
        seconds = _coerce_finite_float(row.get("seconds")) or 0.0
        progress = ""
        total = _coerce_finite_float(row.get("page_total"))
        done = _coerce_finite_float(row.get("pages_done"))
        if total and total > 0 and done is not None:
            progress = f"{int(done)}/{int(total)}"
        if state == "done":
            value = STAGE_STATE_LABELS["done"]
        elif state in {"fused", "unknown"}:
            value = STAGE_STATE_LABELS[state]
        else:
            value = format_duration(seconds)
        entries.append((label, value, progress))
    if not entries:
        return ""

    label_width = max(len(label) for label, _v, _p in entries)
    value_width = max(len(value) for _l, value, _p in entries)
    lines = [QCoreApplication.translate("AutomaticProgress", "단계별 남은 시간")]
    for label, value, progress in entries:
        line = f"  {label.ljust(label_width)}  {value.rjust(value_width)}"
        if progress:
            line = f"{line}  {progress}"
        lines.append(line)
    # 한글과 숫자가 섞이면 비례 글꼴에서 열이 어긋난다. 툴팁은 리치 텍스트를
    # 받으므로 등폭으로 감싸 정렬이 유지되게 한다.
    body = "\n".join(lines)
    escaped = (
        body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"<pre style='margin:0'>{escaped}</pre>"


class AutomaticProgressTracker:
    STARTUP_HISTORY_GROUP = "automatic_progress/startup_history"
    BATCH_HISTORY_GROUP = "automatic_progress/batch_history"
    MAX_HISTORY_ITEMS = 12

    def __init__(self) -> None:
        self.settings = QSettings("ComicLabs", "ComicTranslate")
        self.reset()

    def reset(self, *, page_total: int = 0, run_type: str = "batch") -> None:
        self._supplied_eta_sec = None
        self._supplied_progress_percent = None
        now = time.monotonic()
        self.run_started_at = now
        # 사용자가 '모두 번역'을 누른 벽시계 시각. 실행 리포트가 이 값을 쓴다.
        self.run_started_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        # 전체 예상 시간 = 이미 쓴 시간 + 남은 시간. 남은 시간만으로는 이 실행이
        # 통째로 얼마짜리인지 알 수 없고, 지난 실행과 비교할 수도 없다.
        total_sec = None if eta_sec is None else max(elapsed_sec, 0.0) + max(eta_sec, 0.0)
        event["total_estimate_sec"] = total_sec
        event["total_estimate_text"] = format_duration(total_sec)
        event["eta_breakdown_text"] = format_stage_breakdown(event.get("eta_by_stage"))
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
        # 파이프라인이 남은 시간을 실어 보냈으면 그것이 유일한 출처다. 여기서 다시
        # 추정하면 두 값이 갈린다. 실제로 같은 순간에 2초와 23분이 찍혔다.
        supplied = event.get("eta_seconds")
        if isinstance(supplied, (int, float)) and not isinstance(supplied, bool):
            self._supplied_eta_sec = max(float(supplied), 0.0)
            return self._supplied_eta_sec, AUTOMATIC_PROGRESS_TRANSLATIONS["live_stable"]
        # 미리보기 알림처럼 sweep 이 아닌 이벤트는 추정을 싣지 않는다. 그때 자체
        # 계산으로 되돌아가면 남은 시간이 튄다. 실제로 인페인팅 sweep 이 끝난 직후
        # 미리보기 알림 네 줄에서 3분 33초가 24분 08초로 뛰었다. 파이프라인이 준
        # 마지막 값을 유지한다.
        last_supplied = getattr(self, "_supplied_eta_sec", None)
        if last_supplied is not None:
            return last_supplied, AUTOMATIC_PROGRESS_TRANSLATIONS["live_stable"]
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
            learned_eta, used_history = self._estimate_learned_pipeline_eta(now, recent)
            if learned_eta is not None:
                confidence_key = "recent_history" if used_history else "live_learning"
                return learned_eta, AUTOMATIC_PROGRESS_TRANSLATIONS[confidence_key]
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

    def _estimate_learned_pipeline_eta(self, now: float, history: list[Any]) -> tuple[float | None, bool]:
        if self.page_total <= 0:
            return None, False
        model, used_history = _fit_pipeline_eta_model(history)
        if model is None:
            return None, used_history
        intercept, sec_per_page = model
        estimated_total = intercept + sec_per_page * self.page_total
        elapsed = max(now - self.run_started_at, 0.0)
        return max(estimated_total - elapsed, 0.0), used_history

    def _estimate_progress(self, event: dict[str, Any]) -> float:
        # 남은 시간과 같은 이유로, 파이프라인이 준 진행률을 우선한다. 단계 sweep
        # 모델에서는 페이지 인덱스만으로 진행률을 되계산할 수 없다.
        supplied = event.get("progress_fraction")
        if isinstance(supplied, (int, float)) and not isinstance(supplied, bool):
            self._supplied_progress_percent = min(
                max(float(supplied) * 100.0, 0.0), 100.0
            )
            return self._supplied_progress_percent
        # sweep 이 아닌 이벤트에서 진행률이 뒤로 가지 않게, 마지막 값을 유지한다.
        # 미리보기 알림은 page_index 를 0 으로 보고하므로 되계산하면 진행률이
        # 처음으로 돌아간다.
        last_percent = getattr(self, "_supplied_progress_percent", None)
        if last_percent is not None:
            return last_percent
        if str(event.get("phase") or "") != "pipeline":
            return 0.0
        page_total = int(event.get("page_total") or self.page_total or 0)
        if page_total <= 0:
            return 0.0
        page_index = int(event.get("page_index") or 0)
        stage_name = str(event.get("stage_name") or "")
        stage_order = {
            "page_start": 0,
            "start-image": 0,
            "detect": 1,
            "text-block-detection": 1,
            "detector_overlay": 1,
            "ocr": 2,
            "ocr-processing": 2,
            "inpaint": 3,
            "pre-inpaint-setup": 3,
            "generate-mask": 3,
            "inpainting": 3,
            "raw_mask": 3,
            "mask_overlay": 3,
            "cleanup_delta": 3,
            "inpainted_image": 3,
            "translation": 4,
            "render": 5,
            "text-rendering-prepare": 5,
            "save": 6,
            "save-and-finish": 6,
            "page_done": 7,
            "finalizing_archive": 8,
        }
        units = page_total * 8
        current_units = min(max(page_index * 8 + stage_order.get(stage_name, 0), 0), units)
        return round((current_units / units) * 100.0, 1) if units else 0.0

    # v2: 융합 파이프라인에서 단계 보고가 교대로 들어오는데 추정기가 순차라고
    # 가정하는 바람에, 렌더가 인페인팅의 속도를 자기 것으로 학습했다(1.68초/page,
    # 실제 sweep 은 0.015초). 그 값으로 시드하면 남은 시간이 계속 부풀므로
    # 그룹 이름을 올려 오염된 이력을 버린다.
    STAGE_RATE_GROUP = "automatic_progress/stage_rates_v2"
    STAGE_STARTUP_GROUP = "automatic_progress/stage_startup_v2"

    def read_stage_startups(self) -> dict[str, float]:
        """지난 실행들에서 측정한 단계별 고정 시작 비용의 중앙값(초).

        컨테이너 기동과 모델 적재처럼 페이지 수와 무관한 비용이다. 이 값이 없으면
        아직 시작하지 않은 단계의 시작 비용이 남은 시간에서 통째로 빠진다.
        """

        return self._read_median_group(self.STAGE_STARTUP_GROUP)

    def record_stage_startups(self, startup_by_stage: dict[str, float]) -> None:
        """이번 실행의 단계별 시작 비용을 이력에 더한다."""

        self._append_positive_numbers(self.STAGE_STARTUP_GROUP, startup_by_stage)

    def _read_median_group(self, group: str) -> dict[str, float]:
        self.settings.beginGroup(group)
        keys = list(self.settings.childKeys())
        medians: dict[str, float] = {}
        for key in keys:
            samples = self.settings.value(key, [], type=list) or []
            values = []
            for sample in samples:
                try:
                    parsed = float(sample)
                except (TypeError, ValueError):
                    continue
                if parsed > 0.0:
                    values.append(parsed)
            if values:
                values.sort()
                medians[str(key)] = values[len(values) // 2]
        self.settings.endGroup()
        return medians

    def _append_positive_numbers(self, group: str, values: dict[str, float]) -> None:
        for name, raw in (values or {}).items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            self._append_history(group, str(name), value)

    def read_stage_rates(self) -> dict[str, float]:
        """지난 실행들에서 측정한 단계별 페이지당 속도의 중앙값.

        단계 sweep 추정기는 이 값을 비중으로 써서 첫 페이지부터 맞는 남은 시간을 낸다.
        표본이 없으면 빈 사전을 돌려주고, 추정기는 내장 사전 비중으로 시작한다.
        """

        self.settings.beginGroup(self.STAGE_RATE_GROUP)
        keys = list(self.settings.childKeys())
        rates: dict[str, float] = {}
        for key in keys:
            samples = self.settings.value(key, [], type=list) or []
            values = []
            for sample in samples:
                try:
                    parsed = float(sample)
                except (TypeError, ValueError):
                    continue
                if parsed > 0.0:
                    values.append(parsed)
            if values:
                values.sort()
                rates[str(key)] = values[len(values) // 2]
        self.settings.endGroup()
        return rates

    def record_stage_rates(self, per_page_by_stage: dict[str, float]) -> None:
        """이번 실행의 단계별 속도를 이력에 더한다. 중앙값이라 이상값에 흔들리지 않는다."""

        for name, rate in (per_page_by_stage or {}).items():
            try:
                value = float(rate)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            self._append_history(self.STAGE_RATE_GROUP, str(name), value)

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
