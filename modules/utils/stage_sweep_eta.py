"""stage-batched 파이프라인 전용 남은 시간 추정기.

기존 추정기는 "한 페이지가 모든 단계를 지난다"는 레거시 전제로 만들어져 있었다.
stage-batched 는 반대로 **하나의 단계가 전체 페이지를 훑고, 그 다음 단계로 넘어간다.**
그래서 레거시 추정은 두 가지로 동시에 틀렸다.

* 파이프라인 쪽: 남은 페이지 x 페이지당 평균으로 계산해, 한 단계의 마지막 페이지에서
  `overall=99.8% eta=00:00:02` 를 냈다. 실제로는 그 뒤에 인페인팅·번역·렌더가 남아
  있었다.
* 트래커 쪽: 페이지 완료가 맨 끝에서만 일어나므로 표본이 0이라 과거 이력 기반 선형
  모델로 후퇴해 `eta=00:23:22` 를 냈다. 같은 순간에 두 값이 2초와 23분으로 갈렸다.

여기서는 작업량을 **(단계, 페이지) 칸**으로 모델링한다. 전체 칸 수는 시작 시점에
확정되고, 각 단계의 페이지당 실측 속도로 남은 칸을 환산한다. 아직 시작하지 않은
단계는 사전 비중으로 잡고, 그 단계의 첫 페이지가 끝나는 순간 실측으로 교체한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# 각 단계가 페이지당 대략 얼마나 무거운지에 대한 사전 비중. 상대값만 의미가 있다.
# 첫 표본이 들어오면 그 단계는 실측으로 대체되므로, 이 값은 "아직 한 번도 돌지 않은
# 단계"의 자리만 채운다.
#
# 값은 366장 실측에서 가져왔다. 짐작으로 넣은 첫 값들은 방향이 반대여서, 인페인팅
# sweep 이 시작되는 순간 남은 시간이 2분에서 13분으로 튀었다.
#
#   ocr-all      71s / 366장 = 0.19 s/page
#   inpaint-all 467s / 366장 = 1.28 s/page   (OCR 의 약 6.6배)
#
# OCR 이 가벼운 이유는 Router 안에서 crop 만 추론하고 영구 캐시가 히트하기 때문이고,
# 인페인팅이 무거운 이유는 페이지마다 마스크 생성과 LaMa 통과가 들어가기 때문이다.
# 측정한 값: ocr-all 1.0 : inpaint-all 6.6. 나머지는 아직 실측이 없는 추정치이며,
# 그 단계가 처음 도는 순간 실측으로 대체된다. 두 번째 실행부터는 아래 이력 학습이
# 이 표 전체를 대신하므로, 이 값들은 "첫 실행의 출발점" 역할만 한다.
DEFAULT_STAGE_WEIGHTS: dict[str, float] = {
    "detect-all": 1.0,
    "ocr-all": 1.0,
    "translate-all": 3.0,
    "inpaint-all": 6.6,
    "render-all": 1.5,
    "save-and-finish": 0.2,
}

# 단계가 실제로 도는 순서.
DEFAULT_STAGE_ORDER: tuple[str, ...] = (
    "detect-all",
    "ocr-all",
    "translate-all",
    "inpaint-all",
    "render-all",
    "save-and-finish",
)

# 페이지당 속도를 최근 표본에 맞추는 지수 평활 계수. 값이 클수록 최근을 더 믿는다.
# 페이지 무게는 작품 안에서도 크게 흔들리므로(대사 없는 페이지 대 꽉 찬 페이지)
# 과거를 완전히 버리지 않는 선에서 최근을 우선한다.
PER_PAGE_SMOOTHING = 0.25

# 모델 적재·해제처럼 페이지 수와 무관한 고정 비용. 실측으로 갱신된다.
DEFAULT_RUNTIME_SWAP_SEC = 0.0


@dataclass
class _StageState:
    """한 단계의 진행과 속도."""

    name: str
    pages_done: int = 0
    per_page_sec: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    # 페이지 수와 무관한 고정 시작 비용. 컨테이너 기동과 모델 적재가 여기 들어간다.
    startup_sec: float | None = None

    def observe_page(self, duration_sec: float) -> None:
        if duration_sec <= 0.0:
            return
        if self.per_page_sec is None:
            self.per_page_sec = duration_sec
        else:
            self.per_page_sec = (
                PER_PAGE_SMOOTHING * duration_sec
                + (1.0 - PER_PAGE_SMOOTHING) * self.per_page_sec
            )


@dataclass
class StageSweepEtaEstimator:
    """(단계 x 페이지) 칸 기준으로 남은 시간과 진행률을 계산한다."""

    page_total: int
    stage_order: tuple[str, ...] = DEFAULT_STAGE_ORDER
    stage_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_WEIGHTS)
    )
    run_started_at: float | None = None
    _stages: dict[str, _StageState] = field(default_factory=dict, init=False)
    _current_stage: str = field(default="", init=False)
    _last_page_at: float | None = field(default=None, init=False)
    _runtime_swap_sec: float = field(default=DEFAULT_RUNTIME_SWAP_SEC, init=False)
    _seeded_startup: dict[str, float] = field(default_factory=dict, init=False)
    _last_eta: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.page_total = max(int(self.page_total), 0)
        for name in self.stage_order:
            self._stages[name] = _StageState(name=name)

    def seed_from_history(self, per_page_by_stage: dict[str, float]) -> None:
        """지난 실행에서 측정한 페이지당 속도로 비중을 대체한다.

        사전 비중은 짐작이므로 첫 실행에서 남은 시간이 크게 틀린다. 실제로 짐작한
        비중이 방향까지 반대여서, 인페인팅 sweep 이 시작되는 순간 남은 시간이 2분에서
        13분으로 튀었다. 이력이 있으면 그 값을 비중으로 써서 첫 페이지부터 맞는
        추정을 낸다. 이력은 상대값으로만 쓰이므로 장비가 달라져도 비율이 유지되는 한
        유효하고, 실측이 들어오면 계속 갱신된다.
        """

        rates = {
            str(name): float(value)
            for name, value in (per_page_by_stage or {}).items()
            if isinstance(value, (int, float)) and float(value) > 0.0
        }
        if not rates:
            return
        reference = min(rates.values())
        for name, rate in rates.items():
            if name not in self._stages:
                self._stages[name] = _StageState(name=name)
                self.stage_order = self.stage_order + (name,)
            self.stage_weights[name] = rate / reference

    def measured_per_page_by_stage(self) -> dict[str, float]:
        """이번 실행에서 실제로 측정한 페이지당 속도. 다음 실행의 씨앗이 된다."""

        return {
            name: stage.per_page_sec
            for name, stage in self._stages.items()
            if stage.per_page_sec is not None and stage.per_page_sec > 0.0
        }

    # -- 관측 -------------------------------------------------------------

    def start_run(self, now: float) -> None:
        self.run_started_at = now
        self._last_page_at = now

    def observe(self, stage_name: str, page_index: int, now: float) -> None:
        """단계 ``stage_name`` 의 ``page_index`` 번째 페이지 진행을 기록한다.

        ``page_index`` 는 0부터 세며, 파이프라인이 보고하는 값을 그대로 받는다.
        같은 페이지가 여러 번 보고되어도 진행이 뒤로 가지 않는다.

        단계가 바뀌는 첫 보고는 시작 시각만 남기고 페이지 속도 표본으로 쓰지 않는다.
        그 구간에는 모델 적재·해제 같은 페이지와 무관한 비용이 섞여 있어, 그것을
        페이지당 속도로 환산하면 남은 시간이 크게 튄다. 속도는 두 번째 페이지부터
        측정되고, 그때까지는 사전 비중이 자리를 지킨다.
        """

        stage = self._stages.get(stage_name)
        if stage is None:
            # 사전에 없는 단계는 순서 끝에 붙인다. 알 수 없는 단계를 무시하면
            # 남은 작업을 과소평가한다.
            stage = _StageState(name=stage_name)
            self._stages[stage_name] = stage
            self.stage_order = self.stage_order + (stage_name,)
            self.stage_weights.setdefault(stage_name, 1.0)

        if self.run_started_at is None:
            self.start_run(now)

        if stage_name != self._current_stage:
            if self._current_stage:
                previous = self._stages.get(self._current_stage)
                if previous is not None and previous.finished_at is None:
                    previous.finished_at = now
                    # 이전 단계는 전체 페이지를 마친 것으로 본다.
                    previous.pages_done = max(previous.pages_done, self.page_total)
                # 이전 단계의 마지막 페이지와 이 단계의 첫 보고 사이의 공백은
                # 컨테이너 기동과 모델 적재 시간이다. 페이지와 무관한 고정 비용인데
                # 예전에는 그냥 버려서 남은 시간에서 통째로 빠졌다. 그래서 무거운
                # 단계로 넘어가는 순간 남은 시간이 위로 튀었다.
                if self._last_page_at is not None:
                    self.observe_stage_startup(stage_name, now - self._last_page_at)
            self._current_stage = stage_name
            stage.started_at = now
            self._last_page_at = now

        done = min(max(int(page_index) + 1, 0), self.page_total or (page_index + 1))
        if done > stage.pages_done:
            if self._last_page_at is not None:
                spent = now - self._last_page_at
                pages = done - stage.pages_done
                if pages > 0:
                    stage.observe_page(spent / pages)
            stage.pages_done = done
            self._last_page_at = now

    def observe_stage_startup(self, stage_name: str, elapsed_sec: float) -> None:
        """단계 하나의 고정 시작 비용(컨테이너 기동·모델 적재)을 기록한다."""

        if elapsed_sec <= 0.0:
            return
        stage = self._stages.get(stage_name)
        if stage is None:
            return
        stage.startup_sec = float(elapsed_sec)
        self._runtime_swap_sec += float(elapsed_sec)

    def observe_runtime_swap(self, elapsed_sec: float) -> None:
        """단계 경계 밖에서 일어난 런타임 교체 비용을 현재 단계에 더한다."""

        if elapsed_sec <= 0.0:
            return
        self._runtime_swap_sec += float(elapsed_sec)
        stage = self._stages.get(self._current_stage)
        if stage is not None:
            stage.startup_sec = float(stage.startup_sec or 0.0) + float(elapsed_sec)

    def measured_startup_by_stage(self) -> dict[str, float]:
        """이번 실행에서 측정한 단계별 시작 비용. 다음 실행의 씨앗이 된다."""

        return {
            name: stage.startup_sec
            for name, stage in self._stages.items()
            if stage.startup_sec is not None and stage.startup_sec > 0.0
        }

    def seed_startup_from_history(self, startup_by_stage: dict[str, float]) -> None:
        """지난 실행에서 측정한 시작 비용을 아직 시작하지 않은 단계에 미리 채운다."""

        for name, value in (startup_by_stage or {}).items():
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                continue
            stage = self._stages.get(str(name))
            if stage is None:
                stage = _StageState(name=str(name))
                self._stages[str(name)] = stage
                self.stage_order = self.stage_order + (str(name),)
                self.stage_weights.setdefault(str(name), 1.0)
            self._seeded_startup[str(name)] = float(value)

    def stage_startup_estimate(self, stage_name: str) -> float:
        """단계 시작 비용 추정치(초). 실측이 있으면 실측, 없으면 이력, 없으면 0."""

        stage = self._stages.get(stage_name)
        if stage is not None and stage.startup_sec is not None:
            return float(stage.startup_sec)
        seeded = self._seeded_startup.get(stage_name)
        if seeded is not None:
            return float(seeded)
        return 0.0

    # -- 추정 -------------------------------------------------------------

    def _reference_per_page(self) -> float | None:
        """실측된 단계들로부터 가중치 1.0 에 해당하는 페이지당 시간을 구한다."""

        ratios: list[float] = []
        for name, stage in self._stages.items():
            if stage.per_page_sec is None:
                continue
            weight = float(self.stage_weights.get(name, 1.0))
            if weight > 0.0:
                ratios.append(stage.per_page_sec / weight)
        if not ratios:
            return None
        return sum(ratios) / len(ratios)

    def stage_per_page_estimate(self, stage_name: str) -> float | None:
        stage = self._stages.get(stage_name)
        if stage is not None and stage.per_page_sec is not None:
            return stage.per_page_sec
        reference = self._reference_per_page()
        if reference is None:
            return None
        return reference * float(self.stage_weights.get(stage_name, 1.0))

    def remaining_seconds(self) -> float | None:
        """남은 시간(초). 실측 표본이 없으면 ``None``."""

        if self.page_total <= 0:
            return None
        if self._reference_per_page() is None:
            return None

        remaining = 0.0
        seen_current = False
        for name in self.stage_order:
            stage = self._stages[name]
            per_page = self.stage_per_page_estimate(name)
            if per_page is None:
                continue
            if name == self._current_stage:
                seen_current = True
                remaining += per_page * max(self.page_total - stage.pages_done, 0)
                continue
            if not seen_current and stage.finished_at is not None:
                continue
            if not seen_current and stage.pages_done >= self.page_total:
                continue
            if seen_current:
                # 아직 시작하지 않은 단계는 페이지 비용에 더해 고정 시작 비용도
                # 남아 있다. 이걸 빼놓으면 컨테이너 기동과 모델 적재에 드는
                # 시간만큼 남은 시간을 계속 과소평가한다.
                remaining += self.stage_startup_estimate(name)
                remaining += per_page * self.page_total
            else:
                # 현재 단계보다 앞인데 끝나지 않은 단계. 남은 페이지만 센다.
                remaining += per_page * max(self.page_total - stage.pages_done, 0)

        self._last_eta = max(remaining, 0.0)
        return self._last_eta

    def completed_units(self) -> float:
        units = 0.0
        for name in self.stage_order:
            stage = self._stages[name]
            weight = float(self.stage_weights.get(name, 1.0))
            units += weight * min(stage.pages_done, self.page_total)
        return units

    def total_units(self) -> float:
        return sum(
            float(self.stage_weights.get(name, 1.0)) * self.page_total
            for name in self.stage_order
        )

    def progress_fraction(self) -> float:
        """0.0~1.0 진행률. 단계 경계에서 되돌아가지 않는다."""

        total = self.total_units()
        if total <= 0.0:
            return 0.0
        return min(max(self.completed_units() / total, 0.0), 1.0)
