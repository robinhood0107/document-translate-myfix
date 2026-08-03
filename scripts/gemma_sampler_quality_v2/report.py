"""Sanitized reports for sampler v2; raw text remains in the private run."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .judgment import public_rank_summary
from .protocol import canonical_sha256


def build_phase_report(
    *,
    phase_status: Mapping[str, Any],
    ranked: Sequence[Mapping[str, Any]],
    scope: str,
) -> dict[str, Any]:
    """Build an aggregate with sampler metrics and hashes, never response text."""

    reference_sha256 = str(phase_status.get("reference_sha256") or "")
    summary = public_rank_summary(
        ranked,
        scope=scope,
        reference_sha256=reference_sha256,
    )
    return {
        "schema_version": "gemma-sampler-report-v3",
        "scope": scope,
        "phase": str(phase_status.get("phase") or ""),
        "phase_state": str(phase_status.get("state") or ""),
        "reference_sha256": reference_sha256,
        "expected_logical_slots": int(phase_status.get("expected_logical_slots") or 0),
        "completed_logical_slots": int(phase_status.get("completed_logical_slots") or 0),
        "rank": summary,
        "report_sha256": canonical_sha256(summary),
    }


def render_public_markdown(report: Mapping[str, Any]) -> str:
    """Render only public-safe aggregate values for the tracked latest report."""

    rank = report.get("rank")
    rows = rank.get("rows") if isinstance(rank, Mapping) else []
    lines = [
        "# Gemma sampler quality v2 latest report",
        "",
        "이 파일에는 원문, 번역문, 요청, 응답, 경로를 넣지 않는다.",
        "실제 raw 자료와 판정은 ignored private archive에만 보관한다.",
        "",
        f"- Phase: `{report.get('phase', '')}`",
        f"- State: `{report.get('phase_state', '')}`",
        f"- Reference SHA-256: `{report.get('reference_sha256', '')}`",
        f"- Completed slots: `{report.get('completed_logical_slots', 0)}` / `{report.get('expected_logical_slots', 0)}`",
        "",
        "| sampler | catastrophic | major | minor | unjudged | unique cases | naturalness | latency ms | completion tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def _metric(value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "n/a"
        numeric = float(value)
        return f"{numeric:.4f}" if math.isfinite(numeric) else "n/a"

    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {key} | {cat} | {major} | {minor} | {unjudged} | {cases} | {natural} | {latency} | {tokens} |".format(
                key=str(row.get("sampler_key") or ""),
                cat=int(row.get("catastrophic") or 0),
                major=int(row.get("major") or 0),
                minor=int(row.get("minor") or 0),
                unjudged=int(row.get("unjudged") or 0),
                cases=int(row.get("unique_error_cases") or 0),
                natural=_metric(row.get("naturalness_mean")),
                latency=_metric(row.get("latency_ms_mean")),
                tokens=_metric(row.get("completion_tokens_mean")),
            )
        )
    return "\n".join(lines) + "\n"
