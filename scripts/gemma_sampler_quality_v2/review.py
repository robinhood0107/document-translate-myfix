"""Local-only HTML review board for private reference and judgment packets."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping


class ReviewBoardError(ValueError):
    """Raised when a private review packet cannot be safely rendered."""


def _rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ReviewBoardError("Private review packet has no row list.")
    return [row for row in rows if isinstance(row, Mapping)]


def render_private_review_html(payload: Mapping[str, Any], *, title: str) -> str:
    """Render escaped source/candidate text to a file that remains private."""

    cards: list[str] = []
    for index, row in enumerate(_rows(payload), start=1):
        identity = str(row.get("case_id") or row.get("blind_id") or row.get("cluster_id") or index)
        fields = [
            ("원문", row.get("source_text")),
            ("다음 문맥", row.get("context_after_text")),
            ("정답", row.get("canonical_translation")),
            ("독립 검수", (row.get("blind_review") or {}).get("independent_translation") if isinstance(row.get("blind_review"), Mapping) else row.get("independent_translation")),
            ("후보 출력", row.get("candidate_translation")),
            ("필수 의미", row.get("required_meaning")),
            ("금지 변화", row.get("prohibited_changes")),
            ("flag", row.get("flags")),
        ]
        rendered = []
        for label, value in fields:
            if value in (None, "", []):
                continue
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            rendered.append(
                f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value), quote=False)}</dd>"
            )
        cards.append(
            "<article class='card' data-id='{}'><h2>{:03d}. {}</h2><dl>{}</dl>"
            "<label>검수 메모 <textarea data-id='{}'></textarea></label></article>".format(
                html.escape(identity, quote=True),
                index,
                html.escape(identity),
                "".join(rendered),
                html.escape(identity, quote=True),
            )
        )
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>{escaped_title}</title>
<style>
body {{ margin: 0 auto; max-width: 1080px; padding: 24px; background: #101419; color: #e8edf2; font: 15px/1.5 system-ui, sans-serif; }}
h1 {{ margin-top: 0; }} .notice {{ color: #f5c86b; }} .card {{ border: 1px solid #34414e; border-radius: 10px; padding: 16px; margin: 14px 0; background: #182029; }}
h2 {{ font-size: 16px; margin: 0 0 10px; }} dl {{ display: grid; grid-template-columns: 130px 1fr; gap: 6px 12px; margin: 0; }} dt {{ color: #8dc5ff; }} dd {{ white-space: pre-wrap; margin: 0; }}
textarea {{ display: block; width: 100%; min-height: 54px; margin-top: 5px; box-sizing: border-box; }}
</style></head><body>
<h1>{escaped_title}</h1>
<p class="notice">이 파일은 private validation archive 밖으로 복사하거나 stage하지 마세요. 메모는 브라우저 localStorage에만 저장됩니다.</p>
{''.join(cards)}
<script>
for (const box of document.querySelectorAll('textarea')) {{
  const key = 'gemma-sampler-v2:' + box.dataset.id;
  box.value = localStorage.getItem(key) || '';
  box.addEventListener('input', () => localStorage.setItem(key, box.value));
}}
</script></body></html>
"""
