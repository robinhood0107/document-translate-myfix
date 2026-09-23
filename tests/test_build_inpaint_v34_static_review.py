from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_inpaint_v34_static_review as review


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_review_selection_is_bound_to_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = {
        "candidate_seen": False,
        "page_count": 2,
        "source_inventory_sha256": "a" * 64,
        "pages": [
            {"page_id": "neutral-1", "inventory_number": 1, "source_filename": "neutral-1.png", "corpus": "control"},
            {"page_id": "neutral-2", "inventory_number": 2, "source_filename": "neutral-2.png", "corpus": "development"},
        ],
        "translucent_carrier_annotations": [
            {"page_id": "neutral-2", "bbox_xyxy": [1, 2, 10, 20], "source_only": True}
        ],
    }
    path = tmp_path / "source-inventory.json"
    path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    seal = {
        "manifest_sha256": _sha(path),
        "source_inventory_sha256": inventory["source_inventory_sha256"],
        "candidate_seen": False,
        "page_count": 2,
    }
    path.with_suffix(".json.seal.json").write_text(
        json.dumps(seal, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(review, "SEALED_SOURCE_INVENTORY_FILE_SHA256", _sha(path))
    monkeypatch.setattr(
        review, "SEALED_SOURCE_INVENTORY_SHA256", inventory["source_inventory_sha256"]
    )

    pages, selected, highres = review._sealed_review_expectations(
        path, expected_page_count=2, expected_selected_count=1
    )
    assert pages == {
        "neutral-1": (1, "neutral-1.png"),
        "neutral-2": (2, "neutral-2.png"),
    }
    assert selected == {"neutral-2"}
    assert highres == {("neutral-2", (1, 2, 10, 20))}

    inventory["pages"][1]["page_id"] = "wrong-page"
    path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical source inventory seal differs"):
        review._sealed_review_expectations(
            path, expected_page_count=2, expected_selected_count=1
        )
