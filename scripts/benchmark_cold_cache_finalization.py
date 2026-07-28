#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
PROTOCOL_PATH = (
    ROOT
    / "benchmarks"
    / "cold_cache_finalization"
    / "protocol-v1.json"
)
PIPELINE_RUNNER = SCRIPT_DIR / "benchmark_pipeline.py"
MANAGED_CONTAINERS = (
    "gemma-local-server",
    "paddleocr-server",
    "paddleocr-vllm",
)
PROTOCOL_STATE_NAME = "protocol_state.json"
SUITE_STATE_NAME = "suite_state.json"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "report.md"
REPRODUCIBILITY_FILES = (
    PROTOCOL_PATH,
    SCRIPT_DIR / "benchmark_cold_cache_finalization.py",
    SCRIPT_DIR / "benchmark_pipeline.py",
    SCRIPT_DIR / "benchmark_common.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return str(completed.stdout or "").strip()


def _require_reproducible_checkout() -> None:
    dirty = _git_output("status", "--porcelain")
    if dirty:
        raise RuntimeError(
            "Formal cold/cache runs require a clean Git checkout. Commit the "
            "lab runner first; use benchmark_pipeline.py directly for an "
            "uncommitted development smoke."
        )


def ensure_external_output(path: Path, *, require_new: bool) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "Cold/cache finalization output must be outside the Git repository."
        )
    if require_new and resolved.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing benchmark output directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=not require_new)
    return resolved


def load_protocol() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    if int(protocol.get("protocol_version", 0) or 0) != 1:
        raise ValueError("Unsupported cold/cache finalization protocol version.")
    limits = protocol.get("limits")
    gates = protocol.get("gates")
    if not isinstance(limits, dict) or not isinstance(gates, dict):
        raise ValueError("Protocol limits and gates must be JSON objects.")
    if int(limits.get("pipeline_max_pages", 0) or 0) > 6:
        raise ValueError("Protocol may not select more than six screening pages.")
    if int(limits.get("pipeline_max_blocks", 0) or 0) > 54:
        raise ValueError("Protocol may not select more than 54 screening blocks.")
    if int(limits.get("completion_tokens", 0) or 0) != 512:
        raise ValueError("Translation completion-token limit must remain 512.")
    if int(limits.get("cache_stabilization_pairs", 0) or 0) != 1:
        raise ValueError(
            "Cache verification must use one unscored stabilization pair."
        )
    models = protocol.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("Protocol model contracts are missing.")
    for model_key, model in models.items():
        if not isinstance(model, Mapping):
            raise ValueError(f"Invalid model contract: {model_key}")
        digest = str(model.get("sha256", "") or "").lower()
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or int(model.get("bytes", 0) or 0) <= 0
        ):
            raise ValueError(f"Invalid model digest contract: {model_key}")
    baseline_path = ROOT / str(protocol.get("baseline_preset", ""))
    baseline = _read_json(baseline_path)
    _validate_baseline_contract(baseline)
    return protocol


def _validate_baseline_contract(preset: Mapping[str, Any]) -> None:
    app = preset.get("app")
    gemma = preset.get("gemma")
    cache = preset.get("benchmark_cache_policy")
    if not isinstance(app, Mapping) or not isinstance(gemma, Mapping):
        raise ValueError("Baseline app and Gemma contracts are missing.")
    expected = {
        "model": "gemma-4-26B-IQ4_NL.gguf",
        "chunk_size": 6,
        "context_size": 4096,
        "n_parallel": 1,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "spec_type": "none",
        "max_completion_tokens": 512,
    }
    mismatches = {
        key: {"expected": value, "actual": gemma.get(key)}
        for key, value in expected.items()
        if gemma.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Baseline Gemma contract drifted: {mismatches}")
    if str(app.get("workflow_mode", "")) != "stage_batched_pipeline":
        raise ValueError("Baseline must use the product stage-batched pipeline.")
    if not isinstance(cache, Mapping) or any(
        bool(cache.get(key, False))
        for key in (
            "paddleocr_persistent",
            "translation_persistent",
            "exact_tm",
            "project_checkpoint",
        )
    ):
        raise ValueError("Cold baseline must disable every persistent cache.")
    contract = preset.get("benchmark_contract")
    if (
        not isinstance(contract, Mapping)
        or str(contract.get("request_mode", ""))
        != "contextual-single"
        or not bool(contract.get("prompt_schema_sampler_unchanged", False))
    ):
        raise ValueError(
            "Baseline must lock contextual-single and the unchanged "
            "prompt/schema/sampler contract."
        )


def _validate_cold_candidate_base(preset: Mapping[str, Any]) -> None:
    app = preset.get("app")
    gemma = preset.get("gemma")
    cache = preset.get("benchmark_cache_policy")
    contract = preset.get("benchmark_contract")
    if (
        not isinstance(app, Mapping)
        or not isinstance(gemma, Mapping)
        or not isinstance(cache, Mapping)
        or not isinstance(contract, Mapping)
    ):
        raise ValueError("Candidate base preset is missing a cold contract.")
    if str(app.get("workflow_mode", "")) != "stage_batched_pipeline":
        raise ValueError(
            "Candidate base preset must use the stage-batched pipeline."
        )
    if str(contract.get("request_mode", "")) != "contextual-single":
        raise ValueError(
            "Candidate base preset may not re-enable contextual-grouped."
        )
    if not bool(contract.get("prompt_schema_sampler_unchanged", False)):
        raise ValueError(
            "Candidate base preset may not change prompt/schema/sampler."
        )
    if any(
        bool(cache.get(key, False))
        for key in (
            "paddleocr_persistent",
            "translation_persistent",
            "exact_tm",
            "project_checkpoint",
        )
    ):
        raise ValueError(
            "Candidate base preset must disable every persistent cache."
        )
    expected = {
        "max_completion_tokens": 512,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "spec_type": "none",
    }
    mismatches = {
        key: {"expected": value, "actual": gemma.get(key)}
        for key, value in expected.items()
        if gemma.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Candidate base Gemma safety contract drifted: {mismatches}"
        )


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(
                dict(merged[key]),
                value,
            )
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _family(protocol: Mapping[str, Any], family_id: str) -> dict[str, Any]:
    for family in protocol.get("families", []):
        if (
            isinstance(family, dict)
            and str(family.get("id", "")) == family_id
        ):
            return family
    raise KeyError(f"Unknown benchmark family: {family_id}")


def _axis_patch(axis: str, value: Any) -> dict[str, Any]:
    if axis == "parallel_workers":
        return {"ocr_client": {"parallel_workers": int(value)}}
    if axis in {"max_num_seqs", "max_num_batched_tokens"}:
        return {"ocr_runtime": {axis: int(value)}}
    raise ValueError(f"Unsupported generated pipeline axis: {axis}")


def pipeline_candidates(
    family: Mapping[str, Any],
    *,
    axis: str = "",
) -> tuple[list[dict[str, Any]], str]:
    execution = str(family.get("execution", ""))
    if execution == "pipeline":
        candidates = [
            dict(candidate)
            for candidate in family.get("candidates", [])
            if isinstance(candidate, Mapping)
        ]
        baseline_id = str(family.get("baseline_candidate", ""))
    elif execution == "generated-sequential":
        axes = family.get("axes")
        if not isinstance(axes, Mapping) or axis not in axes:
            raise ValueError(
                f"--axis must select one of: {', '.join(sorted(axes or {}))}"
            )
        values = list(axes[axis])
        candidates = [
            {
                "id": f"{axis}-{value}",
                "patch": _axis_patch(axis, value),
            }
            for value in values
        ]
        baseline_values = family.get("baseline_values")
        if not isinstance(baseline_values, Mapping):
            raise ValueError("Generated family has no baseline_values contract.")
        baseline_id = f"{axis}-{baseline_values[axis]}"
    else:
        raise ValueError(
            f"Family {family.get('id')} is not a pipeline benchmark family."
        )
    candidate_ids = [str(candidate.get("id", "")) for candidate in candidates]
    if not candidates or any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("Every pipeline candidate must have a non-empty id.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Pipeline candidate ids must be unique.")
    if baseline_id not in candidate_ids:
        raise ValueError(
            f"Baseline candidate {baseline_id!r} is not in {candidate_ids}."
        )
    return candidates, baseline_id


def balanced_orders(candidate_ids: Iterable[str], rounds: int) -> list[list[str]]:
    ids = list(candidate_ids)
    if not ids:
        raise ValueError("At least one candidate is required.")
    if rounds < 1:
        raise ValueError("Round count must be positive.")
    orders: list[list[str]] = []
    for round_index in range(rounds):
        if len(ids) == 2:
            orders.append(
                list(ids if round_index % 2 == 0 else reversed(ids))
            )
            continue
        if round_index % 2:
            order = list(reversed(ids))
        else:
            offset = (round_index // 2) % len(ids)
            order = ids[offset:] + ids[:offset]
        orders.append(order)
    return orders


def _stop_managed_containers() -> None:
    subprocess.run(
        [
            "docker",
            "stop",
            "--time",
            "20",
            *MANAGED_CONTAINERS,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def _isolated_environment(user_data_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    user_data_root.mkdir(parents=True, exist_ok=True)
    env["LOCALAPPDATA"] = str(user_data_root)
    env["XDG_DATA_HOME"] = str(user_data_root)
    env["CT_DISABLE_UPDATE_CHECK"] = "1"
    env["CT_BENCH_CLEAR_APP_CACHES"] = "1"
    env["CT_ENABLE_MEMLOG"] = "1"
    env["CT_ENABLE_GPU_BENCH"] = "1"
    return env


def _run_process(
    command: list[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    (stdout_path.parent / "process_elapsed_sec.txt").write_text(
        f"{elapsed:.6f}\n",
        encoding="utf-8",
    )
    return int(completed.returncode)


def _pipeline_command(
    *,
    preset_path: Path,
    input_dir: Path,
    sample_count: int,
    run_dir: Path,
    shared_corpus_dir: Path,
    stage: str,
    source_lang: str,
    project_file: Path | None = None,
    project_action: str = "none",
    save_project_after_run: bool = False,
    project_invalidate_page_index: int = -1,
    project_invalidate_stage: str = "ocr",
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE_RUNNER),
        "--preset",
        str(preset_path),
        "--mode",
        "batch",
        "--repeat",
        "1",
        "--sample-dir",
        str(input_dir),
        "--sample-count",
        str(sample_count),
        "--source-lang",
        source_lang,
        "--target-lang",
        "Korean",
        "--runtime-mode",
        "attach-running",
        "--product-managed-runtime",
        "--output-dir",
        str(run_dir),
        "--shared-corpus-dir",
        str(shared_corpus_dir),
        "--stage-ceiling",
        "ocr" if stage == "ocr" else "render",
        "--runtime-services",
        "ocr-only" if stage == "ocr" else "full",
        "--clear-app-caches",
        "--export-page-snapshots",
    ]
    if project_file is not None:
        command.extend(
            [
                "--project-file",
                str(project_file),
                "--project-action",
                project_action,
            ]
        )
    if save_project_after_run:
        command.append("--save-project-after-run")
    if project_invalidate_page_index >= 0:
        command.extend(
            [
                "--project-invalidate-page-index",
                str(project_invalidate_page_index),
                "--project-invalidate-stage",
                project_invalidate_stage,
            ]
        )
    return command


def _checkpoint_event_contract(
    metrics_path: Path,
    snapshot_path: Path,
) -> dict[str, Any]:
    snapshot = _read_json(snapshot_path)
    pages = snapshot.get("pages")
    if not isinstance(pages, list):
        raise ValueError(
            f"Page snapshot has no page list for event matching: {snapshot_path}"
        )
    page_indexes = {
        str(page.get("image_path", "") or ""): index
        for index, page in enumerate(pages)
        if isinstance(page, Mapping)
    }
    tag_to_stage = {
        "detect_end": "detection",
        "ocr_end": "ocr",
        "translate_end": "translation",
        "inpaint_end": "inpaint",
        "render_end": "render",
    }
    entries: list[dict[str, Any]] = []
    unmatched = 0
    for row in _read_jsonl(metrics_path):
        stage = tag_to_stage.get(str(row.get("tag", "") or ""))
        if not stage:
            continue
        image_path = str(row.get("image_path", "") or "")
        page_index = page_indexes.get(image_path, -1)
        if page_index < 0:
            unmatched += 1
        entries.append(
            {
                "page_index": page_index,
                "stage": stage,
                "project_checkpoint_status": str(
                    row.get("project_checkpoint_status", "") or ""
                ).strip().lower(),
                "cache_status": str(
                    row.get("cache_status", "") or ""
                ).strip().lower(),
                "skip_reason": str(
                    row.get("skip_reason", "") or ""
                ).strip().lower(),
                "render_skipped": bool(row.get("render_skipped", False)),
                "output_materialized": bool(
                    row.get("output_materialized", False)
                ),
            }
        )
    entries.sort(
        key=lambda item: (
            int(item["page_index"]),
            str(item["stage"]),
        )
    )
    return {
        "page_count": len(pages),
        "event_count": len(entries),
        "unmatched_event_count": unmatched,
        "entries": entries,
        "sha256": _canonical_sha256(entries),
    }


def _page_contract(snapshot_path: Path) -> dict[str, Any]:
    payload = _read_json(snapshot_path)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Page snapshot has no page list: {snapshot_path}")
    stage_payloads: dict[str, list[Any]] = {
        "detection": [],
        "ocr": [],
        "translation": [],
        "render": [],
    }
    block_count = 0
    translated_source_count = 0
    empty_translation_count = 0
    page_statuses: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, Mapping):
            raise ValueError("Page snapshot entry must be an object.")
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("Page snapshot blocks must be a list.")
        block_count += len(blocks)
        detection_blocks: list[dict[str, Any]] = []
        ocr_blocks: list[dict[str, Any]] = []
        translation_blocks: list[dict[str, Any]] = []
        render_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ValueError("Page snapshot block must be an object.")
            detection = {
                key: block.get(key)
                for key in (
                    "xyxy",
                    "bubble_xyxy",
                    "angle",
                    "text_class",
                    "block_mask_bbox",
                    "block_mask_source",
                    "block_mask_decision",
                )
            }
            ocr = {
                **detection,
                **{
                    key: block.get(key)
                    for key in (
                        "text",
                        "ocr_status",
                        "ocr_empty_reason",
                        "ocr_attempt_count",
                        "ocr_raw_text",
                        "ocr_sanitized_text",
                        "ocr_reject_reason",
                    )
                },
            }
            translation = {
                **ocr,
                "translation": block.get("translation"),
            }
            if str(block.get("text", "") or "").strip():
                translated_source_count += 1
                if not str(block.get("translation", "") or "").strip():
                    empty_translation_count += 1
            render = {
                **translation,
                **{
                    key: block.get(key)
                    for key in (
                        "render_text",
                        "render_area_source",
                        "render_source_xyxy",
                        "render_anchor_xyxy",
                        "render_bubble_xyxy",
                        "render_normalization_applied",
                        "render_normalization_reasons",
                        "render_normalization_replacements",
                        "render_centered_layout",
                        "render_layout_reasons",
                        "text_fit_status",
                        "text_fit_metrics",
                    )
                },
            }
            detection_blocks.append(detection)
            ocr_blocks.append(ocr)
            translation_blocks.append(translation)
            render_blocks.append(render)
        stage_payloads["detection"].append(
            {"page_index": page_index, "blocks": detection_blocks}
        )
        stage_payloads["ocr"].append(
            {"page_index": page_index, "blocks": ocr_blocks}
        )
        stage_payloads["translation"].append(
            {"page_index": page_index, "blocks": translation_blocks}
        )
        stage_payloads["render"].append(
            {
                "page_index": page_index,
                "blocks": render_blocks,
                "output_sha256": page.get("translated_image_sha256", ""),
            }
        )
        page_statuses.append(
            {
                "page_index": page_index,
                "block_count": len(blocks),
                "failed": bool(page.get("page_failed", False)),
                "stage_status": page.get("stage_status", {}),
                "output_path": str(page.get("translated_image_path", "") or ""),
                "output_exists": bool(
                    page.get("translated_image_exists", False)
                ),
                "output_sha256": str(
                    page.get("translated_image_sha256", "") or ""
                ),
            }
        )
    return {
        "page_count": len(pages),
        "block_count": block_count,
        "translated_source_count": translated_source_count,
        "empty_translation_count": empty_translation_count,
        "detection_sha256": _canonical_sha256(stage_payloads["detection"]),
        "ocr_sha256": _canonical_sha256(stage_payloads["ocr"]),
        "translation_sha256": _canonical_sha256(
            stage_payloads["translation"]
        ),
        "render_sha256": _canonical_sha256(stage_payloads["render"]),
        "pages": page_statuses,
    }


def _write_private_pipeline_review(
    results: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    run_labels: list[str] = []
    for result in sorted(
        results,
        key=lambda item: (
            int(item.get("round", 0) or 0),
            int(item.get("order", 0) or 0),
        ),
    ):
        candidate = str(result.get("candidate", "") or "")
        round_index = int(result.get("round", 0) or 0)
        run_label = f"{candidate}/round-{round_index}"
        if not candidate or round_index <= 0 or run_label in run_labels:
            raise ValueError("Pipeline private review run identity is invalid.")
        run_labels.append(run_label)
        snapshot_path = Path(str(result.get("run_dir", ""))) / "page_snapshots.json"
        snapshot = _read_json(snapshot_path)
        pages = snapshot.get("pages")
        if not isinstance(pages, list):
            raise ValueError(
                f"Pipeline private review snapshot has no pages: {snapshot_path}"
            )
        seen: set[tuple[int, int]] = set()
        for page_index, page in enumerate(pages):
            blocks = page.get("blocks") if isinstance(page, Mapping) else None
            if not isinstance(blocks, list):
                raise ValueError("Pipeline private review page has no blocks.")
            for block_index, block in enumerate(blocks):
                if not isinstance(block, Mapping):
                    raise ValueError(
                        "Pipeline private review block must be an object."
                    )
                key = (page_index, block_index)
                if key in seen:
                    raise ValueError("Duplicate pipeline private review row.")
                seen.add(key)
                row = rows.setdefault(
                    key,
                    {
                        "page_index": page_index,
                        "block_index": block_index,
                        "outputs": {},
                        "candidate_only_regression": "",
                        "regression_type": "",
                        "notes": "",
                    },
                )
                row["outputs"][run_label] = {
                    "source": str(block.get("text", "") or ""),
                    "raw_ocr": str(block.get("ocr_raw_text", "") or ""),
                    "translation": str(block.get("translation", "") or ""),
                    "ocr_status": str(block.get("ocr_status", "") or ""),
                }
        if set(rows) != seen:
            raise ValueError(
                "Pipeline private review page/block structure changed between runs."
            )
    payload = {
        "schema_version": 1,
        "kind": "cold-pipeline-private-meaning-review",
        "run_labels": run_labels,
        "row_count": len(rows),
        "review_status": "pending",
        "rows": [rows[key] for key in sorted(rows)],
    }
    _write_json(output_path, payload)
    return {
        "relative_path": f"private/{output_path.name}",
        "row_count": len(rows),
        "sha256": _sha256_file(output_path),
        "review_status": "pending",
    }


def _write_private_translation_review(
    results: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    run_labels: list[str] = []
    for result in sorted(
        results,
        key=lambda item: (
            int(item.get("round", 0) or 0),
            int(item.get("order", 0) or 0),
        ),
    ):
        candidate = str(result.get("candidate", "") or "")
        round_index = int(result.get("round", 0) or 0)
        run_label = f"{candidate}/round-{round_index}"
        if not candidate or round_index <= 0 or run_label in run_labels:
            raise ValueError("Translation private review run identity is invalid.")
        run_labels.append(run_label)
        private_output = Path(str(result.get("private_output", "") or ""))
        payload = _read_json(private_output)
        outputs = payload.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 54:
            raise ValueError(
                f"Translation private review requires 54 outputs: {private_output}"
            )
        seen: set[tuple[str, int]] = set()
        for item in outputs:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "Translation private review output must be an object."
                )
            language = str(item.get("language", "") or "")
            index = int(item.get("index", -1) or 0)
            key = (language, index)
            if not language or index < 0 or key in seen:
                raise ValueError(
                    "Translation private review output identity is invalid."
                )
            seen.add(key)
            source = str(item.get("source", "") or "")
            case_id = str(item.get("case_id", "") or "")
            row = rows.setdefault(
                key,
                {
                    "language": language,
                    "index": index,
                    "case_id": case_id,
                    "source": source,
                    "outputs": {},
                    "candidate_only_regression": "",
                    "regression_type": "",
                    "notes": "",
                },
            )
            if row["case_id"] != case_id or row["source"] != source:
                raise ValueError(
                    "Translation private review source contract changed between runs."
                )
            row["outputs"][run_label] = str(
                item.get("translation", "") or ""
            )
        if set(rows) != seen:
            raise ValueError(
                "Translation private review key structure changed between runs."
            )
    payload = {
        "schema_version": 1,
        "kind": "cold-translation-private-quality-review",
        "run_labels": run_labels,
        "row_count": len(rows),
        "review_status": "pending",
        "rows": [rows[key] for key in sorted(rows)],
    }
    _write_json(output_path, payload)
    return {
        "relative_path": f"private/{output_path.name}",
        "row_count": len(rows),
        "sha256": _sha256_file(output_path),
        "review_status": "pending",
    }


def _safe_pipeline_result(
    *,
    candidate_id: str,
    round_index: int,
    order_index: int,
    run_dir: Path,
    returncode: int,
) -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_NAME
    snapshot_path = run_dir / "page_snapshots.json"
    metrics_path = run_dir / "metrics.jsonl"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    contract = (
        _page_contract(snapshot_path)
        if snapshot_path.is_file()
        else {
            "page_count": 0,
            "block_count": 0,
        }
    )
    performance = summary.get("performance_stats")
    if not isinstance(performance, Mapping):
        performance = {}
    checkpoint_events = (
        _checkpoint_event_contract(metrics_path, snapshot_path)
        if metrics_path.is_file() and snapshot_path.is_file()
        else {
            "page_count": 0,
            "event_count": 0,
            "unmatched_event_count": 0,
            "entries": [],
            "sha256": "",
        }
    )
    return {
        "candidate": candidate_id,
        "round": int(round_index),
        "order": int(order_index),
        "returncode": int(returncode),
        "status": (
            "passed"
            if returncode == 0
            and int(summary.get("page_failed_count", 0) or 0) == 0
            else "failed"
        ),
        "elapsed_sec": float(summary.get("elapsed_sec", 0.0) or 0.0),
        "page_done_count": int(summary.get("page_done_count", 0) or 0),
        "page_failed_count": int(
            summary.get("page_failed_count", 0) or 0
        ),
        "performance_stats": copy.deepcopy(dict(performance)),
        "page_contract": contract,
        "checkpoint_events": checkpoint_events,
        "run_dir": str(run_dir),
    }


def _stage_elapsed(result: Mapping[str, Any], stage: str) -> float:
    performance = result.get("performance_stats")
    if isinstance(performance, Mapping):
        if stage == "full":
            wall_ms = performance.get("run_wall_ms")
            if isinstance(wall_ms, (int, float)) and not isinstance(
                wall_ms, bool
            ):
                return float(wall_ms) / 1000.0
        stages = performance.get("stages")
        stage_values = stages.get(stage) if isinstance(stages, Mapping) else None
        if isinstance(stage_values, Mapping):
            wall_ms = stage_values.get("wall_ms")
            if isinstance(wall_ms, (int, float)) and not isinstance(
                wall_ms, bool
            ):
                return float(wall_ms) / 1000.0
    return float(result.get("elapsed_sec", 0.0) or 0.0)


def _reduction_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return ((baseline - candidate) / baseline) * 100.0


def analyze_pipeline_results(
    *,
    protocol: Mapping[str, Any],
    family: Mapping[str, Any],
    baseline_id: str,
    results: list[dict[str, Any]],
    full_reference: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    limits = dict(protocol["limits"])
    gates = dict(protocol["gates"])
    stage = str(family.get("stage", "full"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["candidate"])].append(result)
    if baseline_id not in grouped:
        raise ValueError("Pipeline results do not contain the baseline.")

    candidate_summaries: list[dict[str, Any]] = []
    baseline_values = [
        _stage_elapsed(result, stage)
        for result in grouped[baseline_id]
        if str(result.get("status")) == "passed"
    ]
    baseline_median = (
        statistics.median(baseline_values) if baseline_values else 0.0
    )
    baseline_screening_wall_values = [
        _stage_elapsed(result, "full")
        for result in grouped[baseline_id]
        if str(result.get("status")) == "passed"
    ]
    baseline_screening_wall_median = (
        statistics.median(baseline_screening_wall_values)
        if baseline_screening_wall_values
        else 0.0
    )
    baseline_ocr_digests = {
        str((result.get("page_contract") or {}).get("ocr_sha256", ""))
        for result in grouped[baseline_id]
    }
    baseline_detection_digests = {
        str(
            (result.get("page_contract") or {}).get(
                "detection_sha256",
                "",
            )
        )
        for result in grouped[baseline_id]
    }
    quality_gate = str(family.get("quality_gate", ""))
    exact_quality_required = quality_gate == "exact-raw-ocr"
    structural_quality_required = quality_gate in {
        "exact-raw-ocr",
        "structure-and-private-meaning-review",
    }
    baseline_stable = (
        len(baseline_ocr_digests) == 1
        and len(baseline_detection_digests) == 1
        and "" not in baseline_ocr_digests
        and "" not in baseline_detection_digests
    )

    for candidate_id, candidate_results in grouped.items():
        passed_runs = [
            result
            for result in candidate_results
            if str(result.get("status")) == "passed"
        ]
        timings = [
            _stage_elapsed(result, stage)
            for result in passed_runs
        ]
        median_elapsed = statistics.median(timings) if timings else 0.0
        screening_wall_timings = [
            _stage_elapsed(result, "full")
            for result in passed_runs
        ]
        screening_wall_median = (
            statistics.median(screening_wall_timings)
            if screening_wall_timings
            else 0.0
        )
        variance_percent = (
            ((max(timings) - min(timings)) / median_elapsed) * 100.0
            if timings and median_elapsed > 0
            else 0.0
        )
        screening_wall_variance_percent = (
            (
                (max(screening_wall_timings) - min(screening_wall_timings))
                / screening_wall_median
            )
            * 100.0
            if screening_wall_timings and screening_wall_median > 0
            else 0.0
        )
        page_limits_passed = all(
            int((result.get("page_contract") or {}).get("page_count", 0))
            <= int(limits["pipeline_max_pages"])
            and int(
                (result.get("page_contract") or {}).get("block_count", 0)
            )
            <= int(limits["pipeline_max_blocks"])
            for result in passed_runs
        )
        quality_exact = (
            baseline_stable
            and {
                str(
                    (result.get("page_contract") or {}).get(
                        "ocr_sha256",
                        "",
                    )
                )
                for result in passed_runs
            }
            == baseline_ocr_digests
            and {
                str(
                    (result.get("page_contract") or {}).get(
                        "detection_sha256",
                        "",
                    )
                )
                for result in passed_runs
            }
            == baseline_detection_digests
        )
        translation_structure_passed = all(
            int(
                (result.get("page_contract") or {}).get(
                    "empty_translation_count",
                    0,
                )
                or 0
            )
            == 0
            for result in passed_runs
        )
        improvement = _reduction_percent(
            baseline_median,
            median_elapsed,
        )
        expected_full_improvement = (
            improvement
            if stage == "full"
            else (
                (
                    (
                        baseline_screening_wall_median
                        - screening_wall_median
                    )
                    / float(full_reference["full_median_sec"])
                )
                * 100.0
                if (
                    full_reference is not None
                    and float(full_reference["full_median_sec"]) > 0
                )
                else None
            )
        )
        stage_speed_passed = (
            improvement >= float(gates["stage_improvement_percent"])
        )
        expected_full_speed_passed = (
            expected_full_improvement is not None
            and expected_full_improvement
            >= float(gates["expected_full_improvement_percent"])
        )
        speed_passed = (
            candidate_id == baseline_id
            or stage_speed_passed
            or expected_full_speed_passed
        )
        variance_passed = (
            variance_percent
            <= float(gates["cold_variance_percent"])
            and screening_wall_variance_percent
            <= float(gates["cold_variance_percent"])
        )
        automated_passed = (
            len(passed_runs) == int(limits["rounds"])
            and page_limits_passed
            and speed_passed
            and variance_passed
            and (
                quality_exact if structural_quality_required else True
            )
            and (
                translation_structure_passed
                if quality_gate
                == "structure-and-private-meaning-review"
                else True
            )
        )
        candidate_summaries.append(
            {
                "candidate": candidate_id,
                "run_count": len(candidate_results),
                "passed_run_count": len(passed_runs),
                "median_stage_elapsed_sec": round(median_elapsed, 6),
                "median_screening_wall_sec": round(
                    screening_wall_median,
                    6,
                ),
                "variance_percent": round(variance_percent, 3),
                "screening_wall_variance_percent": round(
                    screening_wall_variance_percent,
                    3,
                ),
                "improvement_vs_baseline_percent": round(
                    improvement,
                    3,
                ),
                "expected_full_improvement_percent": (
                    round(expected_full_improvement, 3)
                    if expected_full_improvement is not None
                    else None
                ),
                "page_limits_passed": page_limits_passed,
                "exact_quality_passed": (
                    quality_exact if exact_quality_required else None
                ),
                "structural_quality_passed": (
                    quality_exact and translation_structure_passed
                    if structural_quality_required
                    else None
                ),
                "translation_structure_passed": (
                    translation_structure_passed
                    if quality_gate
                    == "structure-and-private-meaning-review"
                    else None
                ),
                "speed_gate_passed": speed_passed,
                "variance_gate_passed": variance_passed,
                "stage_improvement_gate_percent": float(
                    gates["stage_improvement_percent"]
                ),
                "expected_full_improvement_gate_percent": float(
                    gates["expected_full_improvement_percent"]
                ),
                "automated_gate_passed": automated_passed,
                "requires_private_meaning_review": (
                    quality_gate
                    == "structure-and-private-meaning-review"
                ),
            }
        )
    candidate_summaries.sort(
        key=lambda item: (
            not bool(item["automated_gate_passed"]),
            float(item["median_stage_elapsed_sec"] or 1e30),
            str(item["candidate"]),
        )
    )
    return {
        "family": family.get("id"),
        "stage": stage,
        "baseline_candidate": baseline_id,
        "baseline_median_stage_elapsed_sec": round(
            baseline_median,
            6,
        ),
        "baseline_median_screening_wall_sec": round(
            baseline_screening_wall_median,
            6,
        ),
        "baseline_exact_output_stable": baseline_stable,
        "full_reference": (
            {
                key: round(float(value), 6)
                for key, value in full_reference.items()
            }
            if full_reference is not None
            else None
        ),
        "candidates": candidate_summaries,
        "winner": next(
            (
                item["candidate"]
                for item in candidate_summaries
                if item["candidate"] != baseline_id
                and item["automated_gate_passed"]
                and not item["requires_private_meaning_review"]
            ),
            "",
        ),
        "note": (
            "Meaning-changing candidates remain unpromotable until the "
            "separate private blind quality gate is complete."
        ),
    }


def _render_pipeline_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Cold Pipeline Candidate Report",
        "",
        f"- family: `{analysis.get('family')}`",
        f"- stage: `{analysis.get('stage')}`",
        f"- baseline: `{analysis.get('baseline_candidate')}`",
        f"- baseline exact output stable: `{analysis.get('baseline_exact_output_stable')}`",
        f"- full reference: `{analysis.get('full_reference') or 'none'}`",
        f"- automated winner: `{analysis.get('winner') or 'none'}`",
        "",
        "| candidate | runs | passed | stage sec | screening wall sec | stage improvement % | expected full % | stage variance % | wall variance % | structural quality | speed gate | variance gate | automated gate | meaning review |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analysis.get("candidates", []):
        lines.append(
            "| {candidate} | {runs} | {passed} | {median} | {wall} | "
            "{improvement} | {expected_full} | {variance} | {wall_variance} | "
            "{quality} | {speed} | {variance_gate} | "
            "{gate} | {review} |".format(
                candidate=item.get("candidate", ""),
                runs=item.get("run_count", 0),
                passed=item.get("passed_run_count", 0),
                median=item.get("median_stage_elapsed_sec", ""),
                wall=item.get("median_screening_wall_sec", ""),
                variance=item.get("variance_percent", ""),
                wall_variance=item.get(
                    "screening_wall_variance_percent",
                    "",
                ),
                improvement=item.get(
                    "improvement_vs_baseline_percent",
                    "",
                ),
                expected_full=item.get(
                    "expected_full_improvement_percent",
                    "",
                ),
                quality=item.get("structural_quality_passed", ""),
                speed=item.get("speed_gate_passed", ""),
                variance_gate=item.get("variance_gate_passed", ""),
                gate=item.get("automated_gate_passed", ""),
                review=item.get("requires_private_meaning_review", ""),
            )
        )
    lines.extend(
        [
            "",
            "Private comparison table: "
            f"`{(analysis.get('private_review') or {}).get('relative_path', 'not-required')}`.",
            "",
            "Raw OCR, translations, images, local paths, stdout, and stderr "
            "remain only in this external suite directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _protocol_state(protocol: Mapping[str, Any]) -> dict[str, Any]:
    baseline_path = ROOT / str(protocol["baseline_preset"])
    return {
        "protocol_version": int(protocol["protocol_version"]),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "baseline_preset_sha256": _sha256_file(baseline_path),
        "branch": _git_output("branch", "--show-current"),
        "commit": _git_output("rev-parse", "HEAD"),
        "worktree_clean": not bool(
            _git_output("status", "--porcelain")
        ),
        "code_contract_sha256": _canonical_sha256(
            [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha256_file(path),
                }
                for path in REPRODUCIBILITY_FILES
            ]
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "privacy": (
            "Raw input names, paths, images, OCR, translations, and private "
            "quality reviews are excluded from tracked reports."
        ),
    }


def _input_contract(input_dir: Path, sample_count: int) -> dict[str, Any]:
    from benchmark_common import select_sample_images

    selected = select_sample_images(input_dir, sample_count=sample_count)
    entries = [
        {
            "index": index,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for index, path in enumerate(selected)
    ]
    return {
        "sample_count": len(entries),
        "sha256": _canonical_sha256(entries),
    }


def _load_full_reference(
    path: Path,
    *,
    protocol_state: Mapping[str, Any],
    stage: str,
) -> dict[str, float]:
    state = _read_json(path.expanduser().resolve())
    if str(state.get("status", "")) != "completed":
        raise ValueError("Full reference suite must be completed.")
    reference_protocol = state.get("protocol")
    if not isinstance(reference_protocol, Mapping):
        raise ValueError("Full reference suite has no protocol contract.")
    required_equal = (
        "protocol_sha256",
        "baseline_preset_sha256",
        "commit",
        "sample_count",
        "source_language",
        "input_contract_sha256",
    )
    mismatches = {
        key: {
            "expected": protocol_state.get(key),
            "actual": reference_protocol.get(key),
        }
        for key in required_equal
        if reference_protocol.get(key) != protocol_state.get(key)
    }
    if mismatches:
        raise ValueError(
            f"Full reference suite contract does not match: {mismatches}"
        )
    baseline_id = str(
        reference_protocol.get("baseline_candidate", "") or ""
    )
    results = state.get("results")
    if not baseline_id or not isinstance(results, list):
        raise ValueError("Full reference suite has no baseline results.")
    baseline_results = [
        item
        for item in results
        if isinstance(item, Mapping)
        and str(item.get("candidate", "")) == baseline_id
        and str(item.get("status", "")) == "passed"
    ]
    if len(baseline_results) != 3:
        raise ValueError(
            "Full reference suite must contain three passed baseline runs."
        )
    full_values = [
        _stage_elapsed(item, "full") for item in baseline_results
    ]
    stage_values = [
        _stage_elapsed(item, stage) for item in baseline_results
    ]
    full_median = statistics.median(full_values)
    stage_median = statistics.median(stage_values)
    if full_median <= 0 or stage_median <= 0 or stage_median > full_median:
        raise ValueError("Full reference suite stage timing is invalid.")
    return {
        "full_median_sec": full_median,
        "stage_median_sec": stage_median,
        "stage_share": stage_median / full_median,
    }


def run_pipeline_family(args: argparse.Namespace) -> int:
    _require_reproducible_checkout()
    protocol = load_protocol()
    family = _family(protocol, args.family)
    candidates, baseline_id = pipeline_candidates(
        family,
        axis=args.axis,
    )
    rounds = int(protocol["limits"]["rounds"])
    sample_count = int(args.sample_count)
    if not 1 <= sample_count <= int(
        protocol["limits"]["pipeline_max_pages"]
    ):
        raise ValueError("--sample-count exceeds the protocol page limit.")
    output_dir = ensure_external_output(
        Path(args.output_dir),
        require_new=True,
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory was not found: {input_dir}")
    baseline_path = (
        Path(args.base_preset).expanduser().resolve()
        if args.base_preset
        else ROOT / str(protocol["baseline_preset"])
    )
    baseline_preset = _read_json(baseline_path)
    _validate_baseline_contract(
        _read_json(ROOT / str(protocol["baseline_preset"]))
    )
    _validate_cold_candidate_base(baseline_preset)
    protocol_state = _protocol_state(protocol)
    protocol_state.update(
        {
            "command": "run-pipeline",
            "family": args.family,
            "axis": args.axis,
            "sample_count": sample_count,
            "source_language": args.source_lang,
            "baseline_candidate": baseline_id,
            "input_contract_sha256": _input_contract(
                input_dir,
                sample_count,
            )["sha256"],
        }
    )
    full_reference = None
    if args.full_reference_suite:
        full_reference_path = Path(
            args.full_reference_suite
        ).expanduser().resolve()
        full_reference = _load_full_reference(
            full_reference_path,
            protocol_state=protocol_state,
            stage=str(family.get("stage", "full")),
        )
        protocol_state["full_reference_suite_sha256"] = _sha256_file(
            full_reference_path
        )
    _write_json(output_dir / PROTOCOL_STATE_NAME, protocol_state)

    private_dir = output_dir / "private"
    preset_dir = private_dir / "presets"
    shared_corpus_dir = private_dir / "shared-corpus"
    preset_dir.mkdir(parents=True, exist_ok=True)
    candidate_map: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        preset = _deep_merge(
            baseline_preset,
            candidate.get("patch", {}),
        )
        preset["name"] = (
            f"cold-cache-finalization-{args.family}-{candidate_id}"
        )
        preset_path = preset_dir / f"{candidate_id}.json"
        _write_json(preset_path, preset)
        candidate_map[candidate_id] = {
            "definition": candidate,
            "preset_path": preset_path,
        }

    results: list[dict[str, Any]] = []
    orders = balanced_orders(candidate_map, rounds)
    state: dict[str, Any] = {
        "status": "running",
        "protocol": protocol_state,
        "orders": orders,
        "results": results,
    }
    _write_json(output_dir / SUITE_STATE_NAME, state)
    try:
        for round_index, order in enumerate(orders, start=1):
            for order_index, candidate_id in enumerate(order, start=1):
                run_dir = (
                    private_dir
                    / "runs"
                    / f"round-{round_index:02d}"
                    / f"{order_index:02d}-{candidate_id}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                user_data_root = (
                    private_dir
                    / "isolated-user-data"
                    / f"round-{round_index:02d}"
                    / candidate_id
                )
                _stop_managed_containers()
                returncode = _run_process(
                    _pipeline_command(
                        preset_path=candidate_map[candidate_id][
                            "preset_path"
                        ],
                        input_dir=input_dir,
                        sample_count=sample_count,
                        run_dir=run_dir,
                        shared_corpus_dir=shared_corpus_dir,
                        stage=str(family.get("stage", "full")),
                        source_lang=args.source_lang,
                    ),
                    env=_isolated_environment(user_data_root),
                    stdout_path=run_dir / "stdout.txt",
                    stderr_path=run_dir / "stderr.txt",
                )
                _stop_managed_containers()
                result = _safe_pipeline_result(
                    candidate_id=candidate_id,
                    round_index=round_index,
                    order_index=order_index,
                    run_dir=run_dir,
                    returncode=returncode,
                )
                results.append(result)
                state["results"] = results
                _write_json(output_dir / SUITE_STATE_NAME, state)
                if result["status"] != "passed":
                    raise RuntimeError(
                        f"Pipeline candidate failed: {candidate_id} "
                        f"round={round_index}"
                    )
                contract = result["page_contract"]
                if int(contract.get("page_count", 0)) > int(
                    protocol["limits"]["pipeline_max_pages"]
                ):
                    raise RuntimeError("Pipeline page gate exceeded.")
                if int(contract.get("block_count", 0)) > int(
                    protocol["limits"]["pipeline_max_blocks"]
                ):
                    raise RuntimeError(
                        "Pipeline block gate exceeded; select a shorter corpus."
                    )
        private_review = None
        if (
            str(family.get("quality_gate", ""))
            == "structure-and-private-meaning-review"
        ):
            private_review = _write_private_pipeline_review(
                results,
                private_dir / "pipeline-review.json",
            )
        analysis = analyze_pipeline_results(
            protocol=protocol,
            family=family,
            baseline_id=baseline_id,
            results=results,
            full_reference=full_reference,
        )
        if private_review is not None:
            analysis["private_review"] = private_review
        state["status"] = "completed"
        state["analysis"] = analysis
        _write_json(output_dir / SUITE_STATE_NAME, state)
        _write_json(output_dir / SUMMARY_NAME, analysis)
        (output_dir / REPORT_NAME).write_text(
            _render_pipeline_report(analysis),
            encoding="utf-8",
        )
        return 0
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["analysis"] = (
            analyze_pipeline_results(
                protocol=protocol,
                family=family,
                baseline_id=baseline_id,
                results=results,
            )
            if any(
                result.get("candidate") == baseline_id
                for result in results
            )
            else {}
        )
        _write_json(output_dir / SUITE_STATE_NAME, state)
        raise
    finally:
        _stop_managed_containers()


def _translation_candidate_profiles(
    protocol: Mapping[str, Any],
    family: Mapping[str, Any],
    *,
    axis: str,
    model_key: str,
) -> tuple[list[dict[str, Any]], str]:
    execution = str(family.get("execution", ""))
    models = protocol.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("Protocol model contracts are missing.")
    if execution == "translation":
        candidates = [
            dict(candidate)
            for candidate in family.get("candidates", [])
            if isinstance(candidate, Mapping)
        ]
        for candidate in candidates:
            key = str(candidate.get("model", model_key) or model_key)
            if key not in models:
                raise ValueError(f"Unknown translation model key: {key}")
            candidate["model_key"] = key
            candidate["model_name"] = str(models[key]["name"])
            candidate["model_sha256"] = str(models[key]["sha256"])
            candidate.setdefault("chunk_size", 6)
            candidate.setdefault("context_size", 4096)
            candidate.setdefault("n_parallel", 1)
            candidate.setdefault("concurrency", 1)
            if int(candidate["context_size"]) // int(
                candidate["n_parallel"]
            ) < 4096:
                raise ValueError(
                    "Translation candidate must provide at least 4096 "
                    "context tokens per parallel slot."
                )
            if int(candidate["concurrency"]) > int(
                candidate["n_parallel"]
            ):
                raise ValueError(
                    "Translation concurrency may not exceed llama.cpp slots."
                )
        baseline_id = str(family.get("baseline_candidate", ""))
    elif execution == "generated-translation":
        axes = family.get("axes")
        if not isinstance(axes, Mapping) or axis not in axes:
            raise ValueError(
                f"--axis must select one of: {', '.join(sorted(axes or {}))}"
            )
        if axis != "chunk_size":
            raise ValueError("Only chunk_size is a generated translation axis.")
        if model_key not in models:
            raise ValueError(f"Unknown translation model key: {model_key}")
        candidates = [
            {
                "id": f"chunk-{value}",
                "model_key": model_key,
                "model_name": str(models[model_key]["name"]),
                "model_sha256": str(models[model_key]["sha256"]),
                "chunk_size": int(value),
                "context_size": 4096,
                "n_parallel": 1,
                "concurrency": 1,
            }
            for value in axes[axis]
        ]
        baseline_id = (
            f"chunk-{family['baseline_values']['chunk_size']}"
        )
    else:
        raise ValueError("Selected family is not a translation benchmark.")
    ids = [str(candidate.get("id", "")) for candidate in candidates]
    if baseline_id not in ids:
        raise ValueError("Translation baseline is absent from candidate list.")
    return candidates, baseline_id


def _translation_profile_command(
    *,
    source_summary: Path,
    output_path: Path,
    candidate: Mapping[str, Any],
    language_order: Iterable[str],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_translation-profile",
        "--source-summary",
        str(source_summary),
        "--output",
        str(output_path),
        "--candidate-id",
        str(candidate["id"]),
        "--model",
        str(candidate["model_name"]),
        "--expected-model-sha256",
        str(candidate["model_sha256"]),
        "--chunk-size",
        str(int(candidate.get("chunk_size", 6))),
        "--context-size",
        str(int(candidate.get("context_size", 4096))),
        "--n-parallel",
        str(int(candidate.get("n_parallel", 1))),
        "--concurrency",
        str(int(candidate.get("concurrency", 1))),
        "--language-order",
        *list(language_order),
    ]


def run_translation_family(args: argparse.Namespace) -> int:
    _require_reproducible_checkout()
    protocol = load_protocol()
    family = _family(protocol, args.family)
    candidates, baseline_id = _translation_candidate_profiles(
        protocol,
        family,
        axis=args.axis,
        model_key=args.model_key,
    )
    output_dir = ensure_external_output(
        Path(args.output_dir),
        require_new=True,
    )
    source_summary = Path(args.source_summary).expanduser().resolve()
    if not source_summary.is_file():
        raise FileNotFoundError(
            f"Translation source summary was not found: {source_summary}"
        )
    rounds = int(protocol["limits"]["rounds"])
    candidate_map = {
        str(candidate["id"]): candidate for candidate in candidates
    }
    orders = balanced_orders(candidate_map, rounds)
    languages = list(protocol["limits"]["translation_languages"])
    protocol_state = _protocol_state(protocol)
    protocol_state.update(
        {
            "command": "run-translation",
            "family": args.family,
            "axis": args.axis,
            "baseline_candidate": baseline_id,
            "source_summary_sha256": _sha256_file(source_summary),
        }
    )
    _write_json(output_dir / PROTOCOL_STATE_NAME, protocol_state)
    state: dict[str, Any] = {
        "status": "running",
        "protocol": protocol_state,
        "orders": orders,
        "results": [],
    }
    _write_json(output_dir / SUITE_STATE_NAME, state)
    private_dir = output_dir / "private"
    results: list[dict[str, Any]] = []
    try:
        for round_index, order in enumerate(orders, start=1):
            language_offset = (round_index - 1) % len(languages)
            language_order = (
                languages[language_offset:]
                + languages[:language_offset]
            )
            for order_index, candidate_id in enumerate(order, start=1):
                run_dir = (
                    private_dir
                    / "runs"
                    / f"round-{round_index:02d}"
                    / f"{order_index:02d}-{candidate_id}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                output_path = run_dir / "translation_profile.json"
                candidate = candidate_map[candidate_id]
                env = _isolated_environment(
                    private_dir
                    / "isolated-user-data"
                    / f"round-{round_index:02d}"
                    / candidate_id
                )
                env.update(
                    {
                        "LLAMA_N_PARALLEL": str(
                            int(candidate.get("n_parallel", 1))
                        ),
                        "LLAMA_CTX_SIZE": str(
                            int(candidate.get("context_size", 4096))
                        ),
                        "LLAMA_SPEC_TYPE": "none",
                        "LLAMA_CACHE_TYPE_K": "f16",
                        "LLAMA_CACHE_TYPE_V": "f16",
                    }
                )
                _stop_managed_containers()
                returncode = _run_process(
                    _translation_profile_command(
                        source_summary=source_summary,
                        output_path=output_path,
                        candidate=candidate,
                        language_order=language_order,
                    ),
                    env=env,
                    stdout_path=run_dir / "stdout.txt",
                    stderr_path=run_dir / "stderr.txt",
                )
                _stop_managed_containers()
                payload = (
                    _read_json(output_path)
                    if output_path.is_file()
                    else {}
                )
                public = {
                    key: value
                    for key, value in payload.items()
                    if key != "outputs"
                }
                public.update(
                    {
                        "candidate": candidate_id,
                        "round": round_index,
                        "order": order_index,
                        "returncode": returncode,
                        "private_output": str(output_path),
                    }
                )
                results.append(public)
                state["results"] = results
                _write_json(output_dir / SUITE_STATE_NAME, state)
                if returncode != 0 or str(
                    payload.get("status", "")
                ) != "passed":
                    raise RuntimeError(
                        f"Translation profile failed: {candidate_id} "
                        f"round={round_index}"
                    )
        private_review = _write_private_translation_review(
            results,
            private_dir / "translation-review.json",
        )
        analysis = _analyze_translation_results(
            protocol=protocol,
            family=family,
            baseline_id=baseline_id,
            results=results,
        )
        analysis["private_review"] = private_review
        state["status"] = "completed"
        state["analysis"] = analysis
        _write_json(output_dir / SUITE_STATE_NAME, state)
        _write_json(output_dir / SUMMARY_NAME, analysis)
        (output_dir / REPORT_NAME).write_text(
            _render_translation_report(analysis),
            encoding="utf-8",
        )
        return 0
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_dir / SUITE_STATE_NAME, state)
        raise
    finally:
        _stop_managed_containers()


def _sum_numeric(
    destination: dict[str, int | float],
    source: Mapping[str, Any],
) -> None:
    for key, value in source.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float):
            destination[key] = float(destination.get(key, 0.0)) + value
        else:
            destination[key] = int(destination.get(key, 0)) + int(value)


def _translation_profile(args: argparse.Namespace) -> int:
    import numpy as np

    from benchmark_translation_memory_fast_path import (
        DictionarySettings,
        RuntimeHarness,
        SEVERE_STATS,
        _new_blocks,
        load_multilingual_corpora,
        new_engine,
        stop_managed_container,
    )
    from modules.translation.llm.custom_local_gemma import (
        GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
    )

    expected_model_sha256 = str(args.expected_model_sha256).strip().lower()
    if (
        len(expected_model_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_model_sha256
        )
    ):
        raise ValueError("Expected model SHA-256 contract is invalid.")
    source_summary = Path(args.source_summary).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    corpora = load_multilingual_corpora(source_summary)
    language_order = list(args.language_order)
    if set(language_order) != set(corpora) or len(language_order) != 3:
        raise ValueError("Language order must contain Japanese, Chinese, English once.")
    for language in language_order:
        items = corpora[language]
        if len(items) < 18:
            raise ValueError(
                f"Translation corpus has fewer than 18 blocks: {language}"
            )
        case_ids = [
            str(item.get("case_id", "") or "") for item in items[:18]
        ]
        if any(not case_id for case_id in case_ids) or len(
            set(case_ids)
        ) != 18:
            raise ValueError(
                f"Translation corpus case ids are missing or duplicated: {language}"
            )
    runtime = RuntimeHarness(args.model)
    concurrency = max(1, int(args.concurrency))
    outputs: list[dict[str, Any]] = []
    total_stats: dict[str, int | float] = {}
    stop_managed_container()
    started = time.perf_counter()

    def run_language(language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        items = corpora[language]
        blocks = _new_blocks(items)
        engine = new_engine(
            language=language,
            store=None,
            runtime=runtime,
            persistent_cache_enabled=False,
            exact_tm_enabled=False,
            group_size=int(args.chunk_size),
            max_completion_tokens=512,
        )
        engine.settings = DictionarySettings()
        engine.request_mode = GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE
        if concurrency > 1:
            engine.configure_runtime_hooks(
                ensure_runtime=lambda: None,
                runtime_identity_provider=runtime.identity,
            )
        engine.prepare_translation(blocks, "", requested_indices=range(18))
        engine.translate(
            blocks,
            np.zeros((1, 1, 3), dtype=np.uint8),
            "",
            requested_indices=range(18),
        )
        language_outputs = [
            {
                "language": language,
                "index": index,
                "case_id": items[index]["case_id"],
                "source": items[index]["source"],
                "translation": str(blocks[index].translation or ""),
            }
            for index in range(18)
        ]
        return language_outputs, dict(engine.last_benchmark_stats)

    try:
        if concurrency > 1:
            runtime.ensure()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(run_language, language): language
                    for language in language_order
                }
                indexed: dict[str, list[dict[str, Any]]] = {}
                for future in as_completed(futures):
                    language = futures[future]
                    language_outputs, stats = future.result()
                    indexed[language] = language_outputs
                    _sum_numeric(total_stats, stats)
                for language in language_order:
                    outputs.extend(indexed[language])
        else:
            for language in language_order:
                language_outputs, stats = run_language(language)
                outputs.extend(language_outputs)
                _sum_numeric(total_stats, stats)
        elapsed = time.perf_counter() - started
        runtime_identity = runtime.identity() or {}
        runtime_options = runtime_identity.get("runtime_options")
        if not isinstance(runtime_options, Mapping):
            runtime_options = {}
        model_contract_valid = (
            str(runtime_identity.get("model_name", ""))
            == str(args.model)
            and str(runtime_identity.get("model_sha256", "")).lower()
            == expected_model_sha256
            and str(runtime_options.get("LLAMA_N_PARALLEL", ""))
            == str(int(args.n_parallel))
            and str(runtime_options.get("LLAMA_CTX_SIZE", ""))
            == str(int(args.context_size))
            and str(runtime_options.get("LLAMA_CACHE_TYPE_K", "")).lower()
            == "f16"
            and str(runtime_options.get("LLAMA_CACHE_TYPE_V", "")).lower()
            == "f16"
            and str(runtime_options.get("LLAMA_SPEC_TYPE", "")).lower()
            == "none"
        )
        output_key_sha256 = _canonical_sha256(
            sorted(
                (
                    {
                        "language": item["language"],
                        "index": item["index"],
                        "case_id": item["case_id"],
                    }
                    for item in outputs
                ),
                key=lambda item: (
                    str(item["language"]),
                    int(item["index"]),
                ),
            )
        )
        severe_count = sum(
            int(total_stats.get(key, 0) or 0)
            for key in SEVERE_STATS
        )
        passed = (
            len(outputs) == 54
            and all(str(item["translation"]).strip() for item in outputs)
            and severe_count == 0
            and model_contract_valid
        )
        payload = {
            "status": "passed" if passed else "failed",
            "candidate": args.candidate_id,
            "model": args.model,
            "chunk_size": int(args.chunk_size),
            "context_size": int(args.context_size),
            "n_parallel": int(args.n_parallel),
            "concurrency": concurrency,
            "language_order": language_order,
            "elapsed_sec": round(elapsed, 6),
            "output_count": len(outputs),
            "nonempty_count": sum(
                1
                for item in outputs
                if str(item["translation"]).strip()
            ),
            "severe_telemetry_count": severe_count,
            "output_key_sha256": output_key_sha256,
            "model_contract_valid": model_contract_valid,
            "runtime_contract": {
                "model_name": str(
                    runtime_identity.get("model_name", "") or ""
                ),
                "model_sha256": str(
                    runtime_identity.get("model_sha256", "") or ""
                ),
                "runtime_fingerprint": str(
                    runtime_identity.get("runtime_fingerprint", "") or ""
                ),
                "runtime_options": dict(runtime_options),
            },
            "stats": total_stats,
            "outputs": outputs,
        }
        _write_json(output_path, payload)
        return 0 if passed else 1
    finally:
        runtime.shutdown()


def _analyze_translation_results(
    *,
    protocol: Mapping[str, Any],
    family: Mapping[str, Any],
    baseline_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["candidate"])].append(result)
    baseline_times = [
        float(item.get("elapsed_sec", 0.0) or 0.0)
        for item in grouped[baseline_id]
    ]
    baseline_median = (
        statistics.median(baseline_times) if baseline_times else 0.0
    )
    baseline_key_digests = {
        str(item.get("output_key_sha256", "") or "")
        for item in grouped[baseline_id]
    }
    baseline_keys_stable = (
        len(baseline_key_digests) == 1
        and "" not in baseline_key_digests
    )
    summaries: list[dict[str, Any]] = []
    for candidate_id, items in grouped.items():
        timings = [
            float(item.get("elapsed_sec", 0.0) or 0.0)
            for item in items
        ]
        median_elapsed = statistics.median(timings) if timings else 0.0
        variance_percent = (
            ((max(timings) - min(timings)) / median_elapsed) * 100.0
            if timings and median_elapsed > 0
            else 0.0
        )
        improvement = _reduction_percent(
            baseline_median,
            median_elapsed,
        )
        structural_pass = (
            len(items) == int(protocol["limits"]["rounds"])
            and baseline_keys_stable
            and {
                str(item.get("output_key_sha256", "") or "")
                for item in items
            }
            == baseline_key_digests
            and all(
                int(item.get("output_count", 0) or 0) == 54
                and int(item.get("nonempty_count", 0) or 0) == 54
                and int(
                    item.get("severe_telemetry_count", 0) or 0
                )
                == 0
                and bool(item.get("model_contract_valid", False))
                for item in items
            )
        )
        variance_passed = (
            variance_percent
            <= float(protocol["gates"]["cold_variance_percent"])
        )
        speed_qualified = (
            candidate_id == baseline_id
            or improvement
            >= float(
                protocol["gates"]["stage_improvement_percent"]
            )
        )
        summaries.append(
            {
                "candidate": candidate_id,
                "median_elapsed_sec": round(median_elapsed, 6),
                "variance_percent": round(variance_percent, 3),
                "improvement_vs_baseline_percent": round(
                    improvement,
                    3,
                ),
                "structural_gate_passed": structural_pass,
                "speed_qualified": speed_qualified,
                "variance_gate_passed": variance_passed,
                "promotion_status": (
                    "baseline"
                    if candidate_id == baseline_id
                    else (
                        "requires_blind_quality_review"
                        if (
                            structural_pass
                            and speed_qualified
                            and variance_passed
                        )
                        else "rejected"
                    )
                ),
            }
        )
    summaries.sort(
        key=lambda item: (
            not bool(item["structural_gate_passed"]),
            float(item["median_elapsed_sec"] or 1e30),
        )
    )
    return {
        "family": family.get("id"),
        "baseline_candidate": baseline_id,
        "baseline_median_elapsed_sec": round(baseline_median, 6),
        "baseline_output_keys_stable": baseline_keys_stable,
        "candidates": summaries,
        "auto_promoted_candidate": "",
        "quality_gate": family.get("quality_gate"),
        "note": (
            "No translation candidate is auto-promoted. A speed-qualified "
            "candidate must pass the locked 292-row candidate-only blind gate."
        ),
    }


def _render_translation_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Cold Translation Candidate Report",
        "",
        f"- family: `{analysis.get('family')}`",
        f"- baseline: `{analysis.get('baseline_candidate')}`",
        "- automatic promotion: `disabled`",
        "",
        "| candidate | median sec | variance % | improvement % | structural gate | variance gate | promotion status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in analysis.get("candidates", []):
        lines.append(
            "| {candidate} | {median} | {variance} | {improvement} | "
            "{structural} | {variance_gate} | {status} |".format(
                candidate=item.get("candidate", ""),
                median=item.get("median_elapsed_sec", ""),
                variance=item.get("variance_percent", ""),
                improvement=item.get(
                    "improvement_vs_baseline_percent",
                    "",
                ),
                structural=item.get("structural_gate_passed", ""),
                variance_gate=item.get("variance_gate_passed", ""),
                status=item.get("promotion_status", ""),
            )
        )
    lines.extend(
        [
            "",
            "Private comparison table: "
            f"`{(analysis.get('private_review') or {}).get('relative_path', 'missing')}`.",
            "",
            "Private source text and translations remain under the external "
            "`private/` directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _cache_preset(
    baseline: Mapping[str, Any],
    *,
    paddle: bool,
    project: bool,
) -> dict[str, Any]:
    return _deep_merge(
        baseline,
        {
            "name": (
                "cold-cache-global-ocr"
                if paddle
                else "cold-cache-project-checkpoint"
            ),
            "benchmark_cache_policy": {
                "paddleocr_persistent": paddle,
                "translation_persistent": False,
                "exact_tm": False,
                "project_checkpoint": project,
            },
        },
    )


def _run_cache_pipeline(
    *,
    preset_path: Path,
    input_dir: Path,
    sample_count: int,
    run_dir: Path,
    shared_corpus_dir: Path,
    user_data_root: Path,
    stage: str,
    source_lang: str,
    project_file: Path | None = None,
    project_action: str = "none",
    save_project_after_run: bool = False,
    project_invalidate_page_index: int = -1,
    project_invalidate_stage: str = "ocr",
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    _stop_managed_containers()
    returncode = _run_process(
        _pipeline_command(
            preset_path=preset_path,
            input_dir=input_dir,
            sample_count=sample_count,
            run_dir=run_dir,
            shared_corpus_dir=shared_corpus_dir,
            stage=stage,
            source_lang=source_lang,
            project_file=project_file,
            project_action=project_action,
            save_project_after_run=save_project_after_run,
            project_invalidate_page_index=(
                project_invalidate_page_index
            ),
            project_invalidate_stage=project_invalidate_stage,
        ),
        env=_isolated_environment(user_data_root),
        stdout_path=run_dir / "stdout.txt",
        stderr_path=run_dir / "stderr.txt",
    )
    _stop_managed_containers()
    return _safe_pipeline_result(
        candidate_id=run_dir.name,
        round_index=1,
        order_index=1,
        run_dir=run_dir,
        returncode=returncode,
    )


def _delete_owned_render_outputs(
    result: Mapping[str, Any],
    *,
    owned_root: Path,
) -> list[str]:
    deleted: list[str] = []
    root = owned_root.resolve()
    for page in (result.get("page_contract") or {}).get("pages", []):
        if not isinstance(page, Mapping):
            continue
        value = str(page.get("output_path", "") or "")
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to delete a render outside the suite root: {path}"
            ) from exc
        if path.is_file():
            path.unlink()
            deleted.append(str(path))
    return deleted


def _runtime_count(
    result: Mapping[str, Any],
    service: str,
    key: str,
) -> int:
    performance = result.get("performance_stats")
    runtime = (
        performance.get("runtime")
        if isinstance(performance, Mapping)
        else None
    )
    service_values = (
        runtime.get(service) if isinstance(runtime, Mapping) else None
    )
    return (
        int(service_values.get(key, 0) or 0)
        if isinstance(service_values, Mapping)
        else 0
    )


def _nested_metric(
    result: Mapping[str, Any],
    section: str,
    key: str,
) -> int:
    performance = result.get("performance_stats")
    values = (
        performance.get(section)
        if isinstance(performance, Mapping)
        else None
    )
    return (
        int(values.get(key, 0) or 0)
        if isinstance(values, Mapping)
        else 0
    )


def _result_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _passed_results(value: Any) -> list[Mapping[str, Any]]:
    return [
        result
        for result in _result_list(value)
        if str(result.get("status", "passed")) == "passed"
    ]


def _median_stage(value: Any, stage: str) -> float:
    timings = [
        _stage_elapsed(result, stage)
        for result in _passed_results(value)
        if _stage_elapsed(result, stage) > 0
    ]
    return statistics.median(timings) if timings else 0.0


def _stage_digests(
    values: Iterable[Mapping[str, Any]],
    digest_key: str,
) -> set[str]:
    return {
        str((result.get("page_contract") or {}).get(digest_key, "") or "")
        for result in values
    }


def _page_output_digests(
    result: Mapping[str, Any],
    *,
    exclude_page_indexes: Iterable[int] = (),
) -> dict[int, str]:
    excluded = {int(index) for index in exclude_page_indexes}
    pages = (result.get("page_contract") or {}).get("pages", [])
    return {
        int(page.get("page_index", -1)): str(
            page.get("output_sha256", "") or ""
        )
        for page in pages
        if isinstance(page, Mapping)
        and int(page.get("page_index", -1)) not in excluded
    }


def _timing_variance_percent(
    results: Iterable[Mapping[str, Any]],
    stage: str,
) -> float:
    values = [
        _stage_elapsed(result, stage)
        for result in results
        if _stage_elapsed(result, stage) > 0
    ]
    if not values:
        return float("inf")
    median = statistics.median(values)
    return (
        ((max(values) - min(values)) / median) * 100.0
        if median > 0
        else float("inf")
    )


def _project_checkpoint_event_gate(
    result: Mapping[str, Any],
    *,
    require_render_materialized: bool | None = None,
) -> bool:
    contract = result.get("checkpoint_events")
    if not isinstance(contract, Mapping):
        return False
    page_count = int(contract.get("page_count", 0) or 0)
    entries = contract.get("entries")
    if (
        page_count <= 0
        or not isinstance(entries, list)
        or len(entries) != page_count * 5
        or int(contract.get("unmatched_event_count", 0) or 0) != 0
    ):
        return False
    by_page_stage = {
        (
            int(entry.get("page_index", -1)),
            str(entry.get("stage", "")),
        ): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if len(by_page_stage) != page_count * 5:
        return False
    for page_index in range(page_count):
        for stage in (
            "detection",
            "ocr",
            "inpaint",
            "translation",
            "render",
        ):
            entry = by_page_stage.get((page_index, stage))
            if not isinstance(entry, Mapping):
                return False
            status = str(
                entry.get("project_checkpoint_status", "") or ""
            )
            skip_reason = str(entry.get("skip_reason", "") or "")
            if status != "hit" and not (
                skip_reason == "no_text_detected"
                and stage in {"ocr", "inpaint", "translation"}
            ):
                return False
            if stage == "render":
                if not bool(entry.get("render_skipped", False)):
                    return False
                if (
                    require_render_materialized is not None
                    and bool(entry.get("output_materialized", False))
                    is not require_render_materialized
                ):
                    return False
    return True


def _project_partial_recompute_gate(
    result: Mapping[str, Any],
    *,
    page_index: int,
    invalidated_stage: str,
) -> bool:
    contract = result.get("checkpoint_events")
    if not isinstance(contract, Mapping):
        return False
    page_count = int(contract.get("page_count", 0) or 0)
    entries = contract.get("entries")
    if (
        page_count <= page_index
        or not isinstance(entries, list)
        or len(entries) != page_count * 5
        or int(contract.get("unmatched_event_count", 0) or 0) != 0
    ):
        return False
    by_page_stage = {
        (
            int(entry.get("page_index", -1)),
            str(entry.get("stage", "")),
        ): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if len(by_page_stage) != page_count * 5:
        return False
    stage_order = (
        "detection",
        "ocr",
        "inpaint",
        "translation",
        "render",
    )
    invalidated_position = stage_order.index(invalidated_stage)
    for current_page in range(page_count):
        for position, stage in enumerate(stage_order):
            entry = by_page_stage.get((current_page, stage))
            if not isinstance(entry, Mapping):
                return False
            status = str(
                entry.get("project_checkpoint_status", "") or ""
            )
            skip_reason = str(entry.get("skip_reason", "") or "")
            skipped_no_text = (
                skip_reason == "no_text_detected"
                and stage in {"ocr", "inpaint", "translation"}
            )
            should_recompute = (
                current_page == page_index
                and position >= invalidated_position
            )
            if should_recompute:
                if status == "hit" and not skipped_no_text:
                    return False
            elif status != "hit" and not skipped_no_text:
                return False
    return True


def analyze_cache_results(
    *,
    protocol: Mapping[str, Any],
    scenario: str,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    gates = protocol["gates"]
    disabled = _passed_results(results.get("disabled_cold", []))
    enabled = _passed_results(results.get("enabled_empty_cold", []))
    required_rounds = int(protocol["limits"]["rounds"])
    required_stabilization_runs = (
        int(protocol["limits"]["cache_stabilization_pairs"]) * 2
    )
    stabilization = _passed_results(results.get("stabilization", []))
    stage = "ocr" if scenario == "global-ocr" else "full"
    disabled_median = _median_stage(disabled, stage)
    enabled_median = _median_stage(enabled, stage)
    miss_overhead = (
        ((enabled_median - disabled_median) / disabled_median) * 100.0
        if disabled_median > 0
        else float("inf")
    )
    cold_winner_median = min(
        value for value in (disabled_median, enabled_median) if value > 0
    ) if disabled_median > 0 or enabled_median > 0 else 0.0
    cold = enabled[0] if enabled else {}
    common_checks = {
        "stabilization_runs_complete": (
            len(stabilization) == required_stabilization_runs
        ),
        "disabled_cold_rounds_complete": len(disabled) == required_rounds,
        "enabled_empty_cold_rounds_complete": (
            len(enabled) == required_rounds
        ),
        "cache_miss_overhead_gate": (
            miss_overhead
            <= float(gates["cache_miss_overhead_percent"])
        ),
        "disabled_cold_variance_gate": (
            _timing_variance_percent(disabled, stage)
            <= float(gates["cold_variance_percent"])
        ),
        "enabled_empty_cold_variance_gate": (
            _timing_variance_percent(enabled, stage)
            <= float(gates["cold_variance_percent"])
        ),
    }
    if scenario == "global-ocr":
        hit = results["all_hit"]
        exact_digests = _stage_digests(
            [*disabled, *enabled, hit],
            "ocr_sha256",
        )
        exact = len(exact_digests) == 1 and "" not in exact_digests
        no_runtime = (
            _runtime_count(
                hit,
                "paddleocr_vl",
                "start_count",
            )
            == 0
        )
        no_http = (
            _nested_metric(
                hit,
                "paddleocr_vl",
                "http_attempt_count",
            )
            == 0
        )
        reduction = _reduction_percent(
            cold_winner_median,
            _stage_elapsed(hit, "ocr"),
        )
        checks = {
            **common_checks,
            "all_hit_run_passed": (
                str(hit.get("status", "passed")) == "passed"
            ),
            "raw_ocr_exact": exact,
            "runtime_start_zero": no_runtime,
            "http_attempt_zero": no_http,
            "all_hit_reduction_gate": reduction
            >= float(gates["cache_all_hit_improvement_percent"]),
        }
    else:
        hit = results["all_hit_existing_output"]
        missing = results["all_hit_missing_output"]
        partial = results["single_page_ocr_invalidated"]
        seed = enabled[0] if enabled else {}
        cached_render_digests = _stage_digests(
            [seed, hit, missing],
            "render_sha256",
        )
        cached_render_exact = (
            len(cached_render_digests) == 1
            and "" not in cached_render_digests
        )
        cold_ocr_digests = _stage_digests(
            [*disabled, *enabled, hit, missing, partial],
            "ocr_sha256",
        )
        cold_detection_digests = _stage_digests(
            [*disabled, *enabled, hit, missing, partial],
            "detection_sha256",
        )
        cold_structure_exact = (
            len(cold_ocr_digests) == 1
            and "" not in cold_ocr_digests
            and len(cold_detection_digests) == 1
            and "" not in cold_detection_digests
        )
        invalidated_page_index = int(
            partial.get("invalidated_page_index", -1) or 0
        )
        unaffected_pages_exact = (
            invalidated_page_index >= 0
            and _page_output_digests(
                seed,
                exclude_page_indexes=(invalidated_page_index,),
            )
            == _page_output_digests(
                partial,
                exclude_page_indexes=(invalidated_page_index,),
            )
        )
        no_paddle_http = (
            _nested_metric(
                hit,
                "paddleocr_vl",
                "http_attempt_count",
            )
            == 0
        )
        no_gemma_http = (
            _nested_metric(
                hit,
                "gemma",
                "gemma_http_attempt_count",
            )
            == 0
        )
        all_stage_hits = _project_checkpoint_event_gate(
            hit,
            require_render_materialized=False,
        )
        reduction = _reduction_percent(
            cold_winner_median,
            _stage_elapsed(hit, "full"),
        )
        checks = {
            **common_checks,
            "all_hit_existing_output_run_passed": (
                str(hit.get("status", "passed")) == "passed"
            ),
            "all_hit_missing_output_run_passed": (
                str(missing.get("status", "passed")) == "passed"
            ),
            "partial_recompute_run_passed": (
                str(partial.get("status", "passed")) == "passed"
            ),
            "cold_detection_ocr_exact": cold_structure_exact,
            "cached_render_output_exact": cached_render_exact,
            "unaffected_pages_exact_after_partial_recompute": (
                unaffected_pages_exact
            ),
            "paddle_http_zero": no_paddle_http,
            "gemma_http_zero": no_gemma_http,
            "paddle_runtime_start_zero": (
                _runtime_count(
                    hit,
                    "paddleocr_vl",
                    "start_count",
                )
                == 0
            ),
            "gemma_runtime_start_zero": (
                _runtime_count(hit, "gemma", "start_count") == 0
            ),
            "all_project_stages_hit": all_stage_hits,
            "missing_output_render_only": (
                _project_checkpoint_event_gate(
                    missing,
                    require_render_materialized=True,
                )
            ),
            "missing_output_paddle_http_zero": (
                _nested_metric(
                    missing,
                    "paddleocr_vl",
                    "http_attempt_count",
                )
                == 0
            ),
            "missing_output_gemma_http_zero": (
                _nested_metric(
                    missing,
                    "gemma",
                    "gemma_http_attempt_count",
                )
                == 0
            ),
            "missing_output_paddle_runtime_start_zero": (
                _runtime_count(
                    missing,
                    "paddleocr_vl",
                    "start_count",
                )
                == 0
            ),
            "missing_output_gemma_runtime_start_zero": (
                _runtime_count(missing, "gemma", "start_count") == 0
            ),
            "single_page_ocr_downstream_only": (
                _project_partial_recompute_gate(
                    partial,
                    page_index=invalidated_page_index,
                    invalidated_stage="ocr",
                )
            ),
            "single_page_ocr_http_observed": (
                _nested_metric(
                    partial,
                    "paddleocr_vl",
                    "http_attempt_count",
                )
                > 0
            ),
            "single_page_translation_http_observed": (
                _nested_metric(
                    partial,
                    "gemma",
                    "gemma_http_attempt_count",
                )
                > 0
            ),
            "all_hit_reduction_gate": reduction
            >= float(gates["cache_all_hit_improvement_percent"]),
            "missing_output_recreated_exactly": all(
                bool(page.get("output_exists", False))
                for page in (
                    missing.get("page_contract") or {}
                ).get("pages", [])
            ),
        }
    return {
        "scenario": scenario,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "all_hit_reduction_percent": round(reduction, 3),
        "cache_miss_overhead_percent": (
            round(miss_overhead, 3)
            if miss_overhead != float("inf")
            else None
        ),
        "disabled_cold_median_sec": round(disabled_median, 6),
        "enabled_empty_cold_median_sec": round(enabled_median, 6),
        "cold_elapsed_sec": round(cold_winner_median, 6),
        "all_hit_elapsed_sec": round(
            _stage_elapsed(
                hit,
                "ocr" if scenario == "global-ocr" else "full",
            ),
            6,
        ),
    }


def run_cache_scenario(args: argparse.Namespace) -> int:
    _require_reproducible_checkout()
    protocol = load_protocol()
    output_dir = ensure_external_output(
        Path(args.output_dir),
        require_new=True,
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory was not found: {input_dir}")
    sample_count = int(args.sample_count)
    if not 1 <= sample_count <= int(
        protocol["limits"]["pipeline_max_pages"]
    ):
        raise ValueError("--sample-count exceeds the protocol page limit.")
    baseline = _read_json(ROOT / str(protocol["baseline_preset"]))
    private_dir = output_dir / "private"
    shared_corpus_dir = private_dir / "shared-corpus"
    is_global = args.scenario == "global-ocr"
    disabled_preset_path = private_dir / "cache-disabled-preset.json"
    enabled_preset_path = private_dir / "cache-enabled-preset.json"
    disabled_preset = _cache_preset(
        baseline,
        paddle=False,
        project=False,
    )
    disabled_preset["name"] = "cache-disabled-cold-control"
    enabled_preset = _cache_preset(
        baseline,
        paddle=is_global,
        project=not is_global,
    )
    _write_json(disabled_preset_path, disabled_preset)
    _write_json(
        enabled_preset_path,
        enabled_preset,
    )
    cold_orders = balanced_orders(
        ("disabled_cold", "enabled_empty_cold"),
        int(protocol["limits"]["rounds"]),
    )
    stabilization_orders = balanced_orders(
        ("disabled_cold", "enabled_empty_cold"),
        int(protocol["limits"]["cache_stabilization_pairs"]) + 1,
    )[1:]
    protocol_state = _protocol_state(protocol)
    protocol_state.update(
        {
            "command": "run-cache",
            "scenario": args.scenario,
            "sample_count": sample_count,
            "source_language": args.source_lang,
            "cache_stabilization_orders": stabilization_orders,
            "cold_execution_orders": cold_orders,
        }
    )
    _write_json(output_dir / PROTOCOL_STATE_NAME, protocol_state)
    state: dict[str, Any] = {
        "status": "running",
        "protocol": protocol_state,
        "results": {
            "stabilization": [],
            "disabled_cold": [],
            "enabled_empty_cold": [],
        },
    }
    _write_json(output_dir / SUITE_STATE_NAME, state)
    try:
        seed_user_data_root: Path | None = None
        seed_project_file: Path | None = None
        stage = "ocr" if is_global else "full"

        def run_cold_control(
            *,
            candidate_id: str,
            phase: str,
            round_index: int,
            order_index: int,
        ) -> tuple[dict[str, Any], Path, Path | None]:
            enabled = candidate_id == "enabled_empty_cold"
            preset_path = (
                enabled_preset_path
                if enabled
                else disabled_preset_path
            )
            if phase == "stabilization":
                scope = f"pair-{round_index:02d}"
                run_dir = (
                    private_dir
                    / "runs"
                    / "stabilization"
                    / scope
                    / candidate_id
                )
                user_data_root = (
                    private_dir
                    / "isolated-user-data"
                    / "stabilization"
                    / scope
                    / candidate_id
                )
            else:
                scope = f"round-{round_index:02d}"
                run_dir = (
                    private_dir
                    / "runs"
                    / candidate_id
                    / scope
                )
                user_data_root = (
                    private_dir
                    / "isolated-user-data"
                    / candidate_id
                    / scope
                )
            project_file = None
            project_action = "none"
            save_project = False
            if enabled and not is_global:
                if phase == "stabilization":
                    project_file = (
                        private_dir
                        / "projects"
                        / "stabilization"
                        / scope
                        / "screening.ctpr"
                    )
                else:
                    project_file = (
                        private_dir
                        / "projects"
                        / scope
                        / "screening.ctpr"
                    )
                project_action = "create"
                save_project = True
            result = _run_cache_pipeline(
                preset_path=preset_path,
                input_dir=input_dir,
                sample_count=sample_count,
                run_dir=run_dir,
                shared_corpus_dir=shared_corpus_dir,
                user_data_root=user_data_root,
                stage=stage,
                source_lang=args.source_lang,
                project_file=project_file,
                project_action=project_action,
                save_project_after_run=save_project,
            )
            result.update(
                {
                    "candidate": candidate_id,
                    "phase": phase,
                    "round": round_index,
                    "order": order_index,
                }
            )
            return result, user_data_root, project_file

        for pair_index, order in enumerate(
            stabilization_orders,
            start=1,
        ):
            for order_index, candidate_id in enumerate(order, start=1):
                result, _user_data_root, _project_file = run_cold_control(
                    candidate_id=candidate_id,
                    phase="stabilization",
                    round_index=pair_index,
                    order_index=order_index,
                )
                state["results"]["stabilization"].append(result)
                _write_json(output_dir / SUITE_STATE_NAME, state)
                if result["status"] != "passed":
                    raise RuntimeError(
                        "Cache stabilization failed; stopping without "
                        f"automatic repetition: {candidate_id} "
                        f"pair={pair_index}"
                    )

        for round_index, order in enumerate(cold_orders, start=1):
            for order_index, candidate_id in enumerate(order, start=1):
                result, user_data_root, project_file = run_cold_control(
                    candidate_id=candidate_id,
                    phase="measured",
                    round_index=round_index,
                    order_index=order_index,
                )
                state["results"][candidate_id].append(result)
                enabled = candidate_id == "enabled_empty_cold"
                if enabled and seed_user_data_root is None:
                    seed_user_data_root = user_data_root
                    seed_project_file = project_file
                _write_json(output_dir / SUITE_STATE_NAME, state)
                if result["status"] != "passed":
                    raise RuntimeError(
                        "Cache cold control failed; stopping without "
                        f"automatic repetition: {candidate_id} "
                        f"round={round_index}"
                    )

        if seed_user_data_root is None:
            raise RuntimeError("No enabled cold cache seed run was produced.")
        if is_global:
            hit = _run_cache_pipeline(
                preset_path=enabled_preset_path,
                input_dir=input_dir,
                sample_count=sample_count,
                run_dir=private_dir / "runs" / "all_hit",
                shared_corpus_dir=shared_corpus_dir,
                user_data_root=seed_user_data_root,
                stage="ocr",
                source_lang=args.source_lang,
            )
            state["results"]["all_hit"] = hit
            if hit["status"] != "passed":
                raise RuntimeError("Global OCR all-hit run failed.")
        else:
            if seed_project_file is None:
                raise RuntimeError(
                    "No project checkpoint seed project was produced."
                )
            hit = _run_cache_pipeline(
                preset_path=enabled_preset_path,
                input_dir=input_dir,
                sample_count=sample_count,
                run_dir=private_dir
                / "runs"
                / "all_hit_existing_output",
                shared_corpus_dir=shared_corpus_dir,
                user_data_root=seed_user_data_root,
                stage="full",
                source_lang=args.source_lang,
                project_file=seed_project_file,
                project_action="resume",
            )
            state["results"]["all_hit_existing_output"] = hit
            if hit["status"] != "passed":
                raise RuntimeError(
                    "Project existing-output all-hit run failed."
                )
            deleted = _delete_owned_render_outputs(
                hit,
                owned_root=output_dir,
            )
            state["deleted_render_output_count"] = len(deleted)
            _write_json(output_dir / SUITE_STATE_NAME, state)
            missing = _run_cache_pipeline(
                preset_path=enabled_preset_path,
                input_dir=input_dir,
                sample_count=sample_count,
                run_dir=private_dir
                / "runs"
                / "all_hit_missing_output",
                shared_corpus_dir=shared_corpus_dir,
                user_data_root=seed_user_data_root,
                stage="full",
                source_lang=args.source_lang,
                project_file=seed_project_file,
                project_action="resume",
            )
            state["results"]["all_hit_missing_output"] = missing
            if missing["status"] != "passed":
                raise RuntimeError(
                    "Project missing-output all-hit run failed."
                )
            _write_json(output_dir / SUITE_STATE_NAME, state)
            invalidated_page_index = next(
                (
                    int(page.get("page_index", -1))
                    for page in (
                        hit.get("page_contract") or {}
                    ).get("pages", [])
                    if isinstance(page, Mapping)
                    and int(page.get("block_count", 0) or 0) > 0
                ),
                -1,
            )
            if invalidated_page_index < 0:
                raise RuntimeError(
                    "Project cache corpus has no text-bearing page for "
                    "controlled OCR invalidation."
                )
            partial = _run_cache_pipeline(
                preset_path=enabled_preset_path,
                input_dir=input_dir,
                sample_count=sample_count,
                run_dir=private_dir
                / "runs"
                / "single_page_ocr_invalidated",
                shared_corpus_dir=shared_corpus_dir,
                user_data_root=seed_user_data_root,
                stage="full",
                source_lang=args.source_lang,
                project_file=seed_project_file,
                project_action="resume",
                project_invalidate_page_index=invalidated_page_index,
                project_invalidate_stage="ocr",
            )
            partial["invalidated_page_index"] = invalidated_page_index
            state["results"]["single_page_ocr_invalidated"] = partial
            if partial["status"] != "passed":
                raise RuntimeError(
                    "Project one-page downstream recompute run failed."
                )
        analysis = analyze_cache_results(
            protocol=protocol,
            scenario=args.scenario,
            results=state["results"],
        )
        state["status"] = "completed"
        state["analysis"] = analysis
        _write_json(output_dir / SUITE_STATE_NAME, state)
        _write_json(output_dir / SUMMARY_NAME, analysis)
        (output_dir / REPORT_NAME).write_text(
            _render_cache_report(analysis),
            encoding="utf-8",
        )
        return 0 if analysis["passed"] else 1
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output_dir / SUITE_STATE_NAME, state)
        raise
    finally:
        _stop_managed_containers()


def _render_cache_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Persistent Cache Validation Report",
        "",
        f"- scenario: `{analysis.get('scenario')}`",
        f"- passed: `{analysis.get('passed')}`",
        f"- disabled cold median sec: `{analysis.get('disabled_cold_median_sec')}`",
        f"- enabled empty cold median sec: `{analysis.get('enabled_empty_cold_median_sec')}`",
        f"- cache miss overhead percent: `{analysis.get('cache_miss_overhead_percent')}`",
        f"- cold winner elapsed sec: `{analysis.get('cold_elapsed_sec')}`",
        f"- all-hit elapsed sec: `{analysis.get('all_hit_elapsed_sec')}`",
        f"- all-hit reduction percent: `{analysis.get('all_hit_reduction_percent')}`",
        "",
        "| check | passed |",
        "|---|---:|",
    ]
    for name, passed in (analysis.get("checks") or {}).items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(
        [
            "",
            "Raw cache databases, project sidecars, images, OCR, translations, "
            "and local paths remain only in this external suite directory.",
            "",
        ]
    )
    return "\n".join(lines)


def describe_protocol(args: argparse.Namespace) -> int:
    protocol = load_protocol()
    output_dir = ensure_external_output(
        Path(args.output_dir),
        require_new=True,
    )
    state = _protocol_state(protocol)
    state["command"] = "describe"
    state["protocol"] = protocol
    _write_json(output_dir / PROTOCOL_STATE_NAME, state)
    lines = [
        "# Cold/Cache Finalization Protocol",
        "",
        f"- protocol version: `{protocol['protocol_version']}`",
        f"- commit: `{state['commit']}`",
        f"- worktree clean: `{state['worktree_clean']}`",
        f"- code contract SHA-256: `{state['code_contract_sha256']}`",
        f"- families: `{len(protocol.get('families', []))}`",
        f"- conditional spikes: `{len(protocol.get('conditional_spikes', []))}`",
        "",
        "| family | stage | execution | quality gate |",
        "|---|---|---|---|",
    ]
    for family in protocol.get("families", []):
        lines.append(
            "| {id} | {stage} | {execution} | {quality} |".format(
                id=family.get("id", ""),
                stage=family.get("stage", ""),
                execution=family.get("execution", ""),
                quality=family.get("quality_gate", ""),
            )
        )
    (output_dir / REPORT_NAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final cold-path and persistent-cache screening protocol "
            "through the real offscreen Comic Translate pipeline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser(
        "describe",
        help="Validate and export the locked candidate protocol.",
    )
    describe.add_argument("--output-dir", required=True)
    describe.set_defaults(handler=describe_protocol)

    pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run one pipeline candidate family for three reordered rounds.",
    )
    pipeline.add_argument("--family", required=True)
    pipeline.add_argument("--axis", default="")
    pipeline.add_argument("--base-preset", default="")
    pipeline.add_argument(
        "--full-reference-suite",
        default="",
        help=(
            "Completed full-pipeline suite_state.json used to convert a "
            "stage improvement into an expected whole-pipeline improvement."
        ),
    )
    pipeline.add_argument("--input-dir", required=True)
    pipeline.add_argument("--output-dir", required=True)
    pipeline.add_argument("--sample-count", type=int, default=6)
    pipeline.add_argument("--source-lang", default="Japanese")
    pipeline.set_defaults(handler=run_pipeline_family)

    translation = subparsers.add_parser(
        "run-translation",
        help="Run a 54-block multilingual translation family.",
    )
    translation.add_argument("--family", required=True)
    translation.add_argument("--axis", default="")
    translation.add_argument("--model-key", default="iq4_nl")
    translation.add_argument("--source-summary", required=True)
    translation.add_argument("--output-dir", required=True)
    translation.set_defaults(handler=run_translation_family)

    cache = subparsers.add_parser(
        "run-cache",
        help="Validate global OCR or project all-hit behavior.",
    )
    cache.add_argument(
        "--scenario",
        required=True,
        choices=("global-ocr", "project"),
    )
    cache.add_argument("--input-dir", required=True)
    cache.add_argument("--output-dir", required=True)
    cache.add_argument("--sample-count", type=int, default=6)
    cache.add_argument("--source-lang", default="Japanese")
    cache.set_defaults(handler=run_cache_scenario)

    hidden = subparsers.add_parser("_translation-profile")
    hidden.add_argument("--source-summary", required=True)
    hidden.add_argument("--output", required=True)
    hidden.add_argument("--candidate-id", required=True)
    hidden.add_argument("--model", required=True)
    hidden.add_argument("--expected-model-sha256", required=True)
    hidden.add_argument("--chunk-size", type=int, required=True)
    hidden.add_argument("--context-size", type=int, required=True)
    hidden.add_argument("--n-parallel", type=int, required=True)
    hidden.add_argument("--concurrency", type=int, required=True)
    hidden.add_argument(
        "--language-order",
        nargs=3,
        required=True,
    )
    hidden.set_defaults(handler=_translation_profile)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(
            f"[cold-cache-finalization] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
