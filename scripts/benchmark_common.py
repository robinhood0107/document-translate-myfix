from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_DIR = ROOT / "Sample"
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_SMOKE_COUNT = 5
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
EXCLUDED_SAMPLE_PARENT_NAMES = {
    "translated_images",
    "translated_texts",
    "raw_texts",
    "cleaned_images",
    "ocr_debugs",
}
PRESET_DIRS = [
    ROOT / "benchmarks" / "presets",
    ROOT / "benchmarks" / "paddleocr_vl15" / "presets",
    ROOT / "benchmarks" / "ocr_combo" / "presets",
]
OCR_BUNDLE_DIR = ROOT / "paddleocr_vl_docker_files"
HUNYUAN_OCR_BUNDLE_DIR = ROOT / "hunyuanocr_docker_files"
ROOT_GEMMA_COMPOSE = ROOT / "docker-compose.yaml"
GEMMA_CONTAINER_NAMES = [
    "gemma-local-server",
]
PADDLEOCR_VL_CONTAINER_NAMES = [
    "paddleocr-server",
    "paddleocr-vllm",
]
HUNYUAN_OCR_CONTAINER_NAMES = [
    "hunyuanocr-local-server",
]
DEFAULT_CONTAINER_NAMES = GEMMA_CONTAINER_NAMES + PADDLEOCR_VL_CONTAINER_NAMES
ALL_BENCHMARK_CONTAINER_NAMES = (
    GEMMA_CONTAINER_NAMES + PADDLEOCR_VL_CONTAINER_NAMES + HUNYUAN_OCR_CONTAINER_NAMES
)
GEMMA_HEALTH_URLS = [
    "http://127.0.0.1:18080/health",
    "http://127.0.0.1:18000/v1/models",
]
PADDLEOCR_VL_HEALTH_URLS = [
    "http://127.0.0.1:28118/docs",
]
HUNYUAN_OCR_HEALTH_URLS = [
    "http://127.0.0.1:28080/health",
]


def ocr_runtime_kind(preset: dict[str, Any]) -> str:
    ocr_runtime = preset.get("ocr_runtime", {})
    if isinstance(ocr_runtime, dict):
        kind = str(ocr_runtime.get("kind", "") or "").strip().lower()
        if kind:
            return kind

    app = preset.get("app", {})
    ocr_name = str((app.get("ocr") if isinstance(app, dict) else "") or "").strip()
    if ocr_name == "PaddleOCR VL":
        return "paddleocr_vl"
    if ocr_name == "HunyuanOCR":
        return "hunyuanocr"
    return "internal"


def resolve_runtime_container_names(
    preset: dict[str, Any],
    runtime_services: str = "full",
) -> list[str]:
    names: list[str] = []
    if runtime_services != "ocr-only":
        names.extend(GEMMA_CONTAINER_NAMES)

    kind = ocr_runtime_kind(preset)
    if kind == "paddleocr_vl":
        names.extend(PADDLEOCR_VL_CONTAINER_NAMES)
    elif kind == "hunyuanocr":
        names.extend(HUNYUAN_OCR_CONTAINER_NAMES)
    return names


def resolve_runtime_health_urls(
    preset: dict[str, Any],
    runtime_services: str = "full",
) -> list[str]:
    urls: list[str] = []
    if runtime_services != "ocr-only":
        urls.extend(GEMMA_HEALTH_URLS)

    kind = ocr_runtime_kind(preset)
    if kind == "paddleocr_vl":
        urls.extend(PADDLEOCR_VL_HEALTH_URLS)
    elif kind == "hunyuanocr":
        urls.extend(HUNYUAN_OCR_HEALTH_URLS)
    return urls


def repo_root() -> Path:
    return ROOT


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def benchmark_default_output_root() -> Path:
    env_root = os.getenv("CT_BENCH_OUTPUT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser()

    return ROOT / "banchmark_result_log"


def repo_relative_str(path: str | Path) -> str:
    try:
        return "./" + str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def run_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = (
            f"Command failed (exit={completed.returncode}): {' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"stdout:\n{(completed.stdout or '').strip()}\n"
            f"stderr:\n{(completed.stderr or '').strip()}"
        )
        raise RuntimeError(detail)
    return completed


def remove_containers(container_names: list[str]) -> None:
    for name in container_names:
        subprocess.run(
            ["docker", "rm", "-f", name],
            check=False,
            capture_output=True,
            text=True,
        )


def _python3_yaml_load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _python3_yaml_dump(path: Path, payload: dict[str, Any]) -> None:
    dumped = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(dumped, encoding="utf-8")


def load_preset(preset: str) -> tuple[dict[str, Any], Path]:
    candidate = Path(preset)
    if not candidate.is_file():
        for preset_dir in PRESET_DIRS:
            for suffix in (".json", ".yml", ".yaml"):
                test_path = preset_dir / f"{preset}{suffix}"
                if test_path.is_file():
                    candidate = test_path
                    break
            if candidate.is_file():
                break
    if not candidate.is_file():
        raise FileNotFoundError(f"Unable to find preset: {preset}")

    text = candidate.read_text(encoding="utf-8")
    if candidate.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _python3_yaml_load(candidate)
    if not isinstance(data, dict):
        raise ValueError(f"Preset is not a mapping: {candidate}")
    return data, candidate


def select_sample_images(
    sample_dir: str | Path = DEFAULT_SAMPLE_DIR,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> list[Path]:
    root = Path(sample_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Sample directory does not exist: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not any(part in EXCLUDED_SAMPLE_PARENT_NAMES for part in path.parts)
        and not any(part.startswith("comic_translate_") for part in path.parts)
    )
    if len(files) < sample_count:
        raise RuntimeError(
            f"Need at least {sample_count} benchmark images in {root}, found {len(files)}."
        )
    return files[:sample_count]


def resolve_corpus(
    sample_dir: str | Path = DEFAULT_SAMPLE_DIR,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, list[Path]]:
    representative = select_sample_images(sample_dir, sample_count=sample_count)
    smoke_count = min(DEFAULT_SMOKE_COUNT, len(representative))
    return {
        "smoke": representative[:smoke_count],
        "representative": representative,
    }


def _update_command_option(command: list[Any], option: str, values: list[str]) -> None:
    command_strs = [str(item) for item in command]
    try:
        index = command_strs.index(option)
    except ValueError:
        command.extend([option, *values])
        return

    del command[index : index + 2]
    for offset, value in enumerate([option, *values]):
        command.insert(index + offset, value)


def _stage_gemma_runtime(preset: dict[str, Any], runtime_dir: Path) -> dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    compose = _python3_yaml_load(ROOT_GEMMA_COMPOSE)
    service = compose["services"]["gemma-local-server"]
    command = list(service.get("command") or [])
    volumes = list(service.get("volumes") or [])
    gemma = preset.get("gemma", {})
    testmodel_dir = (ROOT / "testmodel").resolve()

    normalized_volumes: list[Any] = []
    for volume in volumes:
        if isinstance(volume, str) and volume.startswith("./testmodel:"):
            normalized_volumes.append(f"{testmodel_dir.as_posix()}:/models:ro")
        else:
            normalized_volumes.append(volume)
    if normalized_volumes:
        service["volumes"] = normalized_volumes

    if gemma.get("image"):
        service["image"] = str(gemma["image"])
    if gemma.get("pull_policy"):
        service["pull_policy"] = str(gemma["pull_policy"])

    if gemma.get("model_path"):
        _update_command_option(command, "-m", [str(gemma["model_path"])])
    if gemma.get("context_size") is not None:
        _update_command_option(command, "-c", [str(gemma["context_size"])])
    if gemma.get("n_parallel") is not None:
        _update_command_option(command, "-np", [str(gemma["n_parallel"])])
    if gemma.get("threads") is not None:
        _update_command_option(command, "-t", [str(gemma["threads"])])
    if gemma.get("n_gpu_layers") is not None:
        _update_command_option(command, "--n-gpu-layers", [str(gemma["n_gpu_layers"])])
    if gemma.get("reasoning") is not None:
        _update_command_option(command, "--reasoning", [str(gemma["reasoning"])])
    if gemma.get("reasoning_budget") is not None:
        _update_command_option(
            command, "--reasoning-budget", [str(gemma["reasoning_budget"])]
        )
    if gemma.get("reasoning_format") is not None:
        _update_command_option(
            command, "--reasoning-format", [str(gemma["reasoning_format"])]
        )
    if gemma.get("predict") is not None:
        _update_command_option(command, "-n", [str(gemma["predict"])])

    service["command"] = command
    compose_path = runtime_dir / "docker-compose.yaml"
    _python3_yaml_dump(compose_path, compose)
    return {
        "compose_path": str(compose_path.resolve()),
        "service_name": "gemma-local-server",
    }


def _stage_ocr_runtime(preset: dict[str, Any], runtime_dir: Path) -> dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    compose = _python3_yaml_load(OCR_BUNDLE_DIR / "docker-compose.yaml")
    ocr_runtime = preset.get("ocr_runtime", {})
    front_device = str(ocr_runtime.get("front_device", "gpu:0"))
    use_hpip = bool(ocr_runtime.get("use_hpip", False))

    layout_service = compose["services"]["paddleocr-layout"]
    if ocr_runtime.get("layout_image"):
        layout_service["image"] = str(ocr_runtime["layout_image"])
    layout_command = str(layout_service.get("command") or "")
    layout_service["command"] = layout_command.replace("--device gpu:0", f"--device {front_device}")
    layout_command = str(layout_service.get("command") or "")
    if use_hpip:
        if "--use_hpip" not in layout_command:
            layout_command = layout_command.rstrip() + " --use_hpip"
    else:
        layout_command = layout_command.replace(" --use_hpip", "").replace("--use_hpip", "")
    layout_service["command"] = layout_command
    if front_device == "cpu":
        layout_service.pop("gpus", None)
    else:
        layout_service["gpus"] = "all"

    compose_path = runtime_dir / "docker-compose.yaml"
    _python3_yaml_dump(compose_path, compose)

    pipeline_conf = _python3_yaml_load(OCR_BUNDLE_DIR / "pipeline_conf.yaml")
    vl_genai_config = (
        pipeline_conf.setdefault("SubModules", {})
        .setdefault("VLRecognition", {})
        .setdefault("genai_config", {})
    )
    if "max_concurrency" in ocr_runtime:
        vl_genai_config["max_concurrency"] = int(ocr_runtime["max_concurrency"])
    pipeline_path = runtime_dir / "pipeline_conf.yaml"
    _python3_yaml_dump(pipeline_path, pipeline_conf)

    vllm_conf = _python3_yaml_load(OCR_BUNDLE_DIR / "vllm_config.yml")
    for key in ("gpu_memory_utilization", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "dtype"):
        if key in ocr_runtime:
            vllm_conf[key] = ocr_runtime[key]
    vllm_path = runtime_dir / "vllm_config.yml"
    _python3_yaml_dump(vllm_path, vllm_conf)

    return {
        "kind": "paddleocr_vl",
        "compose_path": str(compose_path.resolve()),
        "pipeline_conf_path": str(pipeline_path.resolve()),
        "vllm_config_path": str(vllm_path.resolve()),
        "service_names": ["paddleocr-server", "paddleocr-vllm"],
    }


def _stage_hunyuan_ocr_runtime(preset: dict[str, Any], runtime_dir: Path) -> dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    compose = _python3_yaml_load(HUNYUAN_OCR_BUNDLE_DIR / "docker-compose.yaml")
    service = compose["services"]["hunyuanocr-local-server"]
    command = list(service.get("command") or [])
    volumes = list(service.get("volumes") or [])
    ocr_runtime = preset.get("ocr_runtime", {}) if isinstance(preset.get("ocr_runtime"), dict) else {}
    testmodel_dir = (ROOT / "testmodel").resolve()

    normalized_volumes: list[Any] = []
    for volume in volumes:
        if isinstance(volume, str) and volume.startswith("../testmodel:"):
            normalized_volumes.append(f"{testmodel_dir.as_posix()}:/models:ro")
        else:
            normalized_volumes.append(volume)
    if normalized_volumes:
        service["volumes"] = normalized_volumes

    if ocr_runtime.get("image"):
        service["image"] = str(ocr_runtime["image"])
    if ocr_runtime.get("pull_policy"):
        service["pull_policy"] = str(ocr_runtime["pull_policy"])
    if ocr_runtime.get("model_path"):
        _update_command_option(command, "-m", [str(ocr_runtime["model_path"])])
    if ocr_runtime.get("mmproj_path"):
        _update_command_option(command, "--mmproj", [str(ocr_runtime["mmproj_path"])])
    if ocr_runtime.get("context_size") is not None:
        _update_command_option(command, "-c", [str(ocr_runtime["context_size"])])
    if ocr_runtime.get("n_parallel") is not None:
        _update_command_option(command, "-np", [str(ocr_runtime["n_parallel"])])
    if ocr_runtime.get("threads") is not None:
        _update_command_option(command, "-t", [str(ocr_runtime["threads"])])
    if ocr_runtime.get("n_gpu_layers") is not None:
        _update_command_option(command, "--n-gpu-layers", [str(ocr_runtime["n_gpu_layers"])])

    service["command"] = command
    compose_path = runtime_dir / "docker-compose.yaml"
    _python3_yaml_dump(compose_path, compose)
    return {
        "kind": "hunyuanocr",
        "compose_path": str(compose_path.resolve()),
        "service_names": ["hunyuanocr-local-server"],
    }


def _stage_internal_ocr_runtime(runtime_dir: Path) -> dict[str, Any]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return {
        "kind": "internal",
        "compose_path": "",
        "service_names": [],
    }


def stage_runtime_files(preset: dict[str, Any], runtime_dir: str | Path) -> dict[str, Any]:
    base = Path(runtime_dir)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    gemma_runtime = _stage_gemma_runtime(preset, base / "gemma")
    kind = ocr_runtime_kind(preset)
    if kind == "paddleocr_vl":
        ocr_runtime = _stage_ocr_runtime(preset, base / "ocr")
    elif kind == "hunyuanocr":
        ocr_runtime = _stage_hunyuan_ocr_runtime(preset, base / "ocr")
    else:
        ocr_runtime = _stage_internal_ocr_runtime(base / "ocr")

    app_settings_path = base / "app_settings.json"
    app_settings_path.write_text(
        json.dumps(
            {
                "app": preset.get("app", {}),
                "gemma": preset.get("gemma", {}),
                "ocr_client": preset.get("ocr_client", {}),
                "hunyuan_ocr_client": preset.get("hunyuan_ocr_client", {}),
                "ocr_runtime": preset.get("ocr_runtime", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "runtime_dir": str(base),
        "runtime_dir_abs": str(base.resolve()),
        "gemma": gemma_runtime,
        "ocr": ocr_runtime,
        "app_settings_path": str(app_settings_path.resolve()),
    }


def benchmark_output_root() -> Path:
    root = benchmark_default_output_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_run_dir(
    label: str,
    *,
    root: str | Path | None = None,
    include_timestamp: bool = True,
) -> Path:
    base_root = Path(root) if root is not None else benchmark_output_root()
    base_root.mkdir(parents=True, exist_ok=True)
    dir_name = f"{now_stamp()}_{label}" if include_timestamp else label
    run_dir = base_root / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metrics_path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _primary_gpu(entry: dict[str, Any]) -> dict[str, Any] | None:
    gpu = entry.get("gpu")
    if isinstance(gpu, dict):
        primary = gpu.get("primary")
        if isinstance(primary, dict):
            return primary
    return None


def summarize_metrics(metrics_path: str | Path) -> dict[str, Any]:
    rows = load_metrics(metrics_path)
    if not rows:
        return {
            "entry_count": 0,
            "tag_counts": {},
            "stage_stats": {},
        }

    tag_counts = Counter(str(row.get("tag", "") or "") for row in rows)
    stage_open: dict[tuple[str, str, str], deque[float]] = defaultdict(deque)
    stage_durations: dict[str, list[float]] = defaultdict(list)

    used_values: list[int] = []
    free_values: list[int] = []
    gpu_util_values: list[int] = []
    mem_util_values: list[int] = []
    counters = {
        "gemma_json_retry_count": 0,
        "gemma_chunk_retry_events": 0,
        "gemma_truncated_count": 0,
        "gemma_empty_content_count": 0,
        "gemma_missing_key_count": 0,
        "gemma_reasoning_without_final_count": 0,
        "gemma_schema_validation_fail_count": 0,
        "ocr_total_block_count": 0,
        "ocr_empty_block_count": 0,
        "ocr_low_quality_block_count": 0,
        "ocr_cache_hit_count": 0,
    }

    for row in rows:
        ts = float(row.get("ts", 0.0) or 0.0)
        tag = str(row.get("tag", "") or "")
        image_path = str(row.get("image_path", "") or "")
        pipeline_mode = str(row.get("pipeline_mode", "") or "")

        if tag.endswith("_start"):
            stage = tag[: -len("_start")]
            stage_open[(stage, image_path, pipeline_mode)].append(ts)
        elif tag.endswith("_end"):
            stage = tag[: -len("_end")]
            key = (stage, image_path, pipeline_mode)
            if stage_open[key]:
                started = stage_open[key].popleft()
                stage_durations[stage].append(max(0.0, ts - started))

        primary = _primary_gpu(row)
        if primary:
            used = primary.get("memory_used_mb")
            free = primary.get("memory_free_mb")
            gpu_util = primary.get("gpu_util_percent")
            mem_util = primary.get("memory_util_percent")
            if isinstance(used, int):
                used_values.append(used)
            if isinstance(free, int):
                free_values.append(free)
            if isinstance(gpu_util, int):
                gpu_util_values.append(gpu_util)
            if isinstance(mem_util, int):
                mem_util_values.append(mem_util)

        for key in counters:
            value = row.get(key)
            if isinstance(value, (int, float)):
                counters[key] += int(value)
        if tag == "ocr_end" and str(row.get("cache_status", "") or "").lower() == "hit":
            counters["ocr_cache_hit_count"] += 1

    stage_stats = {}
    for stage, values in sorted(stage_durations.items()):
        if not values:
            continue
        ordered = sorted(values)
        def percentile(p: float) -> float:
            rank = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
            return ordered[rank]
        stage_stats[stage] = {
            "count": len(values),
            "total_sec": round(sum(values), 3),
            "median_sec": round(ordered[len(ordered) // 2], 3),
            "p95_sec": round(percentile(0.95), 3),
            "p99_sec": round(percentile(0.99), 3),
            "max_sec": round(max(values), 3),
        }

    ocr_empty_rate = None
    ocr_low_quality_rate = None
    if counters["ocr_total_block_count"] > 0:
        ocr_empty_rate = round(counters["ocr_empty_block_count"] / counters["ocr_total_block_count"], 4)
        ocr_low_quality_rate = round(counters["ocr_low_quality_block_count"] / counters["ocr_total_block_count"], 4)

    return {
        "entry_count": len(rows),
        "started_at": rows[0].get("ts"),
        "ended_at": rows[-1].get("ts"),
        "elapsed_sec": round(float(rows[-1].get("ts", 0.0)) - float(rows[0].get("ts", 0.0)), 3),
        "tag_counts": dict(sorted(tag_counts.items())),
        "stage_stats": stage_stats,
        "page_done_count": tag_counts.get("page_done", 0),
        "page_failed_count": tag_counts.get("page_failed", 0),
        "gpu_peak_used_mb": max(used_values) if used_values else None,
        "gpu_floor_free_mb": min(free_values) if free_values else None,
        "gpu_peak_util_percent": max(gpu_util_values) if gpu_util_values else None,
        "gpu_peak_mem_util_percent": max(mem_util_values) if mem_util_values else None,
        "detect_total_sec": stage_stats.get("detect", {}).get("total_sec"),
        "ocr_total_sec": stage_stats.get("ocr", {}).get("total_sec"),
        "detect_ocr_total_sec": round(
            float(stage_stats.get("detect", {}).get("total_sec") or 0.0)
            + float(stage_stats.get("ocr", {}).get("total_sec") or 0.0),
            3,
        ),
        "ocr_median_sec": stage_stats.get("ocr", {}).get("median_sec"),
        "ocr_page_p50_sec": stage_stats.get("ocr", {}).get("median_sec"),
        "ocr_page_p95_sec": stage_stats.get("ocr", {}).get("p95_sec"),
        "ocr_page_p99_sec": stage_stats.get("ocr", {}).get("p99_sec"),
        "translate_median_sec": stage_stats.get("translate", {}).get("median_sec"),
        "inpaint_median_sec": stage_stats.get("inpaint", {}).get("median_sec"),
        "gemma_json_retry_count": counters["gemma_json_retry_count"],
        "gemma_chunk_retry_events": counters["gemma_chunk_retry_events"],
        "gemma_truncated_count": counters["gemma_truncated_count"],
        "gemma_empty_content_count": counters["gemma_empty_content_count"],
        "gemma_missing_key_count": counters["gemma_missing_key_count"],
        "gemma_reasoning_without_final_count": counters["gemma_reasoning_without_final_count"],
        "gemma_schema_validation_fail_count": counters["gemma_schema_validation_fail_count"],
        "ocr_total_block_count": counters["ocr_total_block_count"],
        "ocr_empty_block_count": counters["ocr_empty_block_count"],
        "ocr_low_quality_block_count": counters["ocr_low_quality_block_count"],
        "ocr_cache_hit_count": counters["ocr_cache_hit_count"],
        "ocr_empty_rate": ocr_empty_rate,
        "ocr_low_quality_rate": ocr_low_quality_rate,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        "## Run",
        "",
        f"- elapsed_sec: `{summary.get('elapsed_sec')}`",
        f"- page_done_count: `{summary.get('page_done_count')}`",
        f"- page_failed_count: `{summary.get('page_failed_count')}`",
        f"- gpu_peak_used_mb: `{summary.get('gpu_peak_used_mb')}`",
        f"- gpu_floor_free_mb: `{summary.get('gpu_floor_free_mb')}`",
        f"- gpu_peak_util_percent: `{summary.get('gpu_peak_util_percent')}`",
        f"- detect_total_sec: `{summary.get('detect_total_sec')}`",
        f"- ocr_total_sec: `{summary.get('ocr_total_sec')}`",
        f"- detect_ocr_total_sec: `{summary.get('detect_ocr_total_sec')}`",
        f"- ocr_median_sec: `{summary.get('ocr_median_sec')}`",
        f"- ocr_page_p95_sec: `{summary.get('ocr_page_p95_sec')}`",
        f"- ocr_page_p99_sec: `{summary.get('ocr_page_p99_sec')}`",
        f"- translate_median_sec: `{summary.get('translate_median_sec')}`",
        f"- inpaint_median_sec: `{summary.get('inpaint_median_sec')}`",
        f"- gemma_json_retry_count: `{summary.get('gemma_json_retry_count')}`",
        f"- gemma_chunk_retry_events: `{summary.get('gemma_chunk_retry_events')}`",
        f"- gemma_truncated_count: `{summary.get('gemma_truncated_count')}`",
        f"- gemma_empty_content_count: `{summary.get('gemma_empty_content_count')}`",
        f"- gemma_missing_key_count: `{summary.get('gemma_missing_key_count')}`",
        f"- gemma_reasoning_without_final_count: `{summary.get('gemma_reasoning_without_final_count')}`",
        f"- gemma_schema_validation_fail_count: `{summary.get('gemma_schema_validation_fail_count')}`",
        f"- ocr_total_block_count: `{summary.get('ocr_total_block_count')}`",
        f"- ocr_empty_block_count: `{summary.get('ocr_empty_block_count')}`",
        f"- ocr_low_quality_block_count: `{summary.get('ocr_low_quality_block_count')}`",
        f"- ocr_cache_hit_count: `{summary.get('ocr_cache_hit_count')}`",
        f"- ocr_empty_rate: `{summary.get('ocr_empty_rate')}`",
        f"- ocr_low_quality_rate: `{summary.get('ocr_low_quality_rate')}`",
        "",
        "## Stage Stats",
        "",
        "| stage | count | total_sec | median_sec | p95_sec | p99_sec | max_sec |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage, payload in summary.get("stage_stats", {}).items():
        lines.append(
            f"| {stage} | {payload.get('count')} | {payload.get('total_sec')} | {payload.get('median_sec')} | {payload.get('p95_sec')} | {payload.get('p99_sec')} | {payload.get('max_sec')} |"
        )

    lines.extend(
        [
            "",
            "## Tag Counts",
            "",
            "| tag | count |",
            "| --- | --- |",
        ]
    )
    for tag, count in summary.get("tag_counts", {}).items():
        lines.append(f"| {tag} | {count} |")

    lines.append("")
    return "\n".join(lines)
