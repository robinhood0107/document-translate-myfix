#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CT_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("CT_ENABLE_MEMLOG", "1")
os.environ.setdefault("CT_ENABLE_GPU_BENCH", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from app.ui.main_window.constants import supported_source_languages, supported_target_languages
from benchmark_common import (
    DEFAULT_CONTAINER_NAMES,
    create_run_dir,
    load_preset,
    remove_containers,
    render_summary_markdown,
    resolve_corpus,
    run_command,
    stage_runtime_files,
    summarize_metrics,
    write_json,
)
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.gpu_metrics import collect_runtime_snapshot, write_snapshot_json

BENCHMARK_FONT_ROOT = ROOT / "benchmarks-fonts"
BENCHMARK_FONT_EXTENSIONS = {".ttf", ".ttc", ".otf", ".woff", ".woff2"}
LANGUAGE_FONT_FALLBACKS = {
    "Simplified Chinese": ["Simplified Chinese", "Chinese"],
    "Traditional Chinese": ["Traditional Chinese", "Chinese"],
    "Brazilian Portuguese": ["Brazilian Portuguese", "Portuguese"],
}
GEMMA_ENV_OVERRIDES = {
    "temperature": "CT_GEMMA_TEMPERATURE",
    "top_k": "CT_GEMMA_TOP_K",
    "top_p": "CT_GEMMA_TOP_P",
    "min_p": "CT_GEMMA_MIN_P",
    "response_format_mode": "CT_GEMMA_RESPONSE_FORMAT_MODE",
    "response_schema_mode": "CT_GEMMA_RESPONSE_SCHEMA_MODE",
    "think_briefly_prompt": "CT_GEMMA_THINK_BRIEFLY_PROMPT",
}


def _log(message: str) -> None:
    print(f"[benchmark] {message}", flush=True)


def _settings_snapshot() -> dict[str, object]:
    settings = QSettings("ComicLabs", "ComicTranslate")
    return {key: settings.value(key) for key in settings.allKeys()}


def _restore_settings(snapshot: dict[str, object]) -> None:
    settings = QSettings("ComicLabs", "ComicTranslate")
    settings.clear()
    for key, value in snapshot.items():
        settings.setValue(key, value)
    settings.sync()


def _wait_for_url(url: str, timeout_sec: int = 180) -> None:
    started = time.time()
    while time.time() - started < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _benchmark_font_dirs(target_lang: str) -> list[Path]:
    candidates = LANGUAGE_FONT_FALLBACKS.get(target_lang, [target_lang])
    return [BENCHMARK_FONT_ROOT / name for name in candidates]


def _find_benchmark_font(target_lang: str) -> Path | None:
    for directory in _benchmark_font_dirs(target_lang):
        if not directory.is_dir():
            continue
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in BENCHMARK_FONT_EXTENSIONS
        )
        if files:
            return files[0]
    return None


def _apply_benchmark_font(window, target_lang: str) -> None:
    font_path = _find_benchmark_font(target_lang)
    if font_path is not None:
        window.add_custom_font(str(font_path))
        window.set_font(str(font_path))
        _log(
            "벤치 폰트 적용: target={target} font_file={font_file} resolved_family={family}".format(
                target=target_lang,
                font_file=font_path,
                family=window.font_dropdown.currentText(),
            )
        )
        return

    current_font = window.font_dropdown.currentText().strip()
    if current_font:
        _log(
            "벤치 폰트 없음: target={target} searched={dirs} current_font 유지={font}".format(
                target=target_lang,
                dirs=", ".join(str(path) for path in _benchmark_font_dirs(target_lang)),
                font=current_font,
            )
        )
        return

    fallback_font = QApplication.font().family()
    window.set_font(fallback_font)
    _log(
        "벤치 폰트 없음: target={target} searched={dirs} app 기본 폰트 적용={font}".format(
            target=target_lang,
            dirs=", ".join(str(path) for path in _benchmark_font_dirs(target_lang)),
            font=fallback_font,
        )
    )


def _apply_gemma_env(gemma: dict[str, object]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    parts: list[str] = []
    for key, env_name in GEMMA_ENV_OVERRIDES.items():
        snapshot[env_name] = os.environ.get(env_name)
        value = gemma.get(key)
        if value is None:
            os.environ.pop(env_name, None)
            continue
        os.environ[env_name] = str(value)
        parts.append(f"{key}={value}")
    if parts:
        _log("Gemma runtime override 적용: " + ", ".join(parts))
    return snapshot


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _ensure_managed_runtime(run_dir: Path, preset: dict[str, object]) -> None:
    runtime_dir = run_dir / "runtime"
    _log(f"managed runtime staging 시작: {runtime_dir}")
    staged = stage_runtime_files(preset, runtime_dir)
    _log("managed runtime 기존 컨테이너 정리 중...")
    remove_containers(DEFAULT_CONTAINER_NAMES)
    _log(f"Gemma compose 적용: {staged['gemma']['compose_path']}")
    run_command(
        ["docker", "compose", "-f", staged["gemma"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "gemma",
    )
    _log(f"OCR compose 적용: {staged['ocr']['compose_path']}")
    run_command(
        ["docker", "compose", "-f", staged["ocr"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "ocr",
    )
    _log("managed runtime health-check 대기 중...")
    _wait_for_url("http://127.0.0.1:18080/health")
    _wait_for_url("http://127.0.0.1:18000/v1/models")
    _wait_for_url("http://127.0.0.1:28118/docs")
    _log("managed runtime health-check 완료")


def _write_container_logs(run_dir: Path, container_names: list[str]) -> None:
    log_dir = run_dir / "docker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for container_name in container_names:
        completed = run_command(
            ["docker", "logs", "--tail", "200", container_name],
            check=False,
        )
        (log_dir / f"{container_name}.log").write_text(
            (completed.stdout or "") + (completed.stderr or ""),
            encoding="utf-8",
        )


def _configure_window(window, preset: dict[str, object], source_lang: str, target_lang: str) -> None:
    ui = window.settings_page.ui
    app_config = preset.get("app", {})
    gemma = preset.get("gemma", {})
    ocr_client = preset.get("ocr_client", {})

    ui.use_gpu_checkbox.setChecked(bool(app_config.get("use_gpu", True)))
    ui.translator_combo.setCurrentText(str(app_config.get("translator", "Custom Local Server(Gemma)")))
    ui.ocr_combo.setCurrentText(str(app_config.get("ocr", "PaddleOCR VL")))
    ui.detector_combo.setCurrentText(str(app_config.get("detector", "RT-DETR-v2")))
    ui.inpainter_combo.setCurrentText(str(app_config.get("inpainter", "AOT")))

    ui.save_keys_checkbox.setChecked(True)
    ui.extra_context.setPlainText(str(app_config.get("extra_context", "")))
    ui.credential_widgets["Custom Local Server(Gemma)_api_url"].setText(
        str(gemma.get("endpoint_url", "http://127.0.0.1:18080/v1"))
    )
    ui.credential_widgets["Custom Local Server(Gemma)_model"].setText(
        str(gemma.get("model", "gemma-4-26b-a4b-it-heretic.q3_k_m.gguf"))
    )

    ui.paddleocr_vl_server_url_input.setText(
        str(ocr_client.get("server_url", "http://127.0.0.1:28118/layout-parsing"))
    )
    ui.paddleocr_vl_max_new_tokens_spinbox.setValue(int(ocr_client.get("max_new_tokens", 256)))
    ui.paddleocr_vl_parallel_workers_spinbox.setValue(int(ocr_client.get("parallel_workers", 2)))

    ui.gemma_chunk_size_spinbox.setValue(int(gemma.get("chunk_size", 4)))
    ui.gemma_max_completion_tokens_spinbox.setValue(int(gemma.get("max_completion_tokens", 512)))
    ui.gemma_request_timeout_spinbox.setValue(int(gemma.get("request_timeout_sec", 180)))
    ui.gemma_raw_response_logging_checkbox.setChecked(bool(gemma.get("raw_response_logging", False)))
    ui.raw_text_checkbox.setChecked(False)
    ui.translated_text_checkbox.setChecked(True)
    ui.inpainted_image_checkbox.setChecked(False)

    window.s_combo.setCurrentText(source_lang)
    window.t_combo.setCurrentText(target_lang)
    _apply_benchmark_font(window, target_lang)


def _load_images(window, image_paths: list[Path], source_lang: str, target_lang: str) -> list[str]:
    loaded = window.image_ctrl.load_initial_image([str(path) for path in image_paths])
    window.image_ctrl.on_initial_image_loaded(loaded)
    for image_path in window.image_files:
        state = window.image_ctrl.ensure_page_state(image_path)
        state["source_lang"] = source_lang
        state["target_lang"] = target_lang
    return list(window.image_files)


def _stage_selected_images(run_dir: Path, image_paths: list[Path]) -> list[Path]:
    corpus_dir = run_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []
    for path in image_paths:
        target_path = corpus_dir / path.name
        shutil.copy2(path, target_path)
        staged_paths.append(target_path)

    _log(
        "입력 이미지 staging 완료: corpus_dir={corpus_dir} count={count}".format(
            corpus_dir=corpus_dir,
            count=len(staged_paths),
        )
    )
    return staged_paths


def _normalize_ocr_text(text: object) -> str:
    return "".join(str(text or "").split())


def _serialize_page_snapshot_block(blk) -> dict[str, object]:
    bubble_xyxy = getattr(blk, "bubble_xyxy", None)
    text_class = getattr(blk, "text_class", None)
    return {
        "xyxy": [float(value) for value in getattr(blk, "xyxy", [])],
        "bubble_xyxy": [float(value) for value in bubble_xyxy] if bubble_xyxy is not None else None,
        "angle": float(getattr(blk, "angle", 0.0) or 0.0),
        "text_class": text_class if isinstance(text_class, (str, int, float, bool)) or text_class is None else str(text_class),
        "text": str(getattr(blk, "text", "") or ""),
        "normalized_text": _normalize_ocr_text(getattr(blk, "text", "") or ""),
    }


def _page_failed_reason(summary: dict[str, object]) -> str:
    if not isinstance(summary, dict):
        return ""
    last_failure = str(summary.get("last_failure_reason", "") or "").strip()
    if last_failure:
        return last_failure
    stage_status = summary.get("stage_status", {})
    if isinstance(stage_status, dict):
        for stage_name in ("ocr", "detect", "translation", "inpaint", "render", "save"):
            payload = stage_status.get(stage_name)
            if isinstance(payload, dict) and str(payload.get("status", "")) == "failed":
                return str(payload.get("reason", "") or "").strip()
    return ""


def _write_page_snapshots(window, run_dir: Path, loaded_paths: list[str]) -> Path:
    snapshots: list[dict[str, object]] = []
    for image_path in loaded_paths:
        state = window.image_ctrl.ensure_page_state(image_path)
        blk_list = list(state.get("blk_list") or [])
        processing_summary = state.get("processing_summary", {})
        quality = summarize_ocr_quality(blk_list)
        snapshots.append(
            {
                "image_path": str(image_path),
                "image_name": Path(str(image_path)).name,
                "image_stem": Path(str(image_path)).stem,
                "source_lang": str(state.get("source_lang", "")),
                "target_lang": str(state.get("target_lang", "")),
                "page_failed": bool(_page_failed_reason(processing_summary)),
                "page_failed_reason": _page_failed_reason(processing_summary),
                "ocr_quality": {
                    "block_count": int(quality.get("block_count", 0) or 0),
                    "non_empty": int(quality.get("non_empty", 0) or 0),
                    "empty": int(quality.get("empty", 0) or 0),
                    "single_char_like": int(quality.get("single_char_like", 0) or 0),
                },
                "blocks": [_serialize_page_snapshot_block(block) for block in blk_list],
            }
        )

    payload = {
        "generated_at": time.time(),
        "page_count": len(snapshots),
        "pages": snapshots,
    }
    output_path = run_dir / "page_snapshots.json"
    write_json(output_path, payload)
    return output_path


def _run_single_mode(
    *,
    app: QApplication,
    preset: dict[str, object],
    mode: str,
    run_dir: Path,
    source_lang: str,
    target_lang: str,
    image_paths: list[Path],
) -> dict[str, object]:
    os.environ["CT_BENCH_OUTPUT_DIR"] = str(run_dir)
    gemma_env_snapshot = _apply_gemma_env(preset.get("gemma", {}))
    try:
        _log(
            "실행 시작: mode={mode} output={run_dir} images={count} source={source} target={target}".format(
                mode=mode,
                run_dir=run_dir,
                count=len(image_paths) if mode != "one-page" else 1,
                source=source_lang,
                target=target_lang,
            )
        )
        try:
            from controller import ComicTranslate
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown-module"
            raise RuntimeError(
                "Benchmark runtime is missing a required dependency: "
                f"{missing}. Install the full app runtime before running pipeline benchmarks."
            ) from exc

        settings_backup = _settings_snapshot()
        window = ComicTranslate()
        try:
            if os.environ.get("CT_BENCH_CLEAR_APP_CACHES", "").strip() == "1":
                window.pipeline.cache_manager.clear_ocr_cache()
                window.pipeline.cache_manager.clear_translation_cache()
                _log("앱 OCR/번역 캐시 초기화 완료")
            _configure_window(window, preset, source_lang, target_lang)
            _log("앱 설정 적용 완료")
            loaded_paths = _load_images(window, image_paths, source_lang, target_lang)
            _log(f"이미지 로드 완료: {len(loaded_paths)}장")
            window.curr_img_idx = 0
            window._current_batch_run_type = "one_page_auto" if mode == "one-page" else "batch"
            window.emit_memlog(
                "benchmark_run_start",
                benchmark_mode=mode,
                total_images=len(loaded_paths),
            )

            started = time.perf_counter()
            if mode == "one-page":
                _log("one-page 벤치 실행 중...")
                window.pipeline.batch_process([loaded_paths[0]])
            elif mode == "batch":
                _log("batch 벤치 실행 중...")
                window.pipeline.batch_process(loaded_paths)
            elif mode == "webtoon":
                _log("webtoon 벤치 실행 중...")
                window.pipeline.webtoon_batch_process(loaded_paths)
            else:
                raise ValueError(f"Unsupported benchmark mode: {mode}")
            elapsed = time.perf_counter() - started
            _log(f"파이프라인 실행 완료: elapsed={elapsed:.3f}s")

            snapshot_path: Path | None = None
            if os.environ.get("CT_BENCH_EXPORT_PAGE_SNAPSHOTS", "").strip() == "1":
                snapshot_path = _write_page_snapshots(window, run_dir, loaded_paths)
                _log(f"페이지 스냅샷 저장 완료: {snapshot_path}")

            window.pipeline.release_model_caches()
            window.emit_memlog(
                "benchmark_run_finished",
                benchmark_mode=mode,
                elapsed_sec=round(elapsed, 3),
            )
            app.processEvents()
        finally:
            try:
                window._skip_close_prompt = True
                window.close()
                app.processEvents()
            finally:
                _restore_settings(settings_backup)
    finally:
        _restore_env(gemma_env_snapshot)

    metrics_path = run_dir / "metrics.jsonl"
    summary = summarize_metrics(metrics_path)
    summary.update(
        {
            "mode": mode,
            "image_count": len(image_paths) if mode != "one-page" else 1,
            "image_paths": [str(path) for path in image_paths],
        }
    )
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    _log(
        "요약 저장 완료: summary.json={summary_json} summary.md={summary_md} page_done={done} page_failed={failed}".format(
            summary_json=run_dir / "summary.json",
            summary_md=run_dir / "summary.md",
            done=summary.get("page_done_count"),
            failed=summary.get("page_failed_count"),
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offscreen pipeline benchmarks.")
    parser.add_argument("--preset", required=True, help="Preset name or preset file path")
    parser.add_argument(
        "--mode",
        default="all",
        choices=("one-page", "batch", "webtoon", "all"),
        help="Pipeline mode to benchmark",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats")
    parser.add_argument("--sample-dir", default=str(ROOT / "Sample"), help="Local Sample directory")
    parser.add_argument("--sample-count", type=int, default=30, help="Representative corpus size")
    parser.add_argument("--source-lang", default="Chinese")
    parser.add_argument("--target-lang", default="Korean")
    parser.add_argument(
        "--runtime-mode",
        default="attach-running",
        choices=("attach-running", "managed"),
        help="Use already running services or recreate staged runtime files",
    )
    parser.add_argument("--label", default="", help="Optional label suffix for run directory names")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional exact output directory for a single benchmark run",
    )
    parser.add_argument(
        "--export-page-snapshots",
        action="store_true",
        help="Export page-level detect/OCR snapshot JSON for gold/compare workflows.",
    )
    parser.add_argument(
        "--clear-app-caches",
        action="store_true",
        help="Clear app OCR/translation caches before running the pipeline.",
    )
    args = parser.parse_args()

    preset, preset_path = load_preset(args.preset)
    corpus = resolve_corpus(args.sample_dir, sample_count=args.sample_count)
    modes = ["one-page", "batch", "webtoon"] if args.mode == "all" else [args.mode]
    _log(
        "시작: preset={preset} preset_path={preset_path} mode={mode} repeat={repeat} runtime_mode={runtime_mode}".format(
            preset=preset.get("name", args.preset),
            preset_path=preset_path,
            mode=args.mode,
            repeat=args.repeat,
            runtime_mode=args.runtime_mode,
        )
    )
    _log(
        "코퍼스: sample_dir={sample_dir} sample_count={sample_count} smoke={smoke} representative={representative}".format(
            sample_dir=args.sample_dir,
            sample_count=args.sample_count,
            smoke=len(corpus["smoke"]),
            representative=len(corpus["representative"]),
        )
    )
    _log(
        "벤치 폰트 루트: {root} supported_target_langs={count}".format(
            root=BENCHMARK_FONT_ROOT,
            count=len(set(supported_source_languages + supported_target_languages)),
        )
    )

    app = QApplication.instance() or QApplication([])
    if args.export_page_snapshots:
        os.environ["CT_BENCH_EXPORT_PAGE_SNAPSHOTS"] = "1"
    if args.clear_app_caches:
        os.environ["CT_BENCH_CLEAR_APP_CACHES"] = "1"
    aggregated = []
    for repeat_index in range(1, max(1, args.repeat) + 1):
        for mode in modes:
            label = args.label or preset.get("name", args.preset)
            if args.output_dir:
                if len(modes) != 1 or max(1, args.repeat) != 1:
                    parser.error("--output-dir only supports a single mode with repeat=1")
                run_dir = Path(args.output_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
            else:
                run_dir = create_run_dir(f"{label}_{mode}_r{repeat_index}")
            run_dir.mkdir(parents=True, exist_ok=True)
            _log(f"run 디렉터리 준비: {run_dir}")

            selected_paths = corpus["smoke"] if mode == "one-page" else corpus["representative"]
            if mode == "one-page":
                selected_paths = selected_paths[:1]
            _log(f"선택 이미지 수: {len(selected_paths)}")
            staged_paths = _stage_selected_images(run_dir, selected_paths)

            write_json(
                run_dir / "benchmark_request.json",
                {
                    "preset_name": preset.get("name", args.preset),
                    "preset_path": str(preset_path),
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "runtime_mode": args.runtime_mode,
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                    "selected_paths": [str(path) for path in selected_paths],
                    "staged_paths": [str(path) for path in staged_paths],
                },
            )
            write_json(run_dir / "preset_resolved.json", preset)
            write_snapshot_json(
                run_dir / "runtime_snapshot.json",
                collect_runtime_snapshot(DEFAULT_CONTAINER_NAMES),
            )
            _log("런타임 스냅샷 저장 완료")

            if args.runtime_mode == "managed":
                _ensure_managed_runtime(run_dir, preset)
                write_snapshot_json(
                    run_dir / "docker_snapshot.json",
                    collect_runtime_snapshot(DEFAULT_CONTAINER_NAMES),
                )
                _write_container_logs(run_dir, DEFAULT_CONTAINER_NAMES)
            else:
                _log("attach-running 모드: 현재 떠 있는 Docker 서버를 그대로 사용")
                write_snapshot_json(
                    run_dir / "docker_snapshot.json",
                    collect_runtime_snapshot(DEFAULT_CONTAINER_NAMES),
                )
                _write_container_logs(run_dir, DEFAULT_CONTAINER_NAMES)

            try:
                summary = _run_single_mode(
                    app=app,
                    preset=preset,
                    mode=mode,
                    run_dir=run_dir,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    image_paths=staged_paths,
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except Exception:
                traceback.print_exc()
                return 1
            aggregated.append(
                {
                    "run_dir": str(run_dir),
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "summary": summary,
                }
            )
            _log(
                "run 완료: mode={mode} repeat={repeat} elapsed={elapsed} failed={failed}".format(
                    mode=mode,
                    repeat=repeat_index,
                    elapsed=summary.get("elapsed_sec"),
                    failed=summary.get("page_failed_count"),
                )
            )
            print(f"[done] {mode} repeat={repeat_index} -> {run_dir}")

    if aggregated:
        write_json(Path(aggregated[-1]["run_dir"]).parent / "last_benchmark_runs.json", {"runs": aggregated})
        _log(
            "전체 완료: last_benchmark_runs.json={path}".format(
                path=Path(aggregated[-1]["run_dir"]).parent / "last_benchmark_runs.json"
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
