#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_inpaint_debug import _SettingsStub, _iter_sample_images, _process_image
from modules.detection.processor import TextBlockDetector
from modules.utils.pipeline_config import get_inpainter_runtime, inpaint_map
from modules.utils.device import resolve_device


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    inpainter: str
    mask_refiner: str
    keep_existing_lines: bool


def _write_index(root_output: Path, cases: list[BenchmarkCase], summary: dict) -> None:
    lines = [
        '# Inpaint Matrix Benchmark',
        '',
        f"Generated: `{summary['generated_at']}`",
        '',
        '## Cases',
        '',
        '| Case | Inpainter | Mask Refiner | Keep Existing Lines | Output |',
        '| --- | --- | --- | --- | --- |',
    ]
    for case in cases:
        out_dir = f"{case.name}"
        keep_lines = 'yes' if case.keep_existing_lines else 'no'
        lines.append(f"| `{case.name}` | `{case.inpainter}` | `{case.mask_refiner}` | {keep_lines} | [{out_dir}]({out_dir}/index.md) |")
    lines.extend(['', '## Summary', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', ''])
    (root_output / 'index.md').write_text('\n'.join(lines), encoding='utf-8')


def _make_inpainter(settings: _SettingsStub):
    runtime = get_inpainter_runtime(settings, settings.get_tool_selection('inpainter'))
    cls = inpaint_map[runtime['key']]
    device = resolve_device(settings.is_gpu_enabled(), backend=runtime['backend'])
    return cls(
        device,
        backend=runtime['backend'],
        runtime_device=runtime.get('device', device),
        inpaint_size=runtime.get('inpaint_size'),
        precision=runtime.get('precision'),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Run 5-way inpaint benchmark matrix on Sample corpora.')
    parser.add_argument('--glob', default='*', help='Glob pattern for sample filenames.')
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    cases = [
        BenchmarkCase('legacy_bbox_aot', 'AOT', 'legacy_bbox', False),
        BenchmarkCase('ctd_aot', 'AOT', 'ctd', False),
        BenchmarkCase('ctd_protect_aot', 'AOT', 'ctd', True),
        BenchmarkCase('ctd_protect_lama_large_512px', 'lama_large_512px', 'ctd', True),
        BenchmarkCase('ctd_protect_lama_mpe', 'lama_mpe', 'ctd', True),
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root_output = ROOT / 'banchmark_result_log' / 'inpaint_matrix' / f'{timestamp}_sample-matrix'
    root_output.mkdir(parents=True, exist_ok=True)

    overall = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'glob': args.glob,
        'use_gpu': bool(args.use_gpu),
        'cases': {},
    }

    for case in cases:
        settings = _SettingsStub(
            inpainter=case.inpainter,
            use_gpu=args.use_gpu,
            mask_refiner=case.mask_refiner,
            keep_existing_lines=case.keep_existing_lines,
        )
        detector = TextBlockDetector(settings)
        inpainter = _make_inpainter(settings)
        case_root = root_output / case.name
        case_root.mkdir(parents=True, exist_ok=True)
        records_by_corpus: dict[str, list[dict]] = {}
        failures: list[dict] = []
        total_images = 0

        for corpus_name in ('japan', 'China'):
            corpus_dir = ROOT / 'Sample' / corpus_name
            corpus_output = case_root / corpus_name.lower()
            corpus_output.mkdir(parents=True, exist_ok=True)
            records: list[dict] = []
            for image_path in _iter_sample_images(corpus_dir, args.glob):
                total_images += 1
                try:
                    records.append(_process_image(image_path, corpus_output, detector, inpainter, settings))
                except Exception as exc:
                    failures.append({
                        'corpus': corpus_name,
                        'image': image_path.name,
                        'error': str(exc),
                        'traceback': traceback.format_exc(),
                    })
            records_by_corpus[corpus_name.lower()] = records

        summary = {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'case': case.name,
            'detector_key': 'RT-DETR-v2',
            'inpainter': case.inpainter,
            'hd_strategy': 'Resize',
            'mask_refiner': case.mask_refiner,
            'keep_existing_lines': case.keep_existing_lines,
            'use_gpu': bool(args.use_gpu),
            'glob': args.glob,
            'total_images': total_images,
            'success_count': sum(len(v) for v in records_by_corpus.values()),
            'failure_count': len(failures),
            'failures': failures,
            'corpora': {
                corpus: {
                    'image_count': len(records),
                    'cleanup_applied_count': sum(1 for r in records if r['cleanup_applied']),
                    'total_blocks': sum(r['block_count'] for r in records),
                }
                for corpus, records in records_by_corpus.items()
            },
        }
        metrics_dir = case_root / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

        from scripts.export_inpaint_debug import _write_index as write_debug_index
        write_debug_index(case_root, records_by_corpus, summary)
        overall['cases'][case.name] = summary

    _write_index(root_output, cases, overall)
    print(root_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
