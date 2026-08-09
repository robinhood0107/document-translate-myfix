#!/usr/bin/env python3
"""렌더 워커를 늘려도 안전한지, 그리고 실제로 빨라지는지 확인한다.

렌더 융합의 근거는 "렌더 약 300ms 가 인페인팅 1.28s 뒤에 숨는다" 였다. 실측
366장 실행에서 렌더는 페이지당 **2.31초**였고 워커가 하나뿐이라 14.1분의 직렬
작업이 됐다. 그래서 워커를 늘리는 것이 후보가 됐다.

그런데 이 변경은 **조용히 실패할 수 있다.** 렌더가 이벤트 디스패처 없는 스레드로
가면 ``scene.render()`` 가 예외도 경고도 없이 빈 이미지를 만든다
(``pipeline/render_pool.py`` 참고). 그래서 속도만 재서는 안 되고, 결과 픽셀이
워커 수와 무관하게 같은지를 함께 봐야 한다.

이 도구는 워커 수를 바꿔가며 같은 렌더 작업을 돌리고, 매 결과의 픽셀 합을
비교한다. 하나라도 어긋나면 그 워커 수는 쓸 수 없다.

    python scripts/verify_render_worker_scaling.py --pages 24 --workers 1 2 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.canvas.save_renderer import ImageSaveRenderer  # noqa: E402
from pipeline.render_pool import QtRenderPool  # noqa: E402


def _page(height: int = 1430, width: int = 2000) -> np.ndarray:
    """결정적인 합성 페이지. 균일한 색이면 빈 이미지와 구분되지 않는다."""

    rng = np.random.default_rng(20260809)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _render_once(image: np.ndarray) -> int:
    """한 페이지를 렌더하고 결과 픽셀 합을 돌려준다.

    빈 이미지 실패는 합이 0 이거나 크게 달라지는 것으로 드러난다.
    """

    renderer = ImageSaveRenderer(image)
    rendered = renderer.render_to_image()
    return int(np.asarray(rendered).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=24)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    image = _page()

    baseline = _render_once(image)
    print(f"main-thread reference pixel sum: {baseline}")
    if baseline == 0:
        print("FAIL: the reference render is already blank; fix that first.")
        return 1

    results: list[tuple[int, float, bool]] = []
    for workers in args.workers:
        pool = QtRenderPool(max_workers=workers)
        started = time.monotonic()
        futures = [pool.submit(_render_once, image) for _ in range(args.pages)]
        sums = [future.result() for future in futures]
        elapsed = time.monotonic() - started
        pool.shutdown()

        identical = all(value == baseline for value in sums)
        blanks = sum(1 for value in sums if value == 0)
        results.append((workers, elapsed, identical))
        status = "OK  " if identical else "FAIL"
        print(
            f"{status} workers={workers:<2} {elapsed:7.2f}s "
            f"{elapsed / args.pages:6.3f}s/page  blank={blanks}  "
            f"identical={identical}"
        )

    safe = [row for row in results if row[2]]
    if len(safe) != len(results):
        print(
            "\nAt least one worker count produced different pixels. "
            "Do not raise CT_RENDER_WORKERS."
        )
        return 1

    best = min(safe, key=lambda row: row[1])
    slowest = max(safe, key=lambda row: row[1])
    print(
        f"\nAll worker counts rendered identical pixels. "
        f"Fastest: {best[0]} worker(s) at {best[1]:.2f}s "
        f"({slowest[1] / best[1]:.2f}x over {slowest[0]} worker(s))."
    )
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
