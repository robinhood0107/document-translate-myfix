# Japan vLLM Parallel Subset

Curated representative OCR benchmark subset copied from `Sample/japan` for local `PaddleOCR VL + vLLM` parallel tuning.

## Selection goals

- Keep `p_016.jpg` as a mandatory hard anchor.
- Cover both request-count stress and single-crop VRAM stress.
- Preserve all three source families:
  - `094-101.png`
  - `i_099-i_105.jpg`
  - `p_015-p_021.jpg`
- Include mixed pages:
  - dense pages with many blocks
  - sparse pages with few but large crops
  - bubble-heavy pages
  - free-text-heavy pages
  - high-resolution `p_` pages that matter most for local vLLM scheduling

## Coverage

- Selected pages: `13 / 22`
- Selected detected blocks: `212 / 305`
- Detected block coverage: `69.5%`
- Raw profiling data: `banchmark_result_log/japan_sample_curation/2026-04-14_profile.json`

## Why these files

- `094.png`: highest block count in the PNG family and the heaviest total crop coverage.
- `097.png`: lowest bubble ratio in the PNG family, useful for mixed free-text pages.
- `101.png`: sparse PNG page with very large crops.
- `i_099.jpg`: dense `i_` page with large crops and high total crop coverage.
- `i_100.jpg`: fully bubble-based dense page.
- `i_102.jpg`: highest block count in the `i_` family with many tiny crops.
- `i_105.jpg`: lighter `i_` page with relatively large maximum crop size.
- `p_016.jpg`: mandatory hard page, highest difficulty, free-text-heavy, and one of the two highest block-count pages.
- `p_017.jpg`: mixed medium-density page with large crops.
- `p_018.jpg`: bubble-heavy dense page with smaller crops.
- `p_019.jpg`: light high-resolution page with small crops, useful for upper-bound speed checks.
- `p_020.jpg`: high-density high-resolution page with many mixed blocks.
- `p_021.jpg`: sparse high-resolution page with the largest single crop in the whole 22-page set.

## Detector profile summary

`crop_area_ratio_*` is normalized by full page area and uses the OCR crop bbox source that the current PaddleOCR VL path would send:
- `bubble_xyxy` when available
- otherwise `xyxy`

| File | Family | Blocks | Bubble Ratio | Max Crop | P90 Crop | Total Crop Sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 094.png | png_094_101 | 20 | 0.650 | 0.0542 | 0.0265 | 0.3489 |
| 097.png | png_094_101 | 13 | 0.538 | 0.0312 | 0.0263 | 0.2200 |
| 101.png | png_094_101 | 6 | 0.667 | 0.0473 | 0.0394 | 0.1510 |
| i_099.jpg | i_jpg | 17 | 0.824 | 0.0271 | 0.0231 | 0.1750 |
| i_100.jpg | i_jpg | 16 | 1.000 | 0.0273 | 0.0215 | 0.1401 |
| i_102.jpg | i_jpg | 19 | 0.579 | 0.0134 | 0.0094 | 0.0903 |
| i_105.jpg | i_jpg | 10 | 0.700 | 0.0240 | 0.0148 | 0.0846 |
| p_016.jpg | p_jpg | 30 | 0.300 | 0.0274 | 0.0189 | 0.2180 |
| p_017.jpg | p_jpg | 15 | 0.533 | 0.0239 | 0.0217 | 0.1701 |
| p_018.jpg | p_jpg | 18 | 0.833 | 0.0164 | 0.0139 | 0.0860 |
| p_019.jpg | p_jpg | 9 | 0.889 | 0.0135 | 0.0131 | 0.0445 |
| p_020.jpg | p_jpg | 30 | 0.600 | 0.0163 | 0.0116 | 0.1402 |
| p_021.jpg | p_jpg | 9 | 0.889 | 0.0512 | 0.0263 | 0.1228 |
