# Japanese Optimal+ Review Decision Sheet

아이디어 착안자: 사용자

## 채점 규칙

- `O` = 이 페이지는 `MangaLMM`을 그대로 통과시켜도 허용 가능
- `X` = 이 페이지는 `PaddleOCR VL` fallback이 필요

## 권장 해석

- `keep_candidate`는 기본적으로 `O` 후보
- `review_band`는 사용자가 직접 결정
- `fallback_candidate`는 기본적으로 `X` 후보

## Decision Table

| page | detect_box_count | mangallm_bbox_2d_success | miss_count | bbox_mismatch_ratio | suggested_band | reviewer_decision | reviewer_note |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `094.png` | 20 | 15 | 5 | `0.250` | `fallback_candidate_large_gap` |  |  |
| `097.png` | 13 | 12 | 1 | `0.077` | `keep_candidate` |  |  |
| `101.png` | 6 | 4 | 2 | `0.333` | `fallback_candidate` |  |  |
| `i_099.jpg` | 17 | 14 | 3 | `0.176` | `review_band` |  |  |
| `i_100.jpg` | 16 | 14 | 2 | `0.125` | `keep_candidate` |  |  |
| `i_102.jpg` | 19 | 15 | 4 | `0.211` | `review_band` |  |  |
| `i_105.jpg` | 10 | 6 | 4 | `0.400` | `fallback_candidate` |  |  |
| `p_016.jpg` | 30 | 5 | 25 | `0.833` | `fallback_candidate_hard_page` |  |  |
| `p_017.jpg` | 15 | 15 | 0 | `0.000` | `keep_candidate` |  |  |
| `p_018.jpg` | 18 | 15 | 3 | `0.167` | `review_band` |  |  |
| `p_019.jpg` | 9 | 9 | 0 | `0.000` | `keep_candidate` |  |  |
| `p_020.jpg` | 30 | 26 | 4 | `0.133` | `keep_candidate` |  |  |
| `p_021.jpg` | 9 | 8 | 1 | `0.111` | `keep_candidate` |  |  |
