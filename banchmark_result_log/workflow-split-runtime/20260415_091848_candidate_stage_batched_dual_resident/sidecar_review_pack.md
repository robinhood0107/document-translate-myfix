# Stage-Batched Optimal+ Japanese Sidecar Review Pack

아이디어 착안자: 사용자

## OCR Stage Policy

- primary_ocr_engine: `PaddleOCR VL`
- resident_ocr_engines: `PaddleOCR VL, MangaLMM`
- requires_sidecar_collection: `True`

## Page Comparison

| page | detect_box_count | primary_non_empty | primary_empty | sidecar_engine | sidecar_non_empty | sidecar_empty | bbox_2d_success_block_count | hard_page |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 094.png | 20 | 0 | 0 | MangaLMM | 15 | 5 | 15 | False |
| 097.png | 13 | 0 | 0 | MangaLMM | 12 | 1 | 12 | False |
| 101.png | 6 | 0 | 0 | MangaLMM | 4 | 2 | 4 | False |
| i_099.jpg | 17 | 0 | 0 | MangaLMM | 14 | 3 | 14 | False |
| i_100.jpg | 16 | 0 | 0 | MangaLMM | 14 | 2 | 14 | False |
| i_102.jpg | 19 | 0 | 0 | MangaLMM | 15 | 4 | 15 | False |
| i_105.jpg | 10 | 0 | 0 | MangaLMM | 6 | 4 | 6 | False |
| p_016.jpg | 30 | 0 | 0 | MangaLMM | 5 | 25 | 5 | True |
| p_017.jpg | 15 | 0 | 0 | MangaLMM | 15 | 0 | 15 | False |
| p_018.jpg | 18 | 0 | 0 | MangaLMM | 15 | 3 | 15 | False |
| p_019.jpg | 9 | 0 | 0 | MangaLMM | 9 | 0 | 9 | False |
| p_020.jpg | 30 | 0 | 0 | MangaLMM | 26 | 4 | 26 | False |
| p_021.jpg | 9 | 0 | 0 | MangaLMM | 8 | 1 | 8 | False |
