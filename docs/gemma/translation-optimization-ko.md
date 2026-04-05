# Gemma 번역 튜닝 요약

이 문서는 현재 활성 Gemma translation-only 설정만 빠르게 보는 용도입니다.

## 현재 활성 설정

- `temperature=0.6`
- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`
- `n_gpu_layers=23`
- `threads=12`
- `ctx=4096`
- `chunk_size=4`
- `max_completion_tokens=512`
- `paddleocr-server=cpu`

## 왜 이 조합을 쓰는가

- `temperature=0.6`은 representative batch에서 속도와 안정성 균형이 가장 좋았습니다.
- `n_gpu_layers=23`은 `20~24` sweep 중 translate 단계 개선폭이 가장 좋았습니다.
- `paddleocr-server=cpu`는 OCR front GPU 점유를 줄여 전체 파이프라인에 유리했습니다.

## 더 자세한 설명

- 전체 실험 과정: [optimization-journey-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/optimization-journey-ko.md)
- 최신 차트/표: [report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/report-ko.md)
- 설정 이력: [profiles-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/profiles-ko.md)
