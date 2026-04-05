# Benchmark Fonts

이 폴더는 벤치마크 전용 렌더링 폰트 폴더입니다.

- 경로 규칙: `benchmarks-fonts/<언어 이름>/`
- 지원 확장자: `.ttf`, `.ttc`, `.otf`, `.woff`, `.woff2`
- 벤치 실행 시 target language 폴더에서 사전순 첫 번째 폰트 파일을 자동 적용합니다.
- 해당 언어 폴더가 비어 있으면 현재 앱 폰트 설정을 유지합니다.

예시:

- `benchmarks-fonts/Korean/`
- `benchmarks-fonts/Japanese/`
- `benchmarks-fonts/English/`

추가 fallback:

- `Simplified Chinese` -> `Chinese`
- `Traditional Chinese` -> `Chinese`
- `Brazilian Portuguese` -> `Portuguese`
