# Gemma sampler stability workflow

1. 고정된 22개 text corpus의 canonical 번역, 필수 의미, 금지 변화, 허용 말투를 실행 전에 hash로 잠근다.
2. Crop Router pair를 준비하고 OCR 모델을 unload한 뒤 Gemma만 load한다. OCR과 render는 실행하지 않는다.
3. temperature sweep을 세 seed와 세 seed별 순서 변형으로 실행한다.
4. clean 후보가 없더라도 계획된 top-p와 top-k 후속 단계를 수행한다. 각 단계에서 이미 존재하는 sampler identity의 response를 재사용한다.
5. 응답을 원자 파일로 저장하고 resume 시 request contract와 response id를 재검증한다.
6. 구조 gate와 자동 hard gate를 적용한 뒤 prompt별 exact cluster를 만든다.
7. unique output만 고정 reference와 인접 문맥으로 직접 판정한다. 외부 GPT API와 이미지 검수는 사용하지 않는다.
8. clean 후보 중 semantic fail 0, review 최소, seed 안정성, 자연스러움, latency/token 순서로 승자를 정한다.
9. Gemma unload, VRAM 반환 상태, Router container stop을 증명한다. 실패하면 다음 sampler를 실행하지 않고 원인을 기록한다.
10. runner·테스트·sanitized report만 `benchmarking/lab` PR에 넣는다. raw archive는 PR에 넣지 않는다.

제품 승격은 lab merge 이후 별도 `develop` 기반 branch에서 한다. 승인된 sampler tuple만 전역 기본값 migration으로 적용하며, lab adapter·request monkeypatch·ranking 로직은 제품에 들어가지 않는다.

## reasoning 경계

번역 제품은 reasoning을 사용하지 않는다. `reasoning=off`와 reasoning 금지 prompt/schema를 유지하며, `<|channel>thought` 같은 protocol token은 출력 누출로 보고 parser에서 제거한다. 문맥 입력은 번역된 문장의 의미를 보존하기 위한 단일 인접 문맥이며, 원문에 없는 사실을 보충하는 추론 허가가 아니다.
