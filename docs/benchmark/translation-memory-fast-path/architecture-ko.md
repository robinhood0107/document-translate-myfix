# Translation Memory Fast Path 벤치마크 구조

## 구성

```text
multilingual summary (54 blocks)
        |
        v
benchmark_translation_memory_fast_path.py
        |
        +--> CustomLocalGemmaTranslation (real grouped product path)
        |       |
        |       +--> TranslationMemoryStore (scenario-local SQLite)
        |       +--> LocalGemmaRuntimeManager (prepared runtime identity)
        |       `--> llama.cpp OpenAI-compatible endpoint
        |
        +--> public summary (hashes, counts, timings, telemetry)
        `--> private comparison artifact (raw text, ignored locally)
```

## Cache 계획

각 블록의 result-cache key는 전체 정렬 문맥, target index/key, 언어, extra context, group mode/size, prompt/profile/schema, sampler, 모델 SHA-256, runtime fingerprint, sanitizer·guard·TM revision을 포함하는 제품 코드에서 생성됩니다.

mixed 시나리오는 cold 기준선의 짝수 index 결과 27개를 제품 cache plan의 실제 key로 seed합니다. 이후 54블록을 요청하면 seed된 27개는 SQLite hit로 복원하고 홀수 index 27개만 `requested_blocks`로 Gemma에 전송합니다. seed 과정은 성능 측정에서 제외하며, 실제 조회·부분 요청·결과 병합·저장은 제품 경로를 그대로 사용합니다.

## Exact TM 경계

첫 cold 실행의 번역은 승인되지 않은 후보로만 저장됩니다. 러너가 entry id를 명시적으로 승인한 뒤 result cache를 비우고 다시 요청해야 Exact TM 경로가 활성화됩니다. 승인 revision 변경은 이전 result-cache identity를 stale로 만듭니다.

## Runtime 경계

`prepare_translation()`이 miss를 발견한 경우에만 ensure callback이 실행됩니다. all-hit에서는 callback 자체가 0회여야 하며, 중지된 컨테이너 상태도 그대로여야 합니다.

prefix 행렬은 제품 prompt와 decoder를 유지하면서 요청별 `cache_prompt`만 주입하고 runtime contract의 `LLAMA_CACHE_RAM_MIB`만 0 또는 256으로 바꿉니다. 각 후보 뒤에는 `stop`하고 fingerprint 불일치 시 정상 `--force-recreate` 경로를 사용합니다.

## 개인정보와 재현성

공개 summary에는 원문·번역·입력 경로가 없습니다. 재현성은 입력 SHA-256, commit, 모델 SHA-256, image ID/digest, runtime fingerprint/options, group size, completion cap으로 확보합니다. 의미 검수 자료는 로컬 raw 결과로만 유지합니다.
