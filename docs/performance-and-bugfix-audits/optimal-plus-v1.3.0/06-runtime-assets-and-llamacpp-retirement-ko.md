# Runtime 자산, llama.cpp 전용 전환, 메모리 운영 판정

이 문서는 관리형 로컬 runtime의 현재 경계와, 속도·안정성 실험 뒤 유지한 자산 운영 원칙을 기록한다. 핵심은 모델 파일의 **저장 위치**, OS RAM page cache, GPU VRAM residency를 같은 것으로 취급하지 않는 것이다.

## 최종 runtime 경계

관리형 로컬 추론은 모두 llama.cpp를 사용한다.

| 기능 | 관리형 backend | 상태 |
|---|---|---|
| Gemma 번역 | llama.cpp | 제품 기본 |
| Paddle crop OCR | llama.cpp direct `OCR:` | 제품 기본 |
| Paddle full-page Spotting | llama.cpp | Experimental |
| MangaLMM full-page | llama.cpp | Experimental |
| HunyuanOCR | llama.cpp | 관리형 선택지 |

사용자가 직접 지정한 외부 endpoint는 관리형 runtime이 아니다. 앱은 그 endpoint의 backend를 추정하거나 Docker container를 시작하지 않는다.

과거 vLLM 설정·container·이미지는 제품 실행 경로에서 퇴역했다. 호환 migration은 과거 설정을 llama.cpp로 한 번 전환하고, 과거 환경변수는 warning만 남긴 뒤 관리형 Compose에는 전달하지 않는다. 퇴역 확인과 정리에는 소유권을 확인한 manifest를 사용하며, 광범위 prune은 사용하지 않는다.

## 모델 자산과 runtime fingerprint

관리형 모델은 versioned external named volume에 보관한다.

1. 준비 단계에서 source 파일을 `.partial` 이름으로 복사한다.
2. 원본과 복사본의 SHA-256·크기를 확인한다.
3. 같은 volume 안에서 원자적으로 최종 이름으로 바꾼다.
4. GPU smoke가 성공한 뒤에만 ready manifest를 마지막으로 기록한다.
5. 서비스는 volume을 read-only로 mount한다.

정상 실행 시에는 대형 파일 전체를 매번 다시 해시하지 않는다. ready manifest, 파일 크기, 모델 SHA, image digest, compose command hash, volume, runtime 설정을 합친 fingerprint로 재사용 가능 여부를 판단한다. fingerprint가 같으면 stopped container를 `start`로 재사용하고, 다르면 대상 container만 `--force-recreate` 한다.

이 방식은 host bind mount보다 일관된 준비 상태를 제공하지만, 모델을 자동으로 RAM이나 VRAM에 고정하지는 않는다.

## 정상 종료와 GPU handoff

정상 stage 전환과 앱 종료에서는 `docker stop`을 사용한다. 이는 volume과 OS file cache를 보존하면서 container process와 GPU VRAM을 돌려준다.

`down`, volume 삭제, model 재준비, cache drop, 광범위 image/volume prune은 명시적인 초기화·삭제 작업에서만 사용한다. 평상시 성능 경로에 넣지 않는다.

GPU는 다음 순서를 유지한다.

```text
OCR runtime load → 폴더 OCR 완료 → stop·VRAM 반환 확인
→ inpainter → release·VRAM 반환 확인 → Gemma → CPU render
```

모든 모델을 계속 VRAM에 상주시켜 얻는 이득보다 shared GPU memory·WSL swap·OOM 위험이 컸다. 따라서 stage-aware 1회 load와 안전한 release가 기본이다.

## 자연 OS page cache는 유지한다

Gemma GGUF를 named volume에서 읽은 뒤 즉시 정상 재시작한 관측에서는 healthy 시간이 `66.811초 → 16.865초`로 줄었다. 이는 약 74.757% 단축이며, 모델을 GPU에 계속 올려 둔 결과가 아니라 운영체제가 파일 읽기를 RAM page cache에서 재사용한 결과다.

따라서 다음은 유지한다.

- versioned named volume 보관
- 정상 `stop`
- WSL 종료와 cache 강제 비우기 금지
- translation all-hit이면 Gemma runtime 자체를 시작하지 않는 fast path

반대로 인페인트 중 대형 GGUF를 명시적으로 모두 읽어 두는 read-ahead는 제품에 넣지 않았다. 첫 deep-cold 관측에는 이득이 있었지만 반복 AB/BA에서 단측 95% bootstrap 하한이 음수였고, 실행 시작값 대비 WSL swap 증가 0 조건도 지키지 못했다. 자연 cache 이득을 보존하는 것이 더 안정적이다.

## WSL 메모리 프로필

현재 기준 프로필은 아래와 같다.

```ini
[wsl2]
memory=20GB
swap=8GB
maxCrashDumpCount=-1

[experimental]
autoMemoryReclaim=gradual
```

`20GB`는 WSL이 항상 선점하는 양이 아니라 최대 상한이다. 목표는 대형 GGUF의 page cache와 Docker·llama.cpp overhead를 감당하면서 Windows에 충분한 여유를 남기는 것이다. `8GB` swap은 비상 여유일 뿐 정상 벤치마크에서 사용할 성능 자원이 아니다.

운영 검증은 매 실행의 시작값을 기준으로 다음을 확인한다.

- Windows available memory 최저 6GB 이상
- WSL swap 증가량 0
- OOM 및 비정상 shared GPU memory 증가 0
- warm-start page-cache 이득 유지
- S1/S6 전체 pipeline 비회귀

Windows 여유가 부족하면 18GB로 후퇴한다. 22GB 이상은 20GB에서도 page cache가 반복 축출되고 Windows 여유가 충분하다는 실측이 있을 때만 다시 검토한다.

## 관련 문서

- [관리형 로컬 추론 llama.cpp 전용 정책](../../runtime/managed-llamacpp-only-ko.md)
- [공통 진실 명세](00-truth-specification-ko.md)
- [Gemma 번역·모델 판정](03-gemma-translation-and-model-decision-ko.md)
- [캐시·checkpoint·cold path 판정](04-cache-checkpoint-and-cold-path-decision-ko.md)
- [릴리스 검증과 브랜치 정리](07-release-verification-and-branch-closure-ko.md)
