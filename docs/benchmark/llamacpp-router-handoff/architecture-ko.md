# Single llama.cpp Router handoff architecture

각 pair는 공개 OCR 포트와 Gemma 포트를 같은 router의 `8080`으로 연결한다. 이는 포트 호환성 계약이며, OS process 수를 주장하는 계약은 아니다. 계약은 컨테이너 하나, 공개 서버 하나, loaded model 최대 하나다.

router는 Docker built-in `bridge`를 사용한다. one-container/loopback topology에는 별도 Compose network가 필요 없으며, arm마다 project network를 쌓지 않는다.

각 router container에는 protocol·pair·arm 소유 token label을 함께 붙인다. runner와 adapter는 세 label이 현재 arm과 일치할 때만 stop/remove를 허용하므로, 이름이 비슷한 다른 실행의 container를 정리하지 않는다.

```text
Arbiter lease
  → Router command gate: load / status / unload
  → OCR or Gemma HTTP inference outside command gate
  → product Arbiter가 driver-global GPU 반환을 별도로 증명
```

`LocalLlamaRouterCoordinator`의 제품 구현은 lab PASS와 사용자 승인 뒤에만 시작한다. 이 lab은 offscreen pipeline에서 기존 runtime manager를 일시적으로 adapter로 바꿀 뿐이며, product controller나 Compose를 변경하지 않는다.

Paddle crop/Spotting과 MangaLMM의 요청 형식·prompt·model alias·image policy·parser·retry·worker/scheduling·현재 설정은 router lab이 바꾸지 않는다. router가 바꾸는 것은 explicit model unload/load와 그 전후의 lease 소유권뿐이다. 특히 crop의 `parallel_workers=8`, Spotting의 completion `3000`/timeout `360`, MangaLMM의 full-page completion `4096`/worker `1`/resize budget은 원래 계약을 유지한다.

HunyuanOCR은 canonical named volume과 pinned Q8 identity가 먼저 검증되어야 한다. baseline bind와 router volume의 실제 SHA-256을 각각 확인하며, 조건이 없으면 해당 pair는 `NOT ELIGIBLE`이다.

검증용 OCR raw result는 실행마다 달라지는 `TextBlock` UUID를 비교하지 않는다. detection의 exact 순서/geometry에 묶은 block ordinal과 raw result를 private archive에 기록해 block mapping을 비교한다. pipeline 종료 뒤의 OCR 재-load probe는 active/release-failed lease가 없음을 먼저 확인한 뒤 새 Arbiter generation에서만 실행한다.
