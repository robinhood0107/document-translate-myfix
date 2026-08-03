# Gemma sampler quality v2 workflow

1. 승인된 478개 frozen reference와 r6 `9,560` 응답을 read-only로 검증한다.
2. 하나의 CUDA13 campaign에서 아래 고정 순서를 끝까지 실행한다.
   - 온도 10개 `top-p=1`, `top-k=0`, `min-p=0`
   - 나머지 joint top-p/top-k
   - 나머지 min-p
3. Router unload·container stop·GPU 반환이 모두 성공하면
   `WAITING_FOR_FINAL_JUDGMENT`에서 자동 종료한다.
4. 그 뒤 478개 전체를 blind semantic judgment로 판정한다.
5. 사용자가 최종 후보를 명시 승인한 뒤에만 제품 sampler PR을 만든다.

고정 matrix는 r6 재실행 없이 새 응답 `124,280`개를 만든다.

| 구간 | 새 응답 | 재사용 |
| --- | ---: | ---: |
| r6 temperature | 0 | 9,560 |
| joint top-p/top-k | 105,160 | 9,560 중 기본 row |
| min-p | 19,120 | joint의 min-p=0 row |
| 전체 증거 | 124,280 | 9,560 |

중간 결과로 matrix를 바꾸거나, temperature·top-p·top-k·min-p를 자동 선택하지 않는다.
transient HTTP/Python worker 종료만 같은 logical slot으로 자동 resume하며, first-valid
응답 하나만 통계에 포함한다. contract mismatch, foreign runtime, slot drain 실패, VRAM
반환 실패, 5 GiB 미만 여유 공간은 fail-closed다.

응답 판정은 **번역문 품질**이 기준이다. canonical과 글자가 다르다는 이유만으로 실패시키지
않고, 의미·인물·행동·관계·숫자·부정·자연스러움과 민감 표현 보존을 본다. 어투·어순·의성어·동의어
차이는 허용한다. wrapper, channel frame, thought 본문, finish reason은 번역문을 정상 추출할
수 있으면 진단값일 뿐 순위 오류가 아니다. 실제 번역문이 비었거나 혼합 token 손상이 남은 경우는
catastrophic이다.

완료된 raw envelope는 수정하지 않는다. 새 판정기는 raw envelope에서 현재 품질 view를
메모리에서 다시 만들기 때문에, 기존 run은 GPU 재실행 없이 재판정·resume·재사용할 수 있다.
