# Gemma sampler quality v2 workflow

1. 승인된 478개 frozen reference와 r6 `9,560` 응답을 read-only로 검증한다.
2. 하나의 CUDA13 campaign에서 아래 고정 순서를 끝까지 실행한다.
   - 온도 10개 `top-p=1`, `top-k=0`, `min-p=0`
   - 나머지 joint top-p/top-k
   - 나머지 min-p
3. 실행 중에는 완료 인덱스의 원자 snapshot만 읽어 새 unique 번역을 작은 blind 묶음으로
   판정한다. live campaign과 raw 응답은 수정하지 않는다.
4. Router unload·container stop·GPU 반환이 모두 성공하면
   `WAITING_FOR_FINAL_JUDGMENT`에서 자동 종료한다.
5. 누적 ledger를 최종 133,840응답에 다시 결합하고, 남은 새 unique 출력까지 판정하여
   `UNJUDGED=0`을 확인한 뒤 140개 arm을 478개 전체로 순위화한다.
6. 사용자가 최종 후보를 명시 승인한 뒤에만 제품 sampler PR을 만든다.

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

## 분할 판정 ledger 계약

분할 판정은 `faithful-translation-quality-v1` 규칙을 사용한다. 정답과 글자가 같은지는
필수 조건이 아니다. 의미·인물·행동·관계·숫자·부정·질문/선언·정체성·동의/강제·검열/약화와
자연스러움을 판정하고, 의미가 보존된 어투·어순·존댓말·의성어·동의어 차이는 통과시킨다.
정상 번역을 추출할 수 있는 wrapper·channel frame·thought·finish reason은 진단값으로만 남긴다.
실제 번역문에 `나Please세` 같은 혼합 token 손상이 남거나 번역이 비면 catastrophic이다.

ledger key는 sampler 정보가 아니라 `case_id + translation_sha256`으로 고정한다. 그래서 이미
판정한 동일 번역은 이후 arm과 seed에서 재사용할 수 있다. blind packet에는 sampler, arm,
seed, logical slot, 실행 순서를 넣지 않는다. 실행 중 snapshot은 완전히 append된 completion
index만 읽으며, 아직 index에 들어오지 않은 in-flight case를 복구하거나 campaign 파일을
다시 쓰지 않는다. 최종 순위는 campaign 종료 및 cleanup 증명 전에는 만들지 않는다.

재사용한 과거 판정이 현재 semantic rule과 충돌하면 verdict를 직접 덮어쓰지 않는다.
`amend-incremental-judgment`가 기존 verdict hash를 선행 조건으로 확인하고, cluster별 이전값·
새 값·변경 사유·적용 시각을 ledger hash 안에 append-only 감사 기록으로 남긴 뒤에만 정정한다.
validator는 같은 cluster의 연속 정정에서 이전 `after`와 다음 `before`, 마지막 `after`와 현재
verdict가 정확히 이어지는지도 확인한다.
자동 transport verdict, 존재하지 않는 cluster, stale 이전 hash, pending blind batch가 있는
상태의 정정은 fail-closed한다.

## 최종 분석 계약

`analyze-final-campaign`은 campaign이 `WAITING_FOR_FINAL_JUDGMENT`에 도달하고 managed run이
정상 완료된 뒤에만 실행된다. r6 `9,560`개와 신규 `124,280`개를 합쳐, 140개 tuple마다 두 seed와
478개 case가 정확히 한 번씩 있는지 검사한다. reference·plan·runtime·request identity가 하나라도
다르거나 판정이 남아 있으면 보고서를 만들지 않는다.

최종 private 보고서는 오류 번역과 gate 근거를 보존한다. public 집계에는 140개 전체 순위,
온도 10개, joint 120개, min-p 비교 30개, seed별 수치와 오류율만 넣는다. 필수 이름·나이 gate는
별도 private manifest의 문자열 계약으로 검사한다. 보고서 상태는 사용자 승인 대기이며,
분석 명령 자체는 제품 기본값을 변경하지 않는다.
