# Gemma sampler quality v2 workflow

1. private snapshot manifest에서 reference draft를 만든다.
2. canonical 1차 작성, 순서·기존 번역·1차 답을 숨긴 blind 2차 검수, 불일치·저신뢰·OCR 손상 해소를 순서대로 수행한다.
3. non-flagged 24개 층화 표본을 사용자가 PASS한 뒤에만 reference를 freeze한다.
4. temperature를 실행하고 tuning 판정을 끝낸 뒤 상위 두 temperature로 joint top-p/top-k를 실행한다.
5. tuning 상위 세 tuple로 min-p를 실행한다. 기존 row는 재실행하지 않고 first-valid response를 재사용한다.
6. arm/seed를 숨긴 exact-output cluster 판정으로 provisional winner를 정한다.
7. 그 tuple과 기존 기본 tuple의 holdout만 개봉한다. 자동 제품 적용은 하지 않으며, 사용자 명시 승인 없이는 제품 PR을 만들지 않는다.

각 phase는 `WAITING_FOR_JUDGMENT`에서 정상 종료한다. BAT는 transient worker failure만 backoff 후 같은 managed run으로 resume한다. Windows progress checkpoint의 동일 경로 임시 파일 교체가 잠시 거부된 경우만 원래 failed manifest를 감사 기록으로 보존한 뒤 복구할 수 있으며, contract mismatch, foreign runtime, slot drain 실패, VRAM 반환 실패, 5 GiB 미만 여유 공간과 그 밖의 failed manifest는 fail-closed다.

응답 판정은 **번역문 품질만** 기준으로 한다. raw envelope에서 `translation` 값을 찾은 뒤
channel frame·대소문자 변형·그 안의 thought 본문을 제거하고 canonical과 비교한다. choice
개수·finish reason·JSON 여분 key·앞뒤 wrapper·후행 텍스트는 private 진단값으로만 남기며
순위 불이익으로 쓰지 않는다. 번역이 비어 있거나 `나Please세`처럼 실제 문장에 혼합 token
손상이 남은 경우만 catastrophic이다. 번역문을 하나도 추출할 수 없거나 둘 이상이라면
`UNJUDGED`로 기록하고 provisional winner를 자동 선택하지 않는다.

완료된 response의 raw envelope는 수정하지 않는다. 새 판정기는 raw envelope로부터 현재
품질 view를 메모리에서 다시 만들므로, 이전 run은 GPU 재실행 없이 재판정·resume·후속 phase
재사용이 가능하다. raw envelope가 없는 legacy record만 현재 품질 판정에 사용할 수 없다.

정답 작성은 문맥을 이용해 생략된 주어·대상·말투·장르 용어를 faithful하게 해석할 수 있지만, 원문 근거 없는 사건·관계·행동·정체성·동의·화자를 만들면 안 된다. `皮モノ`는 장르명일 때 기본 `가죽물`로, 실제 물체일 때만 문맥에 따라 `스킨슈트`·`인간 가죽`·`가죽옷`으로 판정한다.
