# GPU 인페인트 품질 게이트 결과 이력

이 문서는 중립 case ID와 검증 계약 SHA만 기록한다. 원본 이미지,
좌표, OCR 문자열, annotation, mask, cleaned 이미지, blind key와 완료
검수 CSV는 Git 밖에 보존한다.

## 2026-07-31: Hough 구조 보호 경로 탈락

- frozen contract:
  `ae0773bd1871bf9164c71858c472df1bfdf65bd0bf708596dd98c71e098e1172`
- 역사적 mask screen result:
  `fdf1ebdf690a4c0b60660d074d16434550760efc0b786794ab67614590c106f9`
- 12행 blind review:
  `8e4a85790cec7cdb4ddc9a97f6b6134da32e4d5cd8e7966a304a78dbf82ff7c5`

dilation 1·2·4와 현행 product-mask 기준선은 모두 탈락했다. 구조 보호
마스크가 일본어 획까지 보호해 검은 구멍을 만들었고, 구조 보호를
제거해도 frozen CTD glyph base가 굵은 외곽선 글자의 일부를 놓쳤다.
text bbox 전체를 mask로 쓰는 후보는 휴대폰 UI, 방 구조와 피부 망점을
넓은 사각 패치로 바꿔 탈락했다.

## 2026-07-31: bold-outline·residual mask screen

- result contract:
  `9f7061f5bb6aa315cf27758d6c772a66ee35f2085f5557fc0f2a61a108e296e5`
- 18행 blind review:
  `dea04c4ec13a1bcfced8d4c30b6cd5150d7cba022521095ce48424c9519c3df3`
- blind payload:
  `7eba86dc1ac453248a45afc53a6e12de2c18b5b25dc942250ed6d3015a30a968`

긴 연속 UI 선을 형태학적으로 분리하고 detector text bbox 안의 굵은
dark glyph와 밝은 outline만 선택했다. dilation 2·4·6은 작은 UI를
product mask보다 잘 보존했지만 세 개발 케이스에서 글자 잔상과 국소
복원 자국이 남았다. product-mask FP32와 현행 BF16은 글자 제거 범위가
너무 넓어 구조와 망점을 훼손했다. product 미포함 영역을 두 번째 GPU
pass로 처리한 후보는 추론을 두 번 수행했지만 잔상과 구조 훼손을 함께
해결하지 못했다.

최종 결과:

- `coverage_eligible_candidates`: 0
- `screen_eligible_candidates`: 0
- `promotion_eligible_candidates`: 0
- model screen 진입 가능 mask: bold-outline dilation 2·4·6

model screen에는 세 케이스 blind 평균순위가 가장 높은 dilation 6
계약을 사용했다. 이는 모델 비교를 위한 중간 선택일 뿐 제품 승격
판정이 아니다.

## 2026-07-31: FP32 model screen

- result contract:
  `90dbafb8786ae057d3e87a226d9a452b4e97e80b74c40a06d69a00213ef47900`
- 15행 blind review:
  `1f1758ae9ef071b3ad1da208224de9c23a322844cdc664d144e70e3f5a634805`
- blind payload:
  `e7038826db0347784b9c05e3185f943171bc249c9e8d946eba99709fe1dc6d97`

모든 제품 후보는 실제 CUDA FP32, CPU fallback 0, mask 밖 변경 픽셀
0으로 실행됐다.

| 후보 | 세 케이스 blind 평균순위 | FP32 총 실행시간 | 판정 |
|---|---:|---:|---|
| LaMa Large 1536 | 1.667 | 4.629초 | 잔상·구조·환각 실패 |
| AOT 2048 | 2.000 | 3.661초 | 잔상·구조·환각 실패 |
| LaMa MPE 2048 | 4.333 | 5.203초 | 잔상·구조·환각 실패 |
| LaMa Large 2048 | 4.667 | 5.765초 | 잔상·구조·환각 실패 |

현행 LaMa Large BF16 product-mask 기준선의 평균순위는 2.333이었지만
FP32 승격 후보가 아니며 구조와 외부 보존 게이트도 통과하지 못했다.
FP32 해상도나 모델을 바꾸는 것만으로는 target glyph coverage와
구조 복원을 동시에 해결하지 못했다.

ZITS/ZITS++ feasibility result contract는
`0afc971165c8172ffd8aa976fb5437576ab06ee8a8ae8c413c567041a04313bd`다.
현재 저장소에는 adapter·model SHA·제품 라이선스 계약이 없어 세
케이스 모두 `feasibility_not_implemented`로 기록했고 blind 후보에
넣지 않았다.

## 2026-07-31: 공식 ZITS++ CUDA FP32 feasibility 완료·품질 탈락

- 공식 소스: `ewrfcas/ZITS-PlusPlus`
- source commit:
  `de8dd48b17aedd15824842adb7bcca7535daba84`
- model_512 checkpoint SHA-256:
  `e30d2073ba63af42836ac611214ed984db7ec739a1eef019451df6a34f566f57`
- LSM-HAWP checkpoint SHA-256:
  `6f72a60ec895f11830763069a40cb548dfb0ba77aca5282ffeea8afc72dc1723`
- model_512 config SHA-256:
  `e46dd48b4715f0044debfa0faca1c5af0c149b1cb1dffa29afb042687f13e4f2`
- license: Apache-2.0, file SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- digest-pinned isolated image:
  `sha256:d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5`
- result contract:
  `af05c5f8d64db20382e357ce843a94cc5327baaeefad182fcf6d5a83fce61aad`
- 18행 blind review:
  `43d25197c3a1ab08b0e489e758b01a91b30dacaf8166d7b1fa75a14472aa6bd2`

ZITS++는 RTX 4070 SUPER에서 실제 CUDA FP32로 실행됐다. CPU fallback은
0이고 세 case 모두 mask 밖 변경 픽셀 0이었다. adapter 내부 model load는
8.214초였고 세 case 추론은 각각 1.448초, 0.413초, 0.404초였다.
smoke에서 관측한 peak CUDA allocated memory는 약 1.01GB였다.

그러나 blind 품질에서는 세 case 모두 원문 잔상, 구조·망점 훼손,
보존 실패, 새 얼룩·패치가 발생했다. 평균 순위는 6.0으로 여섯 후보 중
최하위였고 `coverage_eligible_candidates`,
`screen_eligible_candidates`, `promotion_eligible_candidates`는 모두
0이다. 따라서 ZITS++ 제품 도입은 탈락한다. 이전
`feasibility_not_implemented`는 실행 인프라가 없던 역사적 결과이며,
이번 결과가 실제 모델 실행에 기반한 최종 feasibility 판정이다.

## 상류 coverage 판정

두 번째 개발 케이스의 의미 있는 free-text 영역 하나가 frozen detector
block에 존재하지 않았다. 모든 인페인트 후보가 동일하게 남긴 영역이라
모델 상대평가의 residue로 중복 계산하지 않았지만, 전체 파이프라인
품질 게이트에서는 명확한 상류 누락이다. MangaLMM full-page spotting과
공통 semantic-role routing에서 별도 coverage 실패로 검증해야 한다.

## 2026-07-31: 연결 성분·union 후 성분 보정

- result contract:
  `04ea76dd020d02a2e87ef31cbfa6f9a5b9c681a88a643724133b4c76c4abeeac`
- 24행 blind review:
  `09a9de65fdc1bff814c0290c05b295a9980293cdc298b54226e290796099f7f0`
- blind payload:
  `6428387993954d141c250b196d8f3ff3344f81aa9dc75c3da3f77f0a1e278fd9`

bold-outline dilation 6의 최종 mask는 그대로 두고 GPU 호출만
8-connectivity 성분별로 나눴다. 휴대폰 개발 영역에서 21개 성분의
추론시간은 1.860초였고, 넓은 회색 덩어리를 줄였지만 글자형 조각과
흰 패치가 남았다.

먼저 mask 합집합 전체를 한 번 제거하고 같은 성분을 다시 보정한
후보는 세 케이스 평균 blind 순위 1.0으로 상대적으로 가장 나았다.
휴대폰 케이스는 22회 GPU pass와 3.044초가 걸렸다. 그러나 다음 이유로
절대 품질 게이트는 통과하지 못했다.

- 휴대폰 UI와 망점 위에 흰 패치와 짧은 선형 생성물이 남음
- 방·창문 케이스의 상류 의미 텍스트 누락과 구조 단절
- 피부·머리카락 케이스의 흰 조각과 선화 단절

최종 결과:

- `coverage_eligible_candidates`: 0
- `screen_eligible_candidates`: 0
- `promotion_eligible_candidates`: 0

같은 union 후 성분 계약에서 LaMa Large FP32 1536/2048, LaMa MPE
FP32 2048, AOT FP32 2048을 진단 실행한 result contract는
`7073acf1c05a4321e219e194f008061297dd13e64028826ed1455e5ceac739ce`다.
AOT는 가장 빨랐지만 휴대폰 망점과 UI를 더 크게 왜곡했고, 다른
모델도 공통 mask coverage·구조 단절을 해결하지 못했다. 따라서 추가
모델 교체가 아니라 block별 semantic role과 mask strategy를 완전하게
고정한 새 frozen contract가 다음 단계다.

## 2026-07-31: 완전 역할 주석·전경 글자 라우팅

- annotation template SHA-256:
  `4a4f0b99bd1e2f1467cd1983ce1c9905d1c356c2b575434e7a78c8325f97b902`
- annotation decisions SHA-256:
  `e794ae374f01d1a3419b9bb6005a17694dc52baff942ff80ab4c676354f2a80b`
- frozen contract:
  `8f1f6bca5ef28dbeae1ff1741a0dfd2593357f2a6137d9189ddd178e724e0788`
- mask-residual result contract:
  `3db2645f7abb28042a0426edadf669279fa9151f25567d89dbc9ae52051a201d`
- 42행 blind review:
  `0c22e71d3cf0fa6d5f9810185928937422a46ab93975253ac7638d2904999216`
- blind payload:
  `5361ceaa77f6861be209540acf6674e142dbf7b543b9d21a3416eec08ea5bce2`

현재 snapshot의 56개 block을 원본 기준으로 모두 판정했다. `preserve`
block과 SFX는 edit mask에서 제외됐고, 휴대폰 micro-UI block의 전경
mask 픽셀도 0이었다. 불투명 말풍선은 실제 `bubble_xyxy` 밖으로 mask가
나가지 않도록 고정했다.

완전 주석을 사용한 strategy-routed 후보는 foreground dilation
1·2·4와 구조 보호 ON/OFF를 모두 CUDA FP32/2048로 실행했다. 모든
후보의 CPU fallback과 mask 밖 변경 픽셀은 0이었다. 그러나 직접 blind
검수에서는 다음 문제가 남았다.

- 휴대폰 위 대사에서 모든 후보가 읽을 수 있는 일본어 잔상 또는
  글자형 조각을 남김
- 구조 보호 ON은 일본어 획까지 보호해 의미 대사를 대부분 유지함
- 구조 보호 OFF는 작은 UI를 edit mask 밖에 보존했지만 망점에 회색
  구멍·흰 조각·짧은 선형 생성물을 만듦
- 그림 위 의미 있는 free-text를 제거한 후보는 피부에 큰 회색 타원과
  글자형 잔상을 만듦
- 방 배경에서는 일부 후보가 의미 텍스트를 깨끗하게 지웠지만,
  다른 두 필수 케이스를 동시에 통과한 후보가 없음

최종 결과:

- `coverage_eligible_candidates`: 0
- `screen_eligible_candidates`: 0
- `promotion_eligible_candidates`: 0

따라서 완전 역할 라우팅 자체는 보존 범위를 정확히 줄였지만, 현재
학습형 모델과 mask 조합을 제품에 승격하지 않는다. 다음 실험은 dilation
확대나 추가 모델 교체가 아니라 좁은 glyph 영역의 국소 texture 복원과
가려진 선의 결정적 연결을 별도 feasibility로 검증해야 한다.

## 2026-07-31: 구조 배경 국소 복원 feasibility

- structured-repair result contract:
  `7a5b06ec71f5ef4929ec73063d986749fde8be6a2ddc66a7c6e762ea6172a97e`
- 66행 blind review:
  `d53910c1b2b2f4027e337b679a4e478ef1ec2fdf15d51463d5d0989346fb9e58`
- blind payload:
  `8599d442187fecc7b70448101112702ec6a89a4c0c9a36d2953725563610450d`

불투명 말풍선은 LaMa Large CUDA FP32/2048에 남기고, 반투명 화면·피부·
망점 위 전경 글자만 Telea 또는 Navier–Stokes로 국소 복원하는 hybrid
후보를 실행했다. foreground dilation 1·2·4, 복원 radius 1·2·3과
제한된 구조선 재연결을 포함해 기준선과 비교 후보를 합친 22개 프로필을
동일한 세 frozen ROI에 적용했다.

모든 후보가 자동 실행 계약은 통과했다.

- mask 밖 변경 픽셀: 전 후보 0
- 학습형 인페인터 CPU fallback: 0
- 일반 국소 복원 후보 총시간: 약 4.0~4.6초
- 구조선 재연결 후보 총시간: 약 16초

하지만 66행 blind 전수검수에서는 승격 후보가 없었다.

- 휴대폰 화면 케이스는 22개 모두 읽을 수 있는 원문 잔상 또는 회색
  얼룩·글자형 조각을 남김
- 구조선 재연결은 가려진 UI 선을 복원하지 못하고 긴 대각선 생성물을
  추가해 즉시 탈락
- 방 배경과 피부 free-text에서는 일부 후보가 원문을 완전히 지웠지만
  원형의 평탄한 패치와 망점 불연속이 육안으로 남음
- 다른 두 케이스가 좋아져도 필수 휴대폰 케이스를 동시에 통과한
  후보는 없음

최종 결과는 `coverage_eligible_candidates`,
`model_screen_eligible_candidates`, `screen_eligible_candidates`,
`promotion_eligible_candidates`가 모두 0이다. 따라서 단순 dilation,
Telea/Navier–Stokes, Hough 구조선 연결을 더 조정하는 실험은 종료한다.
휴대폰과 피부의 원래 texture를 알 수 없는 큰 글자 영역에서는 현재
입력만으로 절대 품질 목표를 자동 복원할 수 없다는 판정이다.

## 최종 결정

이번 screen에서는 제품에 승격할 GPU FP32 인페인트 후보가 없다.
제품 기본 인페인터와 설정을 변경하지 않는다. benchmark 도구는 다음을
재현하고 차단하는 증거로만 보존한다.

- Hough 구조 보호로 생기는 글자형 구멍
- CTD glyph base의 굵은 outline 누락
- bbox/product mask의 과도한 구조 훼손
- 제한된 2차 GPU pass의 품질 미달
- 모델 교체만으로 해결되지 않는 잔상과 texture 불연속
- detector에 없는 의미 free-text coverage gap
- 결정론적 국소 복원의 회색 패치와 구조선 재연결 오생성
