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

## 상류 coverage 판정

두 번째 개발 케이스의 의미 있는 free-text 영역 하나가 frozen detector
block에 존재하지 않았다. 모든 인페인트 후보가 동일하게 남긴 영역이라
모델 상대평가의 residue로 중복 계산하지 않았지만, 전체 파이프라인
품질 게이트에서는 명확한 상류 누락이다. MangaLMM full-page spotting과
공통 semantic-role routing에서 별도 coverage 실패로 검증해야 한다.

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
