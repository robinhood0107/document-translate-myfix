# 문제 해결 명세서 03 - Candidate Sweep

핵심 문제 해결 방향은 사용자가 착안했다.

## 가설

큰 crop 우선 정렬과 local VRAM 기반 auto worker 계산이 baseline보다 빠른 후보를 만든다.

## 실험 조건

- candidates: `fixed`, `fixed_area_desc`, `auto_v1`
- run shape: `warmup 1 + measured 3`

## 측정값

suite 실행 후 latest spec에서 채운다.

## 해석

suite 실행 후 latest spec에서 채운다.

## 다음 행동

품질 게이트 통과 후보만 winner 비교로 넘긴다.
