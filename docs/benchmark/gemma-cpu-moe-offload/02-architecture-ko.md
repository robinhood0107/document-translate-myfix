# Gemma CPU-MoE / Host-KV Lab Architecture

```text
shipping b10133 + readonly IQ4_NL volume
                 |
       fixed command except two fields
                 |
    no-KV + n_cpu_moe screen
                 |
        physical VRAM fit?
           |             |
          no           yes
           |             |
  physical FAIL    co-resident sentinel
                         |
                  physical PASS / FAIL
                         |
                73-request raw ledger
                    |             |
               non-exact         exact
                    |             |
      raw reproducibility FAIL  speed candidate
      semantic review separate       |
                                   quality + AB/BA
```

`--no-kv-offload`는 KV cache를 GPU에 두지 않는 설정이다. `--n-cpu-moe N`은
첫 N개의 MoE layer weights를 CPU에 둔다. 둘 다 model contents를 변경하지는
않지만 GPU/CPU execution 경로가 달라질 수 있으므로 fixed-seed response ledger
비교가 필요하다. ledger non-exact는 raw 재현성 결과이며 의미 회귀 자체가 아니다.
의미 품질은 원문·인접 대사·페이지 상황을 기준으로 별도 검수하고, 현 v1 protocol은
raw-exact 후보만 속도 gate에 보낸다.

동시 residency sentinel은 product Gemma 또는 product Paddle container를 재사용하지
않는다. loopback 전용, 이름이 분리된 lab container 두 개만 시작하고, 종료 뒤 GPU
반환과 orphan 부재를 확인한다.
