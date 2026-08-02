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
        reject   73-request exact gate
                         |
                   exact output?
                    |          |
                   no         yes
                    |          |
             no promotion  speed candidate
```

`--no-kv-offload`는 KV cache를 GPU에 두지 않는 설정이다. `--n-cpu-moe N`은
첫 N개의 MoE layer weights를 CPU에 둔다. 둘 다 model contents를 변경하지는
않지만 GPU/CPU execution 경로가 달라질 수 있으므로 fixed-seed response ledger
비교가 필요하다.

동시 residency sentinel은 product Gemma 또는 product Paddle container를 재사용하지
않는다. loopback 전용, 이름이 분리된 lab container 두 개만 시작하고, 종료 뒤 GPU
반환과 orphan 부재를 확인한다.
