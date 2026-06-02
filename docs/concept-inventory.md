# tiny-vllm 개념 인벤토리 + 시각화 레퍼런스

> `llm-expert` 에이전트가 저장소 코드(`kernels.cu`, `main.cpp`)를 검증해 작성. 영상 제작 중 개념별 정확성·시각화 아이디어 참고용.
> 각 항목: 정의 / 핵심 / 선수개념 / 난이도(개념·시각화) / 모션 / 시각화 아이디어.
> 등장 에피소드는 [series-scenario.md](./series-scenario.md) 참조.

---

### C0. LLM = 가중치 파일 + 연산 그래프
- 물리적으로 float 다발 파일, 개념적으로 가중치 + 아키텍처가 정의한 연산 순서. 학습(train)≠추론(serve), 우리는 serve만.
- 난이도1 / 시각화2 / 약간동적
- **시각화**: .safetensors "숫자 바다" → 줌아웃하면 연산 노드로 흘러듦, 프롬프트 입력→토큰 출력 파이프라인 펄스.

### C1. Safetensors 포맷
- 3구역: [8B header_size(uint64)] + [JSON 헤더(dtype/shape/offsets)] + [텐서 원시바이트].
- 난이도2 / 시각화2 / 약간동적
- **시각화**: 바이트 띠, 앞 8B 확대 "=정수 N", JSON 한 키에서 offsets 범위로 화살표 날아가 텐서 슬라이스 박스.

### C2. 부동소수점 & bfloat16
- FP16(1/5/10) vs BF16(1/8/7) vs FP32(1/8/23). BF16은 mantissa 3비트↔exponent로 맞바꿔 FP32와 같은 range, 정밀도만 희생.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 16비트 셀, sign/exp/fraction 색 분리. exponent 슬라이더로 소수점이 수직선 위 "떠다님". FP16→BF16은 fraction 칸이 exponent로 이주 + 범위막대 확장.
- [주의] "implicit 1로 1비트 공짜"는 0/비정규수 예외 있음 단주.

### C3. 호스트/디바이스 메모리 (DRAM/HBM/SRAM)
- Host(CPU+DRAM) vs Device(GPU+HBM, __shared__ SRAM). GPU는 DRAM 직접접근 불가 → cudaMalloc+cudaMemcpy. 원칙: 적게할당·많이재사용·드물게복사.
- 난이도2 / 시각화2 / 매우동적
- **시각화**: 좌 CPU/우 GPU 2단, 데이터 패킷이 PCIe 다리를 느리게 건너 복사비용 체감. SRAM은 즉각 점멸 대비.

### C4. CUDA 실행 모델 (thread/block/warp/grid, SIMT)
- 커널=GPU 함수, 다수 thread 병렬. thread→block→grid, 32thread=warp. **block당 최대 1024 thread**. 인덱스 산술이 본질.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: grid=block격자, block=thread점. 줌인하면 32 단위 warp 경계. 1024 함정: 2048칸을 thread당 2개(+1024 stride)로 해결.

### C5. 토큰화
- 텍스트→정수 토큰 ID. 모델은 토큰 단위. HF tokenizer 재사용(BPE 직접구현은 범위 외).
- 난이도1 / 시각화1 / 약간동적
- **시각화**: 문장이 색 블록으로 쪼개지고 아래로 정수 ID 카운터 굴러떨어짐. 채팅 특수토큰 다른 색.

### C6. 임베딩 게더 (첫 CUDA 커널)
- 토큰 ID로 embed_tokens(128256×2048)에서 행 하나(2048 BF16) 추출. block=토큰, thread=원소, thread당 2원소(+1024).
- 난이도2 / 시각화3 / 매우동적
- **시각화**: 임베딩 테이블에서 ID 조명이 해당 행 비추고 슬라이드되어 (N,2048) 출력에 쌓임. GPU 격자 오버레이로 병렬복사(한 thread가 셀 2개 점선).

### C7. RMSNorm
- normalized_i = a_i/RMS(a)·weight_i, RMS=sqrt(mean(a²)+eps). LayerNorm보다 쌈(평균 빼기 없음). eps로 NaN 방지. BF16→float 캐스팅 계산.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 2048 막대 제곱→합산(C8 reduction 연결)→√mean→모든 막대 동시 수축 정렬.

### C8. 병렬 Reduction (tree reduction) ★
- 1024원소 log₂=10단계 합산. stride i인 thread가 self+=self+i, i 2배. __shared__ + __syncthreads(). warp 내부는 __shfl_down_sync.
- 난이도4 / 시각화4 / 매우동적
- **시각화**: 1024잎 토너먼트 이진트리, 매 단계 활성 thread만 발광(절반씩 잠듦), stride 화살표 2배 증가, __syncthreads 게이트. warp-shuffle 버전은 레지스터 직접 교환.

### C9. RoPE
- Q/K의 2D 쌍을 위치·차원 각도로 회전. θ=1/500000^(2i/head_dim), angle=pos·θ. prefill(angle=token위치)/decode(angle=position) 별도 커널.
- 난이도4 / 시각화4 / 매우동적
- **시각화**: head_dim을 2D 쌍 다발로, 위치↑마다 회전↑. 다주파수 시계 다이얼 군집(낮은i=느림, 높은i=빠름). 상대거리=사잇각 보존.

### C10. Residual connection
- output += input, 원소별 덧셈. 정보 소실 방지.
- 난이도1 / 시각화2 / 약간동적
- **시각화**: 메인 흐름에서 skip 샛길 갈라져 블록 우회 후 +노드 합류.

### C11. 행렬곱 (GEMM) & cuBLAS cublasGemmEx
- A(M,K)·B(K,N)=C, c_ij=i행·j열 내적. Q/K/V proj, attention, FFN 핵심. cuBLAS로 텐서코어, compute 32F/데이터 16BF.
- 난이도3 / 시각화4 / 매우동적
- **시각화**: A행이 B열 위로 미끄러지며 내적, 곱셈쌍 점등+합산 카운터, C의 (i,j) 동시 채움 grid fill.

### C12. Row-major ↔ Column-major 전치 트릭 ★
- 모델 row-major, cuBLAS col-major. row를 col로 읽으면=전치. C^T=B^T·A^T 활용해 복사 없이 전치. 전치 안 된 행렬 OP_T, 전치된 것 OP_N.
- 난이도5 / 시각화3 / 매우동적
- **시각화**: 같은 1D 메모리를 row/col로 "접어" 서로 전치임을 보이고, C=A·B^T → C^T=B·A^T 대수 변형. OP_T/OP_N 도장.

### C13. Prefill vs Decode
- 첫 토큰=모든 입력 동시(prefill), 이후=1개씩(decode). 보존: K/V proj + 마지막 토큰. prefill 병렬도 높음, decode 메모리대역폭 바운드.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 타임라인 2구간. prefill 다발 통과, decode 한 토큰씩. 중간계산 박스가 사용 후 증발, K/V만 캐시 잔존.

### C14. KV Cache ★
- 이전 토큰 K/V projection 누적 저장(append only). 없으면 매 스텝 재계산. 속도↑↔메모리 선형증가.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 시간축 따라 K/V 막대가 토큰마다 한 칸 길어짐, 새 Q가 누적 K 전체와 내적 부채살. "캐시없음" 반례 레이스.

### C15. Attention (scaled dot-product)
- softmax(QK^T/√d_k)·V. O(n²·d). head별 cublasGemmEx 2회(score, score·V).
- 난이도5 / 시각화5 / 매우동적
- **시각화**: 토큰 시퀀스 가로, Q×K 내적→(n×n) 히트맵→√d_k 스케일→causal 마스크→행별 softmax→V 가중합. 주인공 토큰 어텐션 분포 빛 굵기.

### C16. GQA (Grouped-Query Attention)
- 여러 Q head가 K/V head 공유. 32 Q : 8 KV (4:1). kv_head=q_head/4. KV 메모리 1/4↔약간 표현력 손실.
- 난이도3 / 시각화3 / 약간동적
- **시각화**: 32 Q head가 4개씩 8 KV head로 수렴. MHA(32:32)/GQA(32:8)/MQA(32:1) 토글로 KV 메모리 비교.

### C17. SiLU
- SiLU(x)=x·σ(x). FFN gate에 적용. ReLU와 달리 음수 근방 작은 음수. in-place 8192차원.
- 난이도2 / 시각화2 / 약간동적
- **시각화**: SiLU 곡선 + ReLU 오버레이, 음수 구간 "딥" 강조. 입력 점이 곡선 타고 출력 매핑.

### C18. Softmax (numerically-stable)
- σ(v)_i = e^(v_i-max)/Σe^(v_j-max). max 빼기로 오버플로 방지. reduction(max)→exp→reduction(sum)→나눔. C8 패턴 재사용.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 막대들→최댓값만큼 평행이동→exp 비선형확대→정규화(합=1). max 없으면 화면밖 폭발 반례.

### C19. Causal mask
- col>row인 상삼각을 -∞→softmax 후 0. 미래 토큰 누출 방지. -HUGE_VALF.
- 난이도2 / 시각화2 / 약간동적
- **시각화**: n×n 격자 대각선 위 셀들이 검게(삼각 커튼 하강), softmax 후 확률 0(투명).

### C20. Argmax (greedy)
- lm_head logits(128256) 최댓값 인덱스=다음 토큰. D2H 복사 후 CPU 선형 스캔(비효율, 개선여지).
- 난이도1 / 시각화2 / 약간동적
- **시각화**: 12만 막대 분포에서 스캐너가 좌→우 훑으며 최댓값 깃발 갱신, 승자 막대 솟구쳐 토큰 디코드.

### C21. Feed-Forward Network (SwiGLU)
- gate=W_gate·x(2048→8192), up=W_up·x(2048→8192), SiLU(gate)*up, down=W_down·(8192→2048). attention=토큰믹싱/FFN=채널믹싱.
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 2048→두 갈래 8192 확장(부채꼴), gate에 SiLU 게이트, 원소곱, down으로 2048 수축(모래시계 역상).

### C22. Buffer reuse
- 수명 안 겹치는 버퍼는 같은 메모리 재사용(buf_2048_1/2). 메모리 절감↔수명분석 필요.
- 난이도3 / 시각화3 / 약간동적
- **시각화**: 타임라인에 버퍼 A·B 생존구간 안 겹침 → 같은 물리 슬롯에 포개기. 겹침 반례=충돌 빨강.

### C23. Static batching
- N개 요청 동시 처리. throughput↑↔latency↑(최장 프롬프트 대기, head-of-line blocking).
- 난이도3 / 시각화3 / 매우동적
- **시각화**: 길이 제각각 요청을 한 배치로, 짧은 것 끝나도 완료게이트가 최장 대기(낭비 빗금).

### C24. Continuous batching ★ (slot 기반)
- 배치를 slot으로 운영. slot 비면 큐의 프롬프트가 즉시 prefill 후 합류. is_slot_free[]/queue/active_slots[]. iteration-level scheduling.
- 난이도4 / 시각화4 / 매우동적
- **시각화**: slot 레인, EOS slot 비워짐(초록 점멸)→큐 프롬프트 슬라이드 입장→prefill→합류. static과 split-screen 가동률 비교.

### C25. Online Softmax (FlashAttention식) ★
- 스트리밍 softmax. running max m, denom d, acc 유지. 새 점수마다 correction=exp(old_max-new_max)로 d·acc 재스케일. 출력=acc/d. 메모리 O(n)→O(1).
- 난이도5 / 시각화5 / 매우동적
- **시각화**: 점수가 컨베이어로 하나씩, 우측 3 레지스터(m/d/acc) 카드 갱신. 새 max 등장 시 기존 d·acc가 correction만큼 재조정 펌프. naive(전체저장)와 대비.

### C26. PagedAttention & Paged KV Cache ★★
- OS 페이징을 KV cache에. 고정블록(16토큰), block_table[slot][layer][logical]→physical 매핑, free_blocks 풀. 커널이 block_table 따라 물리블록 점프. grid=(slot, Q_head), block=HEAD_DIM. 단편화 제거↔간접참조 오버헤드.
- 난이도5 / 시각화5 / 매우동적
- **시각화**: 논리 연속 시퀀스가 물리 KV cache의 흩어진 블록으로 block_table 비연속 점프. free_blocks pop/반환. 커널이 산재 블록 순회+online softmax. naive 연속할당 단편화와 split-screen.

---

## 정확성 메모

- **하드코딩 상수**: EMBEDDING=2048, KV_DIM=512, HEAD_DIM=64, NUM_Q_HEADS=32, KV_HEAD=8(GQA 4:1), N_LAYERS=16, BLOCK_SIZE=16, MAX_SEQ_LEN=2048, MAX_BLOCKS_PER_SEQ=128, BATCH_SIZE=2, SQRT_HEAD_DIM=8, VOCAB=128256, rope base=500000. **모두 Llama 3.2 1B 전용** — 범용 엔진 아님.
- **README는 미완성**: RoPE/Attention/GQA/Causal mask/Buffer reuse/Batching/Online softmax/PagedAttention 섹션이 TODO 스텁. **실제 동작 코드는 kernels.cu/main.cpp에 완성** → 설명은 코드 기준.
- **softmax 2종 공존**: prefill용 tree-reduction softmax(C18) + decode/paged용 online softmax(C25). "왜 두 개?"가 좋은 내러티브 훅(메모리 materialization 차이).
- **단순화 지점**(서사 소재): argmax CPU 선형스캔, RoPE θ/sin/cos 매번 재계산, softmax/mask가 num_tokens≤1024 가정.

## 원본 파일
- `/tmp/tiny-vllm/README.md` (1290줄 강의)
- `/tmp/tiny-vllm/src/kernels.cu`, `kernels.cuh`, `main.cpp`
- `/tmp/tiny-vllm/python/` (tokenizer.py, rms_norm.py, reference.py — 크로스체크 참고)
