# tiny-vllm 영상 시리즈 — 완성형 시나리오 문서

> 출처 저장소: https://github.com/jmaczan/tiny-vllm (C++/CUDA로 Llama 3.2 1B 추론 엔진을 밑바닥부터 구현하는 강의)
> 작성: `llm-expert`(개념·시각화 분석) → `narrative-architect`(시리즈 구성·연출) 2단계 에이전트 파이프라인.
> 색상·폰트는 `src/theme.ts`를 그대로 사용. 규격 1920×1080 / 30fps.

---

## 1. 시리즈 개요

### 결정: 총 9편 (+ 00편 예고편 1개)

전문가의 12편 제안을 **내러티브 밀도** 기준으로 9편으로 재구성. 원칙:
- **한 편 = 하나의 깨달음**: 개념 수가 아니라 통찰 수로 편을 나눈다.
- **반복 모티프는 묶어서 각인**: tree reduction(C8)은 RMSNorm·softmax·attention에 재등장 → 첫 등장을 강하게, 이후 콜백.
- **클라이맥스 보존**: PagedAttention·online softmax·continuous batching은 결말, 다른 개념과 안 섞음.
- **난도5 격리**: row/col-major 전치 트릭(C12)은 혼자 한 편을 버틴다(가장 많이 튕기는 지점).

### 편성표

| EP | 제목 | 핵심 통찰 | 개념 | 길이 | 클라이맥스 시각화 |
|---|---|---|---|---|---|
| **00** | 예고편 | "GPU에게 Llama를 처음부터 가르친다" | 전체 떡밥 | 1:00 | 전 편 몽타주 |
| **01** | LLM은 사실 거대한 숫자 다발이다 | 모델 = 가중치 + 연산그래프, BF16 | C0,C1,C2 | 9:00 | BF16 비트 이주 |
| **02** | GPU는 어떻게 1만 개를 동시에 세나 | 호스트/디바이스, thread/block/warp | C3,C4 | 8:00 | GPU thread 격자 점화 |
| **03** | 첫 커널: 단어를 벡터로 (게더) | 임베딩 게더 = 병렬 룩업(복사) | C5,C6 | 7:30 | 임베딩 게더 병렬 룩업 |
| **04** | 1024개를 한 번에 더하는 법 | 병렬 tree reduction → RMSNorm → softmax | C8,C7,C18 | 11:00 | 1024잎 토너먼트 |
| **05** | 위치를 회전으로 새긴다 (RoPE) | RoPE 다주파수 회전 + residual | C9,C10 | 8:30 | RoPE 다주파수 회전 |
| **06** | cuBLAS와 전치 트릭의 마술 | GEMM + row/col-major 접기 | C11,C12 | 10:00 | 메모리 접기(전치 트릭) |
| **07** | 어텐션, 토큰들이 서로를 본다 | QK^T→mask→softmax→V, GQA(32:8) | C15,C16,C19 | 12:00 | 어텐션 파이프라인 히트맵 |
| **08** | 첫 토큰이 태어나는 순간 | SwiGLU FFN, argmax, prefill→decode | C17,C21,C20,C13 | 10:00 | prefill vs decode 분기 |
| **09** | KV캐시·온라인 소프트맥스·PagedAttention | 캐시→FlashAttn 갱신→페이징→continuous batching | C14,C22,C25,C26,C23,C24 | 14:00 | PagedAttention 블록 매핑 |

**총 ~90분.** EP9는 저장소의 진짜 정체성(vLLM)이므로 "대결말"로 길게.

### 타깃 / 톤
- 타깃: C/C++·Python을 읽을 수 있고 행렬곱·임베딩 용어는 알지만 **CUDA 커널은 안 짜본** 중급 개발자.
- 톤: 3Blue1Brown의 시각적 인내심 + Fireship 페이싱. **화면이 먼저 말하고 내레이션은 보조.**

---

## 2. 관통 서사 (스토리 아크)

### 한 줄 컨셉
> **"GPU라는 백지 상태의 거대한 연산 군대에게, Llama 3.2 1B를 처음부터 끝까지 가르친다. 가중치 파일 한 개에서, 살아 숨 쉬는 추론 서버까지."**

### 3막 구조
```
[ACT 1: 재료] EP1-2  — "ChatGPT의 정체는 숫자 다발과, 그걸 동시에 굴리는 군대다"
[ACT 2: 한 토큰의 여정] EP3-8 — 게더→정규화/합산→위치→행렬곱→어텐션→FFN+첫토큰
[ACT 3: 진짜 엔진] EP9 — 재계산 멈추기(KV캐시)→메모리 안터지게(PagedAttention)→GPU 안놀게(continuous batching)
```

### 콜드오픈 / 엔딩 훅 체인
각 편은 **직전 편 마지막 프레임에서 시작**하는 콜드오픈으로 연결. 시리즈 전체가 하나의 긴 파이프라인을 카메라가 패닝하는 느낌.

| 편 | 콜드오픈 | 엔딩 훅 |
|---|---|---|
| EP1 | "model.safetensors, 2.4GB. 열어보자." | 숫자는 봤다. 누가 굴리지? GPU 실루엣 점등 |
| EP2 | BF16 숫자들이 GPU 격자 위로 쏟아짐 | thread 준비됨. 첫 일거리? → 토큰 |
| EP3 | "Hello"가 토큰 ID로 쪼개짐 | 임베딩 벡터 완성. 2048개를 어떻게 정규화? |
| EP4 | 게더된 벡터가 떨리며 "크기를 재야 한다" | RMS·softmax 합 완성. 근데 위치 정보가 없다! |
| EP5 | 정규화 벡터에 "몇 번째 토큰?" 물음표 | 회전 완료. 진짜 행렬곱(Q,K,V) 만들 시간 |
| EP6 | RoPE 벡터 × 가중치 행렬 = ? | Q,K,V 라인 완성. 토큰들이 서로 볼 차례 |
| EP7 | Q,K,V 세 묶음 입장 | 어텐션 출력 나옴. 근데 "다음 단어"는 아직 없다 |
| EP8 | 어텐션 출력이 FFN 터널 진입 | **첫 토큰 탄생!** 두 번째 토큰은… 다 다시? |
| EP9 | 첫 토큰이 "다시 계산?" 경고 | 서버 가동. "이제 너도 만들 수 있다." (저장소 링크) |

---

## 3. 공통 비주얼 모티프 (시리즈 시각 언어)

편이 바뀌어도 같은 모양·같은 색으로 등장. (theme: accent=#58a6ff 파랑, accent2=#3fb950 초록, warn=#d29922 노랑, danger=#f85149 빨강, bg=#0d1117, surface=#161b22, muted=#8b949e)

| # | 모티프 | 형태 | 색 의미 | Remotion 컴포넌트 |
|---|---|---|---|---|
| M1 | 토큰 블록 | 둥근 사각 칩 + 토큰텍스트/ID | 테두리 accent, 활성 시 accent 채움 | `<TokenChip text id active/>` |
| M2 | 숫자 벡터/행렬 격자 | 정사각 셀 격자, 값=명도 | 양수 accent / 음수 danger / 0 surface | `<Grid rows cols values colorScale/>` |
| M3 | 메모리 보관함 | 가로 주소 슬롯 띠(연속 메모리) | 빈칸 border / 쓰임 accent2 / 재사용직전 warn | `<MemoryStrip slots highlight/>` |
| M4 | GPU thread 격자 | 점32=warp, warp묶음=block | 대기 muted / 실행 accent / warp테두리 accent2 | 절대배치 점 + spring 스태거 |
| M5 | 데이터 흐름 입자 | 셀→셀 광점 트레일 | 흐름 accent / 곱셈 시 warn 펄스 | `<FlowParticle from to delay/>` 베지어 |
| M6 | KV 보관함(캐시) | K칸/V칸 쌍이 시간축 누적 성장 | K accent / V accent2 / 새칸 warn 플래시 | `<KVCache step/>` |
| M7 | 페이지 블록 & 매핑 | 논리블록(연속)↔물리블록(흩어짐) 화살표 | 논리 accent / 물리 accent2 / 매핑선 warn | `<BlockTable logical physical mapping/>` SVG |
| M8 | 레지스터 HUD | 우상단 미니패널 max/d/acc 실시간 | 갱신값 한 프레임 warn 하이라이트 | 고정 오버레이 flash |

**전역 규칙**: 배경 bg, 패널 surface. 제목 Inter / 숫자·코드 JetBrains Mono. 좌하단 **파이프라인 미니맵**(16레이어 막대, 현재 부품 accent 점등, EP3부터 상시). 상단 **스펙 칩** `EMBED=2048·KV=512·HEAD=64·Q:32 K:8·L=16`(Llama 3.2 1B 전용 상기). 전환: 부품 이동=수평 팬, 개념=줌인, 편전환=입자 dissolve.

---

## 4. 에피소드별 상세 시나리오

> 시간=누적 초. 내레이션=실제 대본. Remotion: `useCurrentFrame()`, `interpolate`, `spring`, `<Sequence>`, `theme.colors`.

### EP00 — 예고편 (1:00)

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–8 | "ChatGPT 같은 게 안에서 뭘 하는지, 진짜로 본 적 있나요?" | 검은 화면 커서, `model.safetensors` 타이핑 | 타이핑 후 파일이 수만 숫자(M2)로 폭발 | 타이프라이터 + 격자 스태거 |
| 2 | 8–35 | "우리는 이걸 C++과 CUDA로, 밑바닥부터 만듭니다." | 9편 클라이맥스 0.7초씩 몽타주 | 각 컷 warn 플래시 전환, BPM 컷 | 프리렌더 클립 `<Series>` |
| 3 | 35–60 | "9편이면, 당신도 추론 엔진을 만들 수 있습니다." | 타이틀 + 파이프라인 미니맵 전체 점등 | 미니맵 좌→우 순차 점화 후 로고 | spring 스태거 |

### EP01 — LLM은 사실 거대한 숫자 다발이다 (9:00) ★ BF16 비트 이주

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–12 | "이게 우리가 만들 모델. Llama 3.2 1B. '1B'는 10억 개 숫자라는 뜻이죠." | `model.safetensors` 아이콘, `2.4GB`, `1,235,814,400 params` | 카운터 0→12억 롤업, 아이콘 진동 | interpolate 카운터 |
| 2 | 12–30 | "마법이 아니라 정해진 형식으로 줄 세운 숫자. Safetensors." | `[8B 헤더크기][JSON 헤더][텐서 바이트]`(M3) | 파일 펼쳐지며 3영역 색칠, JSON 줌인 | MemoryStrip, header_size 8B 반영 |
| 3 | 30–48 | "헤더는 지도. offset만 보고 통째로 메모리에." | offsets→텐서 구간 곡선 화살표 | 화살표 입자, 도착 시 구간 채움 | FlowParticle. main.cpp 89–126 근거 |
| 4 | 48–75 | "숫자 하나는? float32가 아니라 BF16, 16비트." | float32(1·8·23) vs BF16(1·8·7) | 가수 16칸 danger로 부서져 떨어짐(비트 이주) | 비트=M2 격자, 중력 interpolate |
| 5 | 75–105 | "지수는 그대로라 범위는 안 줄어요. 정밀도만 희생." | 수직선: float32 촘촘 vs BF16 듬성, 끝점 동일 | BF16 점 듬성 스냅, 끝점 warn 점멸 | spring 스냅 |
| 6 | 105–150 | "실제로 0.7을 BF16 비트로." | 0.7→부호/지수/가수 비트 채움 + 0x3F33 | 비트 좌→우 점등, 완성 후 M3 슬롯으로 빨려듦 | morph scale+translate |
| 7 | 150–200 | "이런 숫자가 12억 개. 전부 BF16 격자." | 여러 가중치 텐서(M2) 격자벽 + 이름표 | 줌아웃, 16레이어 반복 패턴 강조 | Grid 다수 |
| 8 | 200–240 | "LLM = 가중치(숫자) + 연산그래프. 오늘은 숫자를 봤고…" | 좌 숫자벽 / 우 빈 회로도(점선) | 입자가 우측으로 흐르다 멈춤(미완) | split + freeze |
| 9 | 240–270 | "근데 12억 번 곱셈을 누가 하죠? 다음 편, GPU." | GPU 실루엣 한 귀퉁이만 점등 후 블랙아웃 | M4 부분 점등 cut-to-black | M4 점 격자 |

### EP02 — GPU는 어떻게 1만 개를 동시에 세나 (8:00)

핵심: "CPU는 천재 4명, GPU는 평범한 1만 명. LLM엔 1만 명이 맞다."

| # | t(s) | 내레이션(요약) | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–10 | "지난 편 떡밥: 누가 다 곱하나." | EP1 GPU 부분점등에서 이어짐 | 점등이 전체로 번짐 | M4 fade-in |
| 2 | 10–40 | "CPU vs GPU. 소수정예 vs 인해전술." | 좌 CPU(코어4) / 우 GPU(코어수천) | "10000 덧셈" 투입, GPU 압승 race | 카운터 2개 |
| 3 | 40–80 | "데이터는 두 세계. 호스트↔디바이스. 복사 필요." | RAM↔VRAM, cudaMemcpy 다리 | 숫자가 트럭처럼 건너감, 지연 게이지 | M5 대량 이동 |
| 4 | 80–140 | "일꾼 계층: thread→warp(32)→block→grid." | M4 격자 계층 줌아웃 | thread1→32 warp묶음→block→grid 라벨 | scale interpolate |
| 5 | 140–185 | "규칙: block당 최대 1024 thread. 2048 벡터는 2칸씩." | 1024 thread가 i, i+1024 처리 | thread에서 두 화살표 동시 | workIndex 패턴 |
| 6 | 185–210 | "이 패턴 계속 나옵니다. 기억해두세요." | 패턴 도장 박제 | 축소되어 패턴 북마크 | 재사용 컴포넌트 |
| 7 | 210–240 | "일꾼 준비 끝. 첫 일거리는 토큰." | thread 대기, "Hello" 낙하 | thread 기대 깜빡, 컷 | 콜드오픈 훅 |

### EP03 — 첫 커널: 단어를 벡터로 (게더) (7:30) ★ 임베딩 게더 병렬 룩업

핵심: "신경망의 첫 연산은 곱셈이 아니라, 표에서 줄을 베껴오는 복사다."

| # | t(s) | 내레이션(요약) | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–12 | "사람 말을 GPU는 모름. 먼저 토큰으로." | "Hello, world" | 토큰 블록 절단 → ID [9906,11,1917] | 칼선 + spring |
| 2 | 12–35 | "ID로 거대한 표에서 한 줄을 뽑아요. 임베딩 테이블." | embed_tokens[128256×2048](M2) | ID 9906→9906번 행 warn 하이라이트 | Grid 행 하이라이트 |
| 3 | 35–75 | "게더. 그 줄을 통째로 복사해 토큰 벡터로. 2048개." | 행이 떠올라 벡터 슬롯으로 복제 | 행 clone 상승 도킹, 토큰3→벡터3 | clone translate |
| 4 | 75–130 | "GPU: 토큰마다 block, thread t가 값 복사. 두 칸씩." | M4 위 코드 매핑, embed→output 화살표 | thread 동시 입자 발사, +1024 두번째도 동시 | embeddingGatherKernel 1:1 |
| 5 | 130–175 | "곱셈도 더하기도 없어요. 순수 복사. 메모리 대역폭이 전부." | FLOPs≈0 / Memory BW=병목 게이지 | 연산 게이지 바닥, 메모리 풀 | 게이지 interpolate |
| 6 | 175–210 | "decode 땐 토큰 1개만 게더. 같은 커널, block 1개." | prefill vs decode 나란히 | 좌 다발 / 우 단일, "1 token" 펄스 | 두 Sequence |
| 7 | 210–225 | "벡터는 생겼다. 근데 크기가 제멋대로. 1024개를 어떻게 한 번에 더하지?" | 벡터 셀 명도 진동 | 다음 편 떡밥 "+1024?" 물음표 | random jitter |

### EP04 — 1024개를 한 번에 더하는 법 (11:00) ★★ tree reduction

핵심: "순차로 더하면 1024단계, 토너먼트로는 10단계. 병렬 환원은 모든 GPU 합산의 심장."

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–12 | "정규화하려면 모든 원소 제곱합이 필요. 합 하나. 근데 1024개." | EP3 진동 벡터 + `Σx²?` | 콜드오픈, 큰 물음표 | dissolve |
| 2 | 12–40 | "순진하게? thread 0이 1024개 차례로. 나머지 1023명 놀아요." | thread 1줄, 0번만 일함 | thread0 순차 누적, 나머지 muted 잠듦 | 의도적 답답함 |
| 3 | 40–110 | "토너먼트. 1단계에 절반이 짝을 더해요. 1024→512→…10단계." | 1024잎 트리 토너먼트 | stride 1,2,4,8… 살아있는 칸 절반씩, 정점으로 줌인 | rmsNormKernel for(i=1;i<1024;i*=2) 1:1 |
| 4 | 110–140 | "매 단계 __syncthreads(). 전원이 끝낼 때까지 대기." | 동기화 장벽 게이트 | 빠른 thread 대기 → 게이트 열림 → 동시 진행 | 셔터 애니 |
| 5 | 140–175 | "정점에 제곱합. RMSNorm: 평균에 루트, 나누고 가중치." | sqrt(sum/2048+1e-5) 방사 | 정점값이 1024칸 전체로 빛 퍼짐, 재색칠 | output=input/rms*weight |
| 6 | 175–210 | "이 패턴, 이름 붙입시다 — tree reduction. 도장 쾅." | tree reduction 아이콘 박제 | 트리 축소되어 북마크 | 박제 |
| 7 | 210–280 | "softmax도 똑같아요. 단 합이 아니라 최댓값을 먼저." | 같은 트리, 노드 + → max | 노드 라벨 교체 재생, max→빼기→exp→sum→나눔 | softmaxKernel 1:1 |
| 8 | 280–330 | "왜 max를 빼? exp(1000)은 무한대로 터져요." | exp(1000)=∞ danger vs exp(0)=1 | 좌 오버플로우 글리치, 우 차분히 1.0 | 글리치 효과 |
| 9 | 330–390 | "정규화도 softmax도 같은 토너먼트. 다음 편 — 위치를 회전으로." | 정규화 벡터에 회전 화살표 | 회전 아이콘 돌다 멈춤(RoPE 떡밥) | 회전 teaser |

### EP05 — 위치를 회전으로 새긴다 (RoPE) (8:30) ★ 다주파수 회전

핵심: "위치를 더하는 게 아니라 벡터를 돌린다. 차원 쌍마다 다른 속도로."

| # | t(s) | 내레이션(요약) | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–12 | "어텐션은 순서를 몰라요. 'cat sat'='sat cat'. 위치를 새겨야." | 토큰 순서 swap해도 동일 출력 | danger 경고 | 콜드오픈 |
| 2 | 12–55 | "차원을 2개씩 짝지어 2D 점으로 보고, 위치만큼 회전." | (x_2i, x_2i+1)이 2D 좌표 점 | 쌍 추출 → 좌표평면 점 → θ 회전 호 | prev_2i*cos - prev_2i_1*sin |
| 3 | 55–110 | "각도 = 위치 × 주파수. 주파수는 차원마다 달라요." | 다이얼 32개 다른 속도 회전 | 앞=빠름 accent, 뒤=느림 muted, 슬라이더로 비례 | θ=pos/500000^(2i/64) |
| 4 | 110–150 | "위치 커질수록 더 많이 돌아요. 회전 패턴=위치 지문." | 위치 1/5/20 회전량 비교 | 3 복제본 다른 각도, 회전 궤적 잔상 | 3 Sequence |
| 5 | 150–185 | "GPU: thread가 쌍 하나 담당. Q는 1024, K는 512 thread." | M4 thread→쌍 매핑, Q/K 분기 | thread 회전 적용, 두 폭 분기 | ropeKernel, threads=proj_dim/2 |
| 6 | 185–225 | "residual. 레이어 결과에 원본을 다시 더해요." | 입력이 레이어 통과 후 원본과 + 합류 | 두 갈래: 통과 / skip 우회, 출구 합산 | residualKernel input+=embeds |
| 7 | 225–255 | "위치 새겼다. 이제 행렬곱으로 Q,K,V. cuBLAS 등판." | RoPE 벡터 × 큰 행렬 = ? | 곱셈 기호 깜빡, 컷 | GEMM 떡밥 |

### EP06 — cuBLAS와 전치 트릭의 마술 (10:00) ★ 메모리 접기(난도5)

핵심: "우리 데이터는 row-major인데 cuBLAS는 col-major로 본다. 이 '오해'를 역이용해 전치를 공짜로."

| # | t(s) | 내레이션(요약) | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–15 | "직접 커널은 느려요. NVIDIA가 극한 최적화한 cuBLAS." | 손수 커널 vs cuBLAS 게이지 | race, cuBLAS 압승 | 게이지 |
| 2 | 15–60 | "함정. C 배열은 row-major, cuBLAS는 col-major." | 같은 4×4를 두 방식으로 M3에 펼침 | row=가로채움 / col=세로채움 비교 | M2→M3 매핑 |
| 3 | 60–110 | "통찰: 같은 바이트를 col-major로 읽으면, 그게 전치행렬이에요." | A 바이트를 col-major 재해석→A^T | **메모리 접기**: 슬롯 고정, 격자가 90도 접혀 A→A^T morph, 바이트 안 움직임 | transpose interpolate, 띠 정지 |
| 4 | 110–175 | "트릭: Q=X·Wq^T를 OP_T,OP_N으로. 두 번 뒤집혀 원하던 모양." | 수식 변환 체인, 전치 붙었다 상쇄 | "cuBLAS가 보는 것" vs "실제" 토글 | main.cpp 169–182 1:1 |
| 5 | 175–215 | "cublasGemmEx(OP_T,OP_N). Q,K,V 세 번. 전치 비용 0." | 코드 3블록, OP 하이라이트 | X×Wq/Wk/Wv → Q(2048)/K(512)/V(512) | 코드 + 입자 |
| 6 | 215–250 | "진짜 전치 필요한 곳(down/logits)만 OP 바꿔요." | down_proj/logits OP 패턴 다름 | 미니맵 GEMM 노드 점등, OP 라벨 | main.cpp 467–512 |
| 7 | 250–280 | "Q,K,V 라인 완성. 토큰들이 서로 볼 시간 — 어텐션." | 세 묶음 컨베이어 이송 | Q accent/K warn/V accent2 우측 팬 | 카메라 팬 |

### EP07 — 어텐션, 토큰들이 서로를 본다 (12:00) ★★ 어텐션 파이프라인

핵심: "모든 토큰이 다른 모든 토큰에게 '너 얼마나 중요해?'를 묻고 답을 가중평균." + GQA + causal mask.

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–15 | "Q,K,V 왔습니다. Q='질문', K='이름표', V='내용물'." | 세 묶음 + 비유 아이콘 | 콜드오픈, 라벨 카드 flip | dissolve |
| 2 | 15–60 | "1단계: 모든 Q가 모든 K와 내적. 점수 행렬." | Q·K^T → N×N 히트맵(M2) | Q행×K열 교차 셀 점수 채움 | prefill_attn_scores, scale=1/8 |
| 3 | 60–120 | "미래를 보면 반칙. causal mask — 위쪽 삼각을 -∞로." | 상삼각(col>row) -∞ | mask가 대각선 위 danger 칠하고 죽임 | causalMaskKernel col>row→-HUGE_VALF |
| 4 | 120–180 | "2단계: 각 행 softmax. 그 토너먼트 기억나죠?" | 한 행 확대 → EP4 트리 재등장 | 토너먼트 미니 재생(콜백), 확률분포(합1) | softmaxKernel 재사용 |
| 5 | 180–210 | "각 행은 확률. 토큰 i가 누구에게 주목하나." | attention weight 히트맵 | 밝은 셀이 "주목" 입자로 튐 | colormap |
| 6 | 210–270 | "3단계: 확률로 V를 가중평균. 어텐션 출력." | weight × V → 출력 벡터 | weight 셀이 V행에 곱해져(warn) 출력 합산 | attn_scores_v GEMM |
| 7 | 270–330 | "Llama의 절약: GQA. Q 32개, K/V 8개. 4:1 공유." | 32 Q ↔ 8 KV, 4:1 | Q 4개씩 묶여 KV로 수렴, 메모리 1/4 | kv_head=q_head/4 |
| 8 | 330–390 | "헤드 32개가 각자 다른 관계. o_proj로 섞으면 끝." | 32 헤드 concat → o_proj | 타일 합쳐 통과, 단일 벡터 | o_proj GEMM |
| 9 | 390–420 | "어텐션 통과. 근데 '다음 단어'는 아직. 마지막 — FFN." | 출력이 FFN 터널 앞 정지 | 빨려들기 직전 컷 | 터널 떡밥 |

### EP08 — 첫 토큰이 태어나는 순간 (10:00)

핵심: "FFN으로 생각을 마치고, 12만 후보 중 1등을 고른다(argmax). 그리고 prefill/decode가 갈라진다."

| # | t(s) | 내레이션(요약) | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–12 | "어텐션 출력이 FFN으로. Llama는 SwiGLU." | 출력이 터널 진입 | 콜드오픈 | dissolve |
| 2 | 12–70 | "두 갈래로 8192차원 확장: gate, up. gate엔 SiLU. 곱해요." | gate/up 두 GEMM, SiLU 곡선 | 두 갈래 확장(M2), gate에 SiLU(accent2), 원소곱(warn) | gate/up GEMM + siluKernel |
| 3 | 70–110 | "down_proj로 2048로 줄여요. 한 레이어." | 8192→down(2048) | 좁혀짐, "Layer 0 ✓" | down GEMM |
| 4 | 110–160 | "16번 반복. 출력이 다음 입력." | 미니맵 16레이어 순차 점등 | 카메라 빠른 패닝, 가속 | 16 Sequence |
| 5 | 160–210 | "마지막 RMSNorm, 임베딩 행렬로 곱해 12만 점수(logits)." | 벡터 × embed^T → logits[128256] | 12만 칸 막대그래프로 펼침 | embed_proj GEMM, VOCAB=128256 |
| 6 | 210–250 | "argmax — 12만 중 최고 점수. 다음 토큰!" | 최댓값 warn 점등 → 디코드 | 막대 스캔 spotlight, ID→"world" | CPU argmax(531행) |
| 7 | 250 | (첫 토큰 탄생 비트) | "첫 토큰 생성!" | M1 블록 accent2 펄스 | flash |
| 8 | 250–300 | "갈림길. 방금은 prefill — 전체 한 방에. 두 번째부터 decode — 한 토큰씩." | prefill vs decode 분기도 | N 다발(prefill)→단일 루프(decode) | prefill/decode 분리 |
| 9 | 300–360 | "근데 decode에 앞 토큰들의 K·V가 또 필요. 매번 재계산? 다음 편: KV캐시." | decode 루프 "K,V 재계산?!" danger | 이전 K·V 유령 재등장 "다시 ✗" | 클리프행어 |

### EP09 — KV캐시·온라인 소프트맥스·PagedAttention (14:00) ★★★ 대결말

핵심: "진짜 엔진은 세 발명으로 완성 — (1)계산 저장(KV캐시) (2)메모리 안터지게 갱신(online softmax) (3)OS처럼 페이징(PagedAttention) → 여러 요청을 끊김없이(continuous batching)."

#### Part A — KV캐시 (0:00–3:30)

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 1 | 0–12 | "EP8 경고. 매 토큰마다 앞 K·V 다 재계산하면 100번째는 100배." | 토큰수 vs 연산량 O(n²) 폭발 | 막대 기하급수 danger | dissolve |
| 2 | 12–60 | "해결: K·V를 저장해 재사용. KV캐시." | KV 보관함(M6) 시간축 등장 | 스텝마다 K(accent)/V(accent2) 쌍 추가(warn) | KVCache step |
| 3 | 60–130 | "decode: 새 토큰 Q만, 캐시된 모든 K와 내적. n²→n." | 단일 Q가 캐시 K 전체와 내적 | Q에서 K칸들로 부채꼴, 점수→가중합 | O(n²)→O(n) 게이지 |
| 4 | 130–180 | "캐시도 공짜 아냐, 메모리 자라요. 임시버퍼는 돌려막아요." | M6 성장 + buf 재사용(M3) | 길이 증가 + buf_2048_1/2 재배정(warn) | buf_2048_1/2 공유 |

#### Part B — Online softmax (3:30–6:30) ★★ 레지스터 갱신

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 5 | 210–240 | "EP4 softmax는 행 '전체'를 메모리에 펼쳐야 가능. 캐시가 길면?" | EP4 트리 재등장 → "전체 저장" danger | 콜백, 메모리 한계선 초과 | 콜백 |
| 6 | 240–330 | "FlashAttention 묘수: 한 토큰씩 보며 결과를 그때그때 갱신. 새 max 나오면 비율 보정." | 레지스터 HUD(M8) max/d/acc, K/V 스트림 | 토큰마다 correction=exp(old-new), d/acc 갱신. **새 max 순간 기존 acc/d 쪼그라듦** | pagedAttentionKernel 1:1, HUD warn flash |
| 7 | 330–390 | "끝까지 본 뒤 acc/d 한 번. 전체 저장 안 하고 같은 softmax. O(n)→O(1)." | output=acc/d, 전체저장과 일치 | 스트림 끝, acc/d 합쳐 출력 "동일 ✓" | 동일성 강조 |
| 8 | 390 | "이 내적, warp 32명이 __shfl_down 토너먼트. HEAD_DIM=64라 2 warp." | warp shuffle reduction(16→1) | thread 레지스터 직접 교환, 5단계 | __shfl_down_sync, EP4 warp판 콜백 |

#### Part C — PagedAttention (6:30–11:00) ★★★ 블록 매핑

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 9 | 390–430 | "요청마다 '최대 길이'를 통으로 잡으면 낭비가 어마어마." | 연속 블록 통예약, 빈칸 가득 | M3 통예약 muted, "낭비" danger 빗금 | 단편화 |
| 10 | 430–520 | "PagedAttention: OS 가상메모리처럼 페이지(16토큰)로 쪼개요. 논리=연속, 물리=흩어짐 OK." | 논리↔물리 + block_table(M7) | 논리 0,1,2→물리 5,2,9 곡선 매핑, 16토큰 차면 free_blocks pop | block_table, BLOCK_SIZE=16. **최대 클라이맥스** |
| 11 | 520–580 | "커널은 block_table로 물리 주소 계산. K/V는 V_OFFSET으로 분리." | 주소식 논리→물리 점프 | block_table 룩업→물리 점프→token·head·dim 착지, K/V 색분리 | physical_block*BLOCK_BYTES+... |
| 12 | 580–620 | "낭비 거의 0. 같은 GPU로 훨씬 많은 요청." | 페이징 전후 활용률 | 좌(빈칸많음)→우(꽉참) 게이지 급상승 | before/after |

#### Part D — Continuous batching & 피날레 (11:00–14:00) ★ slot 스케줄링

| # | t(s) | 내레이션 | 화면 | 애니메이션 | 구현 힌트 |
|---|---|---|---|---|---|
| 13 | 660–710 | "요청 하나만 처리하면 1만 일꾼이 놀아요. 슬롯에 태워 같이(static batching)." | 슬롯 2개(BATCH_SIZE=2), A·B 탑승 | 두 요청이 slot 0,1, thread 격자 함께 채움 | active_slots, dim3(slots, Q_HEADS) |
| 14 | 710–790 | "static 문제: A 끝나도 B 기다려 놀아요. continuous: 끝난 슬롯에 새 요청 즉시." | 슬롯 타임라인, A종료→C 진입 | A EOS(is_slot_free)→빈칸 warn→큐 C prefill 채움, 가동률 100% | is_slot_free, queue, 간트 |
| 15 | 790–820 | "이게 vLLM의 정체. 페이징 캐시 + 끊김없는 배칭. 밑바닥부터 만들었어요." | 전체 시스템 라이브 다이어그램 | 두 요청 동시 토큰 출력, 미니맵 전체 accent2 | 통합 라이브 뷰 |
| 16 | 820–840 | "9편 전부, 가중치 파일 하나에서 여기까지." | M1~M8 모티프 한 줄 정렬 | 전 모티프 흐르며 연결, 회고 몽타주 | 전 편 콜백 |
| 17 | 840 | "코드는 깃허브에. 이제, 당신 차례입니다." | 저장소 링크 + "Your turn." | git clone 타이핑 멈춤, fade to black | 엔딩 |

---

## 5. 제작 착수 가이드

**재사용 컴포넌트 우선순위(먼저 만들면 전 편 가속):**
1. `<TokenChip>`, `<Grid>`(colorScale), `<MemoryStrip>` — EP1부터 전편
2. `<TreeReduction stride>` — EP4 제작, EP7·EP9 콜백
3. `<FlowParticle>`(베지어+interpolate) — 모든 데이터 흐름
4. `<KVCache>`, `<BlockTable>`, `<RegisterHUD>` — EP9 전용이나 분량 큼, 일찍 착수
5. 좌하단 `<PipelineMinimap currentStage>`, 상단 `<SpecChips>` — EP3부터 상시

**정확성 가드(코드 검증 완료):**
- 상수: EMBED=2048, KV_DIM=512, HEAD_DIM=64, Q:32/K:8(GQA 4:1), N_LAYERS=16, BLOCK_SIZE=16, BATCH_SIZE=2, VOCAB=128256, rope base=500000, attn scale=1/8(=1/√64).
- `kernels.cu`: rmsNorm/softmax의 `for(i=1;i<n;i*=2)` tree reduction, pagedAttention의 online softmax(correction_factor/current_max/acc/d) + warp `__shfl_down_sync`, RoPE `prev_2i*cos - prev_2i_1*sin`, causalMask `col>row→-HUGE_VALF`, silu `a*sigmoid(a)*b`.
- `main.cpp`: cuBLAS 전치 트릭(`CUBLAS_OP_T,OP_N`, 169–182행), block_table/free_blocks 페이징, is_slot_free/queue continuous batching, CPU argmax, buf_2048_1/2 재사용.
- **prefill용 full softmax와 decode/paged online softmax 2종 공존**이 EP9 Part B의 핵심 내러티브 훅.
- README는 다수 섹션이 TODO 스텁이나 **실제 동작 코드는 kernels.cu/main.cpp에 완성** → 설명은 코드 기준.
- **Llama 3.2 1B 전용**(범용 엔진 아님)임을 SpecChips로 상시 명시.
