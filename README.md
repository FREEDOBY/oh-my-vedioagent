# oh-my-videoagent

> **Claude Code로 "어떤 자료든" 모션 그래픽 설명 영상으로 바꾸는 템플릿.**
> 3Blue1Brown · Fireship 스타일의 시각화를, 영상 편집 없이 **코드 + AI 에이전트**로 만듭니다.

이 저장소는 **완성된 영상 한 편이 아니라, 영상을 "만드는 작업 환경"** 입니다.
재사용 컴포넌트 · 디자인 토큰 · 전문 에이전트 · 자가검증(QA) 루프가 세팅돼 있어,
포크한 뒤 Claude Code에게 *"이 자료 시각화해줘"* 라고 시키면 이 발판 위에서 영상이 굴러갑니다.

---

## ⚙️ 전제

- **[Claude Code](https://claude.com/claude-code)** 가 필요합니다. (이 템플릿의 핵심은 `.claude/agents` 와 작업 플레이북)
- **Node.js 18+** — 미리보기/렌더는 Node만 있으면 어디서나 됩니다. (GPU 불필요)

---

## 🚀 빠른 시작

```bash
# 1) 이 저장소에서 "Use this template" → 내 저장소 생성 후 클론
git clone https://github.com/<you>/<your-repo>.git
cd <your-repo>

# 2) 의존성 설치
npm i

# 3) 스튜디오 실행 → 브라우저 http://localhost:3000
npm run dev
```

스튜디오 왼쪽 사이드바에서 **`tiny-vllm > Ep1-Prefill`** 을 열면 **완성 예시**가 재생됩니다.
(사이드바 항목 = `src/Root.tsx`에 등록된 컴포지션. 코드를 실시간으로 그린 미리보기라 항상 최신입니다.)

---

## 📺 완성 예시 — EP1: Prefill

`src/examples/ep1-prefill/Ep1Prefill.tsx` — *"질문을 입력하면 LLM 추론 엔진 안에서 무슨 일이 벌어지나"* 를
토큰화 → Prefill → 모델 구조 → 임베딩 → RMSNorm → K·V 생성 → 어텐션 → KV 캐시 → logits → 첫 토큰
순서로, **실제 숫자 연산까지** 애니메이션으로 보여줍니다.

> 포크한 사람의 동선: **이 예시를 먼저 구경(목표 확인) → 패턴을 참고 → 내 주제로 새 영상 제작.**

---

## 🧩 내 자료를 시각화하기 — Claude Code 플레이북

Claude Code 세션에서 이렇게 시키세요. 에이전트들이 알아서 협업합니다.

```
1) "이 자료 읽고 docs/concept-inventory.md 형식으로 개념·시각화 아이디어 정리해줘"
2) "내용이 기술적으로 정확한지 도메인 전문가 관점에서 검증해줘"   → 도메인 전문가 에이전트
3) "narrative-architect로 에피소드·씬 시나리오를 설계해줘"        → 내러티브 설계
4) "src/components 재사용해서 씬을 구현해줘"                       → 구현
5) "video-visual-qa / motion-flow-qa로 렌더해서 점검·수정해줘"     → 자가검증 루프
```

핵심은 **5번의 QA 루프** — 에이전트가 직접 프레임을 렌더해 보고 잘림·여백·타이밍을 고칩니다.

---

## 📂 프로젝트 구조

```
src/
├─ Root.tsx              # 컴포지션 등록 (사이드바에 뜨는 목록)
├─ theme.ts              # 공유 디자인 토큰 (색·폰트·해상도)
├─ components/           # 재사용 시각 어휘 (TokenChip·NumVector·KVCache·ThreadGrid·ChatPanel …)
└─ examples/
   └─ ep1-prefill/       # 플래그십 완성 예시 (참조용)
docs/
├─ concept-inventory.md  # 개념·시각화 아이디어 (새 주제는 이 양식 복제)
└─ series-scenario.md    # 시리즈 시나리오
.claude/agents/          # ⭐ 전문 에이전트 (이 템플릿의 핵심)
├─ narrative-architect   # 기술 개념 → 에피소드·씬 시나리오 설계
├─ video-visual-qa       # 렌더 → 정지프레임 보고 시각 문제 점검
├─ motion-flow-qa        # 애니메이션 타이밍·전환 점검
└─ llm-expert            # (예시용 도메인 전문가 — 새 주제는 갈아끼우기)
.agents/skills/          # Remotion 모범사례 등 작업 스킬
ref/                     # 예시(EP1)의 출처 자료 — 새 주제선 교체/삭제
```

> **재사용 발판** = `components` · `theme` · `.claude/agents`(narrative-architect, *-qa) · `.agents/skills`
> **갈아끼울 부분** = `examples/*` · `docs/*` · `ref/*` · `llm-expert`(도메인 전문가)

---

## 🎬 영상 파일(mp4)로 뽑기

스튜디오 미리보기와 별개로, 실제 파일이 필요하면 렌더합니다.

```bash
npx remotion render Ep1-Prefill out/ep1-prefill.mp4
```

- `out/` 은 `.gitignore` 처리돼 있어 렌더 결과물은 저장소에 안 올라갑니다.
- 첫 인자 = 컴포지션 id(`Root.tsx`의 `<Composition id="…">`), 둘째 인자 = 출력 경로.

---

## 🛠 명령어

| 명령 | 설명 |
|---|---|
| `npm run dev` | Remotion 스튜디오(미리보기) 실행 |
| `npm run lint` | ESLint + 타입체크 |
| `npx remotion render <id> <out>` | 영상 파일 렌더 |
| `npm run upgrade` | Remotion 버전 업그레이드 |

---

## 📄 크레딧 / 라이선스

[Remotion](https://www.remotion.dev) 기반. 일부 기업 사용은 별도 라이선스가 필요합니다 —
[Remotion 라이선스 약관](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md) 확인.
