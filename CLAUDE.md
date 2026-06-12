# CLAUDE.md — 시각화 작업 플레이북

이 저장소는 **Claude Code로 어떤 자료든 모션 그래픽 설명 영상으로 바꾸는 템플릿**이다.
(Remotion + React. 완성 영상 하나가 아니라 "영상을 만드는 작업 환경".)
사용자가 "이 자료 시각화해줘" 류의 요청을 하면, 아래 절차와 규칙을 따른다.

---

## 새 주제를 시각화하는 절차

1. **개념 추출** — 자료를 읽고 `docs/concept-inventory.md` **형식**으로 새 인벤토리를 만든다.
   각 개념: 정의 / 핵심 / 난이도 / **"어떻게 시각적으로 보여줄지"** 아이디어.
2. **정확성 검증** — 도메인 전문가 에이전트로 기술 내용을 검증한다.
   - 예시(EP1)는 `llm-expert` 사용. **새 주제면 그 도메인 전문가 에이전트를 새로 만들어** 쓴다
     (`.claude/agents/<topic>-expert.md`). 부정확한 비유·틀린 수식은 시각화 전에 잡는다.
3. **시나리오 설계** — `narrative-architect` 에이전트로 에피소드/씬 단위 구성과 연출을 짠다.
4. **구현** — `src/components` 의 재사용 컴포넌트와 `theme` 토큰으로 씬을 만든다.
   새 영상은 `src/<topic>/` 아래, 컴포지션은 `src/Root.tsx` 에 `<Composition>` 등록.
5. **자가검증 루프** — 수정 후 반드시:
   - `video-visual-qa` : 렌더한 정지프레임으로 잘림·오버플로·여백·대비 점검
   - `motion-flow-qa` : 애니메이션 타이밍·전환·등장/퇴장 점검
   - 직접 확인하려면 `npx remotion still <id> out/check.png --frame=N` 로 프레임을 떠서 Read.

> 처음 시작은 `src/starter/HelloVisual.tsx`(최소 패턴), 풍부한 참조는
> `src/examples/ep1-prefill/`(+ 옆의 `NOTES.md`).

---

## 반드시 지킬 규칙

- **프레임 순수성**: 모든 애니메이션은 `useCurrentFrame()` 기반. `Date.now()`·`Math.random()`·
  `setTimeout`·CSS `transition` 금지(렌더 깜빡임 원인). 값은 `interpolate`/`spring` 으로만.
- **theme 사용**: 색·폰트는 `src/theme.ts` 토큰만. 하드코딩 색 지양(시리즈 톤 일관성).
- **씬 길이 관리**: 씬별 길이는 `const S = {...}` 한 곳에, 순서는 `SCENE_LIST`, 총 길이는
  `Object.values(S).reduce(...)` 로 자동. 누적 시작 프레임은 `<Sequence from>` 으로.
- **정확성 우선**: 보여주는 숫자·수식·구조는 실제 동작과 맞아야 한다. 단순화한 부분은
  화면에 명시(예: `(2048개 중 4개만)`). 추측으로 채우지 말고 출처/전문가로 확인.
- **규격**: 1920×1080 / 30fps (`VIDEO_CONFIG`). 텍스트 잘림 방지 — 긴 한글은 `whiteSpace`·여백 확인.
- **수정 후 점검**: `npm run lint`(eslint + tsc) 통과. 시각/모션은 QA 에이전트로.

---

## 디렉터리 지도

| 경로 | 역할 | 새 주제에서 |
|---|---|---|
| `src/Root.tsx` | 컴포지션 등록(사이드바 목록) | 새 `<Composition>` 추가 |
| `src/theme.ts` | 공유 디자인 토큰 | 그대로 사용 |
| `src/components/` | 재사용 시각 어휘 | 그대로 사용 + 필요시 추가 |
| `src/starter/` | 최소 스타터 | 복사해 시작점으로 |
| `src/examples/ep1-prefill/` | 완성 참조 예시 | 패턴 참고(`NOTES.md`) |
| `docs/concept-inventory.md` | 개념·시각화 인벤토리 | **형식 복제**해 새로 작성 |
| `.claude/agents/` | 전문 에이전트 | `*-qa`·`narrative-architect` 재사용, 도메인 전문가는 신규 |
| `ref/` | 예시의 출처 자료 | 교체/삭제 |

---

## 흔한 작업 명령

```bash
npm run dev                                  # 스튜디오 미리보기 (코드 실시간)
npm run lint                                 # eslint + tsc
npx remotion still <id> out/x.png --frame=N  # 특정 프레임 떠서 점검
npx remotion render <id> out/x.mp4           # 영상 파일 렌더 (out/ 은 gitignore)
```
