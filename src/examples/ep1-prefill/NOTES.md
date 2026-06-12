# EP1-Prefill — 이 예시가 보여주는 패턴

`Ep1Prefill.tsx` 는 길지만, 뜯어보면 **재사용 가능한 패턴 몇 개의 반복**입니다.
내 주제로 옮길 때 아래만 이해하면 됩니다.

## 1. 씬 길이를 한 곳에서 관리 — `S` 객체 + `SCENE_LIST`

```ts
const S = { intro: 300, tokenize: 600, ... };          // 씬별 프레임 길이
export const EP1_DURATION = Object.values(S).reduce((a,b)=>a+b, 0);  // 총 길이 자동
const SCENE_LIST = [{ dur: S.intro, Comp: SceneIntro }, ...];        // 순서
```

- 누적 시작 프레임은 루트에서 자동 계산(`<Sequence from={offset}>`). 씬 순서·길이만 바꾸면 됨.
- QA 에이전트도 이 `S`/`SCENE_LIST` 를 읽어 "몇 번째 씬이 몇 프레임부터인지" 계산한다.

## 2. 씬 진입/퇴장 페이드 — `SceneWrap`

모든 씬을 감싸 `frame` 기준으로 앞 15f 페이드인 / 뒤 15f 페이드아웃. 전환이 빈 화면으로 깜빡이지 않게.

## 3. 2층 레이아웃 — `TwoLayer`

상단 22% "사용자가 보는 것(채팅)" / 하단 "엔진 내부". **추상(사용자) ↔ 실제(내부)** 를 한 화면에 대비시키는 장치.
내 주제에 맞으면 재사용, 아니면 안 써도 됨.

## 4. 모든 애니메이션은 `frame → 값`

`interpolate(frame, [입력구간], [출력구간], {clamp})` 와 `spring({frame, fps})` 두 개로 거의 다 만든다.
**절대 `Date.now()`/`Math.random()`/`setTimeout` 쓰지 말 것** — 프레임 순수성이 깨져 렌더가 깜빡인다.

## 5. "실제 숫자 연산"을 보여준다 — 정확성 우선

토큰 ID, 임베딩, RMSNorm, 어텐션 softmax 같은 값을 **진짜 계산해서**(`NumVector`/`NumMatrix`) 보여준다.
숫자는 예시지만 연산 규칙은 실제와 같아야 한다. 단순화한 부분은 화면에 명시(`(2048개 중 4개만)`).

## 6. 진행 맥락 — 상단 진행바(`PrefillProgress`) + `SpecChips`

시청자가 "지금 전체 흐름 중 어디인지" 늘 알 수 있게. 긴 영상일수록 중요.

---

## 갈아끼울 때 체크리스트

- [ ] `QUESTION`/`TOKENS`/예시 숫자 → 내 주제 값으로
- [ ] `S` 객체에서 씬 추가/삭제, `SCENE_LIST` 순서 조정
- [ ] `SpecChips` 의 모델 스펙 칩 → 내 주제 메타데이터로
- [ ] 도메인 정확성은 전용 전문가 에이전트로 검증 (EP1은 `llm-expert` 사용)
- [ ] 수정 후 `video-visual-qa` / `motion-flow-qa` 로 렌더 점검
