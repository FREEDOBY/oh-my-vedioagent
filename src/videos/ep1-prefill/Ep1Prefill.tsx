import {
  AbsoluteFill,
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { theme } from "../../theme";
import { ChatPanel } from "../../components/ChatPanel";
import { TokenChip } from "../../components/TokenChip";
import { KVCache } from "../../components/KVCache";
import { ThreadGrid } from "../../components/ThreadGrid";
import { NumVector } from "../../components/NumVector";

// ========================================================================
// EP1 — 질문을 읽는 순간 (Prefill)
// 추적 예시: 사용자 "대한민국 수도는?" → 모델이 첫 토큰 "서"를 뱉기까지
// ========================================================================

const QUESTION = "대한민국 수도는?";

// 설명용 근사 토큰 분할 (실제 BPE 경계와 다를 수 있음 / 숫자는 예시 값)
const TOKENS = [
  { text: "대한", id: 31495 },
  { text: "민국", id: 80052 },
  { text: "수도", id: 28911 },
  { text: "는", id: 16969 },
  { text: "?", id: 30 },
];

// "수도" 토큰의 임베딩 벡터 (예시 값, 2048차원 중 앞 8개만 표시)
const EMBED_SUDO = [0.21, -0.83, 0.44, 0.1, -0.55, 0.92, -0.07, 0.38];

// RMSNorm 실연산 예시 (가독성 위해 4개 원소만)
const RMS_X = [0.5, -1.2, 0.8, 0.3];
const RMS_SQ = RMS_X.map((v) => v * v); // [0.25, 1.44, 0.64, 0.09]
const RMS_MEAN = RMS_SQ.reduce((a, b) => a + b, 0) / RMS_X.length; // 0.605
const RMS_VAL = Math.sqrt(RMS_MEAN + 1e-5); // ≈ 0.778
const RMS_W = [1.2, 0.8, 1.0, 1.1]; // 학습된 가중치
const RMS_NORM = RMS_X.map((v) => v / RMS_VAL);
const RMS_OUT = RMS_NORM.map((v, i) => v * RMS_W[i]);

// 어텐션 실연산 예시 (4차원). "수도"의 Q가 앞 토큰들의 K와 내적.
const Q_SUDO = [0.9, 0.2, 1.1, -0.3];
const K_VECS: Record<string, number[]> = {
  대한: [1.0, 0.3, 1.2, -0.2],
  민국: [0.8, 0.1, 1.0, -0.3],
  수도: [0.4, -0.1, 0.5, 0.2],
};
const ATTN_ORDER = ["대한", "민국", "수도"];
const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
const ATTN_SCORES = ATTN_ORDER.map((t) => dot(Q_SUDO, K_VECS[t])); // [2.34, 1.93, 0.83]
const SQRT_D = Math.sqrt(Q_SUDO.length); // 2
const ATTN_SCALED = ATTN_SCORES.map((s) => s / SQRT_D);
const ATTN_MAX = Math.max(...ATTN_SCALED);
const ATTN_EXP = ATTN_SCALED.map((s) => Math.exp(s - ATTN_MAX));
const ATTN_SUMEXP = ATTN_EXP.reduce((a, b) => a + b, 0);
const ATTN_PROB = ATTN_EXP.map((e) => e / ATTN_SUMEXP);

// 장면 길이 (frames @ 30fps)
const S = {
  intro: 300,
  tokenize: 600,
  prefill: 600,
  numbers: 960, // 토큰 → 숫자 (임베딩)
  rmsnorm: 1140, // 벡터 연산 (RMSNorm 실연산)
  layers: 720, // 16 레이어 (요약)
  kv: 1050,
  attention: 1500, // 어텐션 (실제 내적·softmax 숫자)
  logits: 1050,
  first: 450,
};
export const EP1_DURATION = Object.values(S).reduce((a, b) => a + b, 0);

// ---------- 공통 헬퍼 ----------

const useCursorOn = () => {
  const frame = useCurrentFrame();
  return Math.floor(frame / 15) % 2 === 0;
};

const SceneWrap: React.FC<{ dur: number; children: React.ReactNode; bg?: string }> = ({
  dur,
  children,
  bg,
}) => {
  const frame = useCurrentFrame();
  const op = interpolate(frame, [0, 15, dur - 15, dur], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill style={{ opacity: op, backgroundColor: bg ?? "transparent" }}>
      {children}
    </AbsoluteFill>
  );
};

// 상단 챗 / 하단 엔진 2층 레이아웃
const TwoLayer: React.FC<{
  top: React.ReactNode;
  bottom: React.ReactNode;
  label?: string;
  badge?: { text: string; color: string } | null;
}> = ({ top, bottom, label, badge }) => {
  return (
    <AbsoluteFill style={{ flexDirection: "column" }}>
      <div
        style={{
          height: "30%",
          borderBottom: `2px solid ${theme.colors.border}`,
          backgroundColor: theme.colors.bg,
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 14,
            left: 28,
            fontSize: 18,
            color: theme.colors.muted,
            fontFamily: theme.fonts.mono,
          }}
        >
          사용자가 보는 것
        </div>
        {top}
      </div>
      <div style={{ flex: 1, position: "relative", backgroundColor: "#0a0e14" }}>
        <div
          style={{
            position: "absolute",
            top: 16,
            left: 28,
            fontSize: 18,
            color: theme.colors.muted,
            fontFamily: theme.fonts.mono,
          }}
        >
          엔진 내부 — 진짜 벌어지는 일
        </div>
        {label && (
          <div
            style={{
              position: "absolute",
              top: 44,
              left: 28,
              fontSize: 30,
              fontWeight: 800,
              color: theme.colors.text,
              fontFamily: theme.fonts.sans,
            }}
          >
            {label}
          </div>
        )}
        {badge && (
          <div
            style={{
              position: "absolute",
              top: 24,
              right: 32,
              padding: "10px 20px",
              borderRadius: 999,
              backgroundColor: badge.color,
              color: theme.colors.bg,
              fontSize: 24,
              fontWeight: 800,
              fontFamily: theme.fonts.mono,
              boxShadow: `0 0 20px ${badge.color}88`,
            }}
          >
            {badge.text}
          </div>
        )}
        {bottom}
      </div>
    </AbsoluteFill>
  );
};

const SpecChips = () => (
  <div
    style={{
      position: "absolute",
      bottom: 18,
      right: 28,
      display: "flex",
      gap: 10,
      fontFamily: theme.fonts.mono,
      fontSize: 16,
      color: theme.colors.muted,
    }}
  >
    {["Llama 3.2 1B", "EMBED=2048", "HEAD=64", "Q:32 K:8", "L=16"].map((t) => (
      <span
        key={t}
        style={{
          padding: "4px 10px",
          border: `1px solid ${theme.colors.border}`,
          borderRadius: 6,
        }}
      >
        {t}
      </span>
    ))}
  </div>
);

const Caption: React.FC<{ children: React.ReactNode; bottom?: number }> = ({
  children,
  bottom = 70,
}) => (
  <div
    style={{
      position: "absolute",
      bottom,
      left: "50%",
      transform: "translateX(-50%)",
      maxWidth: "80%",
      textAlign: "center",
      fontSize: 30,
      lineHeight: 1.35,
      fontWeight: 600,
      color: theme.colors.text,
      fontFamily: theme.fonts.sans,
      backgroundColor: "rgba(13,17,23,0.78)",
      padding: "14px 28px",
      borderRadius: 12,
    }}
  >
    {children}
  </div>
);

// ============================ 장면 1: 챗 인트로 ============================
const SceneIntro = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cursorOn = useCursorOn();

  const enterAt = 165;
  const typedCount = Math.floor(interpolate(frame, [30, 130], [0, QUESTION.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }));
  const sent = frame >= enterAt;

  const titleOp = interpolate(frame, [0, 25], [0, 1], { extrapolateRight: "clamp" });
  const hintOp = interpolate(frame, [200, 230], [0, 1], { extrapolateLeft: "clamp" });
  const hintY = interpolate(frame, [200, 230], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const bubblePop = spring({ frame: frame - enterAt, fps, config: { damping: 12 } });

  return (
    <SceneWrap dur={S.intro} bg={theme.colors.bg}>
      <div
        style={{
          position: "absolute",
          top: 70,
          width: "100%",
          textAlign: "center",
          opacity: titleOp,
          fontFamily: theme.fonts.sans,
        }}
      >
        <div style={{ fontSize: 26, color: theme.colors.muted }}>tiny-vllm · EP1</div>
        <div style={{ fontSize: 56, fontWeight: 800, color: theme.colors.text, marginTop: 8 }}>
          질문을 입력하면, 안에선 무슨 일이?
        </div>
      </div>

      <div style={{ position: "absolute", inset: 0, paddingTop: 180 }}>
        <div style={{ transform: sent ? `scale(${Math.min(bubblePop, 1)})` : "scale(1)", transformOrigin: "bottom right", height: "100%" }}>
          <ChatPanel
            messages={sent ? [{ role: "user", text: QUESTION }] : []}
            inputText={sent ? null : QUESTION.slice(0, typedCount)}
            showCursor={!sent}
            cursorOn={cursorOn}
          />
        </div>
      </div>

      {frame >= 200 && (
        <div
          style={{
            position: "absolute",
            bottom: 60,
            width: "100%",
            textAlign: "center",
            opacity: hintOp,
            transform: `translateY(${hintY}px)`,
            fontSize: 32,
            color: theme.colors.accent2,
            fontWeight: 700,
            fontFamily: theme.fonts.sans,
          }}
        >
          ↓ 엔터를 누르면, 엔진 안으로 ↓
        </div>
      )}
    </SceneWrap>
  );
};

// ============================ 장면 2: 토큰화 ============================
const SceneTokenize = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 문장 → 칼선 → 칩 분리
  const cutProgress = interpolate(frame, [60, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <SceneWrap dur={S.tokenize}>
      <TwoLayer
        label="① 토큰화"
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
            <div style={{ display: "flex", gap: interpolate(cutProgress, [0, 1], [4, 40]), alignItems: "flex-start" }}>
              {TOKENS.map((tok, i) => {
                const appear = spring({ frame: frame - 180 - i * 10, fps, config: { damping: 14 } });
                const showId = frame > 240 + i * 12;
                const idRoll = showId
                  ? tok.id
                  : Math.floor(interpolate(frame, [200 + i * 12, 240 + i * 12], [0, tok.id], {
                      extrapolateLeft: "clamp",
                      extrapolateRight: "clamp",
                    }));
                return (
                  <div key={i} style={{ transform: `translateY(${interpolate(Math.min(appear, 1), [0, 1], [20, 0])}px)` }}>
                    <TokenChip
                      text={tok.text}
                      id={frame > 200 ? idRoll : undefined}
                      scale={cutProgress < 0.5 ? 1 : 0.5 + 0.5 * Math.min(appear, 1) + 0.0}
                    />
                  </div>
                );
              })}
            </div>
            {frame > 360 && (
              <Caption bottom={90}>
                글자도 단어도 아닌, <span style={{ color: theme.colors.accent }}>모델만의 조각(토큰)</span> — 각자 고유 번호를 가진다
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 3: 병렬 Prefill ============================
const ScenePrefill = () => {
  const frame = useCurrentFrame();

  // 토큰이 동시에 grid로 낙하 (stagger = 0, 전부 같은 타이밍 → "병렬" 강조)
  const drop = interpolate(frame, [80, 160], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.ease),
  });
  // grid 점화: 컬럼이 좌→우로 쓸려가며 켜짐, 그러나 모든 행이 동시에
  const sweep = interpolate(frame, [180, 420], [0, 64], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const allRows = [0, 1, 2, 3, 4];

  return (
    <SceneWrap dur={S.prefill}>
      <TwoLayer
        label="② 전부 한꺼번에 — Prefill"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill>
            {/* 토큰 행 */}
            <div
              style={{
                position: "absolute",
                top: interpolate(drop, [0, 1], [120, 110]),
                width: "100%",
                display: "flex",
                justifyContent: "center",
                gap: 30,
              }}
            >
              {TOKENS.map((tok, i) => (
                <TokenChip key={i} text={tok.text} active color={theme.colors.accent} scale={0.8} />
              ))}
            </div>

            {/* 동시 낙하 화살표 */}
            {drop > 0 && drop < 1 && (
              <div style={{ position: "absolute", top: 220, width: "100%", display: "flex", justifyContent: "center", gap: 110 }}>
                {allRows.map((r) => (
                  <div key={r} style={{ fontSize: 40, color: theme.colors.accent, opacity: 1 - drop }}>↓</div>
                ))}
              </div>
            )}

            {/* GPU thread 격자 — 5행(토큰) 동시 점화 */}
            <div style={{ position: "absolute", top: 300, left: "50%", transform: "translateX(-50%)", opacity: drop }}>
              <ThreadGrid
                rows={5}
                cols={Math.round(sweep)}
                activeRows={allRows}
                dot={11}
                gap={6}
              />
            </div>

            {frame > 200 && (
              <Caption>
                한 줄씩이 아니라 <span style={{ color: theme.colors.accent }}>전부 동시에</span> GPU에 올린다.
                {" "}프롬프트를 한 번에 처리 = <b>Prefill</b>
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 N1: 토큰 → 숫자 (임베딩) ============================
const SceneNumbers = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const idPop = spring({ frame: frame - 60, fps, config: { damping: 13 } });
  const cellsShown = Math.floor(
    interpolate(frame, [180, 380], [0, EMBED_SUDO.length], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );

  return (
    <SceneWrap dur={S.numbers}>
      <TwoLayer
        label="④ 토큰은 사실 '숫자'다 — 임베딩"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 44 }}>
            {/* 토큰 → ID */}
            <div style={{ display: "flex", alignItems: "center", gap: 30 }}>
              <TokenChip text="수도" active color={theme.colors.warn} scale={0.9} />
              <div style={{ fontSize: 40, color: theme.colors.muted }}>→</div>
              <div style={{ textAlign: "center", transform: `scale(${Math.min(idPop, 1)})` }}>
                <div style={{ fontSize: 18, color: theme.colors.muted, fontFamily: theme.fonts.sans }}>
                  토큰 번호 (ID)
                </div>
                <div style={{ fontSize: 56, fontWeight: 800, color: theme.colors.accent, fontFamily: theme.fonts.mono }}>
                  28911
                </div>
              </div>
            </div>

            <div style={{ fontSize: 26, color: theme.colors.muted }}>
              ↓ 이 번호로 임베딩 표에서 행 하나를 꺼낸다 ↓
            </div>

            {/* ID → 임베딩 벡터 (실수) */}
            {frame > 160 && (
              <NumVector
                values={EMBED_SUDO.slice(0, cellsShown)}
                tag="임베딩 벡터"
                tagColor={theme.colors.accent2}
                trailingDim={cellsShown >= EMBED_SUDO.length ? 2048 : null}
              />
            )}

            {frame > 420 && (
              <Caption bottom={80}>
                토큰 하나 = <span style={{ color: theme.colors.accent2 }}>2048개의 실수(BF16)</span>.
                {" "}이제부터 모든 게 이 숫자들의 곱셈·덧셈이다
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 N2: 벡터 연산 — RMSNorm 실연산 ============================
const fmt = (n: number) => (n >= 0 ? "+" : "") + n.toFixed(2);
const SceneRMSNorm = () => {
  const frame = useCurrentFrame();

  // 단계별 등장
  const show = (t: number) => frame > t;
  const stepOp = (t: number) =>
    interpolate(frame, [t, t + 20], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <SceneWrap dur={S.rmsnorm}>
      <TwoLayer
        label="⑤ 벡터 연산 — 크기 맞추기 (RMSNorm)"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 14, paddingBottom: 150 }}>
            <div style={{ fontSize: 22, color: theme.colors.muted, marginBottom: 2 }}>
              (가독성 위해 2048개 중 4개만)
            </div>

            <NumVector values={RMS_X} tag="입력 x" format={fmt} cellH={46} fontSize={22} opacity={stepOp(20)} />

            {show(120) && (
              <NumVector
                values={RMS_SQ}
                tag="제곱 x²"
                tagColor={theme.colors.warn}
                format={fmt}
                cellH={46}
                fontSize={22}
                opacity={stepOp(120)}
              />
            )}

            {show(240) && (
              <div
                style={{
                  opacity: stepOp(240),
                  fontFamily: theme.fonts.mono,
                  fontSize: 28,
                  color: theme.colors.text,
                  backgroundColor: theme.colors.surface,
                  padding: "12px 24px",
                  borderRadius: 10,
                  border: `1px solid ${theme.colors.border}`,
                }}
              >
                평균 = (0.25+1.44+0.64+0.09)/4 = <b style={{ color: theme.colors.warn }}>0.605</b>
                {"  →  "}
                √0.605 = <b style={{ color: theme.colors.warn }}>0.778</b> (RMS)
              </div>
            )}

            {show(420) && (
              <NumVector
                values={RMS_NORM}
                tag="÷ 0.778"
                tagColor={theme.colors.accent2}
                format={fmt}
                cellH={46}
                fontSize={22}
                opacity={stepOp(420)}
              />
            )}

            {show(560) && (
              <NumVector
                values={RMS_W}
                tag="× 가중치 w"
                tagColor={theme.colors.muted}
                format={fmt}
                cellH={46}
                fontSize={22}
                opacity={stepOp(560)}
              />
            )}

            {show(680) && (
              <NumVector
                values={RMS_OUT}
                tag="결과"
                tagColor={theme.colors.accent}
                format={fmt}
                cellH={46}
                fontSize={22}
                highlight={[0, 1, 2, 3]}
                opacity={stepOp(680)}
              />
            )}

            {show(820) && (
              <Caption bottom={70}>
                각 숫자를 <span style={{ color: theme.colors.warn }}>RMS(0.778)로 나눠</span> 크기를 맞추고,
                {" "}학습된 가중치를 곱한다 — 이게 한 번의 벡터 연산
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 4: 레이어 통과 (줌인) ============================
const LAYER_STEPS = [
  "임베딩",
  "RMSNorm",
  "RoPE(위치)",
  "어텐션",
  "RMSNorm",
  "FFN",
];
const SceneLayers = () => {
  const frame = useCurrentFrame();
  const { width } = useVideoConfig();

  // 한 토큰("수도")으로 줌인
  const zoom = interpolate(frame, [0, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.ease),
  });

  const NUM_LAYERS = 16;
  // 파이프라인 진행: 패킷은 화면 중앙 고정, 파이프라인이 왼쪽으로 슬라이드
  const blockW = 150;
  const gap = 26;
  const step = blockW + gap;
  const totalBlocks = NUM_LAYERS;
  const progress = interpolate(frame, [80, 620], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.ease),
  });
  const currentBlock = progress * (totalBlocks - 1);
  const centerX = width / 2;
  const panX = centerX - blockW / 2 - currentBlock * step;

  // 스쳐 지나가는 내부 디테일 자막
  const subtitles: { t: [number, number]; text: string }[] = [
    { t: [120, 300], text: "방금 본 벡터 연산(RMSNorm·어텐션·FFN)이 한 레이어" },
    { t: [320, 540], text: "출력 2048개 숫자가 다음 레이어의 입력으로" },
    { t: [560, 720], text: "이 숫자 연산을 레이어 16번 반복" },
  ];

  return (
    <SceneWrap dur={S.layers}>
      <TwoLayer
        label="⑥ 같은 숫자 연산 × 16 레이어"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill>
            {/* 파이프라인 (슬라이드) */}
            <div
              style={{
                position: "absolute",
                top: 230,
                left: 0,
                transform: `translateX(${panX}px)`,
                display: "flex",
                gap,
                opacity: zoom,
              }}
            >
              {Array.from({ length: NUM_LAYERS }).map((_, i) => {
                const isDone = i < currentBlock - 0.3;
                const isActive = Math.abs(i - currentBlock) < 0.6;
                const color = isActive
                  ? theme.colors.accent
                  : isDone
                    ? theme.colors.accent2
                    : theme.colors.border;
                return (
                  <div
                    key={i}
                    style={{
                      width: blockW,
                      height: 150,
                      borderRadius: 12,
                      border: `2px solid ${color}`,
                      backgroundColor: isActive ? theme.colors.accent + "22" : theme.colors.surface,
                      boxShadow: isActive ? `0 0 24px ${theme.colors.accent}88` : "none",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 8,
                      color: theme.colors.text,
                      fontFamily: theme.fonts.sans,
                    }}
                  >
                    <div style={{ fontSize: 20, color: theme.colors.muted, fontFamily: theme.fonts.mono }}>
                      Layer {i}
                    </div>
                    <div style={{ fontSize: 16, color: theme.colors.muted }}>
                      {LAYER_STEPS.slice(1).join(" · ")}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* 패킷 (중앙 고정 토큰) */}
            <div
              style={{
                position: "absolute",
                top: 250,
                left: centerX,
                transform: `translateX(-50%) scale(${0.6 + 0.4 * zoom})`,
                zIndex: 5,
              }}
            >
              <TokenChip text="수도" active color={theme.colors.warn} />
            </div>

            {subtitles.map((s, i) =>
              frame >= s.t[0] && frame <= s.t[1] ? (
                <Caption key={i}>{s.text}</Caption>
              ) : null
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 5: KV 캐시 채우기 ============================
const SceneKV = () => {
  const frame = useCurrentFrame();

  // 5개 토큰의 K/V가 캐시로 (병렬이지만 가독성 위해 빠른 연속 + flash)
  const fillRaw = interpolate(frame, [120, 480], [0, 5], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const filled = Math.floor(fillRaw);
  const flashIndex = filled < 5 && fillRaw - filled > 0.0 && fillRaw - filled < 0.5 ? filled : null;

  return (
    <SceneWrap dur={S.kv}>
      <TwoLayer
        label="④ K·V를 저장 — KV 캐시"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 50 }}>
            {/* 토큰 행 */}
            <div style={{ display: "flex", gap: 24 }}>
              {TOKENS.map((tok, i) => (
                <TokenChip
                  key={i}
                  text={tok.text}
                  active={i < filled}
                  color={theme.colors.accent}
                  scale={0.7}
                  opacity={i < fillRaw + 0.5 ? 1 : 0.4}
                />
              ))}
            </div>

            <div style={{ fontSize: 34, color: theme.colors.muted }}>↓ 각 토큰이 K(키)·V(값)를 만들어 저장 ↓</div>

            {/* KV 캐시: 총 12칸(나중에 더 자랄 공간), 지금 5칸 채움 */}
            <KVCache
              total={12}
              filled={filled}
              flashIndex={flashIndex}
              labels={TOKENS.map((t) => t.text)}
              cellW={50}
              cellH={34}
            />

            {frame > 520 && (
              <Caption bottom={80}>
                재계산을 피하려고 <span style={{ color: theme.colors.warn }}>버리지 않고 저장</span>한다.
                {" "}프롬프트 {TOKENS.length}토큰 → 캐시 {TOKENS.length}칸
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 7: 어텐션 (실제 내적·softmax) ============================
const SceneAttention = () => {
  const frame = useCurrentFrame();
  const { width } = useVideoConfig();

  const n = TOKENS.length;
  const stepX = 200;
  const startX = width / 2 - ((n - 1) * stepX) / 2;
  const rowY = 150;
  const xs = TOKENS.map((_, i) => startX + i * stepX);
  const focus = 2; // "수도"

  const arcReveal = interpolate(frame, [60, 240], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const probsShown = frame > 980;

  const stepOp = (t: number) =>
    interpolate(frame, [t, t + 20], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <SceneWrap dur={S.attention}>
      <TwoLayer
        label="⑦ 어텐션 = 실제 내적 + softmax"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel messages={[{ role: "user", text: QUESTION }]} compact />
          </div>
        }
        bottom={
          <AbsoluteFill>
            {/* 토큰 + 어텐션 호 (확률에 비례한 굵기) */}
            <svg style={{ position: "absolute", inset: 0 }} width="100%" height="100%">
              {ATTN_ORDER.map((t, j) => {
                if (j >= focus) return null;
                const x1 = xs[focus];
                const x2 = xs[j];
                const arcH = rowY - 50 - (focus - j) * 40;
                const thickness = probsShown ? 3 + ATTN_PROB[j] * 22 : 5;
                const dashTotal = 600;
                return (
                  <path
                    key={t}
                    d={`M ${x1} ${rowY + 30} Q ${(x1 + x2) / 2} ${arcH} ${x2} ${rowY + 30}`}
                    fill="none"
                    stroke={theme.colors.accent2}
                    strokeWidth={thickness}
                    strokeDasharray={dashTotal}
                    strokeDashoffset={dashTotal * (1 - arcReveal)}
                    opacity={0.9}
                  />
                );
              })}
            </svg>
            {TOKENS.map((tok, i) => (
              <div key={i} style={{ position: "absolute", left: xs[i], top: rowY, transform: "translate(-50%, -50%)" }}>
                <TokenChip
                  text={tok.text}
                  active={i === focus}
                  color={i === focus ? theme.colors.warn : theme.colors.accent}
                  scale={0.62}
                  opacity={i <= focus ? 1 : 0.3}
                />
              </div>
            ))}

            {/* 숫자 연산 패널 */}
            <div style={{ position: "absolute", top: 232, width: "100%", display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
              {frame > 280 && (
                <div style={{ opacity: stepOp(280) }}>
                  <NumVector values={Q_SUDO} tag="Q (수도)" tagColor={theme.colors.warn} format={fmt} cellW={70} cellH={48} fontSize={22} />
                </div>
              )}
              {frame > 380 && (
                <div style={{ opacity: stepOp(380) }}>
                  <NumVector values={K_VECS["대한"]} tag="K (대한)" tagColor={theme.colors.accent} format={fmt} cellW={70} cellH={48} fontSize={22} />
                </div>
              )}
              {frame > 500 && (
                <div
                  style={{
                    opacity: stepOp(500),
                    fontFamily: theme.fonts.mono,
                    fontSize: 25,
                    color: theme.colors.text,
                    backgroundColor: theme.colors.surface,
                    padding: "12px 22px",
                    borderRadius: 10,
                    border: `1px solid ${theme.colors.border}`,
                  }}
                >
                  Q·K = 0.9×1.0 + 0.2×0.3 + 1.1×1.2 + (−0.3)×(−0.2) = <b style={{ color: theme.colors.warn }}>2.34</b>
                  {"  →  ÷√d = "}
                  <b style={{ color: theme.colors.warn }}>1.17</b>
                </div>
              )}

              {/* 점수 → softmax 표 */}
              {frame > 660 && (
                <div style={{ opacity: stepOp(660), marginTop: 8 }}>
                  <table style={{ borderCollapse: "collapse", fontFamily: theme.fonts.mono, fontSize: 24 }}>
                    <thead>
                      <tr style={{ color: theme.colors.muted }}>
                        <th style={{ padding: "6px 24px", textAlign: "left", fontWeight: 500 }}>토큰</th>
                        <th style={{ padding: "6px 24px" }}>점수 ÷√d</th>
                        <th style={{ padding: "6px 24px", color: probsShown ? theme.colors.accent2 : theme.colors.muted }}>
                          확률 (softmax)
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {ATTN_ORDER.map((t, j) => (
                        <tr key={t} style={{ borderTop: `1px solid ${theme.colors.border}` }}>
                          <td style={{ padding: "5px 24px", color: t === "대한" ? theme.colors.accent2 : theme.colors.text, fontWeight: 700 }}>
                            {t}
                          </td>
                          <td style={{ padding: "5px 24px", textAlign: "center", color: theme.colors.text }}>
                            {ATTN_SCALED[j].toFixed(2)}
                          </td>
                          <td style={{ padding: "5px 24px", textAlign: "center" }}>
                            {probsShown ? (
                              <span
                                style={{
                                  color: theme.colors.bg,
                                  backgroundColor: theme.colors.accent2,
                                  padding: "4px 12px",
                                  borderRadius: 6,
                                  fontWeight: 800,
                                }}
                              >
                                {(ATTN_PROB[j] * 100).toFixed(0)}%
                              </span>
                            ) : (
                              <span style={{ color: theme.colors.muted }}>?</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {frame > 1050 && (
              <Caption bottom={32}>
                <span style={{ color: theme.colors.warn }}>'수도'</span>의 주목은{" "}
                <span style={{ color: theme.colors.accent2 }}>대한 44% · 민국 36% · 수도 21%</span>
                {" "}— 이 확률로 V를 가중평균 (미래 토큰은 못 봄)
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 7: logits → argmax → "서" ============================
const VOCAB_BARS = 48;
const barHeight = (i: number) => 18 + ((i * 41 + 13) % 55); // 결정적 의사난수
const WINNER = 12; // "서" 위치
const SceneLogits = () => {
  const frame = useCurrentFrame();
  const { width, fps } = useVideoConfig();

  const grow = interpolate(frame, [60, 180], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // 스캐너가 좌→우로 훑기
  const scan = interpolate(frame, [220, 520], [0, VOCAB_BARS - 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const scanned = Math.floor(scan);
  const winnerFound = frame > 540;
  const winnerPop = spring({ frame: frame - 540, fps, config: { damping: 10 } });

  const barAreaW = width * 0.82;
  const barGap = 4;
  const barW = barAreaW / VOCAB_BARS - barGap;

  return (
    <SceneWrap dur={S.logits}>
      <TwoLayer
        label="⑥ 다음 토큰 예측 — argmax"
        badge={{ text: "PREFILL", color: theme.colors.accent }}
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel
              messages={[
                { role: "user", text: QUESTION },
                ...(winnerFound ? [{ role: "assistant" as const, text: "서" }] : []),
              ]}
              showCursor={winnerFound}
              cursorOn={useCursorOn()}
              compact
            />
          </div>
        }
        bottom={
          <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
            <div style={{ fontSize: 24, color: theme.colors.muted, marginBottom: 16 }}>
              마지막 토큰 '?'의 출력 → 12만 단어 점수(logits)
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "flex-end",
                gap: barGap,
                height: 280,
                width: barAreaW,
              }}
            >
              {Array.from({ length: VOCAB_BARS }).map((_, i) => {
                const isWinner = i === WINNER;
                const h = (isWinner ? 95 : barHeight(i)) * grow * (isWinner && winnerFound ? 1 + 0.05 * Math.min(winnerPop, 1) : 1);
                const isScanned = i <= scanned && !winnerFound;
                const highlight = isScanned && i === scanned;
                return (
                  <div
                    key={i}
                    style={{
                      height: "100%",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "flex-end",
                      position: "relative",
                    }}
                  >
                    {isWinner && winnerFound && (
                      <div
                        style={{
                          position: "absolute",
                          top: -42,
                          fontSize: 30,
                          fontWeight: 800,
                          color: theme.colors.warn,
                        }}
                      >
                        서
                      </div>
                    )}
                    <div
                      style={{
                        width: barW,
                        height: `${Math.max(h, 1)}%`,
                        borderRadius: 4,
                        backgroundColor:
                          isWinner && winnerFound
                            ? theme.colors.warn
                            : highlight
                              ? theme.colors.accent
                              : theme.colors.border,
                        border: `1px solid ${isWinner && winnerFound ? theme.colors.warn : theme.colors.border}`,
                        boxShadow:
                          isWinner && winnerFound ? `0 0 24px ${theme.colors.warn}` : "none",
                      }}
                    />
                  </div>
                );
              })}
            </div>

            {winnerFound && (
              <Caption bottom={70}>
                가장 점수 높은 1등 = <span style={{ color: theme.colors.warn }}>'서'</span> · 첫 글자가 채팅에 찍힌다
              </Caption>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 장면 8: 첫 토큰 (클리프행어) ============================
const SceneFirstToken = () => {
  const frame = useCurrentFrame();
  const cursorOn = useCursorOn();
  const hookOp = interpolate(frame, [150, 200], [0, 1], { extrapolateLeft: "clamp" });

  return (
    <SceneWrap dur={S.first}>
      <TwoLayer
        label="prefill 끝 — 첫 토큰 완성"
        top={
          <div style={{ height: "100%", paddingTop: 30 }}>
            <ChatPanel
              messages={[
                { role: "user", text: QUESTION },
                { role: "assistant", text: "서" },
              ]}
              showCursor
              cursorOn={cursorOn}
              compact
            />
          </div>
        }
        bottom={
          <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 40 }}>
            <KVCache total={12} filled={5} labels={TOKENS.map((t) => t.text)} cellW={50} cellH={34} />
            <div style={{ fontSize: 30, color: theme.colors.muted }}>
              캐시 {TOKENS.length}칸 채움 · 답은 아직 <span style={{ color: theme.colors.warn }}>한 글자</span>
            </div>
            {frame > 150 && (
              <div
                style={{
                  opacity: hookOp,
                  fontSize: 44,
                  fontWeight: 800,
                  color: theme.colors.accent2,
                  fontFamily: theme.fonts.sans,
                }}
              >
                나머지는 어떻게? → 다음 편: Decode
              </div>
            )}
            <SpecChips />
          </AbsoluteFill>
        }
      />
    </SceneWrap>
  );
};

// ============================ 루트 ============================
// 장면 순서 (각 장면 길이는 S 에서). 누적 시작 프레임은 자동 계산.
const SCENE_LIST: { dur: number; Comp: React.FC }[] = [
  { dur: S.intro, Comp: SceneIntro },
  { dur: S.tokenize, Comp: SceneTokenize },
  { dur: S.prefill, Comp: ScenePrefill },
  { dur: S.numbers, Comp: SceneNumbers },
  { dur: S.rmsnorm, Comp: SceneRMSNorm },
  { dur: S.layers, Comp: SceneLayers },
  { dur: S.kv, Comp: SceneKV },
  { dur: S.attention, Comp: SceneAttention },
  { dur: S.logits, Comp: SceneLogits },
  { dur: S.first, Comp: SceneFirstToken },
];

export const Ep1Prefill: React.FC = () => {
  let offset = 0;
  return (
    <AbsoluteFill style={{ backgroundColor: theme.colors.bg }}>
      {SCENE_LIST.map(({ dur, Comp }, i) => {
        const from = offset;
        offset += dur;
        return (
          <Sequence key={i} from={from} durationInFrames={dur}>
            <Comp />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
