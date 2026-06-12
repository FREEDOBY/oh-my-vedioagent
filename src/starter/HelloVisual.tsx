import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { theme } from "../theme";
import { NumVector } from "../components/NumVector";

// ============================================================================
// 최소 스타터 — "새 영상은 이렇게 시작한다"
// ----------------------------------------------------------------------------
// 핵심 패턴 4가지만 보여줍니다. 이걸 복사해 내 주제로 바꿔 나가세요.
//   1) useCurrentFrame() 으로 현재 프레임을 읽어 모든 애니메이션을 만든다
//   2) interpolate / spring 으로 프레임 → 값(투명도·이동·크기) 매핑
//   3) theme 토큰으로 색·폰트 통일
//   4) src/components 의 재사용 컴포넌트(NumVector 등)를 가져다 쓴다
// 더 풍부한 예시는 src/examples/ep1-prefill 참고.
// ============================================================================

export const HELLO_DURATION = 150; // 프레임 (5초 @30fps)

export const HelloVisual: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 1) 제목 페이드 인 (interpolate: frame 0→20 일 때 opacity 0→1)
  const titleOp = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: "clamp" });

  // 2) 부제 살짝 위로 떠오르기 (spring: 자연스러운 가속/감속)
  const pop = spring({ frame: frame - 25, fps, config: { damping: 14 } });
  const subY = interpolate(Math.min(pop, 1), [0, 1], [20, 0]);

  // 3) 벡터 셀이 하나씩 등장 (frame 50~110 동안 0→6개)
  const cells = Math.floor(
    interpolate(frame, [50, 110], [0, 6], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );
  const VALUES = [0.42, -1.1, 0.8, -0.3, 1.2, -0.6];

  return (
    <AbsoluteFill
      style={{
        backgroundColor: theme.colors.bg,
        alignItems: "center",
        justifyContent: "center",
        gap: 40,
        fontFamily: theme.fonts.sans,
      }}
    >
      <div style={{ textAlign: "center", opacity: titleOp }}>
        <div style={{ fontSize: 26, color: theme.colors.muted }}>starter</div>
        <div style={{ fontSize: 64, fontWeight: 800, color: theme.colors.text, marginTop: 8 }}>
          여기서 시작하세요
        </div>
      </div>

      <div
        style={{
          fontSize: 30,
          color: theme.colors.accent,
          opacity: Math.min(pop, 1),
          transform: `translateY(${subY}px)`,
        }}
      >
        프레임 → 값으로 모든 게 움직인다
      </div>

      {/* 재사용 컴포넌트: 셀이 하나씩 나타남 */}
      {frame > 50 && (
        <NumVector values={VALUES.slice(0, cells)} tag="예시 벡터" tagColor={theme.colors.accent2} />
      )}
    </AbsoluteFill>
  );
};
