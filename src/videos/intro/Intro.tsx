import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Background } from "../../components/Background";
import { theme } from "../../theme";

// === 영상 1: 인트로 (예시 / 패턴 데모) ===
// 새 영상을 만들 때 이 파일을 참고하세요:
//  1) src/videos/<이름>/ 폴더 생성
//  2) 여기처럼 컴포넌트 export
//  3) Root.tsx 의 <Folder> 안에 <Composition> 추가

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 제목 등장 (spring)
  const titleScale = spring({ frame, fps, config: { damping: 14 } });
  const subtitleOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <Background>
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 24,
        }}
      >
        <div
          style={{
            fontSize: 110,
            fontWeight: 800,
            fontFamily: theme.fonts.mono,
            color: theme.colors.accent,
            transform: `scale(${titleScale})`,
          }}
        >
          tiny-vllm
        </div>
        <div
          style={{
            fontSize: 40,
            color: theme.colors.muted,
            opacity: subtitleOpacity,
          }}
        >
          C++ &amp; CUDA로 만드는 LLM 추론 엔진
        </div>
      </div>
    </Background>
  );
};
