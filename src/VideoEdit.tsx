import { AbsoluteFill, OffthreadVideo, useCurrentFrame, useVideoConfig, staticFile, Sequence } from "remotion";

// 자막 데이터 - 여기에 자막을 추가하세요
// startFrame/endFrame은 트리밍 후 기준 프레임입니다
export const subtitles: { text: string; startFrame: number; endFrame: number }[] = [
  // 예시: { text: "안녕하세요", startFrame: 0, endFrame: 300 },
];

// 트리밍 설정 (초 단위)
export const TRIM_START_SEC = 0; // 시작 지점 (초)
export const TRIM_END_SEC = 3407; // 끝 지점 (초)

export const VideoEdit: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // 현재 프레임에 해당하는 자막 찾기
  const currentSubtitle = subtitles.find(
    (s) => frame >= s.startFrame && frame <= s.endFrame
  );

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      <OffthreadVideo
        src={staticFile("2026-03-28 10-46-36.mkv")}
        startFrom={TRIM_START_SEC * fps}
      />

      {/* 자막 표시 */}
      {currentSubtitle && (
        <div
          style={{
            position: "absolute",
            bottom: 80,
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "rgba(0, 0, 0, 0.75)",
            color: "white",
            padding: "12px 24px",
            borderRadius: 8,
            fontSize: 36,
            fontWeight: "bold",
            fontFamily: "'Noto Sans KR', sans-serif",
            textAlign: "center",
            maxWidth: "80%",
          }}
        >
          {currentSubtitle.text}
        </div>
      )}
    </AbsoluteFill>
  );
};
