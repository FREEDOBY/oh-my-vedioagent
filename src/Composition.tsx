import {
  AbsoluteFill,
  Img,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  staticFile,
} from "remotion";

const AIBot: React.FC<{ x: number; y: number; bobOffset: number; legPhase: number }> = ({
  x,
  y,
  bobOffset,
  legPhase,
}) => {
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y + bobOffset,
        width: 80,
        height: 100,
        transform: "translate(-50%, -100%)",
      }}
    >
      {/* Antenna */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: -12,
          width: 3,
          height: 14,
          backgroundColor: "#555",
          transform: "translateX(-50%)",
        }}
      />
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: -18,
          width: 10,
          height: 10,
          borderRadius: "50%",
          backgroundColor: "#FFD700",
          transform: "translateX(-50%)",
          boxShadow: "0 0 8px #FFD700",
        }}
      />

      {/* Head */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 0,
          width: 44,
          height: 36,
          borderRadius: 12,
          backgroundColor: "#E0E0E0",
          border: "2px solid #999",
          transform: "translateX(-50%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {/* Eyes */}
        <div style={{ display: "flex", gap: 8 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              backgroundColor: "#00BFFF",
              boxShadow: "0 0 6px #00BFFF",
            }}
          />
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              backgroundColor: "#00BFFF",
              boxShadow: "0 0 6px #00BFFF",
            }}
          />
        </div>
      </div>

      {/* "AI" label on visor */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 24,
          transform: "translateX(-50%)",
          fontSize: 9,
          fontWeight: "bold",
          color: "#333",
          fontFamily: "monospace",
        }}
      >
        AI
      </div>

      {/* Body */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 38,
          width: 36,
          height: 30,
          borderRadius: 8,
          backgroundColor: "#D0D0D0",
          border: "2px solid #999",
          transform: "translateX(-50%)",
        }}
      />

      {/* Left Arm */}
      <div
        style={{
          position: "absolute",
          left: 8,
          top: 42,
          width: 8,
          height: 24,
          borderRadius: 4,
          backgroundColor: "#C0C0C0",
          border: "1px solid #999",
          transformOrigin: "top center",
          transform: `rotate(${legPhase * -25}deg)`,
        }}
      />

      {/* Right Arm */}
      <div
        style={{
          position: "absolute",
          right: 8,
          top: 42,
          width: 8,
          height: 24,
          borderRadius: 4,
          backgroundColor: "#C0C0C0",
          border: "1px solid #999",
          transformOrigin: "top center",
          transform: `rotate(${legPhase * 25}deg)`,
        }}
      />

      {/* Left Leg */}
      <div
        style={{
          position: "absolute",
          left: 22,
          top: 68,
          width: 8,
          height: 22,
          borderRadius: 4,
          backgroundColor: "#B0B0B0",
          border: "1px solid #888",
          transformOrigin: "top center",
          transform: `rotate(${legPhase * 20}deg)`,
        }}
      />

      {/* Right Leg */}
      <div
        style={{
          position: "absolute",
          right: 22,
          top: 68,
          width: 8,
          height: 22,
          borderRadius: 4,
          backgroundColor: "#B0B0B0",
          border: "1px solid #888",
          transformOrigin: "top center",
          transform: `rotate(${legPhase * -20}deg)`,
        }}
      />
    </div>
  );
};

const SpeechBubble: React.FC<{ x: number; y: number; opacity: number; scale: number }> = ({
  x,
  y,
  opacity,
  scale,
}) => {
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        opacity,
        transform: `translate(-50%, -100%) scale(${scale})`,
        transformOrigin: "bottom center",
      }}
    >
      <div
        style={{
          backgroundColor: "white",
          borderRadius: 16,
          padding: "10px 18px",
          border: "2px solid #666",
          fontSize: 18,
          fontFamily: "'Comic Sans MS', cursive, sans-serif",
          color: "#333",
          whiteSpace: "nowrap",
          position: "relative",
        }}
      >
        Now where is the car?
        {/* Tail */}
        <div
          style={{
            position: "absolute",
            bottom: -14,
            left: 30,
            width: 0,
            height: 0,
            borderLeft: "8px solid transparent",
            borderRight: "8px solid transparent",
            borderTop: "14px solid white",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: -17,
            left: 29,
            width: 0,
            height: 0,
            borderLeft: "9px solid transparent",
            borderRight: "9px solid transparent",
            borderTop: "16px solid #666",
            zIndex: -1,
          }}
        />
      </div>
    </div>
  );
};

export const MyComposition = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();

  // Bot walks from left to right across the scene
  const walkProgress = interpolate(frame, [0, durationInFrames - 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Walking path: left side to right side
  const startX = -60;
  const endX = width + 60;
  const botX = interpolate(walkProgress, [0, 1], [startX, endX]);

  // Y position follows a slight curve (walking on ground)
  const botY = interpolate(
    walkProgress,
    [0, 0.3, 0.5, 0.7, 1],
    [height * 0.82, height * 0.78, height * 0.75, height * 0.72, height * 0.68]
  );

  // Bob up and down while walking
  const bobOffset = Math.sin(frame * 0.6) * 3;

  // Leg/arm swing phase
  const legPhase = Math.sin(frame * 0.6);

  // Speech bubble appears when bot is roughly in the middle
  const bubbleOpacity = interpolate(frame, [40, 50, 110, 120], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const bubbleScale = spring({
    frame: frame - 40,
    fps,
    config: { damping: 12, stiffness: 200 },
  });

  // Caption fade in
  const captionOpacity = interpolate(frame, [100, 115], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const captionSlide = interpolate(frame, [100, 115], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: "#F5F0E8" }}>
      {/* Background image */}
      <Img
        src={staticFile("ChatGPT Image 2026년 3월 22일 오후 12_22_02.png")}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />

      {/* Darken overlay for better visibility of animated elements */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: "rgba(0,0,0,0.15)",
        }}
      />

      {/* Walking AI Bot */}
      <AIBot x={botX} y={botY} bobOffset={bobOffset} legPhase={legPhase} />

      {/* Speech Bubble */}
      {frame >= 40 && (
        <SpeechBubble
          x={botX + 40}
          y={botY - 110 + bobOffset}
          opacity={bubbleOpacity}
          scale={Math.min(bubbleScale, 1)}
        />
      )}

      {/* Bottom Caption */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: "50%",
          transform: `translateX(-50%) translateY(${captionSlide}px)`,
          opacity: captionOpacity,
          backgroundColor: "rgba(255,255,255,0.9)",
          border: "2px solid #666",
          borderRadius: 8,
          padding: "10px 30px",
          fontSize: 28,
          fontWeight: "bold",
          fontFamily: "'Comic Sans MS', cursive, sans-serif",
          color: "#333",
          whiteSpace: "nowrap",
        }}
      >
        Common Sense + Reasoning Failure
      </div>
    </AbsoluteFill>
  );
};
