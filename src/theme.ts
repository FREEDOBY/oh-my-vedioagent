// 모든 영상이 공유하는 디자인 토큰 (색상 / 폰트 / 공통 설정)
// 새 영상을 만들 때 이 값을 사용하면 시리즈 전체 톤이 일관됩니다.

export const theme = {
  colors: {
    bg: "#0d1117", // 기본 배경 (다크)
    surface: "#161b22", // 카드 / 패널 배경
    border: "#30363d",
    text: "#e6edf3", // 기본 텍스트
    muted: "#8b949e", // 보조 텍스트
    accent: "#58a6ff", // 강조 (파랑)
    accent2: "#3fb950", // 보조 강조 (초록)
    warn: "#d29922", // 경고 / 포인트 (노랑)
    danger: "#f85149",
  },
  fonts: {
    sans: "'Inter', 'Malgun Gothic', 'Noto Sans KR', system-ui, -apple-system, sans-serif",
    mono: "'JetBrains Mono', 'Consolas', 'D2Coding', monospace",
  },
} as const;

// 시리즈 공통 영상 규격
export const VIDEO_CONFIG = {
  fps: 30,
  width: 1920,
  height: 1080,
} as const;
