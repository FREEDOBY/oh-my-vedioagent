import { AbsoluteFill } from "remotion";
import { theme } from "../theme";

// 모든 영상이 공유하는 기본 배경. 자식 요소를 그 위에 올립니다.
export const Background: React.FC<{ children?: React.ReactNode }> = ({
  children,
}) => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: theme.colors.bg,
        color: theme.colors.text,
        fontFamily: theme.fonts.sans,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};
