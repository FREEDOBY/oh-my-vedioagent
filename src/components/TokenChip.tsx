import { theme } from "../theme";

// 토큰 블록 (M1 모티프). 시리즈 전체에서 같은 모양으로 재사용.
export const TokenChip: React.FC<{
  text: string;
  id?: number | string;
  active?: boolean;
  color?: string;
  scale?: number;
  opacity?: number;
}> = ({ text, id, active = false, color = theme.colors.accent, scale = 1, opacity = 1 }) => {
  return (
    <div
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        transform: `scale(${scale})`,
        opacity,
      }}
    >
      <div
        style={{
          padding: "12px 18px",
          borderRadius: 12,
          border: `2px solid ${color}`,
          backgroundColor: active ? color : theme.colors.surface,
          color: active ? theme.colors.bg : theme.colors.text,
          fontFamily: theme.fonts.sans,
          fontSize: 34,
          fontWeight: 700,
          whiteSpace: "nowrap",
          boxShadow: active ? `0 0 24px ${color}88` : "none",
          transition: "none",
        }}
      >
        {text}
      </div>
      {id !== undefined && (
        <div
          style={{
            fontFamily: theme.fonts.mono,
            fontSize: 18,
            color: theme.colors.muted,
          }}
        >
          {id}
        </div>
      )}
    </div>
  );
};
