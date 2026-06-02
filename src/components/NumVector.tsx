import { theme } from "../theme";

// 실제 숫자가 담긴 벡터/행. 부호에 따라 색이 다르고, 한글 태그가 따라붙는다.
// trailingDim: 실제 전체 차원(예: 2048) — "…" 로 생략 표시
export const NumVector: React.FC<{
  values: number[];
  tag?: string;
  tagColor?: string;
  trailingDim?: number | null;
  highlight?: number[];
  format?: (n: number) => string;
  cellW?: number;
  cellH?: number;
  fontSize?: number;
  opacity?: number;
}> = ({
  values,
  tag,
  tagColor = theme.colors.accent,
  trailingDim = null,
  highlight = [],
  format = (n) => (n >= 0 ? "+" : "") + n.toFixed(2),
  cellW = 78,
  cellH = 56,
  fontSize = 24,
  opacity = 1,
}) => {
  const hi = new Set(highlight);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14, opacity }}>
      {tag && (
        <div
          style={{
            minWidth: 110,
            textAlign: "right",
            paddingRight: 6,
            fontSize: 22,
            fontWeight: 700,
            color: tagColor,
            fontFamily: theme.fonts.sans,
          }}
        >
          {tag}
        </div>
      )}
      <div style={{ display: "flex", gap: 6 }}>
        {values.map((v, i) => {
          const pos = v >= 0;
          const tint = pos ? theme.colors.accent : theme.colors.danger;
          const isHi = hi.has(i);
          return (
            <div
              key={i}
              style={{
                width: cellW,
                height: cellH,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: 8,
                fontFamily: theme.fonts.mono,
                fontSize,
                fontWeight: 600,
                color: theme.colors.text,
                backgroundColor: tint + (isHi ? "55" : "22"),
                border: `2px solid ${isHi ? theme.colors.warn : tint + "66"}`,
                boxShadow: isHi ? `0 0 16px ${theme.colors.warn}` : "none",
              }}
            >
              {format(v)}
            </div>
          );
        })}
        {trailingDim !== null && (
          <div
            style={{
              height: cellH,
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              fontFamily: theme.fonts.mono,
              fontSize: 20,
              color: theme.colors.muted,
            }}
          >
            … 총 {trailingDim.toLocaleString()}개
          </div>
        )}
      </div>
    </div>
  );
};
