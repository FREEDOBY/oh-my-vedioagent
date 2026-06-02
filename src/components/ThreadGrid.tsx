import { theme } from "../theme";

// GPU thread 격자 (M4 모티프). 각 행(row)이 하나의 토큰을 처리하는 thread 묶음.
// activeRows: 점화된(실행 중) 행의 집합. prefill=여러 행 동시, decode=한 행만.
export const ThreadGrid: React.FC<{
  rows: number;
  cols: number;
  activeRows: number[]; // 켜진 행 인덱스
  rowColors?: Record<number, string>; // 행별 색 (기본 accent)
  dot?: number;
  gap?: number;
}> = ({ rows, cols, activeRows, rowColors = {}, dot = 12, gap = 8 }) => {
  const activeSet = new Set(activeRows);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: gap + 2 }}>
      {Array.from({ length: rows }).map((_, r) => {
        const on = activeSet.has(r);
        const color = rowColors[r] ?? theme.colors.accent;
        return (
          <div key={r} style={{ display: "flex", gap }}>
            {Array.from({ length: cols }).map((_, c) => {
              // 32개마다 warp 경계 살짝 띄우기
              const warpGap = c > 0 && c % 32 === 0 ? 10 : 0;
              return (
                <div
                  key={c}
                  style={{
                    marginLeft: warpGap,
                    width: dot,
                    height: dot,
                    borderRadius: "50%",
                    backgroundColor: on ? color : theme.colors.surface,
                    border: `1px solid ${on ? color : theme.colors.border}`,
                    boxShadow: on ? `0 0 8px ${color}` : "none",
                  }}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
};
