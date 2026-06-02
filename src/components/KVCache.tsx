import { theme } from "../theme";

// KV 캐시 보관함 (M6 모티프). 토큰마다 K(파랑)/V(초록) 칸 한 쌍이 쌓인다.
// filled: 채워진 토큰 수, flashIndex: 방금 추가되어 깜빡일 칸 인덱스(노랑)
export const KVCache: React.FC<{
  total: number; // 표시할 총 슬롯 수
  filled: number; // 채워진 슬롯 수 (소수 가능: 진행 중 애니메이션용)
  labels?: string[]; // 각 슬롯 아래 라벨(토큰 텍스트)
  flashIndex?: number | null;
  cellW?: number;
  cellH?: number;
  gap?: number;
}> = ({ total, filled, labels = [], flashIndex = null, cellW = 44, cellH = 30, gap = 6 }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
        <span style={{ width: 28, color: theme.colors.accent, fontFamily: theme.fonts.mono, fontSize: 20, fontWeight: 700 }}>K</span>
        <div style={{ display: "flex", gap }}>
          {Array.from({ length: total }).map((_, i) => {
            const isFilled = i < filled;
            const isFlash = i === flashIndex;
            return (
              <div
                key={i}
                style={{
                  width: cellW,
                  height: cellH,
                  borderRadius: 6,
                  border: `2px solid ${isFilled ? theme.colors.accent : theme.colors.border}`,
                  backgroundColor: isFlash
                    ? theme.colors.warn
                    : isFilled
                      ? theme.colors.accent + "33"
                      : "transparent",
                  boxShadow: isFlash ? `0 0 18px ${theme.colors.warn}` : "none",
                }}
              />
            );
          })}
        </div>
      </div>
      <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
        <span style={{ width: 28, color: theme.colors.accent2, fontFamily: theme.fonts.mono, fontSize: 20, fontWeight: 700 }}>V</span>
        <div style={{ display: "flex", gap }}>
          {Array.from({ length: total }).map((_, i) => {
            const isFilled = i < filled;
            const isFlash = i === flashIndex;
            return (
              <div
                key={i}
                style={{
                  width: cellW,
                  height: cellH,
                  borderRadius: 6,
                  border: `2px solid ${isFilled ? theme.colors.accent2 : theme.colors.border}`,
                  backgroundColor: isFlash
                    ? theme.colors.warn
                    : isFilled
                      ? theme.colors.accent2 + "33"
                      : "transparent",
                  boxShadow: isFlash ? `0 0 18px ${theme.colors.warn}` : "none",
                }}
              />
            );
          })}
        </div>
      </div>
      {labels.length > 0 && (
        <div style={{ display: "flex", gap, marginLeft: 46 }}>
          {Array.from({ length: total }).map((_, i) => (
            <div
              key={i}
              style={{
                width: cellW,
                textAlign: "center",
                fontSize: 14,
                color: i < filled ? theme.colors.muted : "transparent",
                fontFamily: theme.fonts.sans,
                whiteSpace: "nowrap",
                overflow: "visible",
              }}
            >
              {labels[i] ?? ""}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
