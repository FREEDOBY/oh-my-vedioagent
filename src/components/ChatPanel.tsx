import { theme } from "../theme";

export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

// 챗 UI (상단 레이어). 사용자가 보는 화면.
// showCursor: 마지막 메시지 끝에 깜빡이는 커서를 표시할지
export const ChatPanel: React.FC<{
  messages: ChatMessage[];
  showCursor?: boolean;
  cursorOn?: boolean;
  inputText?: string | null; // 입력창에 타이핑 중인 텍스트 (null이면 입력창 숨김)
  compact?: boolean;
}> = ({ messages, showCursor = false, cursorOn = true, inputText = null, compact = false }) => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: compact ? 12 : 20,
        padding: compact ? 24 : 40,
        height: "100%",
        justifyContent: "flex-end",
        fontFamily: theme.fonts.sans,
      }}
    >
      {messages.map((m, i) => {
        const isUser = m.role === "user";
        const isLast = i === messages.length - 1;
        return (
          <div
            key={i}
            style={{
              alignSelf: isUser ? "flex-end" : "flex-start",
              maxWidth: "72%",
            }}
          >
            <div
              style={{
                fontSize: compact ? 18 : 22,
                color: theme.colors.muted,
                marginBottom: 6,
                textAlign: isUser ? "right" : "left",
              }}
            >
              {isUser ? "사용자" : "LLM"}
            </div>
            <div
              style={{
                padding: compact ? "14px 20px" : "20px 28px",
                borderRadius: 18,
                fontSize: compact ? 28 : 40,
                fontWeight: 600,
                lineHeight: 1.3,
                backgroundColor: isUser ? theme.colors.accent : theme.colors.surface,
                color: isUser ? theme.colors.bg : theme.colors.text,
                border: isUser ? "none" : `2px solid ${theme.colors.border}`,
                borderBottomRightRadius: isUser ? 4 : 18,
                borderBottomLeftRadius: isUser ? 18 : 4,
              }}
            >
              {m.text}
              {isLast && !isUser && showCursor && (
                <span style={{ opacity: cursorOn ? 1 : 0, color: theme.colors.accent2 }}>▌</span>
              )}
            </div>
          </div>
        );
      })}

      {inputText !== null && (
        <div
          style={{
            marginTop: 8,
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "16px 24px",
            borderRadius: 14,
            border: `2px solid ${theme.colors.accent}`,
            backgroundColor: theme.colors.surface,
          }}
        >
          <span style={{ fontSize: 30, color: theme.colors.text }}>{inputText}</span>
          {showCursor && (
            <span style={{ opacity: cursorOn ? 1 : 0, fontSize: 30, color: theme.colors.accent }}>
              ▌
            </span>
          )}
          <div style={{ flex: 1 }} />
          <div
            style={{
              padding: "8px 16px",
              borderRadius: 10,
              backgroundColor: theme.colors.accent,
              color: theme.colors.bg,
              fontSize: 22,
              fontWeight: 700,
            }}
          >
            전송
          </div>
        </div>
      )}
    </div>
  );
};
