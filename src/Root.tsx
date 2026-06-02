import "./index.css";
import { Composition, Folder } from "remotion";
import { VIDEO_CONFIG } from "./theme";
import { Ep1Prefill, EP1_DURATION } from "./videos/ep1-prefill/Ep1Prefill";

// 각 영상은 독립적인 <Composition> 입니다. 따로 미리보기/렌더링됩니다.
//   npx remotion render Ep1-Prefill
//
// 시나리오: docs/series-scenario.md (Prefill / Decode / KV캐시 3편)

export const RemotionRoot: React.FC = () => {
  return (
    <Folder name="tiny-vllm">
      <Composition
        id="Ep1-Prefill"
        component={Ep1Prefill}
        durationInFrames={EP1_DURATION}
        fps={VIDEO_CONFIG.fps}
        width={VIDEO_CONFIG.width}
        height={VIDEO_CONFIG.height}
      />
    </Folder>
  );
};
