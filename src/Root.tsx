import "./index.css";
import { Composition, Folder } from "remotion";
import { VIDEO_CONFIG } from "./theme";
import { Intro } from "./videos/intro/Intro";

// 각 영상은 독립적인 <Composition> 입니다.
// 따로 미리보기 / 렌더링됩니다:  npx remotion render Intro
//
// 새 영상 추가하는 법:
//  1) src/videos/<이름>/<이름>.tsx 에 컴포넌트 작성
//  2) 아래 <Folder> 안에 <Composition> 한 줄 추가
//     (durationInFrames 는 영상마다 따로 지정)

export const RemotionRoot: React.FC = () => {
  return (
    <Folder name="tiny-vllm">
      <Composition
        id="Intro"
        component={Intro}
        durationInFrames={150}
        fps={VIDEO_CONFIG.fps}
        width={VIDEO_CONFIG.width}
        height={VIDEO_CONFIG.height}
      />
    </Folder>
  );
};
