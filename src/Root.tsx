import "./index.css";
import { Composition } from "remotion";
import { MyComposition } from "./Composition";
import { VideoEdit, TRIM_START_SEC, TRIM_END_SEC } from "./VideoEdit";

const FPS = 60;

export const RemotionRoot: React.FC = () => {
  const durationInFrames = Math.round((TRIM_END_SEC - TRIM_START_SEC) * FPS);

  return (
    <>
      <Composition
        id="VideoEdit"
        component={VideoEdit}
        durationInFrames={durationInFrames}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
