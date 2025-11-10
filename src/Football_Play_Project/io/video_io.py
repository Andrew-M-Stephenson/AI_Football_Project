from pathlib import Path
import cv2

def iterate_frames(video_path: str | Path, fps_sample: int = 10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps / fps_sample)), 1)
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        yield frame_idx, frame
    cap.release()
