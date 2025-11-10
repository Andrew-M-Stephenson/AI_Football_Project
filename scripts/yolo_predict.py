# scripts/yolo_predict.py
import argparse
from pathlib import Path

from Football_Play_Project.io.video_io import iterate_frames
from Football_Play_Project.io.yolo_io import YoloRunner
from Football_Play_Project.io.annotations import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="data/meta/game1_people_dets.json")
    args = ap.parse_args()

    # COCO class 0 = person
    yolo = YoloRunner(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        allowed_classes=[0],  # people only
    )

    result = {"video": args.video, "frames": []}

    for idx, frame in iterate_frames(args.video, fps_sample=5):
        dets = yolo.infer_frame(frame)
        result["frames"].append({"idx": idx, "detections": dets})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_json(result, args.out)
    print(f"Saved person detections → {args.out}")

if __name__ == "__main__":
    main()
