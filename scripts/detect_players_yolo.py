# scripts/detect_players_yolo.py

import os
import json
import argparse
from ultralytics import YOLO

IMAGES_DIR_DEFAULT = "data/images"
OUTPUT_PATH_DEFAULT = "data/meta/detections.json"
MODEL_PATH_DEFAULT = "yolov8n.pt"  # you can change to yolov8m.pt or yolov8x.pt


def main():
    ap = argparse.ArgumentParser(description="Run YOLO on images to detect players (persons only).")
    ap.add_argument("--images_dir", default=IMAGES_DIR_DEFAULT,
                    help="Directory of images to run detection on.")
    ap.add_argument("--out", default=OUTPUT_PATH_DEFAULT,
                    help="Where to save detections JSON.")
    ap.add_argument("--model", default=MODEL_PATH_DEFAULT,
                    help="YOLO model path/name (e.g., yolov8n.pt).")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Inference image size (larger helps detect small players).")
    ap.add_argument("--conf", type=float, default=0.20,
                    help="Confidence threshold (lower = more boxes).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = YOLO(args.model)
    detections = {}

    image_files = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".png"))
    )
    if not image_files:
        raise SystemExit(f"No images found in {args.images_dir}")

    print(f"[INFO] Using model={args.model}, imgsz={args.imgsz}, conf={args.conf}")
    print(f"[INFO] Found {len(image_files)} images in {args.images_dir}")

    for fname in image_files:
        img_path = os.path.join(args.images_dir, fname)
        # COCO class 0 = person
        results = model(img_path, imgsz=args.imgsz, conf=args.conf, iou=0.5, classes=[0])

        dets = []
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(float, box[:4])
                dets.append({"bbox": [x1, y1, x2, y2]})

        detections[fname] = dets
        print(f"Processed {fname} ({len(dets)} person detections)")

    with open(args.out, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"[DONE] Saved detections to {args.out}")


if __name__ == "__main__":
    main()
