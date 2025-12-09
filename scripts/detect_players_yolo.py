# scripts/detect_players_yolo.py

import os
import json
import argparse
from ultralytics import YOLO

ROOT_DIR_DEFAULT = "data/images/Plays"
OUTPUT_PATH_DEFAULT = "data/meta/plays_detections.json"
MODEL_PATH_DEFAULT = "yolov8n.pt"


def main():
    ap = argparse.ArgumentParser(
        description="Run YOLO on play images (recursively) to detect players (persons only)."
    )
    ap.add_argument(
        "--root",
        default=ROOT_DIR_DEFAULT,
        help="Root directory containing play folders (Deep_Pass, Short_Pass, etc.).",
    )
    ap.add_argument(
        "--out",
        default=OUTPUT_PATH_DEFAULT,
        help="Where to save detections JSON.",
    )
    ap.add_argument(
        "--model",
        default=MODEL_PATH_DEFAULT,
        help="YOLO model path/name (e.g., yolov8n.pt).",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size (larger helps with small players).",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold (lower = more boxes).",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = YOLO(args.model)
    detections = {}

    #collect all imgs
    image_files = []
    for root, _, files in os.walk(args.root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, args.root)
                image_files.append((rel_path, full_path))

    if not image_files:
        raise SystemExit(f"No images found under {args.root}")

    print(f"[INFO] Using model={args.model}, imgsz={args.imgsz}, conf={args.conf}")
    print(f"[INFO] Found {len(image_files)} images under {args.root}")

    for rel_path, full_path in sorted(image_files):
        results = model(
            full_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.5,
            classes=[0],    #person class
        )

        dets = []
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(float, box[:4])
                dets.append({"bbox": [x1, y1, x2, y2]})

        detections[rel_path] = dets
        print(f"Processed {rel_path} ({len(dets)} person detections)")

    with open(args.out, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"[DONE] Saved detections to {args.out}")


if __name__ == "__main__":
    main()
