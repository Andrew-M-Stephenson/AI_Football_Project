# scripts/build_manual_meta.py

import os
import json

IMAGES_DIR = "data/images"          # or your manual17 folder if different
LOS_PATH = "data/meta/los_data.json"
DETECTIONS_PATH = "data/meta/detections.json"
POSITIONS_PATH = "data/meta/positions.json"
OUTPUT_PATH = "data/meta/manual_people_meta.json"


def main():
    with open(LOS_PATH) as f:
        los_data = json.load(f)
    with open(DETECTIONS_PATH) as f:
        dets_data = json.load(f)
    with open(POSITIONS_PATH) as f:
        pos_data = json.load(f)

    frames = []
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".png"))
    )
    if not image_files:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    for idx, fname in enumerate(image_files):
        img_los = los_data.get(fname)
        img_dets = dets_data.get(fname, [])
        img_pos = pos_data.get(fname, {})

        # convert position keys from str to int just in case
        img_pos_int = {int(k): v for k, v in img_pos.items()}

        detections = []
        for i, d in enumerate(img_dets):
            x1, y1, x2, y2 = d["bbox"]
            role = img_pos_int.get(i)  # e.g. "QB", "WR", etc.

            # simple left/right flag relative to image center (we'll recompute later anyway)
            # for now just store the bbox & role
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "role_pos": role,
                    "side": "offense",   # we can treat all as offense for CNN features
                }
            )

        frame_entry = {
            "idx": idx,           # this is what PlayDataset will use as key_frame_idx
            "fname": fname,
            "los": img_los,
            "detections": detections,
        }
        frames.append(frame_entry)

    meta = {"frames": frames}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[DONE] Wrote {len(frames)} frames to {OUTPUT_PATH}")
    print("Frame indices used (for play_labels.csv):", [f["idx"] for f in frames])


if __name__ == "__main__":
    main()
