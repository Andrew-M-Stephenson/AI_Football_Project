# scripts/build_plays_meta_full.py

import os
import json
import csv

IMAGES_ROOT = "data/images/Plays"
DETECTIONS_PATH = "data/meta/plays_detections.json"
POSITIONS_PATH = "data/meta/plays_positions.json"
LOS_PATH = "data/meta/plays_los.json"

META_OUT = "data/meta/plays_meta_full.json"
LABELS_OUT = "data/meta/play_labels.csv"  # optional but very handy for CNN


def main():
    # Load the three inputs
    if not os.path.exists(DETECTIONS_PATH):
        raise SystemExit(f"Detections file not found: {DETECTIONS_PATH}")
    if not os.path.exists(POSITIONS_PATH):
        raise SystemExit(f"Positions file not found: {POSITIONS_PATH}")
    if not os.path.exists(LOS_PATH):
        raise SystemExit(f"LOS file not found: {LOS_PATH}")

    with open(DETECTIONS_PATH) as f:
        dets_data = json.load(f)
    with open(POSITIONS_PATH) as f:
        pos_data = json.load(f)
    with open(LOS_PATH) as f:
        los_data = json.load(f)

    frames = []
    label_rows = []

    # Use detections keys as master list of frames
    for idx, rel_path in enumerate(sorted(dets_data.keys())):
        img_dets = dets_data.get(rel_path, [])
        img_pos = pos_data.get(rel_path, {})
        img_los = los_data.get(rel_path, None)

        # Normalize roles to int-index keys
        pos_int = {}
        for k, v in img_pos.items():
            try:
                pos_int[int(k)] = v
            except ValueError:
                continue

        detections = []
        for i, d in enumerate(img_dets):
            x1, y1, x2, y2 = d["bbox"]
            role = pos_int.get(i)  # e.g. "QB", "RB", "WR", "OL", "DEF"

            # Side: treat DEF as defense, everyone else as offense
            side = "defense" if role == "DEF" else "offense"

            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "role_pos": role,   # manual position label
                    "side": side,
                }
            )

        # Derive play label from folder name: "Deep_Pass/img001.jpg" -> "Deep_Pass"
        norm_rel = os.path.normpath(rel_path)
        parts = norm_rel.split(os.sep)
        if len(parts) >= 1:
            play_label = parts[0]
        else:
            play_label = "Unknown"

        frame_entry = {
            "idx": idx,
            "fname": rel_path,      # relative path to image under data/images/Plays
            "los": img_los,         # {"p1": [x,y], "p2": [x,y]} or None
            "detections": detections,
            "play_label": play_label,
        }
        frames.append(frame_entry)
        label_rows.append((idx, play_label))

    meta = {"frames": frames}

    os.makedirs(os.path.dirname(META_OUT), exist_ok=True)
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[DONE] Wrote {len(frames)} frames to {META_OUT}")

    # Also write a simple CSV: idx -> play_label (useful for CNN training)
    with open(LABELS_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key_frame_idx", "play_label"])
        for idx, lbl in label_rows:
            writer.writerow([idx, lbl])
    print(f"[DONE] Wrote labels CSV to {LABELS_OUT}")

    if frames:
        print("\nExample frame entry:")
        print(json.dumps(frames[0], indent=4))


if __name__ == "__main__":
    main()
