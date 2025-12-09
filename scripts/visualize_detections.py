# scripts/visualize_meta_los.py

import os
import json
import argparse
import cv2

IMAGES_ROOT_DEFAULT = "data/images/Plays"
META_PATH_DEFAULT = "data/meta/plays_meta_los.json"


def draw_frame(img, frame):
    """
    Draw LOS (if present) and detections on the image.
    frame: dict with keys: fname, los, detections
    """
    vis = img.copy()

    los = frame.get("los")
    if los and "p1" in los and "p2" in los:
        p1 = tuple(los["p1"])
        p2 = tuple(los["p2"])
        cv2.line(vis, p1, p2, (255, 0, 0), 2)  #blue LOS

    #draw boxes
    detections = frame.get("detections", [])
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            str(i),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return vis


def main():
    ap = argparse.ArgumentParser(description="Visualize images with LOS + detections from plays_meta_los.json.")
    ap.add_argument("--root", default=IMAGES_ROOT_DEFAULT,
                    help="Root folder containing play images (Deep_Pass, Short_Pass, etc.).")
    ap.add_argument("--meta", default=META_PATH_DEFAULT,
                    help="Meta JSON with frames, los, and detections.")
    args = ap.parse_args()

    if not os.path.exists(args.meta):
        raise SystemExit(f"Meta file not found: {args.meta}")

    with open(args.meta) as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    if not frames:
        raise SystemExit(f"No frames in {args.meta}")

    frames = sorted(frames, key=lambda f: int(f.get("idx", 0)))

    print(
        "[INFO] Viewing combined LOS + detections.\n"
        "Controls:\n"
        "  - D / Right arrow: next frame\n"
        "  - A / Left arrow : previous frame\n"
        "  - Q or ESC       : quit\n"
    )

    win_name = "LOS + Detections"
    i = 0

    while 0 <= i < len(frames):
        frame = frames[i]
        rel_path = frame.get("fname")
        idx = frame.get("idx")

        if not rel_path:
            print(f"[WARN] Frame {idx} has no fname, skipping.")
            i += 1
            continue

        full_path = os.path.join(args.root, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Could not read {full_path}, skipping.")
            i += 1
            continue

        vis = draw_frame(img, frame)

        overlay = vis.copy()
        dets = frame.get("detections", [])
        has_los = frame.get("los") is not None
        text = f"idx={idx} | {rel_path} | dets={len(dets)} | LOS={'yes' if has_los else 'no'}"
        cv2.rectangle(overlay, (0, 0), (1100, 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
        cv2.putText(
            vis,
            text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  
            break
        elif key in (ord("d"), ord("D"), 83):
            i += 1
        elif key in (ord("a"), ord("A"), 81):
            i -= 1
        else:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
