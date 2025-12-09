# scripts/label_positions_manual.py

import cv2
import json
import os

IMAGES_ROOT = "data/images/Plays"
DETECTIONS_PATH = "data/meta/plays_detections.json"
OUTPUT_PATH = "data/meta/plays_positions.json"

ROLES = ["QB", "RB", "WR", "OL", "DEF"]

ROLE_KEYS = {
    ord("1"): "QB",
    ord("2"): "RB",
    ord("3"): "WR",
    ord("4"): "OL",
    ord("5"): "DEF",
}


def draw_frame(img, boxes, roles, current_idx):
    vis = img.copy()

    for i, d in enumerate(boxes):
        x1, y1, x2, y2 = map(int, d["bbox"])
        if i == current_idx:
            color = (0, 0, 255)  
            thickness = 3
        else:
            color = (0, 255, 0)  
            thickness = 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis,
            str(i),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        role = roles.get(i)
        if role:
            cv2.putText(
                vis,
                role,
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    return vis


def main():
    if not os.path.exists(DETECTIONS_PATH):
        raise SystemExit(f"Detections file not found: {DETECTIONS_PATH}")

    with open(DETECTIONS_PATH) as f:
        dets = json.load(f)

    keys = sorted(dets.keys())
    if not keys:
        raise SystemExit(f"No detections in {DETECTIONS_PATH}")

    print(
        "[INFO] Manual position labeling\n"
        "Controls:\n"
        "  - A / Left arrow : previous player box\n"
        "  - D / Right arrow: next player box\n"
        "  - 1: QB\n"
        "  - 2: RB\n"
        "  - 3: WR\n"
        "  - 4: OL\n"
        "  - 5: DEF (any defensive player)\n"
        "  - S: save this frame's labels and go to NEXT image\n"
        "  - B: save and go to PREVIOUS image\n"
        "  - Q or ESC: save everything and quit\n"
    )

    cached_roles = {}
    i_img = 0
    win_name = "Label Positions"

    while 0 <= i_img < len(keys):
        rel_path = keys[i_img]
        full_path = os.path.join(IMAGES_ROOT, rel_path)

        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Could not read {full_path}, skipping.")
            i_img += 1
            continue

        boxes = dets.get(rel_path, [])
        if not boxes:
            print(f"[WARN] No detections for {rel_path}, skipping.")
            i_img += 1
            continue

        roles_for_frame = cached_roles.get(rel_path, {}).copy()
        current_idx = 0

        print(f"[INFO] Image {i_img+1}/{len(keys)}: {rel_path} ({len(boxes)} boxes)")

        while True:
            vis = draw_frame(img, boxes, roles_for_frame, current_idx)
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(50) & 0xFF

            if key in (ord("q"), 27):
                cached_roles[rel_path] = roles_for_frame
                os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(cached_roles, f, indent=4)
                print(f"[DONE] Early quit. Saved labels to {OUTPUT_PATH}")
                cv2.destroyAllWindows()
                return

            elif key in (ord("d"), ord("D"), 83):
                current_idx = (current_idx + 1) % len(boxes)
            elif key in (ord("a"), ord("A"), 81):
                current_idx = (current_idx - 1) % len(boxes)

            #assign role
            elif key in ROLE_KEYS:
                role = ROLE_KEYS[key]
                roles_for_frame[current_idx] = role
                print(f"[{rel_path}] Box {current_idx} -> {role}")

            #save and next image
            elif key in (ord("s"), ord("S")):
                cached_roles[rel_path] = roles_for_frame
                print(f"[INFO] Saved labels for {rel_path}")
                i_img += 1
                break

            #save and previous image
            elif key in (ord("b"), ord("B")):
                cached_roles[rel_path] = roles_for_frame
                print(f"[INFO] Saved labels for {rel_path}, going back")
                i_img -= 1
                break

        cv2.destroyWindow(win_name)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cached_roles, f, indent=4)
    print(f"[DONE] Labeled positions for {len(cached_roles)} images. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
