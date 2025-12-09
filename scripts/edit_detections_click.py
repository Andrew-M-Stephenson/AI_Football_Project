# scripts/edit_detections_with_positions.py

import os
import json
import argparse
import cv2

IMAGES_ROOT_DEFAULT = "data/images/Plays"
DETECTIONS_PATH_DEFAULT = "data/meta/plays_detections.json"
POSITIONS_PATH_DEFAULT = "data/meta/plays_positions.json"


def draw_boxes(img, boxes, roles, keep_flags):
    vis = img.copy()
    for i, d in enumerate(boxes):
        x1, y1, x2, y2 = map(int, d["bbox"])

        kept = keep_flags[i]
        color = (0, 255, 0) if kept else (0, 0, 255)
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
        role = roles.get(str(i)) or roles.get(i)
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
    ap = argparse.ArgumentParser(
        description="Click to keep/remove detections and keep positions.json in sync."
    )
    ap.add_argument("--root", default=IMAGES_ROOT_DEFAULT,
                    help="Root folder containing play images (Deep_Pass, Short_Pass, etc.).")
    ap.add_argument("--dets", default=DETECTIONS_PATH_DEFAULT,
                    help="Detections JSON (will be updated in-place).")
    ap.add_argument("--pos", default=POSITIONS_PATH_DEFAULT,
                    help="Positions JSON (will be updated in-place).")
    args = ap.parse_args()

    if not os.path.exists(args.dets):
        raise SystemExit(f"Detections file not found: {args.dets}")
    if not os.path.exists(args.pos):
        print(f"[WARN] Positions file not found: {args.pos}, starting with empty roles.")
        positions_data = {}
    else:
        with open(args.pos) as f:
            positions_data = json.load(f)

    with open(args.dets) as f:
        dets_data = json.load(f)

    keys = sorted(dets_data.keys())
    if not keys:
        raise SystemExit(f"No detections in {args.dets}")

    print(
        "[INFO] Edit detections and roles.\n"
        "Controls:\n"
        "  - Left-click on a box: toggle KEEP/DELETE (green=keep, red=delete)\n"
        "  - D / Right arrow: save this frame and NEXT image\n"
        "  - A / Left arrow : save this frame and PREVIOUS image\n"
        "  - Q or ESC      : save all changes and quit\n"
        "\n"
        "NOTE: When you delete a box, any role attached to it is removed too.\n"
        "      Remaining boxes are reindexed (0,1,2,...) and roles are remapped accordingly.\n"
    )

    win_name = "Edit Detections + Positions"
    i_img = 0

    while 0 <= i_img < len(keys):
        rel_path = keys[i_img]
        full_path = os.path.join(args.root, rel_path)

        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Could not read {full_path}, skipping.")
            i_img += 1
            continue

        boxes = dets_data.get(rel_path, [])
        if not boxes:
            print(f"[WARN] No detections for {rel_path}, skipping.")
            i_img += 1
            continue

        roles = positions_data.get(rel_path, {})

        keep_flags = [True] * len(boxes)

        def on_mouse(event, x, y, flags, param):
            nonlocal keep_flags
            if event == cv2.EVENT_LBUTTONDOWN:
                #find first box containing point
                for idx, d in enumerate(boxes):
                    x1, y1, x2, y2 = d["bbox"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        keep_flags[idx] = not keep_flags[idx]
                        state = "KEEP" if keep_flags[idx] else "DELETE"
                        print(f"[{rel_path}] Box {idx} -> {state}")
                        break

        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, on_mouse)

        print(f"[INFO] Image {i_img+1}/{len(keys)}: {rel_path} ({len(boxes)} boxes)")

        while True:
            vis = draw_boxes(img, boxes, roles, keep_flags)
            overlay = vis.copy()
            text = f"{rel_path} | boxes={len(boxes)} (green=keep, red=delete)"
            cv2.rectangle(overlay, (0, 0), (1000, 30), (0, 0, 0), -1)
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
            key = cv2.waitKey(50) & 0xFF

            if key in (ord("q"), 27):
                #save current frame changes then write files and quit
                _save_frame(rel_path, boxes, keep_flags, roles, dets_data, positions_data)
                _write_files(args.dets, args.pos, dets_data, positions_data)
                cv2.destroyAllWindows()
                print(f"[DONE] Saved and quit.")
                return

            elif key in (ord("d"), ord("D"), 83):
                _save_frame(rel_path, boxes, keep_flags, roles, dets_data, positions_data)
                i_img += 1
                break

            elif key in (ord("a"), ord("A"), 81):
                _save_frame(rel_path, boxes, keep_flags, roles, dets_data, positions_data)
                i_img -= 1
                break

        cv2.destroyWindow(win_name)

    _write_files(args.dets, args.pos, dets_data, positions_data)
    print(f"[DONE] Finished all images and saved updated JSONs.")


def _save_frame(rel_path, boxes, keep_flags, roles, dets_data, positions_data):
    """Apply keep_flags to boxes and roles for a single image."""
    filtered_boxes = [b for k, b in zip(keep_flags, boxes) if k]

    roles_old = {}
    for k, v in roles.items():
        try:
            roles_old[int(k)] = v
        except ValueError:
            continue

    roles_new = {}
    new_idx = 0
    for old_idx, keep in enumerate(keep_flags):
        if keep:
            if old_idx in roles_old:
                roles_new[str(new_idx)] = roles_old[old_idx]
            new_idx += 1

    dets_data[rel_path] = filtered_boxes
    positions_data[rel_path] = roles_new
    print(f"[SAVE] {rel_path}: kept {len(filtered_boxes)} boxes, {len(roles_new)} roles")


def _write_files(dets_path, pos_path, dets_data, positions_data):
    os.makedirs(os.path.dirname(dets_path), exist_ok=True)
    os.makedirs(os.path.dirname(pos_path), exist_ok=True)
    with open(dets_path, "w") as f:
        json.dump(dets_data, f, indent=4)
    with open(pos_path, "w") as f:
        json.dump(positions_data, f, indent=4)
    print(f"[WRITE] Updated {dets_path} and {pos_path}")


if __name__ == "__main__":
    main()
