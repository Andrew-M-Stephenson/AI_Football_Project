import cv2
import json
import os

IMAGES_DIR = "data/images"
LOS_PATH = "data/meta/los_data.json"
DETECTIONS_PATH = "data/meta/detections.json"
OUTPUT_PATH = "data/meta/positions.json"

ROLES = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S"]

ROLE_KEYS = {
    ord("1"): "QB",
    ord("2"): "RB",
    ord("3"): "WR",
    ord("4"): "TE",
    ord("5"): "OL",
    ord("6"): "DL",
    ord("7"): "LB",
    ord("8"): "CB",
    ord("9"): "S",
}

def draw_frame(img, boxes, roles, current_idx, los_for_frame=None):
    """Draw LOS, boxes, indices, and assigned roles on a copy of the image."""
    vis = img.copy()

    # Draw LOS if we have it
    if los_for_frame is not None:
        p1, p2 = los_for_frame["p1"], los_for_frame["p2"]
        cv2.line(vis, tuple(p1), tuple(p2), (255, 0, 0), 2)  # blue line

    for i, d in enumerate(boxes):
        x1, y1, x2, y2 = map(int, d["bbox"])
        # Highlight the currently selected box
        if i == current_idx:
            color = (0, 0, 255)  # red
            thickness = 3
        else:
            color = (0, 255, 0)  # green
            thickness = 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        # Index label
        cv2.putText(
            vis,
            str(i),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        # Role label (if assigned)
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
    # Load LOS + detections
    if not os.path.exists(LOS_PATH):
        raise SystemExit(f"LOS file not found: {LOS_PATH}")
    if not os.path.exists(DETECTIONS_PATH):
        raise SystemExit(f"Detections file not found: {DETECTIONS_PATH}")

    with open(LOS_PATH) as f:
        los_data = json.load(f)
    with open(DETECTIONS_PATH) as f:
        dets = json.load(f)

    data_out = {}

    image_files = sorted(
        fname for fname in os.listdir(IMAGES_DIR)
        if fname.lower().endswith((".jpg", ".png"))
    )
    if not image_files:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    print(
        "[INFO] Controls:\n"
        "  - Left/Right arrows or A/D: move between players\n"
        "  - Number keys 1–9: assign role\n"
        "      1=QB, 2=RB, 3=WR, 4=TE, 5=OL, 6=DL, 7=LB, 8=CB, 9=S\n"
        "  - S: save labels for this frame and go to next image\n"
        "  - B: go back to previous image (if any, keeps labels)\n"
        "  - Q or ESC: quit early (saves progress so far)\n"
    )

    i_img = 0
    # Track labels even if you move back/forth
    cached_roles = {}

    while 0 <= i_img < len(image_files):
        fname = image_files[i_img]
        img_path = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            i_img += 1
            continue

        boxes = dets.get(fname, [])
        if not boxes:
            print(f"[WARN] No detections for {fname}, skipping.")
            i_img += 1
            continue

        # Get LOS for this frame if available
        los_for_frame = los_data.get(fname)

        # Restore roles if we already labeled this frame earlier
        roles_for_frame = cached_roles.get(fname, {}).copy()

        current_idx = 0

        while True:
            vis = draw_frame(img, boxes, roles_for_frame, current_idx, los_for_frame)
            cv2.imshow("Label Players", vis)
            key = cv2.waitKey(50) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                # Save everything done so far and exit
                data_out.update(cached_roles)
                os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(data_out, f, indent=4)
                print(f"[DONE] Early quit. Saved partial labels to {OUTPUT_PATH}")
                cv2.destroyAllWindows()
                return

            # Next / previous player
            elif key in (ord("d"), ord("D"), 83):  # right arrow or D
                current_idx = (current_idx + 1) % len(boxes)
            elif key in (ord("a"), ord("A"), 81):  # left arrow or A
                current_idx = (current_idx - 1) % len(boxes)

            # Assign role via number keys
            elif key in ROLE_KEYS:
                role = ROLE_KEYS[key]
                roles_for_frame[current_idx] = role
                print(f"[{fname}] Player {current_idx} -> {role}")

            # Save and go to next frame
            elif key in (ord("s"), ord("S")):
                cached_roles[fname] = roles_for_frame
                print(f"[INFO] Saved labels for {fname}")
                i_img += 1
                break

            # Go back to previous image
            elif key in (ord("b"), ord("B")):
                cached_roles[fname] = roles_for_frame
                print(f"[INFO] Going back from {fname}")
                i_img -= 1
                break

        cv2.destroyAllWindows()

    # Done all images
    data_out.update(cached_roles)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data_out, f, indent=4)
    print(f"[DONE] Saved labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
