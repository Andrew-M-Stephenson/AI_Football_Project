# scripts/mark_los_manual_images.py

import os
import json
import cv2

IMAGES_ROOT = "data/images/Plays"
OUTPUT_PATH = "data/meta/plays_los.json"


def collect_images(root):
    image_files = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(r, f)
                rel_path = os.path.relpath(full_path, root)
                image_files.append((rel_path, full_path))
    return sorted(image_files)


def main():
    images = collect_images(IMAGES_ROOT)
    if not images:
        raise SystemExit(f"No images found under {IMAGES_ROOT}")

    print(
        "[INFO] Mark LOS for each image.\n"
        "Controls:\n"
        "  - Left-click twice: set two points along the LOS\n"
        "  - R: reset points for this image\n"
        "  - S: save LOS for this image and go to NEXT\n"
        "  - N: skip this image (no LOS saved)\n"
        "  - Q or ESC: quit early (saves progress so far)\n"
    )

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            los_data = json.load(f)
    else:
        los_data = {}

    idx = 0
    window_name = "Mark LOS"

    while idx < len(images):
        rel_path, full_path = images[idx]
        img = cv2.imread(full_path)
        if img is None:
            print(f"[WARN] Could not read {full_path}, skipping.")
            idx += 1
            continue

        points = []
        existing = los_data.get(rel_path)

        def on_mouse(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 2:
                    points.append((x, y))

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)

        print(f"[INFO] Image {idx+1}/{len(images)}: {rel_path}")
        if existing:
            print("  -> Existing LOS found; press S to keep it or R to redraw.")

        while True:
            vis = img.copy()

            if len(points) == 2:
                cv2.line(vis, points[0], points[1], (255, 0, 0), 2)
            elif existing and len(points) == 0:
                p1 = tuple(existing["p1"])
                p2 = tuple(existing["p2"])
                cv2.line(vis, p1, p2, (0, 255, 255), 2)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(50) & 0xFF

            if key in (ord("q"), 27):
                os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(los_data, f, indent=4)
                print(f"[DONE] Early quit. Saved LOS to {OUTPUT_PATH}")
                cv2.destroyAllWindows()
                return

            elif key in (ord("r"), ord("R")):
                points = []
                print("  -> Reset points for this image.")

            elif key in (ord("n"), ord("N")):
                print("  -> Skipped this image (no LOS).")
                break

            elif key in (ord("s"), ord("S")):
                if len(points) == 2:
                    los_data[rel_path] = {
                        "p1": [points[0][0], points[0][1]],
                        "p2": [points[1][0], points[1][1]],
                    }
                    print("  -> Saved LOS for this image.")
                    break
                elif existing:
                    print("  -> Keeping existing LOS for this image.")
                    break
                else:
                    print("  -> Need 2 points before saving.")

        cv2.destroyWindow(window_name)
        idx += 1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(los_data, f, indent=4)
    print(f"[DONE] Labeled LOS for {len(los_data)} images. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
