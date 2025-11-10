import cv2
import os
import json

# Paths
IMAGES_DIR = "data/images"
OUTPUT_PATH = "data/meta/los_data.json"

los_data = {}
points = []
current_img = None
current_name = None

def click_event(event, x, y, flags, params):
    global points, current_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(current_img, (x, y), 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(current_img, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow("Frame", current_img)

def main():
    global current_img, current_name, points

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(IMAGES_DIR, fname)
        current_img = cv2.imread(img_path)
        current_name = fname
        points = []

        cv2.imshow("Frame", current_img)
        cv2.setMouseCallback("Frame", click_event)
        print(f"[INFO] Click 2 points for LOS on {fname}. Press 's' to save or 'n' to skip.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(points) == 2:
                los_data[fname] = {"p1": points[0], "p2": points[1]}
                print(f"Saved LOS for {fname}")
                break
            elif key == ord('n'):
                print(f"Skipped {fname}")
                break

        cv2.destroyAllWindows()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(los_data, f, indent=4)
    print(f"[DONE] LOS data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
