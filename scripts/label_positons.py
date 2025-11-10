import cv2
import json
import os
from ultralytics import YOLO

# --- Paths ---
frames_dir = "data/images"
meta_dir = "data/meta"
os.makedirs(meta_dir, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define position classes (you can modify or add as needed)
POSITIONS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S"]

def label_positions_on_frame(frame_name):
    frame_path = os.path.join(frames_dir, frame_name)
    img = cv2.imread(frame_path)

    # --- Load LOS data if available ---
    json_path = os.path.join(meta_dir, frame_name.replace(".jpg", ".json"))
    los_y = None
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_data = json.load(f)
            los_y = existing_data.get("line_of_scrimmage_y", None)
    else:
        existing_data = {"frame": frame_name}

    # --- Draw LOS line if found ---
    if los_y is not None:
        cv2.line(img, (0, los_y), (img.shape[1], los_y), (0, 0, 255), 2)
        print(f"Line of scrimmage loaded at y={los_y}")
    else:
        print(f"No LOS found for {frame_name}. Run mark_los_manual.py first!")

    # --- Run YOLO detections ---
    results = model.predict(img, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    detections = []

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "label": None
        })

    # --- Manual labeling loop ---
    current_index = 0
    while True:
        display = img.copy()

        # Draw LOS line again each frame
        if los_y is not None:
            cv2.line(display, (0, los_y), (display.shape[1], los_y), (0, 0, 255), 2)

        # Draw bounding boxes and labels
        for i, d in enumerate(detections):
            color = (0, 255, 0) if d["label"] is None else (255, 0, 0)
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label_text = d["label"] if d["label"] else f"{i}"
            cv2.putText(display, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Label Player Positions", display)
        key = cv2.waitKey(0) & 0xFF

        # Controls
        if key == ord('q'):   # Quit early
            break
        elif key == 13:       # Enter → next player
            current_index += 1
            if current_index >= len(detections):
                break
        elif key in [ord(str(i)) for i in range(len(POSITIONS))]:
            idx = int(chr(key))
            detections[current_index]["label"] = POSITIONS[idx]
            print(f"Labeled player {current_index} as {POSITIONS[idx]}")
        else:
            print("Keys: 0–8 for positions, Enter to move, q to quit")

    # --- Save combined data (LOS + detections) ---
    existing_data["detections"] = detections
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"✅ Saved labeled data for {frame_name}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

    for frame_name in frame_files:
        print(f"\nNow labeling: {frame_name}")
        label_positions_on_frame(frame_name)

    print("\n✅ All frames labeled.")
