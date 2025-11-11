# scripts/realtime_play_predict.py

import os
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from Football_Play_Project.cnn.model import PlayCNN
from Football_Play_Project.cnn.dataset import ROLE_ORDER  # ["QB","RB","WR","OL","DEF"]


def role_one_hot(role: str):
    role = role or "DEF"
    if role not in ROLE_ORDER:
        role = "DEF"
    vec = [0.0] * len(ROLE_ORDER)
    idx = ROLE_ORDER.index(role)
    vec[idx] = 1.0
    return vec


def build_features_from_dets(dets, los, frame_shape, max_players: int):
    """
    dets: list of dicts with "box": [x1,y1,x2,y2]
    los: {"p1":[x,y], "p2":[x,y]} or None
    frame_shape: (H, W, C)
    return: torch tensor [1, max_players, feature_dim]
    """
    H, W = frame_shape[:2]

    offense_players = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cx = 0.5 * (x1 + x2) / (W + 1e-6)
        cy = 0.5 * (y1 + y2) / (H + 1e-6)
        offense_players.append({"cx": cx, "cy": cy, "box": d["box"]})

    if not offense_players:
        feat_dim = 3 + len(ROLE_ORDER)
        return torch.zeros((1, max_players, feat_dim), dtype=torch.float32)

    # sort left->right by cx
    offense_players.sort(key=lambda p: p["cx"])

    # rough y threshold for LOS row: median cy
    cys = [p["cy"] for p in offense_players]
    row_y = float(np.median(cys))
    tol = 0.05

    row_players = [p for p in offense_players if abs(p["cy"] - row_y) < tol]
    if row_players:
        row_indices = [offense_players.index(p) for p in row_players]
        row_indices.sort()
        mid = len(row_indices) // 2
        ol_indices = [row_indices[mid]]
        left = mid - 1
        right = mid + 1
        while len(ol_indices) < min(5, len(row_indices)):
            if left >= 0:
                ol_indices.append(row_indices[left])
                left -= 1
            if right < len(row_indices) and len(ol_indices) < 5:
                ol_indices.append(row_indices[right])
                right += 1
        ol_indices = sorted(ol_indices)
    else:
        ol_indices = []

    # QB: just behind OL row and near OL center
    qb_index = None
    if ol_indices:
        ol_xs = [offense_players[i]["cx"] for i in ol_indices]
        ol_center_x = float(np.mean(ol_xs))
        candidates = []
        for i, p in enumerate(offense_players):
            if i in ol_indices:
                continue
            if p["cy"] > row_y + tol:  # behind LOS
                dist_y = p["cy"] - row_y
                dist_x = abs(p["cx"] - ol_center_x)
                score = dist_y + dist_x
                candidates.append((score, i))
        if candidates:
            qb_index = min(candidates, key=lambda t: t[0])[1]

    # RB: behind QB
    rb_index = None
    if qb_index is not None:
        qb = offense_players[qb_index]
        candidates = []
        for i, p in enumerate(offense_players):
            if i == qb_index or i in ol_indices:
                continue
            if p["cy"] > qb["cy"]:
                dist_y = p["cy"] - qb["cy"]
                dist_x = abs(p["cx"] - qb["cx"])
                score = dist_y + dist_x
                candidates.append((score, i))
        if candidates:
            rb_index = min(candidates, key=lambda t: t[0])[1]

    # WR: wide players not OL/QB/RB
    wr_indices = []
    ol_center_x = float(np.mean([offense_players[i]["cx"] for i in ol_indices])) if ol_indices else 0.5
    for i, p in enumerate(offense_players):
        if i in ol_indices or i == qb_index or i == rb_index:
            continue
        if abs(p["cx"] - ol_center_x) > 0.15:
            wr_indices.append(i)

    feats = []
    for i, p in enumerate(offense_players):
        cx = p["cx"]
        cy = p["cy"]
        side_lr_val = 0.0 if cx < 0.5 else 1.0

        if i in ol_indices:
            role = "OL"
        elif i == qb_index:
            role = "QB"
        elif i == rb_index:
            role = "RB"
        elif i in wr_indices:
            role = "WR"
        else:
            role = "OL"

        role_vec = role_one_hot(role)
        feat = [cx, cy, side_lr_val] + role_vec
        feats.append(feat)

    feat_dim = len(feats[0])
    X = np.zeros((max_players, feat_dim), dtype=np.float32)
    n = min(len(feats), max_players)
    X[:n, :] = np.asarray(feats[:n], dtype=np.float32)
    X = torch.from_numpy(X).unsqueeze(0)  # [1, max_players, feat_dim]
    return X


def main():
    ap = argparse.ArgumentParser(description="Real-time play prediction from video using YOLO + CNN.")
    ap.add_argument("--video", default="data/raw/Test_Vid.mp4",
                    help="Path to input video or camera index (0,1,...)")
    ap.add_argument("--yolo", default="yolov8n.pt",
                    help="YOLO model weights.")
    ap.add_argument("--ckpt", default="runs/cnn/model.pt",
                    help="Trained CNN checkpoint.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_players", type=int, default=11)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load YOLO
    yolo = YOLO(args.yolo)

    # 2) Load CNN
    ckpt = torch.load(args.ckpt, map_location=device)
    label2idx = ckpt["label2idx"]
    idx2label = ckpt["idx2label"]
    feature_dim = ckpt["feature_dim"]
    max_players = ckpt["max_players"]
    model = PlayCNN(feature_dim=feature_dim, num_classes=len(idx2label), max_players=max_players).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Classes:", idx2label)
    print("ROLE_ORDER:", ROLE_ORDER)

    # 3) Video capture
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    los_line = None  # {"p1":[x,y], "p2":[x,y]}
    click_points = []
    paused = False
    last_frame = None

    def on_mouse(event, x, y, flags, param):
        nonlocal click_points, los_line
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append((x, y))
            if len(click_points) == 2:
                p1, p2 = click_points
                los_line = {"p1": [p1[0], p1[1]], "p2": [p2[0], p2[1]]}
                print(f"[LOS] Set LOS from {p1} to {p2}")
                click_points = []

    cv2.namedWindow("Real-time Play Prediction")
    cv2.setMouseCallback("Real-time Play Prediction", on_mouse)

    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            frame_idx += 1
        else:
            # When paused, keep using the last_frame
            if last_frame is None:
                # nothing grabbed yet: try to grab a frame
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame
                frame_idx += 1
            frame = last_frame

        H, W = frame.shape[:2]
        vis = frame.copy()

        # Draw LOS if set
        if los_line is not None:
            p1 = tuple(los_line["p1"])
            p2 = tuple(los_line["p2"])
            cv2.line(vis, p1, p2, (255, 0, 0), 2)

        # Run YOLO (even when paused we can keep predicting on the current frame)
        results = yolo.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.5,
            classes=[0],
            verbose=False,
        )
        dets = []
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(float, box[:4])
                dets.append({"box": [x1, y1, x2, y2]})

        # Draw boxes
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prediction logic
        if los_line is None:
            pred_text = "Press L, then click 2 points to set LOS"
            probs = None
            pred_idx = None
        else:
            pred_text = "Predicting..."
            probs = None
            pred_idx = None
            if dets:
                X = build_features_from_dets(dets, los_line, frame.shape, max_players).to(device)
                if X.shape[2] == feature_dim:
                    with torch.no_grad():
                        logits = model(X)
                        probs = torch.softmax(logits, dim=1)[0]
                        pred_idx = int(torch.argmax(probs).item())
                        pred_label = idx2label[pred_idx]
                        pred_text = f"Pred: {pred_label}"
                else:
                    pred_text = "Feature dim mismatch"

        # Overlay UI
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (W, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(
            vis,
            f"{status} | {pred_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            "Keys: P=pause/resume | L=set LOS (click 2 points) | Q=quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # show class probabilities
        if probs is not None and pred_idx is not None:
            y0 = 90
            for i, lbl in enumerate(idx2label):
                p = probs[i].item()
                cv2.putText(
                    vis,
                    f"{lbl}: {p:.2f}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if i == pred_idx else (200, 200, 200),
                    1,
                )
                y0 += 20

        cv2.imshow("Real-time Play Prediction", vis)
        # slower refresh when paused so clicks feel smoother
        delay = 30 if paused else 1
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("p"), ord("P")):
            paused = not paused
            print(f"[STATE] paused = {paused}")
        elif key in (ord("l"), ord("L")):
            # clear LOS; user will click two points (even while paused)
            los_line = None
            click_points = []
            print("[LOS] Click two points on the LOS in the window.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
