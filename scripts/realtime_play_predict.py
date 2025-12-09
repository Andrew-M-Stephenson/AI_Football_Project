# scripts/realtime_play_predict.py
import os
import argparse
import csv
import json
import cv2
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from Football_Play_Project.cnn.model import PlayCNN
from Football_Play_Project.cnn.dataset import ROLE_ORDER

#helper functions
def role_one_hot(role: str):
    role = role or "DEF"
    if role not in ROLE_ORDER:
        role = "DEF"
    vec = [0.0] * len(ROLE_ORDER)
    idx = ROLE_ORDER.index(role)
    vec[idx] = 1.0
    return vec

def build_features_from_dets(dets, los, frame_shape, max_players: int, feature_dim_hint=None):
    H, W = frame_shape[:2]
    offense_players = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cx = 0.5 * (x1 + x2) / (W + 1e-6)
        cy = 0.5 * (y1 + y2) / (H + 1e-6)
        offense_players.append({"cx": cx, "cy": cy, "box": d["box"]})

    if not offense_players:
        feat_dim = feature_dim_hint or (3 + len(ROLE_ORDER))
        return torch.zeros((1, max_players, feat_dim), dtype=torch.float32)

    offense_players.sort(key=lambda p: p["cx"])

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
                ol_indices.append(row_indices[left]); left -= 1
            if right < len(row_indices) and len(ol_indices) < 5:
                ol_indices.append(row_indices[right]); right += 1
        ol_indices = sorted(ol_indices)
    else:
        ol_indices = []

    qb_index = None
    if ol_indices:
        ol_xs = [offense_players[i]["cx"] for i in ol_indices]
        ol_center_x = float(np.mean(ol_xs))
        candidates = []
        for i, p in enumerate(offense_players):
            if i in ol_indices:
                continue
            if p["cy"] > row_y + tol:
                dist_y = p["cy"] - row_y
                dist_x = abs(p["cx"] - ol_center_x)
                score = dist_y + dist_x
                candidates.append((score, i))
        if candidates:
            qb_index = min(candidates, key=lambda t: t[0])[1]

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

    wr_indices = []
    ol_center_x = float(np.mean([offense_players[i]["cx"] for i in ol_indices])) if ol_indices else 0.5
    for i, p in enumerate(offense_players):
        if i in ol_indices or i == qb_index or i == rb_index:
            continue
        if abs(p["cx"] - ol_center_x) > 0.15:
            wr_indices.append(i)

    feats = []
    for i, p in enumerate(offense_players):
        cx = p["cx"]; cy = p["cy"]
        side_lr_val = 0.0 if cx < 0.5 else 1.0
        if i in ol_indices: role = "OL"
        elif i == qb_index: role = "QB"
        elif i == rb_index: role = "RB"
        elif i in wr_indices: role = "WR"
        else: role = "OL"
        role_vec = role_one_hot(role)
        feat = [cx, cy, side_lr_val] + role_vec
        feats.append(feat)

    feat_dim = len(feats[0])
    X = np.zeros((max_players, feat_dim), dtype=np.float32)
    n = min(len(feats), max_players)
    X[:n, :] = np.asarray(feats[:n], dtype=np.float32)
    X = torch.from_numpy(X).unsqueeze(0)
    return X

def ensure_log(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","video","frame_idx","method","final_pred","top_prob","probs_json","correct","true_label"])

def append_log(log_path: str, row: dict):
    ensure_log(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("timestamp",""),
            row.get("video",""),
            row.get("frame_idx",""),
            row.get("method",""),
            row.get("final_pred",""),
            f'{row.get("top_prob", ""):.4f}' if isinstance(row.get("top_prob",None),(float,int)) else row.get("top_prob",""),
            json.dumps(row.get("probs",{})),
            row.get("correct",""),
            row.get("true_label","")
        ])

def crop_around_los_mask(shape, los_line, pad=40):
    H,W = shape[:2]
    mask = np.zeros((H,W), dtype=np.uint8)
    if los_line is None:
        mask[:] = 255
        return mask
    p1 = np.array(los_line["p1"], dtype=np.int32)
    p2 = np.array(los_line["p2"], dtype=np.int32)
    v = p2 - p1
    if np.linalg.norm(v) < 1:
        mask[:] = 255
        return mask
    v = v / (np.linalg.norm(v) + 1e-6)
    n = np.array([-v[1], v[0]])
    a = p1 + n*pad; b = p1 - n*pad; c = p2 - n*pad; d = p2 + n*pad
    poly = np.array([a,b,c,d], dtype=np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask

#main
def main():
    ap = argparse.ArgumentParser(description="Real-time play prediction with finalization + feedback logging.")
    ap.add_argument("--video", default="data/raw/Demo_1.mp4")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--ckpt", default="runs/cnn/model.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_players", type=int, default=11)
    ap.add_argument("--auto_snap", type=int, default=0, help="1 to enable motion-triggered finalization")
    ap.add_argument("--motion_thresh", type=float, default=12.0, help="Mean abs diff threshold (0-255 scale)")
    ap.add_argument("--log_csv", default="runs/cnn/prediction_log.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.yolo)
    ckpt = torch.load(args.ckpt, map_location=device)
    idx2label = ckpt["idx2label"]
    feature_dim = ckpt["feature_dim"]
    max_players = ckpt["max_players"]
    model = PlayCNN(feature_dim=feature_dim, num_classes=len(idx2label), max_players=max_players).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    show_boxes = True
    show_pred  = True

    #state
    los_line = None
    click_points = []
    paused = False
    last_frame = None
    last_gray = None
    final_pred = None
    final_top_prob = None
    final_method = None
    selected_true_label = ""
    frame_idx = 0

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
    ensure_log(args.log_csv)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            last_frame = frame
            frame_idx += 1
        else:
            if last_frame is None:
                ret, frame = cap.read()
                if not ret: break
                last_frame = frame
                frame_idx += 1
            frame = last_frame

        H, W = frame.shape[:2]
        vis = frame.copy()

        #los
        if los_line is not None:
            p1 = tuple(los_line["p1"]); p2 = tuple(los_line["p2"])
            cv2.line(vis, p1, p2, (255, 0, 0), 2)

        #yolo
        results = yolo.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=0.5, classes=[0], verbose=False)
        dets = []
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(float, box[:4])
                dets.append({"box": [x1, y1, x2, y2]})

        if show_boxes:
            for d in dets:
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #predict
        probs = None; pred_idx = None; pred_label = None
        if los_line is not None and dets:
            X = build_features_from_dets(dets, los_line, frame.shape, max_players, feature_dim_hint=feature_dim).to(device)
            if X.shape[2] == feature_dim:
                with torch.no_grad():
                    logits = model(X)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_label = idx2label[pred_idx]

        #motion detect
        if args.auto_snap and final_pred is None and los_line is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                diff = cv2.absdiff(gray, last_gray)
                mask = crop_around_los_mask(gray.shape, los_line, pad=40)
                mean_diff = (diff[mask > 0]).mean() if np.any(mask) else diff.mean()
                if mean_diff > args.motion_thresh and pred_label is not None:
                    final_pred = pred_label
                    final_top_prob = float(probs[pred_idx]) if probs is not None else None
                    final_method = "auto"
                    selected_true_label = ""
                    print(f"[FINAL][AUTO] {final_pred} (p={final_top_prob:.3f}) at frame {frame_idx}")
            last_gray = gray

        #hud
        overlay = vis.copy()
        if show_pred:
            cv2.rectangle(overlay, (0, 0), (W, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

            status = "PAUSED" if paused else "PLAYING"
            headline = f"{status}"
            if pred_label:
                headline += f" | Pred: {pred_label}"
            if final_pred:
                headline += f" | FINAL: {final_pred} ({final_method})"
            cv2.putText(vis, headline, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            y0 = 60
            if probs is not None:
                for i, lbl in enumerate(idx2label):
                    p = float(probs[i])
                    cv2.putText(vis, f"{lbl}: {p:.2f}", (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if pred_idx == i else (200, 200, 200), 1)
                    y0 += 18

            cv2.putText(vis, "Keys: P=pause | L=set LOS | F=finalize | U=unfinalize | H=toggle HUD/boxes | Y/N=correct? | 1-4=true label | Q=quit",
                        (10, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
        else:
            cv2.putText(vis, "HUD OFF (H to toggle) | P pause | L set LOS | F finalize | U unfinalize | Q quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 220, 255), 2)

        cv2.imshow("Real-time Play Prediction", vis)
        key = cv2.waitKey(30 if paused else 1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("p"), ord("P")):
            paused = not paused
            print(f"[STATE] paused = {paused}")
        elif key in (ord("l"), ord("L")):
            los_line = None; click_points = []; print("[LOS] Click two points on the LOS in the window.")
        elif key in (ord("f"), ord("F")):
            if pred_label is not None:
                final_pred = pred_label
                final_top_prob = float(probs[pred_idx]) if probs is not None else None
                final_method = "manual"
                selected_true_label = ""
                print(f"[FINAL][MANUAL] {final_pred} (p={final_top_prob:.3f}) at frame {frame_idx}")
        elif key in (ord("u"), ord("U")):
            final_pred = None; final_top_prob = None; final_method = None; selected_true_label = ""
            print("[FINAL] cleared.")
        elif key in (ord("h"), ord("H")):
            show_boxes = not show_boxes
            show_pred  = not show_pred
            print(f"[TOGGLE] show_boxes={show_boxes} | show_pred={show_pred}")
        elif key in (ord("y"), ord("Y")):
            if final_pred:
                append_log(args.log_csv, {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "video": args.video, "frame_idx": frame_idx,
                    "method": final_method or "",
                    "final_pred": final_pred,
                    "top_prob": final_top_prob,
                    "probs": {lbl: float(probs[i]) for i, lbl in enumerate(idx2label)} if probs is not None else {},
                    "correct": 1, "true_label": selected_true_label
                })
                print("[LOG] recorded: correct=1")
        elif key in (ord("n"), ord("N")):
            if final_pred:
                append_log(args.log_csv, {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "video": args.video, "frame_idx": frame_idx,
                    "method": final_method or "",
                    "final_pred": final_pred,
                    "top_prob": final_top_prob,
                    "probs": {lbl: float(probs[i]) for i, lbl in enumerate(idx2label)} if probs is not None else {},
                    "correct": 0, "true_label": selected_true_label
                })
                print("[LOG] recorded: correct=0")
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            idx = key - ord("1")
            if 0 <= idx < len(idx2label):
                selected_true_label = idx2label[idx]
                print(f"[GT] true_label set to: {selected_true_label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
