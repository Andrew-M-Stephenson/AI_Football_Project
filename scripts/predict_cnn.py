# scripts/predict_cnn.py

import argparse
import json
import os

import torch

from Football_Play_Project.cnn.model import PlayCNN
from Football_Play_Project.cnn.dataset import ROLE_ORDER  # same roles as training: ["QB","RB","WR","OL","DEF"]


def load_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)


def role_one_hot(role: str):
    """
    Same behavior as in PlayDataset:
      - ROLE_ORDER e.g. ["QB","RB","WR","OL","DEF"]
      - Anything not in ROLE_ORDER becomes "DEF" (defense fallback)
    """
    role = role or "DEF"
    if role not in ROLE_ORDER:
        role = "DEF"
    vec = [0.0] * len(ROLE_ORDER)
    idx = ROLE_ORDER.index(role)
    vec[idx] = 1.0
    return vec


def frame_to_tensor(frame, img_root, max_players: int):
    """
    Build the same tensor as in PlayDataset:
      X: [max_players, feature_dim]
    using only offensive players, sorted left->right.
    """
    # Load image to get actual width/height
    fname = frame["fname"]
    import cv2

    full_path = os.path.join(img_root, fname)
    img = cv2.imread(full_path)
    if img is None:
        # fallback: approximate from boxes
        W, H = 1920, 1080
    else:
        H, W = img.shape[:2]

    dets = frame.get("detections", [])
    players = []

    for d in dets:
        if d.get("side") != "offense":
            continue

        x1, y1, x2, y2 = d["box"]
        cx = 0.5 * (x1 + x2) / (W + 1e-6)
        cy = 0.5 * (y1 + y2) / (H + 1e-6)

        side_lr_val = 0.0 if cx < 0.5 else 1.0  # left vs right half

        role = d.get("role_pos", None)
        role_vec = role_one_hot(role)

        feat = [cx, cy, side_lr_val] + role_vec
        players.append(feat)

    import numpy as np
    import torch as th

    if not players:
        # no offense players? return zeros
        return th.zeros(max_players, 3 + len(ROLE_ORDER), dtype=th.float32)

    players.sort(key=lambda f: f[0])  # sort by cx (left -> right)
    feat_dim = len(players[0])
    X = np.zeros((max_players, feat_dim), dtype=np.float32)
    n = min(len(players), max_players)
    X[:n, :] = np.asarray(players[:n], dtype=np.float32)
    return th.from_numpy(X)


def main():
    ap = argparse.ArgumentParser(description="Predict play label for a single frame using trained CNN.")
    ap.add_argument("--meta", default="data/meta/plays_meta_full.json",
                    help="Meta JSON with LOS/detections/roles/play_label.")
    ap.add_argument("--ckpt", default="runs/cnn/model.pt",
                    help="Trained CNN checkpoint.")
    ap.add_argument("--root", default="data/images/Plays",
                    help="Root for images referenced in meta[fname].")
    ap.add_argument("--frame_idx", type=int, required=True,
                    help="Frame idx to predict (0..N-1, must match meta).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    label2idx = ckpt["label2idx"]
    idx2label = ckpt["idx2label"]
    feature_dim = ckpt["feature_dim"]
    max_players = ckpt["max_players"]

    print("Classes:", idx2label)
    print(f"Model expects feature_dim={feature_dim}, max_players={max_players}")
    print(f"ROLE_ORDER used at train time: {ROLE_ORDER}")

    # 2) Build model
    model = PlayCNN(
        feature_dim=feature_dim,
        num_classes=len(idx2label),
        max_players=max_players,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) Load meta & select frame
    meta = load_meta(args.meta)
    frames = meta.get("frames", [])
    if not frames:
        raise SystemExit("No frames in meta JSON.")

    frame = None
    for f in frames:
        if int(f["idx"]) == args.frame_idx:
            frame = f
            break
    if frame is None:
        raise SystemExit(f"Frame idx={args.frame_idx} not found in meta.")

    # 4) Build input tensor
    X = frame_to_tensor(frame, img_root=args.root, max_players=max_players)
    X = X.unsqueeze(0).to(device)  # [1, max_players, feature_dim]

    if X.shape[2] != feature_dim:
        raise SystemExit(
            f"Feature dim mismatch: X has dim={X.shape[2]}, "
            f"but model was trained with feature_dim={feature_dim}"
        )

    # 5) Predict
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = idx2label[pred_idx]

    print(f"\nPrediction for frame idx={args.frame_idx}: {pred_label}")
    print("Class probabilities:")
    for i, lbl in enumerate(idx2label):
        print(f"  {lbl:>12s}: {probs[i].item():.3f}")


if __name__ == "__main__":
    main()
