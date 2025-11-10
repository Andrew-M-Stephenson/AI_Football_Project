from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Football_Play_Project.io.annotations import load_json


ROLE_ORDER = ["QB", "RB", "WR", "OL", "SKILL"]


@dataclass
class PlaySample:
    key_frame_idx: int
    frame_meta: Dict[str, Any]
    label_str: str
    label_id: int


class PlayDataset(Dataset):
    """
    Dataset that turns each labeled play into a fixed-size tensor:

        X: [max_players, feature_dim]
        y: scalar class index

    Features per offensive player:
        - cx_norm (0..1)
        - cy_norm (0..1)
        - side_lr (0 = L, 1 = R)
        - role one-hot over ROLE_ORDER (len=5)

    Players are sorted left→right by cx and padded/truncated to max_players.
    """

    def __init__(
        self,
        meta_path: str,
        labels_csv: str,
        max_players: int = 11,
    ) -> None:
        super().__init__()

        self.meta_path = meta_path
        self.labels_csv = labels_csv
        self.max_players = max_players

        # 1) Load meta JSON
        meta = load_json(meta_path)
        frames = meta.get("frames", [])
        if not frames:
            raise RuntimeError(f"No frames in meta JSON: {meta_path}")

        # Map frame idx -> frame_meta
        self._frames_by_idx: Dict[int, Dict[str, Any]] = {
            int(f["idx"]): f for f in frames
        }

        # 2) Estimate image width/height from boxes (if not stored)
        self.W, self.H = self._estimate_image_size(frames)

        # 3) Load labels CSV
        df = pd.read_csv(labels_csv)
        if "key_frame_idx" not in df.columns or "play_label" not in df.columns:
            raise RuntimeError(
                f"Expected columns 'key_frame_idx' and 'play_label' in {labels_csv}"
            )

        # Build label string -> id mapping
        unique_labels = sorted(df["play_label"].unique().tolist())
        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx2label: List[str] = unique_labels
        self.num_classes = len(unique_labels)

        # 4) Build PlaySample list
        samples: List[PlaySample] = []
        missing = 0

        for _, row in df.iterrows():
            key_idx = int(row["key_frame_idx"])
            label_str = str(row["play_label"])
            if key_idx not in self._frames_by_idx:
                missing += 1
                continue
            frame_meta = self._frames_by_idx[key_idx]
            label_id = self.label2idx[label_str]
            samples.append(
                PlaySample(
                    key_frame_idx=key_idx,
                    frame_meta=frame_meta,
                    label_str=label_str,
                    label_id=label_id,
                )
            )

        if not samples:
            raise RuntimeError(
                "No matching plays found between labels CSV and meta frames."
            )
        if missing > 0:
            print(
                f"[PlayDataset] Warning: {missing} labeled key_frame_idx "
                f"values not found in meta JSON and were skipped."
            )

        self.samples = samples
        # Infer feature_dim from first sample
        example_X, _ = self._build_features_for_sample(self.samples[0])
        self.feature_dim = example_X.shape[1]
        print(
            f"[PlayDataset] Loaded {len(self.samples)} plays | "
            f"max_players={self.max_players} | feature_dim={self.feature_dim} | "
            f"num_classes={self.num_classes}"
        )

    def _estimate_image_size(self, frames: List[Dict[str, Any]]) -> Tuple[int, int]:
        max_x = 0.0
        max_y = 0.0
        for f in frames[:200]:  # sample first 200 frames
            for d in f.get("detections", []):
                x1, y1, x2, y2 = d["box"]
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
        W = int(max_x) if max_x > 0 else 1920
        H = int(max_y) if max_y > 0 else 1080
        return W, H

    # ----- feature construction -----

    def _role_one_hot(self, role: str) -> List[float]:
        """
        Role one-hot over ROLE_ORDER.
        Unknown roles are treated as SKILL.
        """
        role = role or "SKILL"
        if role not in ROLE_ORDER:
            role = "SKILL"
        vec = [0.0] * len(ROLE_ORDER)
        idx = ROLE_ORDER.index(role)
        vec[idx] = 1.0
        return vec

    def _frame_to_tensor(self, frame: Dict[str, Any]) -> torch.Tensor:
        """
        Extract offensive players, build features, pad/truncate to max_players.

        Returns: tensor [max_players, feature_dim]
        """
        W, H = float(self.W), float(self.H)
        dets = frame.get("detections", [])
        # offense only
        players: List[List[float]] = []
        for d in dets:
            if d.get("side") != "offense":
                continue
            x1, y1, x2, y2 = d["box"]
            cx = 0.5 * (x1 + x2) / (W + 1e-6)
            cy = 0.5 * (y1 + y2) / (H + 1e-6)

            side_lr = d.get("side_lr")
            side_lr_val = 0.0 if side_lr == "L" else 1.0  # default R if unknown

            role = d.get("role_pos", "SKILL")
            role_vec = self._role_one_hot(role)

            feat = [cx, cy, side_lr_val] + role_vec
            players.append(feat)

        # If there are no offensive players, return zeros
        if not players:
            return torch.zeros(self.max_players, 3 + len(ROLE_ORDER), dtype=torch.float32)

        # Sort left-to-right by cx
        players.sort(key=lambda f: f[0])

        feat_dim = len(players[0])
        X = np.zeros((self.max_players, feat_dim), dtype=np.float32)

        n = min(len(players), self.max_players)
        X[:n, :] = np.asarray(players[:n], dtype=np.float32)

        return torch.from_numpy(X)

    def _build_features_for_sample(self, sample: PlaySample) -> Tuple[torch.Tensor, int]:
        X = self._frame_to_tensor(sample.frame_meta)
        y = sample.label_id
        return X, y

    # ----- Dataset API -----

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        X, y = self._build_features_for_sample(sample)
        return X, torch.tensor(y, dtype=torch.long)
