from __future__ import annotations

import torch
import torch.nn as nn


class PlayCNN(nn.Module):
    """
    Simple 1D conv + MLP over per-player feature sequences.

    Input shape:  [B, max_players, feature_dim]
    Output shape: [B, num_classes] (logits)
    """

    def __init__(self, feature_dim: int, num_classes: int, max_players: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_players = max_players
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # aggregate over players
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, max_players, feature_dim]
        """
        x = x.transpose(1, 2)  # [B, feature_dim, max_players]
        x = self.conv(x)       # [B, 64, 1]
        x = x.squeeze(-1)      # [B, 64]
        logits = self.fc(x)    # [B, num_classes]
        return logits
