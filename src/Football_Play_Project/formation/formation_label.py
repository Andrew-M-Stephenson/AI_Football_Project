# src/footballplaypredictor/formation/formation_label.py

from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter

from .role_assign import extract_offense_features_for_frame


def formation_from_counts(num_left: int, num_right: int) -> str:
    """
    Map (#WR_left, #WR_right) -> simple formation name.
    Only looks at wide WRs, not TE/RB/QB details.
    """
    key = (num_left, num_right)

    mapping = {
        (2, 2): "2x2",
        (3, 1): "TripsLeft",
        (1, 3): "TripsRight",
        (3, 0): "TripsLeftTight",
        (0, 3): "TripsRightTight",
        (1, 2): "2x1_RightHeavy",
        (2, 1): "2x1_LeftHeavy",
    }

    return mapping.get(key, f"Other_{num_left}x{num_right}")


def infer_formation_for_frame(
    frame_meta: Dict[str, Any],
    image_width: int,
    image_height: int,
    min_offense_players: int = 7,
) -> str | None:
    """
    For a single frame, infer a rough formation label based on WR distribution.

    Side effects:
      - Also annotates each offensive detection with:
            det["role_pos"], det["side_lr"], det["is_wide"]
        via extract_offense_features_for_frame().
    """
    players = extract_offense_features_for_frame(
        frame_meta,
        image_width=image_width,
        image_height=image_height,
        min_offense_players=min_offense_players,
    )

    if not players:
        return None

    # Count WRs that we marked as wide
    wr_left = 0
    wr_right = 0

    for p in players:
        if p.get("pos") != "WR":
            continue
        side_lr = p.get("side_lr")
        if side_lr == "L":
            wr_left += 1
        elif side_lr == "R":
            wr_right += 1

    total_wr = wr_left + wr_right
    if total_wr < 2:
        return None

    return formation_from_counts(wr_left, wr_right)


def infer_formation_for_play(
    play_frames: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> str:
    """
    Given frames belonging to the same play, infer a formation via majority vote.
    """
    labels = []
    for f in play_frames:
        label = infer_formation_for_frame(f, image_width, image_height)
        if label:
            labels.append(label)

    if not labels:
        return "Unknown"

    counts = Counter(labels)
    return counts.most_common(1)[0][0]
