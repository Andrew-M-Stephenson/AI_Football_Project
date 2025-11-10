# src/footballplaypredictor/formation/role_assign.py

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math


def _los_axes(los: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Given LOS segment, return:
      u  = unit vector along LOS (p1 -> p2)
      n  = unit normal (perpendicular to LOS)
      p1 = origin point on LOS
    """
    x1, y1 = los["p1"]
    x2, y2 = los["p2"]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / length, dy / length
    # normal (one orientation); sign resolved later by OL
    nx, ny = -uy, ux
    return (ux, uy), (nx, ny), (x1, y1)


def _project_point(
    cx: float,
    cy: float,
    u: Tuple[float, float],
    n: Tuple[float, float],
    p1: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Project a point onto LOS coordinate system:
      t = coordinate along LOS (in pixels, left/right)
      d = signed depth perpendicular to LOS (in pixels, front/back)
    """
    ux, uy = u
    nx, ny = n
    x1, y1 = p1
    vx = cx - x1
    vy = cy - y1
    t = vx * ux + vy * uy   # along LOS
    d = vx * nx + vy * ny   # perpendicular to LOS
    return t, d


def extract_offense_features_for_frame(
    frame_meta: Dict[str, Any],
    image_width: int,
    image_height: int,
    min_offense_players: int = 6,
    line_min_count: int = 5,
    line_max_count: int = 7,
) -> List[Dict[str, Any]]:
    """
    Assign roles to offensive players using only geometry + LOS.

    Roles:
      - OL: on LOS, central horizontally
      - QB: “where the ball is snapped to”:
            closest to OL center line, just behind LOS (or deeper for gun)
      - RB: deeper than QB, in same central lane
      - WR: near LOS, wide horizontally, and especially near top/bottom of screen
      - SKILL: everyone else (TE / FB / misc)

    Writes back into each offensive detection:
      det["role_pos"] = "OL" | "QB" | "RB" | "WR" | "SKILL"
      det["side_lr"]  = "L" | "R"
      det["is_wide"]  = bool

    Returns list of player dicts with geometry & role info.
    """
    los = frame_meta.get("los")
    if not isinstance(los, dict) or los.get("type") != "segment":
        return []

    dets = frame_meta.get("detections", [])
    if not dets:
        return []

    u, n, p1 = _los_axes(los)

    offense: List[Dict[str, Any]] = []
    for d in dets:
        if d.get("side") != "offense":
            continue
        x1, y1, x2, y2 = d["box"]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        t, d_signed = _project_point(cx, cy, u, n, p1)
        offense.append(
            {
                "det": d,
                "cx": cx,
                "cy": cy,
                "t_along": t,
                "d_signed": d_signed,
                "depth_abs": abs(d_signed),
            }
        )

    if len(offense) < min_offense_players:
        return []

    W = float(image_width)
    H = float(image_height)

    # ---------- 1) Find OL: close to LOS + strongly central ----------
    by_depth = sorted(offense, key=lambda p: p["depth_abs"])

    core_for_center = by_depth[:max(line_min_count, 3)]
    ol_center_t_guess = sum(p["t_along"] for p in core_for_center) / max(len(core_for_center), 1)

    # Tighter thresholds so WRs don't get sucked into OL
    depth_close_thresh = 0.03 * H        # “on the line”
    lateral_core_band = 0.14 * W         # narrower central band for OL
    wr_definitely_wide = 0.22 * W        # beyond this = WR, never OL

    ol_candidates: List[Dict[str, Any]] = []
    for p in by_depth:
        depth_ok = p["depth_abs"] <= depth_close_thresh
        lateral = abs(p["t_along"] - ol_center_t_guess)
        central_ok = lateral <= lateral_core_band
        if depth_ok and central_ok:
            ol_candidates.append(p)

    if len(ol_candidates) < line_min_count:
        ol_candidates = by_depth[:line_min_count]

    final_ol: List[Dict[str, Any]] = []
    for p in ol_candidates:
        lateral = abs(p["t_along"] - ol_center_t_guess)
        if lateral >= wr_definitely_wide:
            continue
        final_ol.append(p)

    final_ol = sorted(final_ol, key=lambda p: p["depth_abs"])[:line_max_count]
    if not final_ol:
        return []

    # OL stats
    ol_center_t = sum(p["t_along"] for p in final_ol) / max(len(final_ol), 1)
    ol_depth_abs_mean = sum(p["depth_abs"] for p in final_ol) / max(len(final_ol), 1)
    ol_depth_signed_mean = sum(p["d_signed"] for p in final_ol) / max(len(final_ol), 1)

    # Direction from LOS into offensive backfield (QB/RB side)
    sign_ol = 1.0 if ol_depth_signed_mean >= 0 else -1.0

    # Approximate "ball spot": OL center point on LOS in image coords
    # This is our proxy for where the ball is snapped from.
    ball_t = ol_center_t
    ball_x = p1[0] + u[0] * ball_t
    ball_y = p1[1] + u[1] * ball_t

    for p in offense:
        p["pos"] = "SKILL"
    for p in final_ol:
        p["pos"] = "OL"

    for p in offense:
        p["depth_off"] = p["d_signed"] * sign_ol

    # ---------- 2) QB: “player the ball is snapped to” ----------
    # Score candidates by:
    #   - being behind LOS into backfield
    #   - being close horizontally to OL center / ball spot
    #   - being not insanely deep (pistol/under-center/gun)
    qb_candidates: List[Dict[str, Any]] = []
    qb_lateral_band = 0.12 * W
    qb_min_depth = ol_depth_abs_mean + 0.01 * H      # just behind LOS
    qb_max_depth = ol_depth_abs_mean + 0.12 * H      # not way downfield

    for p in offense:
        if p in final_ol:
            continue
        depth_off = p["depth_off"]
        if depth_off < qb_min_depth or depth_off > qb_max_depth:
            continue

        # distance from ball spot in image space
        dist_ball = math.hypot(p["cx"] - ball_x, p["cy"] - ball_y)
        lateral = abs(p["t_along"] - ol_center_t)
        if lateral > qb_lateral_band:
            continue

        qb_candidates.append((p, dist_ball, lateral, depth_off))

    qb_player = None
    rb_player = None

    if qb_candidates:
        # pick the one closest to the ball spot, then most central
        qb_candidates.sort(key=lambda item: (item[1], item[2]))
        qb_player = qb_candidates[0][0]
        qb_player["pos"] = "QB"

        qb_depth = qb_player["depth_off"]

        # ---------- 3) RB: deeper than QB in same central lane ----------
        rb_min_extra = 0.02 * H
        rb_lateral_max = 0.20 * W

        rb_cands: List[Dict[str, Any]] = []
        for p in offense:
            if p in final_ol or p is qb_player:
                continue
            depth_off = p["depth_off"]
            lateral = abs(p["t_along"] - ol_center_t)
            if depth_off >= qb_depth + rb_min_extra and lateral <= rb_lateral_max:
                rb_cands.append(p)

        if rb_cands:
            rb_player = max(rb_cands, key=lambda p: p["depth_off"])
            rb_player["pos"] = "RB"

    # ---------- 4) Gun/pistol fallback if no QB found at all ----------
    if qb_player is None:
        backfield_min_offset = 0.035 * H
        backfield_lateral_max = 0.22 * W

        backfield_candidates: List[Dict[str, Any]] = []
        for p in offense:
            if p in final_ol:
                continue
            depth_off = p["depth_off"]
            lateral = abs(p["t_along"] - ol_center_t)
            if depth_off >= ol_depth_abs_mean + backfield_min_offset and lateral <= backfield_lateral_max:
                backfield_candidates.append(p)

        backfield_candidates.sort(key=lambda p: p["depth_off"], reverse=True)

        if len(backfield_candidates) >= 2:
            rb_player = backfield_candidates[0]
            qb_player = backfield_candidates[1]
            rb_player["pos"] = "RB"
            qb_player["pos"] = "QB"
        elif len(backfield_candidates) == 1:
            rb_player = backfield_candidates[0]
            rb_player["pos"] = "RB"

    # ---------- 5) WR: wide & near LOS, + extra sideline boost ----------
    wr_depth_extra = 0.08 * H
    wr_lateral_min = 0.14 * W

    for p in offense:
        if p["pos"] != "SKILL":  # skip OL/QB/RB already assigned
            continue
        depth_abs = p["depth_abs"]
        lateral = abs(p["t_along"] - ol_center_t)

        if depth_abs <= ol_depth_abs_mean + wr_depth_extra and lateral >= wr_lateral_min:
            p["pos"] = "WR"

    # EXTRA: very strong bias for sideline WRs.
    # Offense near top/bottom of frame that isn't OL/QB/RB is almost certainly a WR.
    sideline_band = 0.28 * H       # larger band near top/bottom
    sideline_depth_extra = 0.18 * H

    for p in offense:
        if p["pos"] not in ("SKILL", "WR"):
            continue
        cy = p["cy"]
        depth_abs = p["depth_abs"]
        near_top_or_bottom = cy <= sideline_band or cy >= (H - sideline_band)
        if near_top_or_bottom and depth_abs <= ol_depth_abs_mean + sideline_depth_extra:
            p["pos"] = "WR"

    # ---------- 6) Final annotations: side_lr, is_wide, push into det ----------
    for p in offense:
        pos = p.get("pos", "SKILL")
        side_lr = "L" if p["t_along"] < ol_center_t else "R"
        is_wide = pos == "WR"

        p["side_lr"] = side_lr
        p["is_wide"] = is_wide

        det = p["det"]
        det["role_pos"] = pos
        det["side_lr"] = side_lr
        det["is_wide"] = bool(is_wide)

    return sorted(offense, key=lambda p: p["depth_abs"])
