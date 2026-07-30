"""Spatial-memory retrieval audit for the Memory-Maze adaptation of Lyra 2.0.

Three offline modes, all driven by ground-truth geometry (no diffusion, no DA3):

``--mode pitch``
    Tests whether ``camera_w2c`` is missing the Memory-Maze camera's fixed downward mount
    tilt. The depth renderer applies ``pitch_deg=5.71`` (``MemoryMazeDataGen/.../render/depth.py``)
    but ``_agent_pose_to_c2w`` builds a level camera, so RGB/depth and the pose would disagree.
    Two independent estimators sweep a candidate pitch and should agree on the true value:

    - **structure** (single-frame, absolute): unproject GT depth and measure the height of every
      point above the camera eye. The maze is axis-aligned, so heights must concentrate at the
      floor (``-eye_height``) and stay below the wall top (``wall_height - eye_height``). A wrong
      pitch tilts each frame's reconstruction and smears both.
    - **reprojection** (pairwise, relative): warp frame ``a``'s GT RGBD into frame ``b`` and score
      rendered depth against ``b``'s GT depth, bucketed by yaw difference. A pose pitch error
      cancels when two views share orientation and grows with relative yaw, so a wrong pitch shows
      up as error rising across yaw buckets.

``--mode retrieval``
    Scores what ``Sparse3DCache.retrieve`` actually picks against three oracles: true co-visibility
    (measured against the target's OWN GT depth, which correctly rejects behind-wall frames),
    maze-layout line of sight, and pose deltas. Compares the training cache rule (recency dead-zone,
    includes future frames) against the AR-inference rule (stride only, causal).

``--mode offset``
    Sweeps a forward camera offset and compares where DEPTH warp accuracy peaks against where RGB
    warp accuracy peaks. The MuJoCo walker mounts the camera at ``pos = (0, 0.15, 0.3)`` in the
    head-body frame -- 0.075 cells ahead of the root body that ``agent_pos`` reports -- while
    ``render/depth.py`` casts its rays from ``agent_pos`` exactly, so RGB and depth may come from
    different viewpoints. A separation between the two optima is the evidence; a difference of
    optima is robust to whatever systematic error both modalities share.

``--dataset-pitch-deg`` sets the tilt the LOADER bakes into the poses (now 5.71 by default, i.e.
correct); ``--pitch-deg`` is a RESIDUAL sweep on top of that, so a correctly-configured loader
should show its optimum at 0. Both are needed to A/B the fix against the old level-camera poses.

Usage (CWD must be the lyra repo root):
  python -m lyra_2.eval_retrieval_audit --mode pitch --num-scenes 4
  python -m lyra_2.eval_retrieval_audit --mode retrieval --num-scenes 4 --dataset-pitch-deg 0.0
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import torch

from lyra_2._src.datasets.maze_streaming import MazeLyraDataset, build_maze_dataloader
from lyra_2.eval_warp_geometry import _warp

# Memory-Maze 15x15 scene constants, in maze cells (see MemoryMazeDataGen/.../render/depth.py).
EYE_HEIGHT = 0.45  # camera height above the floor
WALL_HEIGHT = 0.75  # wall top above the floor
TRUE_PITCH_DEG = 5.71  # atan(0.1), the renderer's fixed downward mount tilt


def _pitch_matrix(pitch_deg: float, device, dtype=torch.float32) -> torch.Tensor:
    """(4,4) camera-local rotation that tilts the optical axis down by ``pitch_deg``.

    Mirrors the depth renderer's basis construction: with camera axes (right, down, forward),
    ``forward = cos(p)*forward + sin(p)*down`` and ``down = -sin(p)*forward + cos(p)*down``,
    i.e. a rotation about the camera's right axis. Right-multiplying a c2w by this matrix
    applies the tilt in the camera's own frame.
    """
    p = math.radians(float(pitch_deg))
    cp, sp = math.cos(p), math.sin(p)
    m = torch.eye(4, device=device, dtype=dtype)
    m[1, 1], m[1, 2] = cp, sp
    m[2, 1], m[2, 2] = -sp, cp
    return m


def _apply_pitch(w2c: torch.Tensor, pitch_deg: float) -> torch.Tensor:
    """Re-express ``w2c`` [T,4,4] as if the camera carried a ``pitch_deg`` downward tilt.

    The tilt is camera-local, so ``c2w' = c2w @ M`` and ``w2c' = M^-1 @ w2c``. Because it is
    applied on the camera side, the WORLD frame is untouched: the dataset anchors it to frame 0's
    level (agent-heading) orientation, so world-up stays ``(0,-1,0)`` for every candidate pitch.
    """
    if float(pitch_deg) == 0.0:
        return w2c
    m = _pitch_matrix(pitch_deg, w2c.device, w2c.dtype)
    return torch.linalg.inv(m) @ w2c


def _apply_forward_offset(w2c: torch.Tensor, offset_cells: float) -> torch.Tensor:
    """Move the camera ``offset_cells`` along its own forward (+z) axis.

    ``c2w' = c2w @ T(0,0,d)`` puts the eye at ``t + d * forward``; hence ``w2c' = T(0,0,-d) @ w2c``.
    Used to test whether the RGB was rendered from a viewpoint displaced from ``agent_pos``: the
    MuJoCo walker mounts the camera at ``pos = (0, 0.15, 0.3)`` in the head-body frame, i.e. 0.15
    world units = 0.075 cells AHEAD of the root body that ``agent_pos`` reports, while
    ``render/depth.py`` casts its rays from ``agent_pos`` exactly.
    """
    if float(offset_cells) == 0.0:
        return w2c
    m_inv = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
    m_inv[2, 3] = -float(offset_cells)
    return m_inv @ w2c


def _up_direction(dataset_pitch_deg: float, device, dtype=torch.float32) -> torch.Tensor:
    """World-up unit vector, in the world frame the dataset actually produces.

    The dataset rebases so ``c2w[0] = I``, i.e. the world axes ARE frame 0's camera axes. When the
    loader bakes in a downward mount tilt ``p``, that camera is tilted, so true world-up expressed
    in its (right, down, forward) basis is ``(0, -cos p, -sin p)`` rather than ``(0, -1, 0)``.
    A residual pitch applied by :func:`_apply_pitch` is camera-local and does NOT move the world
    frame, so only the dataset's own pitch enters here.
    """
    p = math.radians(float(dataset_pitch_deg))
    return torch.tensor([0.0, -math.cos(p), -math.sin(p)], device=device, dtype=dtype)


def _unproject(depth_1hw: torch.Tensor, w2c_44: torch.Tensor, K_33: torch.Tensor) -> torch.Tensor:
    """GT depth -> (N,3) world points, using the same math as Sparse3DCache.add / the warp path."""
    from lyra_2._src.datasets.forward_warp_utils_pytorch import unproject_points

    pts = unproject_points(
        depth=depth_1hw.unsqueeze(0),  # [1,1,H,W]
        w2c=w2c_44.unsqueeze(0),
        intrinsic=K_33.unsqueeze(0),
        is_depth=True,
        is_ftheta=False,
        mask=(depth_1hw.unsqueeze(0) > 0),
        return_sparse=True,
    )
    return pts[0]


def _half_pixel_K(K: torch.Tensor) -> torch.Tensor:
    """Shift the principal point by -0.5 px so integer-index unprojection samples pixel centers.

    The maze depth renderer casts rays through pixel CENTERS (``(i + 0.5) / width``), and so does
    the Plucker ray embedding, but ``unproject_points`` / ``project_points`` use raw integer pixel
    indices. Subtracting 0.5 from cx/cy makes ``i - cx'`` equal ``i + 0.5 - cx``, reconciling them.
    """
    K = K.clone()
    K[..., 0, 2] -= 0.5
    K[..., 1, 2] -= 0.5
    return K


def _yaw_deg(agent_dir: np.ndarray) -> np.ndarray:
    """(T,2) unit heading -> (T,) yaw in degrees."""
    return np.degrees(np.arctan2(agent_dir[:, 1], agent_dir[:, 0]))


def _yaw_delta(a: float, b: float) -> float:
    """Absolute yaw difference in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------------------------
# Mode: pitch
# ---------------------------------------------------------------------------------------------

YAW_BUCKETS = [(0, 15), (15, 45), (45, 90), (90, 135), (135, 180)]


def _structure_stats(depth: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor, pitch_deg: float,
                     frames: list[int], dataset_pitch_deg: float) -> dict:
    """Height-distribution statistics of the reconstructed point cloud for one candidate pitch.

    Returns the fraction of points landing on the floor plane (a sharp spike iff the pose is
    right), the fraction outside the physically possible height band, and the spread of the
    near-floor band.
    """
    w2c_p = _apply_pitch(w2c, pitch_deg)
    up = _up_direction(dataset_pitch_deg, depth.device)
    heights = []
    for t in frames:
        pts = _unproject(depth[t], w2c_p[t], K[t])
        if pts.numel() == 0:
            continue
        heights.append(pts @ up)  # height above the eye, in cells
    if not heights:
        return {"floor_spike": float("nan"), "frac_out": float("nan"), "floor_std": float("nan")}
    h = torch.cat(heights)

    floor_h, top_h = -EYE_HEIGHT, WALL_HEIGHT - EYE_HEIGHT
    floor_spike = float(((h - floor_h).abs() < 0.03).float().mean().item())
    frac_out = float(((h < floor_h - 0.03) | (h > top_h + 0.03)).float().mean().item())
    band = h[h < floor_h + 0.15]  # near-floor band; smears when the reconstruction is tilted
    floor_std = float(band.std().item()) if band.numel() > 16 else float("nan")
    return {"floor_spike": floor_spike, "frac_out": frac_out, "floor_std": floor_std}


def _reproj_stats(video: torch.Tensor, depth: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor,
                  pairs: list[tuple[int, int]], yaws: np.ndarray, pitch_deg: float) -> dict:
    """Per-yaw-bucket warp accuracy for one candidate pitch (inlier fraction of rendered depth)."""
    w2c_p = _apply_pitch(w2c, pitch_deg)
    per_bucket = defaultdict(list)
    for a, b in pairs:
        f1 = video[:, :, a].unsqueeze(1)
        d1 = depth[:, a].unsqueeze(1)
        _, wd, wm = _warp(
            f1, d1, w2c_p[None, a].unsqueeze(1), K[None, a].unsqueeze(1),
            w2c_p[None, b], K[None, b], clean_continuity=False, ratio_thresh=None,
        )
        gt_d = depth[0, b, 0]
        m = wm & (gt_d > 0) & (wd > 0)
        if int(m.sum()) < 32:
            continue
        rel = (wd[m] - gt_d[m]).abs() / gt_d[m]
        inlier = float((rel < 0.10).float().mean().item())
        dy = _yaw_delta(float(yaws[a]), float(yaws[b]))
        for lo, hi in YAW_BUCKETS:
            if lo <= dy < hi or (hi == 180 and dy == 180):
                per_bucket[(lo, hi)].append(inlier)
                break
    return {k: float(np.mean(v)) for k, v in per_bucket.items() if v}


def _offset_stats(video: torch.Tensor, depth: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor,
                  pairs: list[tuple[int, int]], offset_cells: float) -> dict:
    """Depth AND RGB warp accuracy for one candidate forward camera offset.

    Both modalities are scored from the SAME warp so the only thing that differs between them is
    what they are compared against. If depth is anchored at ``agent_pos`` but RGB is rendered from
    a viewpoint ahead of it, the two curves peak at different offsets, and that separation is the
    measurement -- a difference of optima is robust to whatever systematic error the two share.
    """
    from lyra_2.eval_warp_geometry import _masked_psnr

    w2c_o = _apply_forward_offset(w2c, offset_cells)
    d_inl, rgb_psnr, npix = [], [], []
    for a, b in pairs:
        f1 = video[:, :, a].unsqueeze(1)
        d1 = depth[:, a].unsqueeze(1)
        w_rgb, wd, wm = _warp(
            f1, d1, w2c_o[None, a].unsqueeze(1), K[None, a].unsqueeze(1),
            w2c_o[None, b], K[None, b], clean_continuity=False, ratio_thresh=None,
        )
        gt_d = depth[0, b, 0]
        m = wm & (gt_d > 0) & (wd > 0)
        if int(m.sum()) < 64:
            continue
        rel = (wd[m] - gt_d[m]).abs() / gt_d[m]
        d_inl.append(float((rel < 0.10).float().mean().item()))
        rgb_psnr.append(_masked_psnr(w_rgb, video[0, :, b], m))
        npix.append(int(m.sum()))
    if not d_inl:
        return {"depth_inlier": float("nan"), "rgb_psnr": float("nan"), "npix": 0.0}
    return {
        "depth_inlier": float(np.mean(d_inl)),
        "rgb_psnr": float(np.nanmean(rgb_psnr)),
        "npix": float(np.mean(npix)),
    }


def run_offset(args, device: str) -> None:
    """Sweep a forward camera offset and compare where depth vs RGB warp accuracy peaks.

    Tests the MuJoCo walker's unmodelled ``pos[1] = 0.15`` (= 0.075 cells) camera mount offset:
    the RGB comes from that displaced viewpoint while ``render/depth.py`` casts from ``agent_pos``.
    """
    offsets = [float(x) for x in args.offset_sweep.split(",") if x.strip()]

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed,
        camera_pitch_deg=args.dataset_pitch_deg,
    )
    it = iter(build_maze_dataloader(ds, batch_size=1, num_workers=0))
    acc = {o: defaultdict(list) for o in offsets}

    for i in range(args.num_scenes):
        batch = next(it)
        video = batch["video"].to(device)
        depth = batch["depth"].to(device).float()
        w2c = batch["camera_w2c"][0].to(device).float()
        K = batch["intrinsics"][0].to(device).float()
        if args.half_pixel:
            K = _half_pixel_K(K)
        pos = batch["agent_pos"][0].numpy()
        T = int(video.shape[2])

        frames = list(range(0, T, max(1, T // args.num_probe_frames)))[: args.num_probe_frames]
        rng = np.random.default_rng(args.seed + i)
        pairs = []
        for a in frames:
            near = np.flatnonzero(np.linalg.norm(pos - pos[a], axis=1) < args.max_pair_dist)
            near = near[near != a]
            if near.size == 0:
                continue
            take = rng.choice(near, size=min(args.pairs_per_frame, near.size), replace=False)
            pairs.extend((a, int(b)) for b in take)

        for o in offsets:
            for k, v in _offset_stats(video, depth, w2c, K, pairs, o).items():
                acc[o][k].append(v)
        print(f"[offset] scene {i} done ({len(pairs)} pairs)")

    def mn(o, k):
        vals = [x for x in acc[o][k] if not math.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n==== forward camera offset sweep ({args.num_scenes} scenes) ====")
    print("Hypothesis: depth is cast from agent_pos (peak at 0.000) but RGB comes from the mount")
    print("position 0.075 cells ahead (peak at ~+0.075). A separation of the two optima is the signal.")
    def sd(o, k):
        vals = [x for x in acc[o][k] if not math.isnan(x)]
        return float(np.std(vals)) if len(vals) > 1 else float("nan")

    print("offset_cells\tdepth_inlier\trgb_psnr\trgb_sd\tpixels")
    for o in offsets:
        print(f"{o:+.4f}\t\t{mn(o,'depth_inlier'):.4f}\t\t{mn(o,'rgb_psnr'):.3f}\t\t"
              f"{sd(o,'rgb_psnr'):.3f}\t{mn(o,'npix'):.0f}")
    best_d = max(offsets, key=lambda o: mn(o, "depth_inlier"))
    best_r = max(offsets, key=lambda o: mn(o, "rgb_psnr"))
    print(f"-> grid argmax: depth = {best_d:+.4f} ; RGB = {best_r:+.4f} "
          f"(separation {best_r - best_d:+.4f})")

    # Per-scene argmax: the spread across scenes is the honest uncertainty on the estimate,
    # and it does not assume the grid happens to contain the true value.
    n_scenes = len(acc[offsets[0]]["rgb_psnr"])
    per_scene = []
    for i in range(n_scenes):
        vals = [(acc[o]["rgb_psnr"][i], o) for o in offsets if not math.isnan(acc[o]["rgb_psnr"][i])]
        if vals:
            per_scene.append(max(vals)[1])
    if per_scene:
        print(f"-> per-scene RGB argmax: {[f'{x:+.4f}' for x in per_scene]}")
        print(f"   mean {np.mean(per_scene):+.4f} +/- {np.std(per_scene):.4f} (n={len(per_scene)})")

    # Parabola through the grid argmax and its two neighbours -> sub-grid peak estimate.
    gi = offsets.index(best_r)
    if 0 < gi < len(offsets) - 1:
        x0, x1, x2 = offsets[gi - 1], offsets[gi], offsets[gi + 1]
        y0, y1, y2 = (mn(o, "rgb_psnr") for o in (x0, x1, x2))
        d1, d2 = (y1 - y0) / (x1 - x0), (y2 - y1) / (x2 - x1)
        if d1 != d2:
            peak = ((x0 + x1) / 2 * d2 - (x1 + x2) / 2 * d1) / (d2 - d1)
            print(f"-> parabolic sub-grid RGB peak = {peak:+.4f} cells")
    print("[offset] done.")


def run_pitch(args, device: str) -> None:
    """Sweep a candidate camera pitch and report where each estimator bottoms out."""
    pitches = [float(x) for x in args.pitch_sweep.split(",") if x.strip()]

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed,
        camera_pitch_deg=args.dataset_pitch_deg,
    )
    it = iter(build_maze_dataloader(ds, batch_size=1, num_workers=0))

    struct_acc = {p: defaultdict(list) for p in pitches}
    reproj_acc = {p: defaultdict(list) for p in pitches}

    for i in range(args.num_scenes):
        batch = next(it)
        video = batch["video"].to(device)
        depth = batch["depth"].to(device).float()  # [1,T,1,H,W]
        w2c = batch["camera_w2c"][0].to(device).float()  # [T,4,4]
        K = batch["intrinsics"][0].to(device).float()  # [T,3,3]
        if args.half_pixel:
            K = _half_pixel_K(K)
        yaws = _yaw_deg(batch["agent_dir"][0].numpy())
        pos = batch["agent_pos"][0].numpy()
        T = int(video.shape[2])

        frames = list(range(0, T, max(1, T // args.num_probe_frames)))[: args.num_probe_frames]
        # Pair each probe frame with nearby-in-SPACE frames only: co-visibility needs bounded
        # translation, and this is exactly the regime spatial-memory retrieval targets (same place,
        # different time/heading). Unbounded pairs are dominated by disocclusion and carry no signal.
        rng = np.random.default_rng(args.seed + i)
        pairs = []
        for a in frames:
            near = np.flatnonzero(np.linalg.norm(pos - pos[a], axis=1) < args.max_pair_dist)
            near = near[near != a]
            if near.size == 0:
                continue
            take = rng.choice(near, size=min(args.pairs_per_frame, near.size), replace=False)
            pairs.extend((a, int(b)) for b in take)

        for p in pitches:
            s = _structure_stats(depth[0], w2c, K, p, frames, args.dataset_pitch_deg)
            for k, v in s.items():
                struct_acc[p][k].append(v)
            for k, v in _reproj_stats(video, depth, w2c, K, pairs, yaws, p).items():
                reproj_acc[p][k].append(v)
        print(f"[pitch] scene {i} done ({len(frames)} probe frames, {len(pairs)} pairs)")

    def mn(d, k):
        return float(np.nanmean(d[k])) if d.get(k) else float("nan")

    print(f"\n==== structure estimator (single-frame, {args.num_scenes} scenes) ====")
    print("floor_spike = frac of points within 0.03 cells of the floor plane (HIGHER is better)")
    print("frac_out    = frac of points outside [floor, wall_top] (LOWER is better)")
    print("floor_std   = spread of the near-floor band (LOWER is better)")
    print("pitch_deg\tfloor_spike\tfrac_out\tfloor_std")
    for p in pitches:
        d = struct_acc[p]
        print(f"{p:+.2f}\t\t{mn(d,'floor_spike'):.4f}\t\t{mn(d,'frac_out'):.4f}\t\t{mn(d,'floor_std'):.4f}")
    best_spike = max(pitches, key=lambda p: mn(struct_acc[p], "floor_spike"))
    best_out = min(pitches, key=lambda p: mn(struct_acc[p], "frac_out"))
    print(f"-> argmax floor_spike = {best_spike:+.2f} deg ; argmin frac_out = {best_out:+.2f} deg")

    print("\n==== reprojection estimator (pairwise, inlier frac at |absrel| < 0.10; HIGHER better) ====")
    hdr = "\t".join(f"{lo}-{hi}" for lo, hi in YAW_BUCKETS)
    print(f"pitch_deg\t{hdr}")
    for p in pitches:
        row = "\t".join(f"{mn(reproj_acc[p], b):.3f}" for b in YAW_BUCKETS)
        print(f"{p:+.2f}\t\t{row}")
    for b in YAW_BUCKETS:
        best = max(pitches, key=lambda p: mn(reproj_acc[p], b))
        print(f"-> yaw {b[0]}-{b[1]} deg: best pitch = {best:+.2f} deg")
    print("[pitch] done.")


# ---------------------------------------------------------------------------------------------
# Mode: retrieval
# ---------------------------------------------------------------------------------------------


def _oracle_covis(pts_world: torch.Tensor, w2c_now: torch.Tensor, K_now: torch.Tensor,
                  gt_depth_now: torch.Tensor, tol: float) -> torch.Tensor:
    """Target pixels a cached frame TRULY shares with the target view, as a boolean [H,W] mask.

    Projects the candidate's GT point cloud into the target and keeps a pixel only when the
    projected depth agrees with the target's OWN ground-truth depth. This is what
    ``Sparse3DCache.retrieve``'s cache-only z-buffer is approximating: a frame on the far side of a
    wall projects points into the target's frustum, but at a depth far behind the wall the target
    actually sees, so it is correctly rejected here and (often) not by the model.
    """
    H, W = gt_depth_now.shape[-2:]
    p_cam = (w2c_now[:3, :3] @ pts_world.T).T + w2c_now[:3, 3]
    z = p_cam[:, 2]
    uv = (K_now @ p_cam.T).T
    x = torch.round(uv[:, 0] / (z + 1e-7)).long()
    y = torch.round(uv[:, 1] / (z + 1e-7)).long()
    ok = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not bool(ok.any()):
        return torch.zeros((H, W), dtype=torch.bool, device=gt_depth_now.device)
    x, y, z = x[ok], y[ok], z[ok]
    gt = gt_depth_now[y, x]
    agree = (gt > 0) & ((z - gt).abs() / gt.clamp(min=1e-6) < tol)
    mask = torch.zeros((H, W), dtype=torch.bool, device=gt_depth_now.device)
    mask[y[agree], x[agree]] = True
    return mask


def _line_of_sight(maze: np.ndarray, pa: np.ndarray, pb: np.ndarray, samples: int = 64) -> bool:
    """True if the straight segment between two agent positions crosses no wall cell.

    Wall grid convention copied from the depth raycaster: ``maze == 0`` is solid, indexed
    ``[row=y, col=x]``, and out-of-bounds counts as the maze's enclosing boundary wall. Walls are
    taller than the eye, so a solid cell always blocks.
    """
    Hg, Wg = maze.shape
    ts = np.linspace(0.0, 1.0, samples)[:, None]
    pts = pa[None] * (1 - ts) + pb[None] * ts
    col = np.floor(pts[:, 0]).astype(int)
    row = np.floor(pts[:, 1]).astype(int)
    oob = (col < 0) | (col >= Wg) | (row < 0) | (row >= Hg)
    solid = np.zeros(pts.shape[0], dtype=bool)
    inb = ~oob
    solid[inb] = maze[row[inb], col[inb]] == 0
    return not bool((solid | oob).any())


def _build_cache(cache_cls, depth, w2c, K, downsample, rule, now, stride, skip_recent, step, T,
                 inf_skip_recent=0):
    """Populate a Sparse3DCache with the frames the given rule admits; returns (cache, ids)."""
    if rule == "inference":
        # AR inference (lyra2_ar_inference.py:1237-1250): frame 0 is seeded, then every stride-th
        # frame that has already entered history. No recency dead-zone exists on this path;
        # ``inf_skip_recent`` > 0 simulates adding training's dead-zone causally.
        newest = max(0, now - step - int(inf_skip_recent))
        ids = [0] + [t for t in range(1, newest + 1) if t % stride == 0]
    else:
        # Training (lyra2_model.py:2423-2435): stride lattice minus frame 0, minus the buffer
        # frame, minus a +/-skip_recent dead-zone around the generated window -- and future
        # frames ARE admitted.
        t0, t1 = now - step + 1, now
        buf = t0 - 1
        ids = [
            t for t in range(T)
            if t != 0 and t != buf and (t < t0 - skip_recent or t > t1 + skip_recent) and t % stride == 0
        ]
    cache = cache_cls(downsample=downsample, store_device=str(depth.device), store_values=True)
    for t in ids:
        cache.add(depth[:, t], w2c[:, t], K[:, t], latent_index=t, frame_id=t)
    return cache, ids


def run_retrieval(args, device: str) -> None:
    """Score what retrieve() picks against true co-visibility, line of sight, and pose deltas."""
    import json
    import tempfile

    from lyra_2._src.models.lyra2_model import Sparse3DCache

    # Dogfood the LYRA_RETRIEVE_DEBUG instrumentation to recover per-candidate coverage.
    dbg_path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    os.environ["LYRA_RETRIEVE_DEBUG"] = dbg_path

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed,
        camera_pitch_deg=args.dataset_pitch_deg,
    )
    it = iter(build_maze_dataloader(ds, batch_size=1, num_workers=0))
    rows = []

    print(f"\n==== retrieval audit | rule={args.cache_rule} stride={args.stride} "
          f"skip_recent={args.skip_recent} slots={args.slots} pitch={args.pitch_deg:+.2f} deg ====")
    print("scene\tnow\tcands\tfilled\tstop\t\tsel_covis\tsel_LOS\toracle_covis\tunion\tor_union\tspearman")

    for i in range(args.num_scenes):
        batch = next(it)
        depth = batch["depth"].to(device).float()  # [1,T,1,H,W]
        w2c = _apply_pitch(batch["camera_w2c"][0].to(device).float(), args.pitch_deg)[None]
        K = batch["intrinsics"].to(device).float()
        if args.half_pixel:
            K = _half_pixel_K(K)
        pos = batch["agent_pos"][0].numpy()
        yaws = _yaw_deg(batch["agent_dir"][0].numpy())
        maze = batch["maze_layout"][0].numpy()
        T, H, W = int(depth.shape[1]), int(depth.shape[-2]), int(depth.shape[-1])

        if args.nows.strip():
            nows = [int(x) for x in args.nows.split(",") if x.strip()]
        else:
            nows = [int(f * (T - 1)) for f in (0.4, 0.55, 0.7, 0.85, 0.98)]
        for now in nows:
            cache, ids = _build_cache(
                Sparse3DCache, depth, w2c, K, args.downsample, args.cache_rule,
                now, args.stride, args.skip_recent, args.step, T,
                inf_skip_recent=args.inf_skip_recent,
            )
            if not ids:
                print(f"{i}\t{now}\t0\t-\t(no candidates)")
                continue

            open(dbg_path, "w").close()  # truncate; we only want this call's record
            retrieved = cache.retrieve(
                w2c[:, now], K[:, now], (H, W), num_latents=args.slots, max_coverage=True,
                debug_tag=now,
            )
            sel = [int(f) for (_li, f) in retrieved]
            with open(dbg_path) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            rec = json.loads(lines[-1]) if lines else {}

            gt_now = depth[0, now, 0]
            n_valid = int((gt_now > 0).sum())
            covis = {}
            for t in ids:
                pts = _unproject(depth[0, t], w2c[0, t], K[0, t])
                covis[t] = _oracle_covis(pts, w2c[0, now], K[0, now], gt_now, args.covis_tol)
            frac = {t: float(m.sum()) / max(n_valid, 1) for t, m in covis.items()}

            los = {t: _line_of_sight(maze, pos[t], pos[now]) for t in ids}
            oracle_top = sorted(ids, key=lambda t: -frac[t])[: args.slots]

            def union(fids, _covis=covis, _n=n_valid):
                """Fraction of the target's valid-depth pixels covered by ANY frame in ``fids``."""
                if not fids:
                    return 0.0
                u = torch.zeros_like(_covis[fids[0]])
                for t in fids:
                    u |= _covis[t]
                return float(u.sum()) / max(_n, 1)

            # An empty retrieval delivers zero co-visibility to the model, so it scores 0 rather
            # than dropping out of the mean -- otherwise the worst failure mode is invisible and
            # the selected-vs-oracle comparison is taken over different row sets.
            sel_covis = float(np.mean([frac[t] for t in sel])) if sel else 0.0
            sel_los = float(np.mean([los[t] for t in sel])) if sel else 0.0
            spearman = float("nan")
            cand_px = rec.get("cand_px")
            if cand_px and len(cand_px) == len(ids):
                model_rank = np.argsort(np.argsort(-np.asarray(cand_px, dtype=float)))
                oracle_rank = np.argsort(np.argsort(-np.asarray([frac[t] for t in ids])))
                if model_rank.size > 2 and model_rank.std() > 0 and oracle_rank.std() > 0:
                    spearman = float(np.corrcoef(model_rank, oracle_rank)[0, 1])

            # "Dead" = a slot that shares essentially nothing with the target view, i.e. exactly the
            # useless retrieval the rollout videos show. Counting the unfilled slots as dead makes
            # this the fraction of the model's spatial-memory budget that carries no information.
            n_slots = int(args.slots)
            dead_sel = (n_slots - len(sel) + sum(frac[t] < args.dead_thresh for t in sel)) / n_slots
            dead_pool = float(np.mean([frac[t] < args.dead_thresh for t in ids]))
            dead_oracle = sum(frac[t] < args.dead_thresh for t in oracle_top) / n_slots

            rows.append({
                "cands": len(ids), "filled": len(sel), "stop": rec.get("stop", "?"),
                "sel_covis": sel_covis, "sel_los": sel_los,
                "dead_sel": dead_sel, "dead_pool": dead_pool, "dead_oracle": dead_oracle,
                "oracle_covis": float(np.mean([frac[t] for t in oracle_top])),
                "union": union(sel), "or_union": union(oracle_top), "spearman": spearman,
                "sel_dist": float(np.mean([np.linalg.norm(pos[t] - pos[now]) for t in sel])) if sel else np.nan,
                "sel_dyaw": float(np.mean([_yaw_delta(yaws[t], yaws[now]) for t in sel])) if sel else np.nan,
                "sel_age": float(np.mean([now - t for t in sel])) if sel else np.nan,
            })
            r = rows[-1]
            print(f"{i}\t{now}\t{r['cands']}\t{r['filled']}\t{r['stop']:<12}\t{sel_covis:.3f}\t\t"
                  f"{sel_los:.2f}\t{r['oracle_covis']:.3f}\t\t{r['union']:.3f}\t{r['or_union']:.3f}\t{spearman:+.2f}")

    if not rows:
        print("[retrieval] no rows collected.")
        return

    def mn(k):
        v = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    stops = defaultdict(int)
    for r in rows:
        stops[r["stop"]] += 1
    print(f"\n==== aggregate over {len(rows)} (scene x target) positions ====")
    print(f"slots filled (of {args.slots}):\t\t{mn('filled'):.2f}\tstop reasons: {dict(stops)}")
    print(f"selected slots -> true covis:\t{mn('sel_covis'):.3f}\tline-of-sight frac: {mn('sel_los'):.2f}")
    print(f"oracle-best slots -> true covis:\t{mn('oracle_covis'):.3f}")
    print(f"DEAD slot frac (covis < {args.dead_thresh}):\tselected {mn('dead_sel'):.2f}"
          f"\toracle-best {mn('dead_oracle'):.2f}\twhole pool {mn('dead_pool'):.2f}")
    print(f"union covis selected / oracle:\t{mn('union'):.3f} / {mn('or_union'):.3f}"
          f"\t(regret {mn('or_union') - mn('union'):+.3f})")
    print(f"model-vs-oracle rank corr:\t{mn('spearman'):+.3f}")
    print(f"selected slot stats -> dist {mn('sel_dist'):.2f} cells, "
          f"|dyaw| {mn('sel_dyaw'):.0f} deg, age {mn('sel_age'):.0f} frames")
    print("[retrieval] done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Spatial-memory retrieval audit on Memory-Maze GT geometry.")
    ap.add_argument("--mode", choices=["pitch", "retrieval", "offset"], default="pitch")
    ap.add_argument("--eval-root", default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval")
    ap.add_argument("--num-frames", type=int, default=201)
    ap.add_argument("--num-scenes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # pitch mode
    ap.add_argument("--pitch-sweep", default="-8,-5.71,-3,0,2,4,5.71,8,11",
                    help="Comma-separated candidate pitches in degrees.")
    ap.add_argument("--num-probe-frames", type=int, default=12)
    ap.add_argument("--pairs-per-frame", type=int, default=6)
    ap.add_argument("--offset-sweep", default="-0.15,-0.075,0,0.0375,0.075,0.15,0.30",
                    help="Forward camera offsets in maze cells to sweep (offset mode).")
    ap.add_argument("--max-pair-dist", type=float, default=3.0,
                    help="Only pair frames whose agent positions are within this many maze cells.")
    ap.add_argument("--dataset-pitch-deg", type=float, default=5.71,
                    help="Camera mount tilt the LOADER bakes into the poses (0.0 = the old level "
                    "camera). --pitch-deg is a residual applied on top of this.")
    ap.add_argument("--half-pixel", action="store_true",
                    help="Shift cx/cy by -0.5 px so integer-index unprojection samples pixel centers.")
    # retrieval mode
    ap.add_argument("--cache-rule", choices=["inference", "training"], default="inference")
    ap.add_argument("--slots", type=int, default=3, help="num_spatial_hist.")
    ap.add_argument("--stride", type=int, default=4, help="spatial_memory_stride.")
    ap.add_argument("--skip-recent", type=int, default=16, help="spatial_memory_skip_recent (training rule).")
    ap.add_argument("--inf-skip-recent", type=int, default=0,
                    help="Causal recency dead-zone for the inference rule (0 = as shipped).")
    ap.add_argument("--step", type=int, default=12, help="framepack_num_new_video_frames (AR chunk).")
    ap.add_argument("--downsample", type=int, default=1, help="spatial_memory_downsample.")
    ap.add_argument("--pitch-deg", type=float, default=0.0,
                    help="RESIDUAL pitch applied on top of the loader's poses; 0 = use them as-is.")
    ap.add_argument("--nows", default="",
                    help="Explicit target frame indices (comma-separated); empty = spread over the clip. "
                    "Use the AR chunk boundaries to cross-check against a live rollout's JSONL.")
    ap.add_argument("--dead-thresh", type=float, default=0.01,
                    help="A slot covering less than this fraction of the target view counts as dead.")
    ap.add_argument("--covis-tol", type=float, default=0.10,
                    help="Relative depth agreement for a pixel to count as truly co-visible.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.mode == "pitch":
        run_pitch(args, args.device)
    elif args.mode == "offset":
        run_offset(args, args.device)
    else:
        run_retrieval(args, args.device)


if __name__ == "__main__":
    main()
