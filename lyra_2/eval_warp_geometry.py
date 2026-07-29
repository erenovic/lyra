"""Forward-warp geometry sanity check for the Memory-Maze pipeline.

Validates the camera geometry (``_agent_pose_to_c2w`` poses, pixel-space intrinsics,
unprojection, and forward splatting) *independently* of the diffusion model and DA3, by
warping each source frame's GROUND-TRUTH RGBD into chosen target camera views with the
same ``forward_warp_multiframes`` call the model uses in ``_apply_camera_controls`` and
comparing the warped depth/RGB to the GT at the target view.

Because the input depth is perfect, residual error at co-visible pixels reflects only
splatting/quantization -- it should be near zero. Large error concentrated in newly
disoccluded regions is expected and correct. Big error at co-visible pixels means a
pose/intrinsics/warp bug that must be fixed before DA3 depth numbers are meaningful.

Usage (CWD must be the lyra repo root; run on a GPU node):
  python -m lyra_2.eval_warp_geometry --num-scenes 4 --src-frame 40 \
      --target-offsets 4,8,16,32 --out outputs/warp_geometry
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lyra_2._src.datasets.maze_streaming import MazeLyraDataset, build_maze_dataloader
from lyra_2._src.datasets.forward_warp_utils_pytorch import (
    forward_warp_multiframes,
    reliable_depth_mask_range_batch,
)


def _reliable_clean_depth(depth1: torch.Tensor, ratio_thresh: float) -> torch.Tensor:
    """Zero source-depth pixels whose 5x5 neighborhood depth spread exceeds ``ratio_thresh``.

    ``depth1`` is [B,V,1,H,W]. Returns a masked copy so a subsequent
    ``forward_warp_multiframes(clean_points=False)`` reproduces the model's continuity
    cleaning at an arbitrary threshold (the built-in call hardcodes ratio_thresh=0.05).
    """
    B, V = depth1.shape[:2]
    d = depth1.reshape(B * V, 1, depth1.shape[3], depth1.shape[4])
    mask = reliable_depth_mask_range_batch(d, ratio_thresh=ratio_thresh)  # [B*V,1,H,W]
    d = d * mask.to(d.dtype)
    return d.reshape(B, V, 1, depth1.shape[3], depth1.shape[4])


def _warp(
    frame1: torch.Tensor,
    depth1: torch.Tensor,
    w2c1: torch.Tensor,
    K1: torch.Tensor,
    w2c_t: torch.Tensor,
    K_t: torch.Tensor,
    *,
    clean_continuity: bool,
    ratio_thresh: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-warp source RGBD (b,v=1,...) into a target view. Returns (rgb[3,H,W], depth[H,W], mask[H,W]).

    If ``ratio_thresh`` is not None, apply continuity cleaning at that threshold via
    source-depth pre-masking and disable the warp's built-in cleaning; otherwise defer to
    the warp's own flags (``clean_continuity`` toggles both clean_points/continuity).
    """
    if ratio_thresh is not None:
        depth1 = _reliable_clean_depth(depth1, ratio_thresh)
        cp = cpc = False
    else:
        cp = cpc = clean_continuity
    w_img, w_mask, w_depth, _ = forward_warp_multiframes(
        frame1,
        mask1=None,
        depth1=depth1,
        transformation1=w2c1,
        transformation2=w2c_t,
        intrinsic1=K1,
        intrinsic2=K_t,
        is_image=True,
        render_depth=True,
        clean_points=cp,
        clean_points_continuity=cpc,
    )
    return w_img[0], w_depth[0], (w_mask[0, 0] > 0.5)


def _depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
    """Standard monocular-depth metrics over ``mask`` (all tensors broadcast-compatible)."""
    p = pred[mask].double()
    g = gt[mask].double()
    if p.numel() == 0:
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "count": 0}
    absrel = (torch.abs(p - g) / g).mean().item()
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    ratio = torch.maximum(p / g, g / p)
    delta1 = (ratio < 1.25).double().mean().item()
    return {"absrel": absrel, "rmse": rmse, "delta1": delta1, "count": int(p.numel())}


def _masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR between two [-1,1] RGB images [3,H,W] over a [H,W] mask (0 if empty)."""
    m = mask.unsqueeze(0).expand_as(pred)
    if m.sum() == 0:
        return float("nan")
    p = (pred * 0.5 + 0.5).clamp(0, 1)
    g = (gt * 0.5 + 0.5).clamp(0, 1)
    mse = ((p - g) ** 2)[m].double().mean()
    return float((-10.0 * torch.log10(mse)).item())


def _rgb_to_uint8(rgb_3hw: torch.Tensor) -> np.ndarray:
    """[-1,1] [3,H,W] -> (H,W,3) uint8."""
    v = rgb_3hw.permute(1, 2, 0).float().clamp(-1, 1)
    return ((v + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()


def _depth_to_uint8(depth_hw: torch.Tensor, valid_hw: torch.Tensor, vmin: float, vmax: float) -> np.ndarray:
    """Metric depth [H,W] -> (H,W,3) uint8, shared [vmin,vmax] scaling; invalid pixels black."""
    d = depth_hw.float().cpu()
    norm = ((d - vmin) / max(vmax - vmin, 1e-6)).clamp(0, 1)
    try:
        import matplotlib.cm as cm

        rgb = cm.get_cmap("turbo")(norm.numpy())[..., :3]
        rgb = (rgb * 255).astype(np.uint8)
    except Exception:
        g = (norm.numpy() * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    rgb[~valid_hw.cpu().numpy()] = 0
    return rgb


def run_sweep(args, device: str) -> None:
    """Sweep the depth-continuity ``ratio_thresh`` and report coverage vs warp accuracy.

    Collects source/target pairs across scenes x offsets once, then warps each pair at
    every threshold. Higher coverage means fewer points discarded; AbsRel/delta1 measure
    warp fidelity on the surviving pixels. The knee (coverage rises with error still low)
    is the value to use for the maze config.
    """
    offsets = [int(x) for x in args.target_offsets.split(",") if x.strip()]
    thresholds: list[float | None] = []
    for tok in args.sweep.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        thresholds.append(None if tok in ("off", "none", "raw", "inf") else float(tok))

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed
    )
    dl = build_maze_dataloader(ds, batch_size=1, num_workers=0)
    it = iter(dl)

    pairs: list[dict] = []
    for _ in range(args.num_scenes):
        batch = next(it)
        video = batch["video"].to(device)
        depth = batch["depth"].to(device).float()
        w2c = batch["camera_w2c"].to(device).float()
        K = batch["intrinsics"].to(device).float()
        T = video.shape[2]
        s = args.src_frame
        for o in offsets:
            t = s + o
            if t >= T:
                continue
            pairs.append(
                dict(
                    frame1=video[:, :, s].unsqueeze(1),
                    depth1=depth[:, s].unsqueeze(1),
                    w2c1=w2c[:, s].unsqueeze(1),
                    K1=K[:, s].unsqueeze(1),
                    w2c_t=w2c[:, t],
                    K_t=K[:, t],
                    gt_d=depth[0, t, 0],
                    gt_rgb=video[0, :, t],
                )
            )

    print(
        f"\n==== ratio_thresh sweep | {args.num_scenes} scenes x offsets {offsets} "
        f"= {len(pairs)} pairs (src={args.src_frame}) ===="
    )
    print("ratio_thresh\tcoverage\tabsrel\tdelta1\tpsnr")
    for thr in thresholds:
        cov, absrel, d1, psnr = [], [], [], []
        for p in pairs:
            # thr=None -> raw (no continuity cleaning); numeric -> pre-mask at that threshold.
            w_rgb, w_d, w_m = _warp(
                p["frame1"], p["depth1"], p["w2c1"], p["K1"], p["w2c_t"], p["K_t"],
                clean_continuity=False, ratio_thresh=thr,
            )
            valid = w_m & (p["gt_d"] > 0) & (w_d > 0)
            m = _depth_metrics(w_d, p["gt_d"], valid)
            cov.append(float(valid.float().mean().item()))
            absrel.append(m["absrel"])
            d1.append(m["delta1"])
            psnr.append(_masked_psnr(w_rgb, p["gt_rgb"], valid))
        label = "off" if thr is None else f"{thr:g}"
        print(
            f"{label}\t{np.nanmean(cov):.3f}\t\t{np.nanmean(absrel):.4f}\t"
            f"{np.nanmean(d1):.4f}\t{np.nanmean(psnr):.2f}"
        )
    print("[warp] sweep done.")


def run_retrieval_render(args, device: str) -> None:
    """Compare spatial-memory RETRIEVAL health vs warp RENDER health on GT geometry.

    For each clip and target ("now") frame, build the Sparse3DCache exactly as the model does
    (``stride`` / ``skip_recent`` / ``downsample``), retrieve the top-k slots against the "now"
    view (``max_coverage`` greedy, as at inference), then forward-warp each retrieved frame into
    that view twice:
      - RAW (no continuity cleaning) -> the geometric ceiling the retrieval delivers = RETRIEVAL
        health (does memory find frames that actually overlap the current view?).
      - continuity-cleaned at ``--rr-render-thresh`` -> what survives = RENDER health.
    Reports per-slot / union coverage, render yield (rendered / geometric), and warp accuracy.

    Cache mode: causal (past-only, mirrors AR inference) by default; ``--rr-full-clip`` includes
    future frames (mirrors training) so the train/inference asymmetry can be quantified.
    """
    from lyra_2._src.models.lyra2_model import Sparse3DCache

    stride, skip_recent = int(args.rr_stride), int(args.rr_skip_recent)
    downsample, num_slots, thr = int(args.rr_downsample), int(args.rr_slots), float(args.rr_render_thresh)

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed
    )
    dl = build_maze_dataloader(ds, batch_size=1, num_workers=0)
    it = iter(dl)

    agg = {k: [] for k in ("slots", "geom_s1", "geom_union", "rend_union", "yield", "absrel", "psnr")}
    mode = "full-clip (train-like)" if args.rr_full_clip else "causal (inference-like)"
    print(
        f"\n==== retrieval vs render | stride={stride} skip_recent={skip_recent} downsample={downsample} "
        f"slots={num_slots} render_thresh={thr} | cache={mode} ===="
    )
    print("scene\tnow\tslots\tgeom_s1\tgeom_union\trend_union\tyield\tabsrel\tpsnr")

    for i in range(args.num_scenes):
        batch = next(it)
        video = batch["video"].to(device)              # [1,3,T,H,W]
        depth = batch["depth"].to(device).float()      # [1,T,1,H,W]
        w2c = batch["camera_w2c"].to(device).float()   # [1,T,4,4]
        K = batch["intrinsics"].to(device).float()     # [1,T,3,3]
        T = video.shape[2]
        H, W = int(video.shape[-2]), int(video.shape[-1])

        if args.rr_now.strip():
            nows = [int(x) for x in args.rr_now.split(",") if x.strip()]
        else:
            nows = [int(f * (T - 1)) for f in (0.5, 0.7, 0.9)]

        for now in nows:
            if now <= skip_recent or now >= T:
                continue
            cache = Sparse3DCache(downsample=downsample, store_device=str(device), store_values=True)
            for t in range(T):
                if args.rr_full_clip:
                    if abs(t - now) <= skip_recent:  # exclude the window neighborhood (train-like)
                        continue
                elif t >= now - skip_recent:  # causal: only frames strictly in the past-minus-recent
                    continue
                if t == 0 or (t % stride != 0):
                    continue
                cache.add(depth[:, t], w2c[:, t], K[:, t], latent_index=t, frame_id=t)

            if len(cache._frame_ids) == 0:
                print(f"{i}\t{now}\t0\t(no cache candidates)")
                continue
            retrieved = cache.retrieve(
                w2c[:, now], K[:, now], (H, W), num_latents=num_slots, max_coverage=True
            )
            n_filled = len(retrieved)
            gt_d = depth[0, now, 0]
            gt_rgb = video[0, :, now]
            gt_valid = gt_d > 0

            geom_union = torch.zeros((H, W), dtype=torch.bool, device=device)
            rend_union = torch.zeros_like(geom_union)
            geom_s1, absrels, psnrs = float("nan"), [], []
            for si, (_li, fid) in enumerate(retrieved):
                f1, d1 = video[:, :, fid].unsqueeze(1), depth[:, fid].unsqueeze(1)
                w1, k1 = w2c[:, fid].unsqueeze(1), K[:, fid].unsqueeze(1)
                _, wd_raw, wm_raw = _warp(
                    f1, d1, w1, k1, w2c[:, now], K[:, now], clean_continuity=False, ratio_thresh=None
                )
                v_raw = wm_raw & gt_valid & (wd_raw > 0)
                w_rgb, wd, wm = _warp(
                    f1, d1, w1, k1, w2c[:, now], K[:, now], clean_continuity=False, ratio_thresh=thr
                )
                v_rend = wm & gt_valid & (wd > 0)
                geom_union |= v_raw
                rend_union |= v_rend
                if si == 0:
                    geom_s1 = float(v_raw.float().mean().item())
                m = _depth_metrics(wd, gt_d, v_rend)
                absrels.append(m["absrel"])
                psnrs.append(_masked_psnr(w_rgb, gt_rgb, v_rend))

            gu = float(geom_union.float().mean().item())
            ru = float(rend_union.float().mean().item())
            yld = (ru / gu) if gu > 0 else float("nan")
            absrel = float(np.nanmean(absrels)) if absrels else float("nan")
            psnr = float(np.nanmean(psnrs)) if psnrs else float("nan")
            for k, v in zip(agg, (n_filled, geom_s1, gu, ru, yld, absrel, psnr), strict=True):
                agg[k].append(v)
            print(
                f"{i}\t{now}\t{n_filled}\t{geom_s1:.3f}\t{gu:.3f}\t\t{ru:.3f}\t\t"
                f"{yld:.2f}\t{absrel:.4f}\t{psnr:.2f}"
            )

    def mn(k):
        return float(np.nanmean(agg[k])) if agg[k] else float("nan")

    print(f"\n==== aggregate (mean over {len(agg['slots'])} scene x now positions) ====")
    print(f"avg slots filled (of {num_slots}):            {mn('slots'):.2f}")
    print(f"RETRIEVAL health -> geom coverage slot1={mn('geom_s1'):.3f}  union={mn('geom_union'):.3f}")
    print(f"RENDER health    -> rendered union coverage={mn('rend_union'):.3f}  yield(rend/geom)={mn('yield'):.2f}")
    print(f"warp accuracy on rendered pixels -> absrel={mn('absrel'):.4f}  psnr={mn('psnr'):.2f}")
    print("[retrieval-render] done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward-warp geometry check on Memory-Maze GT depth.")
    ap.add_argument("--eval-root", default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval")
    ap.add_argument("--num-frames", type=int, default=201)
    ap.add_argument("--num-scenes", type=int, default=4)
    ap.add_argument("--src-frame", type=int, default=40, help="Source frame index within the clip.")
    ap.add_argument("--target-offsets", default="1,2,4,8", help="Comma-separated target offsets from src.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/eval_warp_geometry")
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable point cleaning / depth-continuity masking (default mirrors the model: on).",
    )
    ap.add_argument(
        "--ratio-thresh",
        type=float,
        default=None,
        help="Viz mode: continuity-clean at this threshold instead of the model's 0.05.",
    )
    ap.add_argument(
        "--sweep",
        default=None,
        help="Sweep mode: comma-separated ratio_thresh values ('off' = no continuity cleaning), "
        "e.g. 'off,0.05,0.1,0.2,0.3'. Prints a coverage/accuracy table and skips image output.",
    )
    # Retrieval-vs-render mode: builds the real Sparse3DCache, retrieves top-k, warps them.
    ap.add_argument("--retrieval-render", action="store_true", help="Retrieval-vs-render health mode.")
    ap.add_argument("--rr-now", default="", help="Comma target frame indices; empty = auto (0.5/0.7/0.9 of T).")
    ap.add_argument("--rr-stride", type=int, default=8)
    ap.add_argument("--rr-skip-recent", type=int, default=16)
    ap.add_argument("--rr-downsample", type=int, default=1)
    ap.add_argument("--rr-slots", type=int, default=3, help="num_spatial_hist (spatial slots).")
    ap.add_argument("--rr-render-thresh", type=float, default=0.45, help="warp_continuity_ratio_thresh.")
    ap.add_argument("--rr-full-clip", action="store_true", help="Cache whole clip (train-like) vs causal past-only.")
    args = ap.parse_args()
    clean = not args.no_clean

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.retrieval_render:
        run_retrieval_render(args, device)
        return
    if args.sweep:
        run_sweep(args, device)
        return
    offsets = [int(x) for x in args.target_offsets.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed
    )
    dl = build_maze_dataloader(ds, batch_size=1, num_workers=0)
    it = iter(dl)

    # Accumulators keyed by target offset.
    agg = {o: {"absrel": [], "rmse": [], "delta1": [], "psnr": [], "coverage": []} for o in offsets}

    for i in range(args.num_scenes):
        batch = next(it)
        video = batch["video"].to(device)          # [1,3,T,H,W] in [-1,1]
        depth = batch["depth"].to(device).float()  # [1,T,1,H,W], metric (maze cells), 0 = invalid
        w2c = batch["camera_w2c"].to(device).float()   # [1,T,4,4]
        K = batch["intrinsics"].to(device).float()     # [1,T,3,3] pixel-space
        T = video.shape[2]
        s = args.src_frame
        assert 0 <= s < T, f"--src-frame {s} out of range [0,{T})"

        # Shared depth colour scale from the source frame's valid depth (stable across the row).
        src_d = depth[0, s, 0]
        src_valid = src_d > 0
        vmin = float(src_d[src_valid].min()) if src_valid.any() else 0.0
        vmax = float(src_d[src_valid].max()) if src_valid.any() else 1.0

        # Source RGBD/camera as (b, v=1, ...).
        frame1 = video[:, :, s].unsqueeze(1)        # [1,1,3,H,W]
        depth1 = depth[:, s].unsqueeze(1)           # [1,1,1,H,W]
        w2c1 = w2c[:, s].unsqueeze(1)               # [1,1,4,4]
        K1 = K[:, s].unsqueeze(1)                   # [1,1,3,3]

        rows_rgb = [_rgb_to_uint8(video[0, :, s])]
        rows_depth = [_depth_to_uint8(src_d, src_valid, vmin, vmax)]

        key = batch["__key__"][0] if isinstance(batch["__key__"], list) else str(batch["__key__"])
        for o in offsets:
            t = s + o
            if t >= T:
                # Off the end of this clip; pad viz with black and skip metrics.
                rows_rgb.append(np.zeros_like(rows_rgb[0]))
                rows_depth.append(np.zeros_like(rows_depth[0]))
                continue

            w_img_3hw, w_depth_hw, w_mask_hw = _warp(
                frame1, depth1, w2c1, K1, w2c[:, t], K[:, t],
                clean_continuity=clean, ratio_thresh=args.ratio_thresh,
            )

            gt_d = depth[0, t, 0]                    # [H,W]
            gt_rgb = video[0, :, t]                  # [3,H,W]
            gt_valid = gt_d > 0
            valid = w_mask_hw & gt_valid & (w_depth_hw > 0)

            m = _depth_metrics(w_depth_hw, gt_d, valid)
            psnr = _masked_psnr(w_img_3hw, gt_rgb, valid)
            coverage = float(valid.float().mean().item())
            # Coverage breakdown to localise where valid pixels are lost:
            #   warp_frac  -- target pixels that received any splat (source visibility + motion)
            #   gt_frac    -- target pixels with valid GT depth (non-sky)
            warp_frac = float(w_mask_hw.float().mean().item())
            gt_frac = float(gt_valid.float().mean().item())
            agg[o]["absrel"].append(m["absrel"])
            agg[o]["rmse"].append(m["rmse"])
            agg[o]["delta1"].append(m["delta1"])
            agg[o]["psnr"].append(psnr)
            agg[o]["coverage"].append(coverage)

            rows_rgb.append(_rgb_to_uint8(w_img_3hw))
            rows_depth.append(_depth_to_uint8(w_depth_hw, w_mask_hw, vmin, vmax))
            print(
                f"[warp] scene {i} ({key}) s={s} t={t} (o={o})\t"
                f"absrel={m['absrel']:.4f}\trmse={m['rmse']:.4f}\tdelta1={m['delta1']:.4f}\t"
                f"psnr={psnr:.2f}\tcoverage={coverage:.3f}\twarp_frac={warp_frac:.3f}\tgt_frac={gt_frac:.3f}"
            )

        # Two stacked rows: RGB (src | warped@offsets) and depth (src | warped@offsets).
        grid = np.concatenate(
            [np.concatenate(rows_rgb, axis=1), np.concatenate(rows_depth, axis=1)], axis=0
        )
        path = out_dir / f"scene_{i:02d}_{key}.png"
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, grid)
        except ImportError:
            np.save(path.with_suffix(".npy"), grid)

        print(f"[warp] scene {i} ({key}) -> {path}")

    print("\n==== aggregate (mean over scenes, per target offset) ====")
    print("offset\tabsrel\trmse\tdelta1\tpsnr\tcoverage")
    for o in offsets:
        a = agg[o]
        if not a["absrel"]:
            continue
        print(
            f"{o}\t{np.nanmean(a['absrel']):.4f}\t{np.nanmean(a['rmse']):.4f}\t"
            f"{np.nanmean(a['delta1']):.4f}\t{np.nanmean(a['psnr']):.2f}\t{np.nanmean(a['coverage']):.3f}"
        )
    print("[warp] done.")


if __name__ == "__main__":
    main()
