"""Depth Anything 3 vs ground-truth depth benchmark on Memory Maze.

Runs DA3 monocularly on 64x64 maze eval frames and compares predicted depth to the
simulator's ground-truth depth, to quantify the DA3 -> maze domain gap (the baseline that
motivates finetuning DA3, and the error of the depth that would feed ``_warp_multisrc`` if
maze inference ran with ``depth_backend="da3"``).

DA3 monocular depth is only defined up to scale (and often shift), so metrics are reported
in three regimes per frame: raw ``metric`` (as returned; DA3's nested model sets is_metric=1
but its scale won't match maze-cell units), ``scale`` (least-squares scale-only), and
``scale_shift`` (least-squares affine) alignment to GT over valid pixels. The scale_shift
number is the fair, resolution/scale-independent measure of DA3's intrinsic depth quality.

The model loads from the LOCAL checkpoint bundled in the Lyra-2.0 snapshot
(``checkpoints/recon/model.pt`` -> ``da3nested-giant-large``); no Hugging Face download is
needed, which matters because the compute nodes are offline (run with HF_HUB_OFFLINE=1).

Usage (CWD must be the lyra repo root; run on a GPU node, offline-safe):
  HF_HUB_OFFLINE=1 python -m lyra_2.eval_da3_depth --num-scenes 5 --frames-per-scene 8 \
      --out outputs/eval_da3_depth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lyra_2._src.inference.depth_utils import load_da3_model
from lyra_2._src.datasets.maze_streaming import MazeLyraDataset, build_maze_dataloader
from lyra_2.eval_warp_geometry import _depth_metrics, _rgb_to_uint8, _depth_to_uint8


def _da3_depth(model, img_uint8_hwc: np.ndarray, out_hw: tuple[int, int]) -> torch.Tensor:
    """Run DA3 monocular inference on one HWC uint8 image; return depth [H,W] at ``out_hw``.

    DA3 pads the input to a patch multiple (e.g. 64 -> 70), so the prediction is bilinearly
    resized back to the GT resolution before comparison.
    """
    pred = model.inference(
        image=[img_uint8_hwc],
        extrinsics=None,
        intrinsics=None,
        infer_gs=False,
        process_res=max(out_hw),
        process_res_method="upper_bound_resize",
        export_dir=None,
    )
    d = torch.from_numpy(np.asarray(pred.depth)[0]).float()  # [H',W']
    d = F.interpolate(d[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    return d


def _align(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Align ``pred`` to ``gt`` over ``mask`` by least squares. mode in {metric, scale, scale_shift}."""
    if mode == "metric":
        return pred
    p = pred[mask].double()
    g = gt[mask].double()
    if p.numel() < 2:
        return pred
    if mode == "scale":
        s = (p * g).sum() / (p * p).sum().clamp_min(1e-8)
        return pred * s
    if mode == "scale_shift":
        n = torch.tensor(float(p.numel()), dtype=torch.float64, device=p.device)
        Spp, Sp, Spg, Sg = (p * p).sum(), p.sum(), (p * g).sum(), g.sum()
        A = torch.stack([torch.stack([Spp, Sp]), torch.stack([Sp, n])])
        b = torch.stack([Spg, Sg])
        s, t = torch.linalg.solve(A, b)
        return pred.double() * s + t
    raise ValueError(f"unknown align mode {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3-vs-GT depth benchmark on Memory Maze.")
    ap.add_argument("--eval-root", default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval")
    ap.add_argument("--num-frames", type=int, default=201)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--frames-per-scene", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-name", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--checkpoint", default="checkpoints/recon/model.pt", help="Local DA3 checkpoint.")
    ap.add_argument("--out", default="outputs/eval_da3_depth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["metric", "scale", "scale_shift"]

    model = load_da3_model(
        da3_model_name=args.model_name, da3_model_path_custom=args.checkpoint, device=device
    )

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed
    )
    dl = build_maze_dataloader(ds, batch_size=1, num_workers=0)
    it = iter(dl)

    agg = {mode: {"absrel": [], "rmse": [], "delta1": []} for mode in modes}

    for i in range(args.num_scenes):
        batch = next(it)
        video = batch["video"]              # [1,3,T,H,W] in [-1,1]
        depth = batch["depth"].float()      # [1,T,1,H,W], metric maze cells, 0 = invalid/sky
        T, H, W = video.shape[2], video.shape[3], video.shape[4]
        frame_ids = np.linspace(0, T - 1, args.frames_per_scene).round().astype(int)
        key = batch["__key__"][0] if isinstance(batch["__key__"], list) else str(batch["__key__"])

        rows = []
        for f in frame_ids:
            rgb = video[0, :, f]                                   # [3,H,W]
            gt = depth[0, f, 0].to(device)                        # [H,W]
            valid = gt > 0
            if valid.sum() < 16:
                continue

            img = ((rgb * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
            pred = _da3_depth(model, img, (H, W)).to(device)      # [H,W]

            frame_metrics = {}
            for mode in modes:
                aligned = _align(pred, gt, valid, mode).to(device)
                m_valid = valid & torch.isfinite(aligned) & (aligned > 0)
                m = _depth_metrics(aligned, gt, m_valid)
                frame_metrics[mode] = (m, aligned)
                agg[mode]["absrel"].append(m["absrel"])
                agg[mode]["rmse"].append(m["rmse"])
                agg[mode]["delta1"].append(m["delta1"])

            ss_metrics, ss_aligned = frame_metrics["scale_shift"]
            print(
                f"[da3] scene {i} ({key}) f={f:3d}\t"
                + "  ".join(
                    f"{mode}:absrel={frame_metrics[mode][0]['absrel']:.3f}/d1={frame_metrics[mode][0]['delta1']:.3f}"
                    for mode in modes
                )
            )

            # Viz row: RGB | GT depth | DA3 (scale_shift aligned) | abs error
            vmin = float(gt[valid].min())
            vmax = float(gt[valid].max())
            err = (ss_aligned.float() - gt).abs()
            row = np.concatenate(
                [
                    _rgb_to_uint8(rgb),
                    _depth_to_uint8(gt, valid, vmin, vmax),
                    _depth_to_uint8(ss_aligned.float(), valid, vmin, vmax),
                    _depth_to_uint8(err, valid, 0.0, max(float(err[valid].max()), 1e-6)),
                ],
                axis=1,
            )
            rows.append(row)

        if rows:
            grid = np.concatenate(rows, axis=0)
            path = out_dir / f"scene_{i:02d}_{key}.png"
            try:
                import imageio.v2 as imageio

                imageio.imwrite(path, grid)
            except ImportError:
                np.save(path.with_suffix(".npy"), grid)
            print(f"[da3] scene {i} ({key}) -> {path}  (panels: RGB | GT | DA3(scale_shift) | |err|)")

    print("\n==== aggregate (mean over all frames) ====")
    print("regime\t\tabsrel\trmse\tdelta1\tn")
    for mode in modes:
        a = agg[mode]
        if not a["absrel"]:
            continue
        label = mode if len(mode) >= 8 else mode + "\t"
        print(
            f"{label}\t{np.nanmean(a['absrel']):.4f}\t{np.nanmean(a['rmse']):.4f}\t"
            f"{np.nanmean(a['delta1']):.4f}\t{len(a['absrel'])}"
        )
    print("[da3] done.")


if __name__ == "__main__":
    main()
