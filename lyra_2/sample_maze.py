"""Autoregressive rollout evaluation for maze-trained Lyra 2.0 checkpoints.

Drives the SAME AR inference engine that ``lyra2_zoomgs_inference.py`` uses --
``run_lyra2_sample`` / ``Lyra2InferencePipeline`` -- so maze rollouts match the
production path exactly: seed from frame 0, then generate chunk-by-chunk via
``model.inference`` with spatial-memory cache updates, warp conditioning, and streaming
decode. The only maze-specific piece is ``depth_backend="gt"`` (added to the pipeline):
the camera path is the clip's GT trajectory, so the cache is updated from GT sim depth
instead of DA3-predicted depth (DA3 is poor on maze; the model trains on GT depth).

Because the rollout follows the GT trajectory from frame 0, generated frame t aligns to
GT frame t, so the two are compared directly. Writes one video per element -- GT, gen,
(if present) warp, and a top-down camera-trajectory map (``src.visualization.topdown_maze_map``)
-- as separate mp4s (gif fallback), plus a stacked grid (PNG). With ``--slot-viz`` (default on)
it also writes ``slots.mp4`` (the 3 retrieved spatial-memory frames per AR step, gen-over-GT,
id-labeled, color-bordered) and ``topdown_slots.mp4`` (the map with color-matched rings at those
retrieved cameras). A pixel-MSE and early-vs-late drift signal are printed. ``--num-gen-frames``
sets the length. Drift accumulates over the rollout -- that's expected for a world-model generation.

The checkpoint path is the DCP checkpoint DIRECTORY (contains ``model/``, ``optim/``,
...), e.g. ``.../checkpoints/iter_000041500`` -- the loader reads DCP, not a ``.pt``.

Output lands under the model's own experiment dir, keyed by training step and length:
``<exp_root>/eval/step{step:06d}_f{num_gen_frames:06d}/scene_{i:04d}/`` (``exp_root`` is the
checkpoint's grandparent when it sits in a ``checkpoints/`` dir; ``step`` is parsed from the ckpt
name). Each scene dir holds ``{gt,gen,warp,topdown,slots,topdown_slots}.mp4`` plus ``grid.png``
and ``retrieval.jsonl``; the episode key is printed in the per-scene summary line.

Usage (CWD must be the lyra repo root; run on a GPU node):
  python -m lyra_2.sample_maze --experiment maze_small \
      --checkpoint outputs/lyra2_from_bidir/memorymaze/finetuned/checkpoints/iter_000009700 \
      --num-gen-frames 397
"""

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from lyra_2._src.datasets.maze_streaming import MazeLyraDataset, build_maze_dataloader
from lyra_2._src.inference.lyra2_ar_inference import run_lyra2_sample
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

# The top-down maze renderer lives in the parent MemoryKrea repo (src/visualization); add its
# root to sys.path so the `src.*` import resolves when running from this lyra subrepo.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.visualization.topdown_maze_map import topdown_video  # noqa: E402


def _to_uint8(video_b3thw: torch.Tensor) -> np.ndarray:
    """(B,3,T,H,W) [-1,1] -> (T,H,W,3) uint8 for the first batch element."""
    v = video_b3thw[0].permute(1, 2, 3, 0).float().clamp(-1, 1)
    return ((v + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()


def _write_video(frames_thwc: np.ndarray, path: Path, fps: int) -> Path:
    """Write (T,H,W,3) uint8 frames as h264 mp4; fall back to gif on any ffmpeg failure.

    Returns the path actually written (``.mp4`` or ``.gif``).
    """
    import imageio.v2 as imageio

    try:
        # macro_block_size=1 avoids the "resized to a multiple of 16" behavior for small frames.
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        try:
            for f in frames_thwc:
                writer.append_data(f)
        finally:
            writer.close()
        return path
    except Exception:
        gif = path.with_suffix(".gif")
        imageio.mimsave(gif, list(frames_thwc), fps=fps)
        return gif


# Consistent per-slot colors: slot 0 = red, 1 = green, 2 = blue. Drives both the strip borders/labels
# and the topdown rings so strip-slot-k always matches topdown-ring-k.
SLOT_COLORS = [(230, 60, 60), (60, 200, 60), (60, 120, 240)]


def _slots_strip(gen_full: np.ndarray, gt_full: np.ndarray, per_frame_slots: np.ndarray, colors) -> np.ndarray:
    """Retrieved-spatial-slot strip video.

    For each output frame, the n slots side by side; each slot is the model's generated frame (top)
    over the GT frame (bottom) at the retrieved index -- id-labeled, color-bordered. A ``-1`` id
    (padding / no retrieval / seed frame) renders as a gray placeholder.

    Args:
        gen_full: (L,H,W,3) uint8 generated frames (result["video"], the RGB the model encoded).
        gt_full: (L',H,W,3) uint8 GT frames (same absolute index space).
        per_frame_slots: (T, n) int64 retrieved frame ids per output frame; -1 = placeholder.

    Returns:
        (T, 2*H, n*W, 3) uint8.
    """
    T, n = int(per_frame_slots.shape[0]), int(per_frame_slots.shape[1])
    H, W = int(gen_full.shape[1]), int(gen_full.shape[2])
    out = np.empty((T, 2 * H, n * W, 3), dtype=np.uint8)
    for t in range(T):
        for k in range(n):
            fid = int(per_frame_slots[t, k])
            col = colors[k % len(colors)]
            if fid < 0:
                sub = np.full((2 * H, W, 3), 96, dtype=np.uint8)  # gray placeholder
                label = "--"
            else:
                fg = min(fid, gen_full.shape[0] - 1)
                fgt = min(fid, gt_full.shape[0] - 1)
                sub = np.concatenate([gen_full[fg], gt_full[fgt]], axis=0)  # gen over gt -> (2H,W,3)
                label = str(fid)
            img = Image.fromarray(sub)
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, W - 1, 2 * H - 1], outline=col, width=2)  # slot-color border
            d.text((2, 1), label, fill=(0, 0, 0))  # dark backing for legibility
            d.text((3, 2), label, fill=col)
            out[t, :, k * W:(k + 1) * W] = np.array(img)
    return out


def _ar_args(args, num_frames: int) -> SimpleNamespace:
    """Build the argparse-like namespace ``run_lyra2_sample`` / the pipeline read.

    Maze specifics: ``depth_backend="gt"`` (use the clip's GT depth, no DA3), no
    context/offload/DMD/multiview. ``guidance`` is inert here (cross-attn is disabled) but
    passed through for API parity.
    """
    return SimpleNamespace(
        context_parallel_size=1,
        num_frames=num_frames,
        guidance=args.guidance,
        shift=args.shift,
        num_sampling_step=args.num_steps,
        seed=args.seed,
        fps=args.fps,
        offload=False,
        use_dmd_scheduler=False,
        ablate_same_t5=False,
        multiview_ids=None,
        num_retrieval_views=1,
        warp_chunk_size=None,
        disable_cache_update=False,
        depth_backend="gt",
        offload_da3_diffusion=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample a maze-trained Lyra2 checkpoint.")
    ap.add_argument("--experiment", default="lyra2_maze_small")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-root", default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval")
    ap.add_argument("--num-frames", type=int, default=401, help="frames loaded per eval clip")
    ap.add_argument(
        "--num-gen-frames",
        type=int,
        default=401,
        help="target generated video length; autoregressive rollout of "
        "ceil((n-1)/frames_per_segment) segments (1 segment = 20 frames). Capped by clip length.",
    )
    ap.add_argument("--num-scenes", type=int, default=2)
    ap.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=5.71,
        help="Camera downward mount tilt baked into the poses. 5.71 (= atan(0.1)) is the camera's "
        "true tilt and matches checkpoints trained from 2026-07-30 on; pass 0.0 to evaluate the "
        "older level-pose checkpoints.",
    )
    ap.add_argument("--num-steps", type=int, default=35, help="diffusion sampling steps per chunk")
    ap.add_argument("--shift", type=float, default=1.0, help="flow-match scheduler shift")
    # CFG is inert for this config: disable_cross_attn removes the text/CLIP branch, so the
    # conditional and unconditional passes are identical and guidance*(cond-uncond)==0. Kept
    # for API parity; 1.0 matches the (guidance-free) training objective.
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=8, help="frames/sec for saved videos")
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-grid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--slot-viz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render retrieved-spatial-slot videos (slots strip + topdown with color-matched rings).",
    )
    ap.add_argument(
        "--experiment-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra config override applied at load time, e.g. "
        "model.config.spatial_memory_stride=1. Repeatable.",
    )
    ap.add_argument(
        "--retrieval-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dump one JSON line per spatial retrieval (candidates, coverage, greedy gains, stop "
        "reason) next to the videos. Skipped if LYRA_RETRIEVE_DEBUG is already set.",
    )
    args = ap.parse_args()

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        config_file="lyra_2/_src/configs/config.py",
        instantiate_ema=False,
        strict=True,
        # self_aug is a TRAINING augmentation: its tokenizer branch uses one-shot
        # ``_stage_a_*`` handshake keys that leak across repeated inference calls (KeyError
        # on the 3rd rollout segment). Disable it so tokenization takes the plain encode
        # path -- identical latents, no training-only state. (cur_segment_id is always
        # supplied, so the self_aug max_segments-1 branch never runs either.)
        experiment_opts=["model.config.self_aug_enabled=False", *args.experiment_opt],
    )
    model.eval()

    ds = MazeLyraDataset(
        [args.eval_root], num_frames=args.num_frames, stage="eval", resampled=False, seed=args.seed,
        camera_pitch_deg=args.camera_pitch_deg,
    )
    dl = build_maze_dataloader(ds, batch_size=1, num_workers=0)

    # Output lands under the model's experiment dir: <exp_root>/eval/step{step}_f{frames}.
    # exp_root = the checkpoint's grandparent when it lives in a `checkpoints/` dir; step is
    # parsed from the checkpoint name (e.g. iter_000009700 -> 9700).
    ckpt = Path(args.checkpoint)
    m = re.search(r"(\d+)", ckpt.name)
    step = int(m.group(1)) if m else 0
    exp_root = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    out_dir = exp_root / "eval" / f"step{step:06d}_f{args.num_gen_frames:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sample_maze] output dir: {out_dir}")
    # Captured once: an externally-set sink wins for the whole run, and checking the live env var
    # per scene would see the value we set for scene 0 and funnel every scene into one file.
    external_retrieve_debug = os.environ.get("LYRA_RETRIEVE_DEBUG")
    it = iter(dl)

    for i in range(args.num_scenes):
        batch = next(it)
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # GT reference (the clip the rollout's camera trajectory follows) before the
        # pipeline mutates the batch. neg prompt == the constant t5 (inert; cross-attn off).
        gt_v = batch["video"].clone()
        # Raw agent poses for the top-down map (grabbed before the pipeline runs).
        agent_pos = batch["agent_pos"][0].cpu().numpy() if "agent_pos" in batch else None
        agent_dir = batch["agent_dir"][0].cpu().numpy() if "agent_dir" in batch else None
        maze_layout = batch["maze_layout"][0].cpu().numpy() if "maze_layout" in batch else None
        batch.setdefault("neg_t5_text_embeddings", batch["t5_text_embeddings"])
        key = batch["__key__"][0] if isinstance(batch["__key__"], list) else str(batch["__key__"])
        scene_dir = out_dir / f"scene_{i:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Per-scene retrieval diagnostics sink (read by Sparse3DCache.retrieve at call time).
        # An externally-set LYRA_RETRIEVE_DEBUG wins, so a manual sink is never clobbered.
        if args.retrieval_debug and not external_retrieve_debug:
            dbg = scene_dir / "retrieval.jsonl"
            dbg.unlink(missing_ok=True)  # records are appended; start each rollout clean
            os.environ["LYRA_RETRIEVE_DEBUG"] = str(dbg)

        ar_args = _ar_args(args, args.num_gen_frames)
        with torch.no_grad():
            result = run_lyra2_sample(
                model, batch, ar_args, da3_model=None, show_progress=True,
                log_prefix=f"maze scene {i} ({key})",
            )
        gen_v = result["video"]  # (B,3,L,H,W) [-1,1], seed frame + generated, cpu
        warp_v = result.get("warp_video")  # (B,3,L,H,W) or None
        slot_ids = result.get("spatial_slot_ids")  # (num_steps, n_slots) int64 or None

        gen_full = _to_uint8(gen_v)  # (L,H,W,3) full -- retrieved-slot lookups by absolute id
        gt_full = _to_uint8(gt_v)  # (L',H,W,3) full
        T = min(gen_full.shape[0], gt_full.shape[0])
        gen, gt = gen_full[:T], gt_full[:T]
        # Separate named panels, each saved as its own video (error discarded).
        panels = {"gt": gt, "gen": gen}
        if warp_v is not None:
            panels["warp"] = _to_uint8(warp_v)[:T]  # spatial-memory warp conditioning
        # Top-down camera-trajectory panel from the GT agent poses (blank canvas, no walls -- same
        # call as src/inference/eval_diffusion.py). Best-effort; rendered at the frame resolution so
        # it stacks with the other panels.
        if agent_pos is not None and agent_dir is not None:
            try:
                td = topdown_video(maze_layout, agent_pos[:T], agent_dir[:T], size=int(gt.shape[1]))
                panels["topdown"] = td.transpose(0, 2, 3, 1)  # (T,3,H,W) -> (T,H,W,3)
            except Exception as e:  # noqa: BLE001 -- top-down panel is best-effort
                print(f"[sample_maze] scene {i} topdown panel skipped: {e}")

        # Retrieved-spatial-slot videos (separate from the 64x64 grid panels: the strip is 3*W wide).
        # Retrieval is per AR chunk (F frames), so repeat each chunk's ids across its F frames; frame 0
        # is the seed (no retrieval). Best-effort.
        extra_videos = {}
        if args.slot_viz and slot_ids is not None and slot_ids.shape[0] > 0:
            F = int(model.framepack_num_new_video_frames)  # px frames per AR chunk
            S = int(slot_ids.shape[0])
            per_frame_slots = np.full((T, slot_ids.shape[1]), -1, dtype=np.int64)  # frame 0 = seed
            for t in range(1, T):
                per_frame_slots[t] = slot_ids[min((t - 1) // F, S - 1)]
            try:
                extra_videos["slots"] = _slots_strip(gen_full, gt_full, per_frame_slots, SLOT_COLORS)
            except Exception as e:  # noqa: BLE001
                print(f"[sample_maze] scene {i} slots strip skipped: {e}")
            if agent_pos is not None and agent_dir is not None:
                try:
                    td2 = topdown_video(
                        maze_layout, agent_pos[:T], agent_dir[:T], size=2 * int(gt.shape[1]),
                        highlight_indices_per_frame=per_frame_slots.tolist(),
                        highlight_colors=SLOT_COLORS,
                    )
                    extra_videos["topdown_slots"] = td2.transpose(0, 2, 3, 1)  # (T,H,W,3)
                except Exception as e:  # noqa: BLE001
                    print(f"[sample_maze] scene {i} topdown_slots skipped: {e}")

        written = []
        if args.save_grid:
            # Grid of up to 8 evenly spaced frames, one row per panel (GT / gen / warp).
            idx = np.linspace(0, T - 1, min(8, T)).astype(int)
            grid = np.concatenate([np.concatenate(list(p[idx]), axis=1) for p in panels.values()], axis=0)
            png = scene_dir / "grid.png"
            try:
                import imageio.v2 as imageio

                imageio.imwrite(png, grid)
            except ImportError:
                png = png.with_suffix(".npy")
                np.save(png, grid)
            written.append(png.name)

        if args.save_video:
            # One video per element: <scene_dir>/{gt,gen,warp,topdown}.mp4
            for name, frames in panels.items():
                vid = _write_video(frames, scene_dir / f"{name}.mp4", args.fps)
                written.append(vid.name)
            # Extra (non-grid) videos: <scene_dir>/{slots,topdown_slots}.mp4
            for name, frames in extra_videos.items():
                vid = _write_video(frames, scene_dir / f"{name}.mp4", args.fps)
                written.append(vid.name)

        mse = torch.nn.functional.mse_loss(gen_v[:, :, :T].float().cpu(), gt_v[:, :, :T].float().cpu()).item()
        # Coarse drift signal: mean pixel MSE over the first vs last fifth of the rollout.
        q = max(1, T // 5)
        early = float(np.square(gen[:q].astype(np.float32) / 127.5 - gt[:q].astype(np.float32) / 127.5).mean())
        late = float(np.square(gen[-q:].astype(np.float32) / 127.5 - gt[-q:].astype(np.float32) / 127.5).mean())
        print(
            f"[sample_maze] scene {i} ({key}): frames={T} mse={mse:.4f} "
            f"drift[early={early:.4f} late={late:.4f}] -> {', '.join(written)}"
        )

    print("[sample_maze] done.")


if __name__ == "__main__":
    main()
