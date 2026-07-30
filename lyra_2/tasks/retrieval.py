"""Lyra 2.0 memory-RETRIEVAL / reverse-recall probe on Memory-Maze.

The Lyra2 counterpart of the host repo's RavenFull probe (``src/tasks/retrieval.py``): same
trajectory-reversal episodes (legs ``fwd`` / ``turn`` / ``bwd`` sharing one global camera frame),
same question -- how much of the outbound path can the model reconstruct on the way back? -- but
built on Lyra2's own AR inference path instead of an SSM coarse read plus DiT sharpening.

Per episode:

  1. Trim the forward leg's first ``(n_fwd - 1) % 12`` pixel frames so generation starts on the DiT
     chunk grid, i.e. exactly on the first turn frame (``lyra_2/_src/datasets/maze_reversal.py``).
  2. **Prefill both of Lyra2's memory buffers with ground truth** over the trimmed forward leg
     (``Lyra2InferencePipeline.prefill_from_ground_truth``): the past-latent buffer (the anchor plus
     the 6 most-recent temporal latents) and the spatial buffer (the ``Sparse3DCache`` geometry and
     the GT pixels its 3 retrieved slots are encoded from). After the prefill the pipeline state is
     indistinguishable from having autoregressed the forward leg perfectly.
  3. AR-generate every turn+backward chunk. The only ground truth the model sees is that prefix;
     DEPLOYMENT SEMANTICS -- generated frames enter the spatial cache too, so later retrievals may
     pick the model's own output as well as GT forward frames (this matches the RavenFull probe,
     whose memory likewise absorbs generated content).
  4. Score the generation against GT over the turn+backward span: per-frame image PSNR / SSIM /
     LPIPS, latent PSNR, a VAE round-trip ceiling, and spatial-slot diagnostics (how many slots got
     filled, how often they came from the GT forward leg, and how much of the target view they
     actually share).

Metric parity with the RavenFull probe is deliberate, since the two are meant to be compared:
identical episodes in identical order, and image metrics computed on frames bilinear-upsampled to
``--metric-size`` (256, matching that probe's ``decode_uint8``) because SSIM and especially LPIPS
are resolution-dependent. Latent PSNR is reported but is NOT comparable across the two probes --
the RavenFull one scores raw Wan latents while Lyra2's history holds mean/std-normalized latents
that have additionally been through a decode/re-encode round trip.

Outputs land in ``<exp_root>/eval/step{step:06d}_recall{variant}/`` (``exp_root`` is the
checkpoint's grandparent when it sits in a ``checkpoints/`` dir): per-scene ``metrics.json``,
``retrieval.jsonl`` and mp4s (``forward_gt`` / ``reverse_gt`` / ``reverse_gen`` / ``warp`` /
``slots`` / ``forward_map`` / ``reverse_map`` / ``reverse_map_slots``, the last being the reverse
map with the retrieved slot cameras ringed in the matching slot colors), plus the aggregate
``per_frame_psnr.json`` and ``per_frame_image_metrics.png``.

Usage (CWD must be the lyra repo root; run on a GPU node):
  python -m lyra_2.tasks.retrieval --experiment maze_small --checkpoint outputs/lyra2_from_bidir/memorymaze/finetuned/checkpoints/iter_000009700
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import tqdm

from lyra_2._src.datasets.maze_reversal import MazeReversalDataset, build_reversal_dataloader
from lyra_2._src.inference.lyra2_ar_inference import (
    Lyra2InferencePipeline,
    _decode_new_latent_chunk,
    _get_vae_handles,
    _prime_encoder_cache_with_history,
)
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

# The maze visualization helpers and the top-down renderer
from lyra_2.sample_maze import SLOT_COLORS, _slots_strip, _to_uint8, _write_video

_LYRA_ROOT = Path(__file__).resolve().parents[2]  # .../src/external/lyra
if str(_LYRA_ROOT.parents[2]) not in sys.path:  # .../MemoryKrea
    sys.path.insert(0, str(_LYRA_ROOT.parents[2]))

from src.metrics import per_frame_video_metrics  # noqa: E402
from src.training.raven.metrics import latent_psnr_from_mse  # noqa: E402
from src.visualization.topdown_maze_map import topdown_video  # noqa: E402


def _metric_block(m: dict[str, np.ndarray]) -> dict:
    """JSON-ready mean + framewise block, matching src.logger.json_utils.metric_block.

    Reimplemented rather than imported: ``src/logger/__init__.py`` eagerly pulls accelerate and
    the wandb/tensorboard wrappers, which have no business in this process.
    """
    return {
        "mean": {k: round(float(np.nanmean(m[k])), 4) for k in ("psnr", "ssim", "lpips")},
        "framewise": {k: [round(float(v), 4) for v in m[k]] for k in ("psnr", "ssim", "lpips")},
    }


def _to_uint8_scaled(video_b3thw: torch.Tensor, size: int) -> np.ndarray:
    """(B,3,T,H,W) [-1,1] -> (T,size,size,3) uint8 for the first batch element.

    Replicates the RavenFull probe's ``decode_uint8``: convert to [0,1], bilinear-resize, THEN
    quantize. Resizing after quantization would give different SSIM/LPIPS, so the order matters
    for cross-probe comparability.
    """
    v01 = (video_b3thw[0].permute(1, 0, 2, 3).float().clamp(-1, 1) + 1.0) * 0.5  # (T,3,H,W)
    if int(v01.shape[-1]) != size or int(v01.shape[-2]) != size:
        v01 = torch.nn.functional.interpolate(v01, size=(size, size), mode="bilinear", align_corners=False)
    u8 = v01.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()  # (T,3,size,size)
    return u8.transpose(0, 2, 3, 1)


def _ar_args(args, num_frames: int) -> SimpleNamespace:
    """Build the argparse-like namespace the AR pipeline reads.

    Mirrors ``sample_maze._ar_args``: ``depth_backend="gt"`` (the camera path is the clip's GT
    trajectory, so scene depth is known and DA3 is not needed), no offload/DMD/multiview.
    ``disable_cache_update=False`` gives deployment semantics -- generated frames join the spatial
    cache. ``guidance`` is inert here (cross-attn is disabled) but passed for API parity.
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


def _prefill_self_check(model, pipe, video: torch.Tensor, n_prefill: int, chunk_px: int) -> None:
    """Assert the prefilled VAE stream caches continue correctly into the first generated chunk.

    The priming decode is self-consistent by construction, so agreement with ``history_frames``
    proves little. What actually discriminates a mis-primed cache is CONTINUITY across the
    prefill/generation seam: encoding the next GT chunk from the primed encoder cache must give the
    same latents as a single from-scratch stream over the concatenation. A stale cache passes the
    prefix check and fails the continuation one.

    Raises:
        AssertionError: If the prefix or the continuation disagrees with the reference stream.
    """
    n_lat = int(pipe.history_latents.shape[2])
    nxt = video[:, :, n_prefill : n_prefill + chunk_px]
    ref, _ = _prime_encoder_cache_with_history(
        torch.cat([pipe.history_frames, nxt], dim=2), pipe.vae_wrap, pipe.vae_core, model, False
    )
    ref = ref.float().cpu()
    assert torch.allclose(ref[:, :, :n_lat], pipe.history_latents.float().cpu(), atol=1e-2), (
        "prefill latents disagree with a from-scratch stream over the same pixels"
    )

    feats, _ = model.vae_encode_with_cache(
        enc_cache=pipe.enc_feat_cache, video=nxt, start_t=0, end_t=int(nxt.shape[2]), return_cache=True
    )
    cont = model._encoder_feats_to_normalized_latents(feats).float().cpu()
    assert torch.allclose(ref[:, :, n_lat:], cont, atol=1e-2), (
        "encoder cache is not continuous across the prefill seam -- the first generated chunk "
        "would be encoded from the wrong VAE state"
    )

    # Decoder-cache check on a CLONE: the real cache must stay at the prefill boundary.
    dec_clone = [c.clone() if torch.is_tensor(c) else c for c in pipe.dec_feat_cache]
    rt = _decode_new_latent_chunk(
        pipe.vae_wrap, pipe.vae_core, dec_clone, cont.to(pipe.history_latents), n_lat, model, False
    )
    mse = torch.nn.functional.mse_loss(rt.float().cpu(), nxt.float().cpu()).item()
    psnr = 10.0 * np.log10(4.0 / max(mse, 1e-12))
    assert psnr > 20.0, (
        f"decoder round trip across the prefill seam is only {psnr:.2f} dB -- dec_feat_cache is "
        "likely out of phase with history_latents"
    )
    print(f"[retrieval] prefill self-check OK (seam round trip {psnr:.2f} dB)")


def _pad_frames(model) -> tuple[int, int]:
    """The AR pipeline's history pad: (repeat_pixels, start_index).

    ``repeat_pixels`` seed copies occupy latent slots 0..T_hist-1, and absolute frame ``f`` then
    lives at pixel position ``start_index + f`` -- the index space shared by ``history_frames``,
    ``build_outputs()["video"]`` and :func:`_clip_latents`.
    """
    fpl = int(model.framepack_num_frames_per_latent)
    t_hist = int(model.framepack_total_max_num_latent_frames) - int(model.framepack_num_new_latent_frames)
    return (t_hist - 1) * fpl + 1, (t_hist - 1) * fpl


def _clip_latents(model, video: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Encode ``video[:, :, :n_frames]`` on the pipeline's padded latent grid.

    Reproduces the prefill's front pad so the returned latents share an index space with
    ``pipeline.history_latents``. Used for the GT reference latents and the round-trip ceiling.
    Call only after the rollout: it resets the live VAE encoder cache.
    """
    repeat_pixels, _ = _pad_frames(model)
    _, vae_wrap, vae_core = _get_vae_handles(model)
    padded = torch.cat(
        [video[:, :, :1].repeat(1, 1, repeat_pixels, 1, 1), video[:, :, 1:n_frames]], dim=2
    )
    latents, _ = _prime_encoder_cache_with_history(padded, vae_wrap, vae_core, model, False)
    return latents


def _decode_clip_latents(model, latents: torch.Tensor) -> torch.Tensor:
    """Stream-decode a padded latent stack and strip the pad -> (B,3,T,H,W), absolute-frame indexed.

    The pad must be dropped here: without it the returned frame axis is offset by ``start_index``
    and every downstream slice silently reads the wrong frames.
    """
    _, vae_wrap, vae_core = _get_vae_handles(model)
    vae_core.clear_cache()
    px = _decode_new_latent_chunk(
        vae_wrap, vae_core, [None] * vae_core._conv_num, latents, 0, model, False
    )
    return px[:, :, _pad_frames(model)[1] :]


def _slot_stats(slot_ids: np.ndarray, batch: dict, gen_start: int, chunk_px: int, device: str) -> dict:
    """Per-AR-step spatial-slot diagnostics.

    Args:
        slot_ids: (S, n_slots) absolute retrieved frame ids; -1 = unfilled (left-padding).
        batch: The episode batch (needs ``depth``, ``camera_w2c``, ``intrinsics``, ``agent_pos``,
            ``agent_dir``).
        gen_start: First generated absolute frame index (== prefill length).
        chunk_px: Pixel frames per AR step.

    Returns:
        Scalar means plus the per-step ``filled`` curve. ``covis`` is the fraction of the target
        view's valid-depth pixels a slot truly shares with it (measured against the target's OWN GT
        depth, so behind-wall frames are correctly rejected); ``dead`` counts slots below
        ``covis < 0.01``, including unfilled ones, i.e. the wasted fraction of the memory budget.
    """
    from lyra_2.eval_retrieval_audit import _oracle_covis, _unproject, _yaw_deg, _yaw_delta

    depth = batch["depth"].to(device).float()
    w2c = batch["camera_w2c"][0].to(device).float()
    K = batch["intrinsics"][0].to(device).float()
    pos = batch["agent_pos"][0].cpu().numpy()
    yaws = _yaw_deg(batch["agent_dir"][0].cpu().numpy())
    n_slots = int(slot_ids.shape[1])

    filled, covis, dead, ages, dists, dyaws, from_gt = [], [], [], [], [], [], []
    for s in range(int(slot_ids.shape[0])):
        target = min(gen_start + chunk_px * s + chunk_px - 1, int(w2c.shape[0]) - 1)
        ids = [int(f) for f in slot_ids[s] if int(f) >= 0]
        filled.append(len(ids))
        gt_now = depth[0, target, 0]
        n_valid = max(int((gt_now > 0).sum()), 1)
        n_dead = n_slots - len(ids)  # unfilled slots carry no information either
        for f in ids:
            pts = _unproject(depth[0, f], w2c[f], K[f])
            frac = float(_oracle_covis(pts, w2c[target], K[target], gt_now, 0.10).sum()) / n_valid
            covis.append(frac)
            n_dead += int(frac < 0.01)
            ages.append(target - f)
            dists.append(float(np.linalg.norm(pos[f] - pos[target])))
            dyaws.append(_yaw_delta(float(yaws[f]), float(yaws[target])))
            from_gt.append(float(f < gen_start))
        dead.append(n_dead / n_slots)

    def _mn(v):
        return float(np.mean(v)) if v else float("nan")

    return {
        "slots_filled": _mn(filled),
        "slots_filled_per_step": [int(x) for x in filled],
        "frac_from_gt_forward": _mn(from_gt),
        "slot_covis": _mn(covis),
        "dead_slot_frac": _mn(dead),
        "slot_age": _mn(ages),
        "slot_dist_cells": _mn(dists),
        "slot_abs_dyaw_deg": _mn(dyaws),
    }


@torch.no_grad()
def _run_episode(model, batch: dict, args, *, gen_start: int, n_chunks: int, log_prefix: str) -> dict:
    """Prefill the buffers with the GT forward leg, then AR-generate turn+backward.

    Returns:
        ``build_outputs``' dict plus ``gen_latents`` (the re-encoded generated latents, captured
        before ``build_outputs`` frees the history) and ``n_lat_prefill``.
    """
    chunk_px = int(model.framepack_num_new_video_frames)
    ar_args = _ar_args(args, gen_start + chunk_px * n_chunks)
    pipe = Lyra2InferencePipeline(
        model=model,
        args=ar_args,
        first_frame=batch["video"][:, :, :1],
        first_depth=batch["depth"][:, 0],
        first_cam_w2c=batch["camera_w2c"][:, 0],
        first_intrinsics=batch["intrinsics"][:, 0],
        base_t5_text_embeddings=batch["t5_text_embeddings"],
        base_neg_t5_text_embeddings=batch["neg_t5_text_embeddings"],
        padding_mask=batch.get("padding_mask", None),
        fps=batch.get("fps", None),
        gt_depth=batch["depth"],
    )
    pipe.prefill_from_ground_truth(
        video=batch["video"],
        depth=batch["depth"],
        camera_w2c=batch["camera_w2c"],
        intrinsics=batch["intrinsics"],
        n_prefill=gen_start,
    )
    n_lat_prefill = int(pipe.history_latents.shape[2])
    if args.self_check:
        _prefill_self_check(model, pipe, batch["video"], gen_start, chunk_px)

    for i in tqdm.tqdm(range(n_chunks), desc=log_prefix):
        s = 1 + pipe.ar_idx * chunk_px
        pipe.autoregressive_step(
            cam_w2c_chunk=batch["camera_w2c"][:, s : s + chunk_px],
            intrinsics_chunk=batch["intrinsics"][:, s : s + chunk_px],
            t5_text_embeddings=batch["t5_text_embeddings"],
            neg_t5_text_embeddings=batch["neg_t5_text_embeddings"],
            is_last_step=(i == n_chunks - 1),
        )

    # build_outputs deletes history_latents, so capture the generated tail first.
    gen_latents = pipe.history_latents[:, :, n_lat_prefill:].detach().float().cpu()
    result = pipe.build_outputs(None, log_prefix)
    result["gen_latents"] = gen_latents
    result["n_lat_prefill"] = n_lat_prefill
    return result


def _plot_curves(out_dir: Path, curves: dict[str, np.ndarray], n_turn: int, n_eps: int, title: str) -> None:
    """Write the per-frame metric panels (best-effort; plotting must never fail the probe)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("lpips", "LPIPS"), ("filled", "slots filled /3")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (key, ylabel) in zip(axes.flat, panels, strict=True):
            if key in curves:
                ax.plot(curves[key], lw=1.0, label="generation")
            if key == "psnr" and "psnr_ceiling" in curves:
                ax.plot(curves["psnr_ceiling"], lw=1.0, ls="--", label="VAE round-trip ceiling")
            if n_turn > 0:
                ax.axvline(n_turn, color="k", ls=":", lw=0.8, label="turn -> bwd")
            ax.set_xlabel("generated frame index (0 = first turn frame)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{title} ({n_eps} episodes)")
        fig.tight_layout()
        fig.savefig(out_dir / "per_frame_image_metrics.png", dpi=150)
        plt.close(fig)
        print(f"[retrieval] wrote {out_dir / 'per_frame_image_metrics.png'}")
    except Exception as e:  # noqa: BLE001 -- plotting is best-effort
        print(f"[retrieval] per-frame plot skipped: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lyra2 reverse-recall probe on Memory-Maze.")
    ap.add_argument("--experiment", default="maze_small")
    ap.add_argument("--checkpoint", required=True, help="DCP checkpoint DIRECTORY, e.g. .../iter_000009700")
    ap.add_argument(
        "--data-root",
        default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/retrieval_data",
        help="Parent dir of the per-length reversal variants.",
    )
    ap.add_argument("--variant", default="401", choices=["101", "401", "801", "1601"],
                    help="Forward-leg length; picks <data-root>/<variant>x64x64.")
    ap.add_argument("--num-scenes", type=int, default=8, help="Episodes to probe (RavenFull default: 8).")
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
    ap.add_argument("--guidance", type=float, default=1.0, help="inert: cross-attn is disabled")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=8, help="frames/sec for saved videos")
    ap.add_argument(
        "--metric-size",
        type=int,
        default=256,
        help="Frames are bilinear-upsampled to this before PSNR/SSIM/LPIPS, matching the RavenFull "
        "probe's decode_uint8(size=256). Changing it breaks cross-probe comparability.",
    )
    ap.add_argument("--max-chunks", type=int, default=-1, help="cap AR chunks per episode (-1 = all); smoke knob")
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--slot-viz", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--self-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert the prefilled VAE stream caches continue into the first generated chunk (~5 s).",
    )
    ap.add_argument("--retrieval-debug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--experiment-opt", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        config_file="lyra_2/_src/configs/config.py",
        instantiate_ema=False,
        strict=True,
        # self_aug is a TRAINING augmentation whose one-shot handshake keys leak across repeated
        # inference calls; see sample_maze.py for the full rationale.
        experiment_opts=["model.config.self_aug_enabled=False", *args.experiment_opt],
    )
    model.eval()
    device = "cuda"
    chunk_px = int(model.framepack_num_new_video_frames)

    root = Path(args.data_root) / f"{args.variant}x64x64"
    ds = MazeReversalDataset(root, camera_pitch_deg=args.camera_pitch_deg, chunk_px=chunk_px)
    dl = build_reversal_dataloader(ds)

    ckpt = Path(args.checkpoint)
    m = re.search(r"(\d+)", ckpt.name)
    step = int(m.group(1)) if m else 0
    exp_root = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    out_dir = exp_root / "eval" / f"step{step:06d}_recall{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.num_scenes, len(ds))
    print(
        f"[retrieval] {ckpt} @ step {step} | {args.variant}x64x64 ({n}/{len(ds)} episodes) | "
        f"{args.num_steps}-step shift={args.shift} pitch={args.camera_pitch_deg} | -> {out_dir}"
    )
    external_retrieve_debug = os.environ.get("LYRA_RETRIEVE_DEBUG")

    img_acc: list[dict[str, np.ndarray]] = []
    ceil_acc: list[dict[str, np.ndarray]] = []
    lat_acc: list[np.ndarray] = []
    slot_acc: list[dict] = []
    filled_acc: list[np.ndarray] = []
    n_turn_ref = 0

    it = iter(dl)
    for ep in range(n):
        batch = next(it)
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model._normalize_video_databatch_inplace(batch)  # asserts float + [-1,1] (is_preprocessed)
        batch.setdefault("neg_t5_text_embeddings", batch["t5_text_embeddings"])

        key = batch["__key__"][0] if isinstance(batch["__key__"], list) else str(batch["__key__"])
        legs = tuple(int(x) for x in batch["legs"][0])
        gen_start = int(batch["gen_start"][0])
        n_chunks_full = int(batch["n_chunks"][0])
        n_chunks = min(n_chunks_full, args.max_chunks) if args.max_chunks >= 0 else n_chunks_full
        n_gen = chunk_px * n_chunks
        n_out = gen_start + n_gen
        n_turn_ref = legs[1]
        # The genuine whole-chunk remainder, independent of any --max-chunks cap.
        tail = (int(batch["num_frames"][0]) - gen_start) - chunk_px * n_chunks_full

        scene_dir = out_dir / f"scene_{ep:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        if args.retrieval_debug and not external_retrieve_debug:
            dbg = scene_dir / "retrieval.jsonl"
            dbg.unlink(missing_ok=True)  # records are appended; start each episode clean
            os.environ["LYRA_RETRIEVE_DEBUG"] = str(dbg)

        gt_v = batch["video"].clone()
        result = _run_episode(
            model, batch, args, gen_start=gen_start, n_chunks=n_chunks,
            log_prefix=f"recall ep{ep} ({key})",
        )
        gen_v = result["video"]  # (B,3,n_out,H,W) [-1,1] cpu; [:gen_start] is the GT prefill
        # Free check that the prefill really is the GT we fed in.
        assert torch.allclose(
            gen_v[:, :, :gen_start], gt_v[:, :, :gen_start].float().cpu(), atol=1e-3
        ), "prefilled history_frames do not match the GT prefix"

        gen_u8 = _to_uint8_scaled(gen_v[:, :, gen_start:n_out], args.metric_size)
        gt_u8 = _to_uint8_scaled(gt_v[:, :, gen_start:n_out], args.metric_size)
        m_gen = per_frame_video_metrics(gen_u8, gt_u8, device)
        img_acc.append(m_gen)

        # Latent-space error, plus the VAE round-trip ceiling that makes the absolute image numbers
        # interpretable. Both use the pipeline's padded latent grid; run after the rollout because
        # _clip_latents resets the live encoder cache.
        gt_lat = _clip_latents(model, gt_v, n_out)
        gt_gen_lat = gt_lat[:, :, int(result["n_lat_prefill"]) :].float().cpu()
        t_lat = min(int(gt_gen_lat.shape[2]), int(result["gen_latents"].shape[2]))
        pf_mse = (
            (result["gen_latents"][0, :, :t_lat] - gt_gen_lat[0, :, :t_lat])
            .pow(2)
            .mean(dim=(0, 2, 3))  # over channels and space -> one value per latent frame
            .numpy()
        )
        lat_acc.append(pf_mse)

        ceil_v = _decode_clip_latents(model, gt_lat.to(**model.tensor_kwargs))
        ceil_u8 = _to_uint8_scaled(ceil_v[:, :, gen_start:n_out].cpu(), args.metric_size)
        m_ceil = per_frame_video_metrics(ceil_u8, gt_u8, device)
        ceil_acc.append(m_ceil)
        del ceil_v, gt_lat

        slot_ids = result.get("spatial_slot_ids")
        stats = _slot_stats(slot_ids, batch, gen_start, chunk_px, device) if slot_ids is not None else {}
        slot_acc.append(stats)
        if stats.get("slots_filled_per_step"):
            filled_acc.append(np.repeat(np.asarray(stats["slots_filled_per_step"], float), chunk_px))

        report = {
            "scene": key,
            "num_frames": int(len(m_gen["psnr"])),
            "legs": list(legs),
            "gen_start": gen_start,
            "unused_tail_frames": tail,
            "gen": _metric_block(m_gen),
            "vae_ceiling": _metric_block(m_ceil),
            "latent_psnr": round(latent_psnr_from_mse(float(np.nanmean(pf_mse))), 3),
            "slots": {k: v for k, v in stats.items() if k != "slots_filled_per_step"},
        }
        (scene_dir / "metrics.json").write_text(json.dumps(report, indent=2))

        written = []
        if args.save_video:
            gen_full = _to_uint8(gen_v)  # native 64x64, absolute frame axis
            gt_full = _to_uint8(gt_v[:, :, :n_out].float().cpu())
            vids = {
                "forward_gt": gt_full[:gen_start],
                "reverse_gt": gt_full[gen_start:n_out],
                "reverse_gen": gen_full[gen_start:n_out],
            }
            warp_v = result.get("warp_video")
            if warp_v is not None:
                # build_outputs prepends frame 0, so warp_video is 1 + n_gen long while `video` is
                # gen_start + n_gen: drop that seed frame to realign onto the generated span.
                vids["warp"] = _to_uint8(warp_v)[1 : 1 + n_gen]
            # Retrieval is per AR chunk, so each chunk's ids repeat across its chunk_px frames.
            per_frame = None
            if args.slot_viz and slot_ids is not None and slot_ids.shape[0] > 0:
                per_frame = np.repeat(slot_ids, chunk_px, axis=0)[:n_gen]

            pos = batch["agent_pos"][0].cpu().numpy()
            dirs = batch["agent_dir"][0].cpu().numpy()
            layout = batch["maze_layout"][0].cpu().numpy()
            try:
                vids["forward_map"] = topdown_video(layout, pos[:gen_start], dirs[:gen_start]).transpose(0, 2, 3, 1)
                vids["reverse_map"] = topdown_video(
                    layout, pos[gen_start:n_out], dirs[gen_start:n_out], prior_pos=pos[:gen_start]
                ).transpose(0, 2, 3, 1)
                if per_frame is not None:
                    # Same map with the retrieved slot cameras ringed in the slot colors. The ids are
                    # ABSOLUTE and mostly point into the forward leg, and topdown_video resolves
                    # highlights against the very array it is handed -- so this must render over the
                    # FULL clip (not reverse_map's backward slice) and then keep the generated tail.
                    hl = [[-1] * int(slot_ids.shape[1])] * gen_start + per_frame.tolist()
                    vids["reverse_map_slots"] = topdown_video(
                        layout, pos[:n_out], dirs[:n_out],
                        highlight_indices_per_frame=hl, highlight_colors=SLOT_COLORS,
                    )[gen_start:].transpose(0, 2, 3, 1)
            except Exception as e:  # noqa: BLE001 -- top-down maps are best-effort
                print(f"[retrieval] ep {ep}: topdown maps skipped: {e}")

            if per_frame is not None:
                try:
                    vids["slots"] = _slots_strip(gen_full, gt_full, per_frame, SLOT_COLORS)
                except Exception as e:  # noqa: BLE001
                    print(f"[retrieval] ep {ep}: slots strip skipped: {e}")

            for name, frames in vids.items():
                written.append(_write_video(frames, scene_dir / f"{name}.mp4", args.fps).name)

        print(
            f"[retrieval] ep {ep:03d} ({key}): gen psnr/ssim/lpips="
            f"{np.nanmean(m_gen['psnr']):.2f}/{np.nanmean(m_gen['ssim']):.3f}/{np.nanmean(m_gen['lpips']):.3f}"
            f" | ceiling psnr={np.nanmean(m_ceil['psnr']):.2f}"
            f" | latent psnr={report['latent_psnr']:.2f}"
            f" | slots {stats.get('slots_filled', float('nan')):.2f}/3"
            f" gt-fwd={stats.get('frac_from_gt_forward', float('nan')):.2f}"
            f" covis={stats.get('slot_covis', float('nan')):.3f}"
            f" dead={stats.get('dead_slot_frac', float('nan')):.2f}"
            f" | {len(written)} videos"
        )
        if tail:
            print(f"[retrieval] ep {ep:03d}: {tail} clip frames left ungenerated (partial chunk)")
        if n_chunks < n_chunks_full:
            print(
                f"[retrieval] ep {ep:03d}: --max-chunks capped generation at {n_chunks}/{n_chunks_full} "
                f"chunks ({chunk_px * (n_chunks_full - n_chunks)} further frames not generated)"
            )

    if not img_acc:
        return

    def _agg(acc: list[dict[str, np.ndarray]]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        scalars, per_frame = {}, {}
        for k in ("psnr", "ssim", "lpips"):
            scalars[k] = float(np.nanmean([np.nanmean(e[k]) for e in acc]))
            if len({e[k].shape[0] for e in acc}) == 1:
                per_frame[k] = np.nanmean(np.stack([e[k] for e in acc], 0), 0)
        return scalars, per_frame

    img, img_pf = _agg(img_acc)
    ceil, ceil_pf = _agg(ceil_acc)
    lat_mean = float(np.nanmean(np.concatenate(lat_acc)))
    summary = {
        "variant": args.variant,
        "episodes": len(img_acc),
        "num_denoise_steps": args.num_steps,
        "shift": args.shift,
        "camera_pitch_deg": args.camera_pitch_deg,
        "metric_size": args.metric_size,
        "image_metrics": {"gen": {k: round(v, 4) for k, v in img.items()},
                          "vae_ceiling": {k: round(v, 4) for k, v in ceil.items()}},
        "latent_psnr": round(latent_psnr_from_mse(lat_mean), 3),
        "slots": {
            k: round(float(np.nanmean([s[k] for s in slot_acc if k in s])), 4)
            for k in ("slots_filled", "frac_from_gt_forward", "slot_covis", "dead_slot_frac",
                      "slot_age", "slot_dist_cells", "slot_abs_dyaw_deg")
            if any(k in s for s in slot_acc)
        },
    }
    if "psnr" in img_pf:
        summary["per_frame_psnr"] = [round(float(v), 3) for v in img_pf["psnr"]]
    (out_dir / "per_frame_psnr.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\n[retrieval] over {len(img_acc)} episodes:\n"
        f"  generation   psnr {img['psnr']:.2f}  ssim {img['ssim']:.3f}  lpips {img['lpips']:.3f}\n"
        f"  VAE ceiling  psnr {ceil['psnr']:.2f}  ssim {ceil['ssim']:.3f}  lpips {ceil['lpips']:.3f}\n"
        f"  latent psnr  {summary['latent_psnr']:.2f} (NOT comparable to the RavenFull probe)\n"
        f"  slots        {summary['slots']}"
    )

    curves = dict(img_pf)
    if "psnr" in ceil_pf:
        curves["psnr_ceiling"] = ceil_pf["psnr"]
    if filled_acc and len({c.shape[0] for c in filled_acc}) == 1:
        curves["filled"] = np.nanmean(np.stack(filled_acc, 0), 0)
    _plot_curves(out_dir, curves, n_turn_ref, len(img_acc), f"Lyra2 reverse recall @ step {step}")
    print("[retrieval] done.")


if __name__ == "__main__":
    main()
