"""Lyra2Model subclass with a real validation_step (upstream's is a stub).

Runs the standard sampling path (``generate_samples_from_batch`` -> ``decode``, the same
entry the inference scripts use) with ``return_condition_state=True`` so the window-aligned
GT pixels, the depth-warp condition renders, and the plucker rays come back with the sample.
Writes one row-stacked frame grid per validation sample and returns a generated-vs-GT pixel
MSE as the validation loss so the trainer's ``validate()`` contract
(``output_batch, loss = model.validation_step(...)``) holds.

Grid rows (top to bottom; columns are ALL generated frames of the sampled window -- 12 at the
g3 framepack. The generation tail is decoded with a short re-encoded GT-history context so all
12 px frames come back aligned; a standalone decode of the 3-latent chunk would yield only 9):
  gt          window-aligned GT pixels (``_latest_gt_gen_pixels``; the window start/segment
              are random, so the full-clip tail is NOT the right reference)
  gen         decoded generation
  err         |gen - gt| mean over RGB (bright = wrong)
  warp        buffer depth-warp render fed to the DiT as camera control
  warp_sN     per-spatial-slot warp renders (first two slots)
  warp_depth  buffer warp's rendered depth (normalized; holes = 0)
  gt_depth    dataset metric depth at the generated frames' absolute indices
  ray_dir     plucker ray-direction condition
  slots       retrieved spatial-slot GT frames in the first columns (color-bordered, frame-id
              labeled; gray tile = left-padding, i.e. no retrieval for that slot)
  topdown     top-down maze map per generated frame (trajectory so far + agent glyph), with the
              retrieved slot cameras ringed in the matching slot colors

Sample grids land in ``$LYRA_VAL_SAMPLE_DIR`` (exported by ``lyra_2/train.py`` as
``<job.path_local>/val_samples``; falls back to ``./val_samples``). Rank 0 writes only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lyra_2._ext.imaginaire.utils import distributed, log
from lyra_2._src.models.lyra2_model import Lyra2Model

# Sampling settings for validation: modest step count keeps a validation sweep ~1 min;
# fixed seed makes grids comparable across iterations.
VAL_NUM_STEPS = 15
VAL_GUIDANCE = 1.5
VAL_SEED = 0
MAX_SPATIAL_ROWS = 2  # cap per-slot warp rows so the grid stays readable

# Per-slot colors, kept identical to sample_maze.py's SLOT_COLORS so validation grids and
# rollout videos read the same: slot 0 = red, 1 = green, 2 = blue.
SLOT_COLORS = [(230, 60, 60), (60, 200, 60), (60, 120, 240)]


def _slot_tiles(video_b3thw: torch.Tensor, slot_ids: list[int], n_real: int, t: int) -> np.ndarray:
    """(t,H,W,3) slots row: one GT frame per retrieved slot, color-bordered and id-labeled.

    Slot ids are LEFT-padded (lyra2_model pads with the window's first frame id), so slot k is
    real iff ``k >= len(slot_ids) - n_real``; padding renders as a gray placeholder. Columns
    beyond the slot count stay black.
    """
    H, W = int(video_b3thw.shape[-2]), int(video_b3thw.shape[-1])
    out = np.zeros((t, H, W, 3), dtype=np.uint8)
    for k, fid in enumerate(slot_ids[:t]):
        color = SLOT_COLORS[k % len(SLOT_COLORS)]
        if k < len(slot_ids) - n_real:
            tile = np.full((H, W, 3), 96, dtype=np.uint8)  # gray placeholder
            label = "--"
        else:
            v = video_b3thw[0, :, int(fid)].permute(1, 2, 0).float().clamp(-1, 1)
            tile = ((v + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
            label = str(int(fid))
        img = Image.fromarray(tile)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, H - 1], outline=color, width=2)
        d.text((2, 1), label, fill=(0, 0, 0))  # dark backing for legibility
        d.text((3, 2), label, fill=color)
        out[k] = np.array(img)
    return out


def _topdown_row(
    maze_layout: np.ndarray,
    agent_pos: np.ndarray,
    agent_dir: np.ndarray,
    gen_idx: list[int],
    slot_ids: list[int],
    n_real: int,
    size: int,
) -> np.ndarray:
    """(len(gen_idx),size,size,3) top-down map at each generated frame, slot cameras ringed.

    Uses the same ``src.visualization.topdown_maze_map.topdown_video`` renderer as
    ``sample_maze.py`` (parent-repo import resolved lazily; callers wrap in try/except).
    """
    import sys

    root = Path(__file__).resolve().parents[6]  # the MemoryKrea repo root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.visualization.topdown_maze_map import topdown_video

    gen_idx = [int(i) for i in gen_idx]
    last = max(gen_idx)
    hl = None
    if slot_ids and n_real > 0:
        # -1 marks left-padding; topdown_video skips negatives, keeping slot-color alignment.
        marks = [int(s) if k >= len(slot_ids) - n_real else -1 for k, s in enumerate(slot_ids)]
        hl = [marks] * (last + 1)
    td = topdown_video(
        maze_layout,
        agent_pos[: last + 1],
        agent_dir[: last + 1],
        size=size,
        highlight_indices_per_frame=hl,
        highlight_colors=SLOT_COLORS,
    )  # (last+1, 3, size, size) uint8
    return td[gen_idx].transpose(0, 2, 3, 1)


def _to_uint8(video_b3thw: torch.Tensor) -> np.ndarray:
    """(B,3,T,H,W) [-1,1] -> (T,H,W,3) uint8 for the first batch element."""
    v = video_b3thw[0].permute(1, 2, 3, 0).float().clamp(-1, 1)
    return ((v + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()


def _gray_to_uint8(x_thw: torch.Tensor) -> np.ndarray:
    """(T,H,W) [-1,1] -> (T,H,W,3) uint8 grayscale."""
    g = ((x_thw.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
    return np.repeat(g[..., None], 3, axis=-1)


def _metric_depth_to_uint8(d_thw: torch.Tensor) -> np.ndarray:
    """(T,H,W) metric depth -> (T,H,W,3) uint8, robust 1-99 percentile normalization."""
    d = d_thw.float()
    flat = d.flatten()
    lo, hi = torch.quantile(flat, 0.01), torch.quantile(flat, 0.99)
    d = ((d - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    return _gray_to_uint8(d * 2.0 - 1.0)


class MazeLyra2Model(Lyra2Model):
    """Lyra2Model + sample-saving validation (drop-in for the ddp/fsdp model nodes)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        # Re-apply framepack_trainable_modules. build_net freezes the net (lyra2_model.py:301-333),
        # but WANDiffusionModel.__init__ then calls self.net.requires_grad_(True)
        # (wan_t2v_model.py:204), which recursively undoes it -- so upstream's whitelist only ever
        # changed a log line. Nothing else in the tree re-applies it.
        self._apply_framepack_trainable_modules()

    def _apply_framepack_trainable_modules(self) -> None:
        """Freeze the net and unfreeze only the whitelisted modules.

        A VERBATIM transcription of ``Lyra2Model.build_net``'s whitelist block
        (``lyra2_model.py:301-333``), including its three DIFFERENT guard tests -- the
        ``clean_patch_embeddings`` sweep is guarded by a substring test over the whitelist entries,
        while the ``patch_embedding`` / ``patch_embedding_buffer`` sweeps are guarded by EXACT list
        membership. Those are not interchangeable: with a whitelist of just
        ``["clean_patch_embeddings"]`` the original skips the ``patch_embedding`` sweep, whereas a
        substring guard would run it (``"patch_embedding"`` is a substring of
        ``"clean_patch_embeddings"``). Keep this in lockstep with upstream.
        """
        config = self.config
        net = self.net
        if config.framepack_trainable_modules:
            whitelist = [p.strip() for p in config.framepack_trainable_modules.split(",") if p.strip()]
            if whitelist:
                log.info(f"Freezing model and unfreezing Lyra2AttentionBlock layers matching: {whitelist}")

                for param in net.parameters():
                    param.requires_grad = False

                trainable_param_names = set()

                for name, module in net.named_modules():
                    if type(module).__name__ == "Lyra2AttentionBlock":
                        for sub_name, param in module.named_parameters():
                            if any(pattern in sub_name for pattern in whitelist):
                                param.requires_grad = True
                                full_name = f"{name}.{sub_name}"
                                trainable_param_names.add(full_name)

                if any("clean_patch_embeddings" in p for p in whitelist):
                    for name, param in net.named_parameters():
                        if "clean_patch_embeddings" in name:
                            param.requires_grad = True
                            trainable_param_names.add(name)

                if "patch_embedding" in whitelist:
                    for name, param in net.named_parameters():
                        if "patch_embedding" in name:
                            param.requires_grad = True
                            trainable_param_names.add(name)
                if "patch_embedding_buffer" in whitelist:
                    for name, param in net.named_parameters():
                        if "patch_embedding_buffer" in name:
                            param.requires_grad = True
                            trainable_param_names.add(name)

                log.info(
                    f"Enabled gradients for {len(trainable_param_names)} parameters: "
                    f"{sorted(list(trainable_param_names))}"
                )
                # Extra (not upstream): the param totals are what actually prove the freeze took --
                # upstream emits the line above even though wan_t2v_model.py:204 undoes the freeze.
                n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in net.parameters())
                log.info(f"[maze] freeze re-applied: {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M params trainable")

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # Snapshot before the model mutates the batch; needed to align GT depth to the window.
        gt_depth_full = data_batch.get("depth", None)  # (B,T,1,H,W) metric
        gt_video_full = data_batch["video"]  # (B,3,N,H,W); slot tiles + context decode read it
        agent_pos = data_batch.get("agent_pos", None)  # (B,N,2) maze-cell xy
        agent_dir = data_batch.get("agent_dir", None)  # (B,N,2) unit heading
        maze_layout = data_batch.get("maze_layout", None)  # (B,15,15) wall grid

        out = self.generate_samples_from_batch(
            data_batch,
            guidance=VAL_GUIDANCE,
            seed=VAL_SEED,
            num_steps=VAL_NUM_STEPS,
            # Sample on the training schedule; the arg default is 5.0 and would silently diverge
            # from a config that trains at a different shift.
            shift=float(self.config.shift),
            return_condition_state=True,
        )
        # Return shape varies with what was collected: latents | (latents, cond) |
        # (latents, cond, gt) -- see Lyra2Model.generate_samples_from_batch.
        if isinstance(out, tuple):
            latents = out[0]
            cond_state = out[1] if len(out) > 1 else None
            gt_vis = out[2] if len(out) > 2 else None
        else:
            latents, cond_state, gt_vis = out, None, None

        # Side channels stashed by lyra2_model during collection (read once, then clear).
        warp_depth = getattr(self, "_latest_condition_state_depth", None)  # (B,1,F,H,W) [-1,1]
        spatial_depth = getattr(self, "_latest_spatial_warped_depth", None)  # (B,N,F,H,W) [-1,1]
        gen_idx = getattr(self, "_latest_gt_gen_indices", None)  # (T_gen,) absolute frame ids
        # Padded slot frame ids + how many are real retrievals (left-padding uses frame ids too,
        # so the count is the only way to tell a genuine frame-0 retrieval from padding).
        slot_ids_val = list(getattr(self, "_latest_spatial_slot_ids", None) or [])
        slot_real = int(getattr(self, "_latest_spatial_retrieved_count", 0) or 0)
        self._latest_condition_state_depth = None
        self._latest_spatial_warped_depth = None
        self._latest_gt_gen_indices = None
        self._latest_spatial_slot_ids = None
        self._latest_spatial_retrieved_count = None

        # Decode WITH a short re-encoded GT-history context so all framepack_num_new_video_frames
        # (12 at g3) generated px frames come back aligned: a standalone decode of the 3-latent
        # chunk treats gen latent 0 as an I-latent and yields only 1+2*4=9 frames, the first of
        # them ~3 px frames early. Mirrors the self-aug stage's stitch decode (lyra2_model.py).
        frames = None
        F_new = int(self.framepack_num_new_video_frames)
        if gen_idx is not None and int(gen_idx[0]) > 0:
            try:
                ctx_px = min(9, int(gen_idx[0]))  # 9 px frames -> 3 context latents
                ctx_px = 1 + ((ctx_px - 1) // 4) * 4  # snap to the causal VAE's 1+4k grid
                s0 = int(gen_idx[0]) - ctx_px
                ctx = gt_video_full[:, :, s0 : s0 + ctx_px].to(
                    device=latents.device, dtype=self.tensor_kwargs["dtype"]
                )
                ctx_lat = self.encode(ctx)
                frames = self.decode(torch.cat([ctx_lat, latents.to(ctx_lat.dtype)], dim=2))
                frames = frames[:, :, -F_new:]
            except Exception as e:  # noqa: BLE001 -- alignment nicety; plain decode still works
                log.info(f"[val] context decode failed ({e}); falling back to plain decode")
                frames = None
        if frames is None:
            frames = self.decode(latents)  # (B,3,T_gen,H,W) in [-1,1]

        # Val loss: pixel MSE vs the WINDOW-ALIGNED GT (the window start/segment are random,
        # so data_batch["video"]'s tail is generally the wrong reference).
        gt = gt_vis if gt_vis is not None else data_batch["video"]
        gt = gt.to(device=frames.device)
        t = min(frames.shape[2], gt.shape[2])
        val_loss = torch.nn.functional.mse_loss(frames[:, :, -t:].float(), gt[:, :, -t:].float())

        if distributed.is_rank0():
            out_dir = Path(os.environ.get("LYRA_VAL_SAMPLE_DIR", "val_samples"))
            out_dir.mkdir(parents=True, exist_ok=True)

            rows: list[tuple[str, np.ndarray]] = []
            gen_u8 = _to_uint8(frames[:, :, -t:])
            gt_u8 = _to_uint8(gt[:, :, -t:])
            rows.append(("gt", gt_u8))
            rows.append(("gen", gen_u8))
            err = (frames[0, :, -t:].float() - gt[0, :, -t:].float()).abs().mean(dim=0)  # (t,H,W) in [0,2]
            rows.append(("err", _gray_to_uint8(err - 1.0)))

            if cond_state is not None:
                # Channel layout: buffer warp RGB(3) | spatial warp RGB(3 x N) | plucker(6).
                C = int(cond_state.shape[1])
                has_plucker = C >= 9 and (C - 6) % 3 == 0
                n_spatial = max(0, ((C - 6 if has_plucker else C) - 3) // 3)
                rows.append(("warp", _to_uint8(cond_state[:, :3, -t:])))
                for i in range(min(n_spatial, MAX_SPATIAL_ROWS)):
                    rows.append((f"warp_s{i}", _to_uint8(cond_state[:, 3 + 3 * i : 6 + 3 * i, -t:])))
                if has_plucker:
                    rows.append(("ray_dir", _to_uint8(cond_state[:, -3:, -t:])))

            if warp_depth is not None:
                rows.append(("warp_depth", _gray_to_uint8(warp_depth[0, 0, -t:])))
            elif spatial_depth is not None:  # fall back to slot-0 warp depth
                rows.append(("warp_depth", _gray_to_uint8(spatial_depth[0, 0, -t:])))

            if gt_depth_full is not None and gen_idx is not None:
                idx = gen_idx[-t:].to(gt_depth_full.device).long()
                d = gt_depth_full[0, idx, 0]  # (t,H,W) metric
                rows.append(("gt_depth", _metric_depth_to_uint8(d)))

            if slot_ids_val:
                rows.append(("slots", _slot_tiles(gt_video_full, slot_ids_val, slot_real, t)))

            if agent_pos is not None and agent_dir is not None and gen_idx is not None:
                try:
                    layout_np = maze_layout[0].cpu().numpy() if maze_layout is not None else None
                    rows.append((
                        "topdown",
                        _topdown_row(
                            layout_np,
                            agent_pos[0].cpu().numpy(),
                            agent_dir[0].cpu().numpy(),
                            gen_idx[-t:].tolist(),
                            slot_ids_val,
                            slot_real,
                            size=int(gt.shape[-2]),
                        ),
                    ))
                except Exception as e:  # noqa: BLE001 -- topdown row is best-effort viz
                    log.info(f"[val] topdown row skipped: {e}")

            # ALL generated frames as columns (12 at the g3 framepack) -- no subsampling.
            col = np.arange(t)
            grid = np.concatenate([np.concatenate(list(r[col]), axis=1) for _, r in rows], axis=0)
            key = data_batch.get("__key__", ["sample"])
            key = key[0] if isinstance(key, (list, tuple)) else str(key)
            path = out_dir / f"iter_{iteration:09d}_{key}.png"

            try:
                import imageio.v2 as imageio

                imageio.imwrite(path, grid)
            except ImportError:
                np.save(path.with_suffix(".npy"), grid)

            row_names = "/".join(name for name, _ in rows)
            log.info(f"[val] iter {iteration} key={key}: val_mse={val_loss.item():.4f} rows={row_names} -> {path}")

        return {}, val_loss
