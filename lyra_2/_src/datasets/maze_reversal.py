"""Trajectory-reversal Memory-Maze episodes as Lyra2 batches, for the recall probe.

Reads the offline trajectory-reversal WebDataset shards
(``data/memory-maze-15x15/retrieval_data/<n_fwd>x64x64/train-*.tar``), where each episode is
three pre-baked legs -- ``fwd`` (a "tour" walk out), ``turn`` (a 21-frame in-place spin) and
``bwd`` (the walk back along roughly the same path). ``lyra_2/tasks/retrieval.py`` prefills the
model's memory with the forward leg and asks it to generate turn+bwd, so recall of the outbound
path is what is being scored.

Two things this loader does that ``MazeLyraDataset`` cannot:

  1. **Leg awareness.** ``MazeLyraDataset``'s ``_sanitize_maze`` maps every member ending in
     ``image.npy`` to the bare key ``image.npy``, so ``<key>.fwd.image.npy`` / ``.turn.`` /
     ``.bwd.`` collide into one slot (last write wins) -- the leg dimension is structurally
     invisible to it. Its shard glob is ``shard-*.tar`` too, not ``train-*.tar``.
  2. **Chunk-grid trimming.** Lyra2 generates in 12-pixel-frame chunks starting at absolute index
     ``1 + 12k``, so generation can only begin on the first turn frame if ``(n_fwd - 1) % 12 == 0``.
     ``px_trim`` frames are dropped from the FRONT of the clip to make that hold; only forward
     frames -- the ones farthest from the turnaround -- are lost. See :func:`chunk_trim`.

Everything else matches ``MazeLyraDataset`` exactly (same key set, same conventions: ``video`` in
[-1,1] as (3,N,H,W), PIXEL-space ``intrinsics``, OpenCV ``camera_w2c`` rebased so ``w2c[0] = I``,
metric ``depth`` in maze cells with sky -> 0, constant T5 embedding). The geometry helpers are
imported from ``maze_streaming`` rather than copied -- they are parity-checked against the host
repo's ``src/data/dataset_maze.py`` and must not be forked.

The three legs share ONE camera frame: poses are built by a single ``_agent_pose_to_c2w`` call on
the CONCATENATED ``[fwd|turn|bwd]`` trajectory, so a backward frame's pose is directly comparable
to the forward pose it revisits (this is what makes spatial-memory retrieval across the reversal
meaningful). Ported from the host repo's ``src/tasks/reversal_common.PixelReversalDataset``, whose
episode ordering is matched exactly so episode ``i`` is the same episode in both probes.

Smoke test (CPU):
  python -m lyra_2._src.datasets.maze_reversal --root /cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/retrieval_data/401x64x64
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lyra_2._src.datasets.maze_streaming import (
    _DEFAULT_T5_PATH,
    _agent_pose_to_c2w,
    _build_intrinsics,
    _dict_collation_fn,
)

LEGS = ("fwd", "turn", "bwd")
_PER_FRAME_FIELDS = ("image", "depth", "agent_pos", "agent_dir")


def chunk_trim(n_fwd: int, chunk_px: int = 12) -> int:
    """Pixel frames to drop from the clip front so generation starts on the chunk grid.

    Lyra2 generates ``chunk_px`` frames per AR step beginning at absolute index ``1 + k*chunk_px``,
    so the forward leg must end on that lattice for the first generated frame to be the first turn
    frame. Trimming the front (rather than the back) keeps the frames adjacent to the turnaround,
    which are the ones the model most needs in its recent-history buffer.

    Args:
        n_fwd: Forward-leg length before trimming.
        chunk_px: ``framepack_num_new_video_frames`` (12 for the ``g3`` framepack).

    Returns:
        The number of leading pixel frames to drop.
    """
    assert (n_fwd - 1) % 4 == 0, f"n_fwd={n_fwd} is off the causal VAE's 1+4k grid"
    return (n_fwd - 1) % int(chunk_px)


class MazeReversalDataset(Dataset):
    """Map-style reader over the trajectory-reversal tars, emitting Lyra2 batch dicts.

    Args:
        root: Directory holding the variant's ``*.tar`` shards, e.g.
            ``.../retrieval_data/401x64x64``.
        height: Frame height; used only to build the intrinsics (frames are not resized).
        width: Frame width; same.
        fov_vertical_deg: Vertical FoV the renderer used. Must match training.
        eye_height: Camera height above the floor, in maze cells. Must match training.
        camera_pitch_deg: Downward mount tilt baked into the poses. 5.71 (= atan(0.1)) is the
            camera's true tilt and what the depth renderer applies; pass 0.0 only to serve
            checkpoints fitted on the old level poses.
        chunk_px: ``framepack_num_new_video_frames``; 0 disables front trimming.
        t5_embedding_path: The constant UMT5 embedding used as text conditioning (no captions).
        fps: Conditioning scalar only; maze steps are not real fps.

    Attributes:
        index: ``(tar_path, episode_key)`` pairs in the same order as the host repo's
            ``PixelReversalDataset`` -- ``sorted(tars)`` then ``sorted(keys)`` within each.
    """

    def __init__(
        self,
        root,
        *,
        height: int = 64,
        width: int = 64,
        fov_vertical_deg: float = 80.0,
        eye_height: float = 0.45,
        camera_pitch_deg: float = 5.71,
        chunk_px: int = 12,
        t5_embedding_path: str = _DEFAULT_T5_PATH,
        fps: int = 24,
    ) -> None:
        self.height, self.width = int(height), int(width)
        self.fov_vertical_deg = float(fov_vertical_deg)
        self.eye_height = float(eye_height)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.chunk_px = int(chunk_px)
        self.t5_embedding_path = str(t5_embedding_path)
        self.fps = int(fps)

        tars = sorted(Path(root).glob("*.tar"))
        if not tars:
            raise FileNotFoundError(f"No *.tar found under {root}")

        self.index: list[tuple[Path, str]] = []
        for tp in tars:
            with tarfile.open(tp) as tf:
                keys = sorted(
                    {m.name.split(".", 1)[0] for m in tf.getmembers() if m.name.endswith(".fwd.image.npy")}
                )
            self.index += [(tp, k) for k in keys]

        self._t5: torch.Tensor | None = None  # lazy per-worker load

    def _t5_embedding(self) -> torch.Tensor:
        if self._t5 is None:
            d = torch.load(self.t5_embedding_path, map_location="cpu", weights_only=False)
            emb = d["t5_text_embeddings"] if isinstance(d, dict) else d
            # bf16: the net runs bfloat16 and text_embedding has no autocast (wan2pt1_lyra2.py:968).
            self._t5 = emb.reshape(-1, emb.shape[-1])[:512].to(torch.bfloat16)  # (512, 4096)
        return self._t5

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        tp, key = self.index[idx]
        H, W = self.height, self.width

        with tarfile.open(tp) as tf:

            def _load(suffix: str) -> np.ndarray:
                return np.load(io.BytesIO(tf.extractfile(f"{key}.{suffix}").read()))

            per_leg = {f: {leg: _load(f"{leg}.{f}.npy") for leg in LEGS} for f in _PER_FRAME_FIELDS}
            maze = _load("maze_layout.npy")
            meta = json.loads(tf.extractfile(f"{key}.json").read().decode())

        legs_raw = tuple(int(per_leg["agent_pos"][leg].shape[0]) for leg in LEGS)
        n_fwd = legs_raw[0]
        px_trim = chunk_trim(n_fwd, self.chunk_px) if self.chunk_px else 0

        # Concatenate the legs, THEN trim the front: every per-frame array must be sliced
        # identically or the top-down maps desynchronise from the video.
        cat = {f: np.concatenate([per_leg[f][leg] for leg in LEGS], 0)[px_trim:] for f in _PER_FRAME_FIELDS}
        n_clip = int(cat["image"].shape[0])
        legs = (n_fwd - px_trim, legs_raw[1], legs_raw[2])
        assert sum(legs) == n_clip, f"leg lengths {legs} do not sum to clip length {n_clip}"

        # ONE pose build over the concatenated trimmed trajectory: rebases to the NEW frame 0 and
        # puts all three legs in a single camera frame.
        c2w = _agent_pose_to_c2w(
            cat["agent_pos"], cat["agent_dir"], eye_height=self.eye_height, pitch_deg=self.camera_pitch_deg
        )  # (n_clip, 4, 4)
        w2c = torch.linalg.inv(c2w)

        # Lyra2 consumes PIXEL-space K (ray_condition / forward_warp use fx,fy,cx,cy in px).
        K_pix = _build_intrinsics(n_clip, H, W, self.fov_vertical_deg)
        K_pix[:, 0, :] *= W
        K_pix[:, 1, :] *= H

        video = torch.from_numpy(cat["image"].copy()).permute(3, 0, 1, 2).contiguous()  # (3,N,H,W)
        video = video.float().div_(127.5).sub_(1.0)  # [-1, 1]

        d = cat["depth"].astype(np.float32)
        d[~np.isfinite(d)] = 0.0  # sky/inf -> invalid; units = maze cells (match c2w translation)
        depth = torch.from_numpy(d).unsqueeze(1)  # (N,1,H,W)

        return {
            "video": video,
            "camera_w2c": w2c.float().contiguous(),
            "intrinsics": K_pix,
            "depth": depth,
            "agent_pos": torch.from_numpy(cat["agent_pos"].astype(np.float32).copy()),  # (N,2) cells
            "agent_dir": torch.from_numpy(cat["agent_dir"].astype(np.float32).copy()),  # (N,2) unit
            "maze_layout": torch.from_numpy(maze.astype(np.uint8).copy()),  # (15,15)
            "is_preprocessed": True,
            "t5_text_embeddings": self._t5_embedding().clone(),  # (512,4096) constant (no captions)
            "t5_text_mask": torch.ones(512),
            # Absolute indices in the UNTRIMMED episode, so a trimmed clip stays traceable.
            "sample_frame_indices": torch.arange(px_trim, px_trim + n_clip, dtype=torch.long),
            "num_frames": n_clip,
            "fps": self.fps,
            "image_size": torch.tensor([H, W], dtype=torch.long),
            "padding_mask": torch.zeros(1, H, W),
            # Probe bookkeeping: the trim / leg split / generation start live here so the index
            # arithmetic exists in exactly one place.
            "legs": legs,
            "px_trim": px_trim,
            "gen_start": legs[0],  # first turn frame == prefill length
            "n_chunks": (n_clip - legs[0]) // max(self.chunk_px, 1) if self.chunk_px else 0,
            "seed": int(meta.get("seed", -1)),
            "__key__": key,
            "clip_name": f"mazerev-{key}-t{px_trim:03d}",
        }


def build_reversal_dataloader(
    dataset: MazeReversalDataset, batch_size: int = 1, num_workers: int = 0
) -> DataLoader:
    """DataLoader over the reversal dataset with Lyra2's dict collation.

    ``num_workers=0`` by default: one sample is a whole multi-hundred-frame episode, so the probe
    is bound by generation, not loading, and worker processes would just duplicate the tar reads.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_dict_collation_fn,
    )


if __name__ == "__main__":
    import argparse
    import math

    from lyra_2._src.datasets.forward_warp_utils_pytorch import unproject_points

    # Memory-Maze 15x15 scene constants in maze cells (MemoryMazeDataGen/.../render/depth.py).
    EYE_HEIGHT = 0.45
    WALL_HEIGHT = 0.75
    TRUE_PITCH_DEG = 5.71

    ap = argparse.ArgumentParser(description="Smoke test the Lyra2 maze reversal loader.")
    ap.add_argument(
        "--root",
        default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/retrieval_data/401x64x64",
    )
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    ds = MazeReversalDataset(args.root)
    print(f"episodes: {len(ds)} from {len({t for t, _ in ds.index})} shards")
    dl = build_reversal_dataloader(ds)

    it = iter(dl)
    for i in range(min(args.episodes, len(ds))):
        b = next(it)
        n = int(b["num_frames"][0])
        legs = tuple(int(x) for x in b["legs"][0])
        trim, gen_start = int(b["px_trim"][0]), int(b["gen_start"][0])

        assert b["video"].shape == (1, 3, n, 64, 64), b["video"].shape
        v = b["video"]
        assert -1.0001 <= float(v.min()) and float(v.max()) <= 1.0001, (v.min(), v.max())
        assert b["camera_w2c"].shape == (1, n, 4, 4)
        # frame-0 c2w = I  =>  frame-0 w2c = I (the single-call rebase).
        assert torch.allclose(b["camera_w2c"][0, 0], torch.eye(4), atol=1e-4), "w2c[0] != I"
        fx = float(b["intrinsics"][0, 0, 0, 0])
        assert abs(fx - (0.5 / math.tan(math.radians(80.0) / 2.0)) * 64) < 1e-2, f"pixel fx off: {fx}"
        assert b["depth"].shape == (1, n, 1, 64, 64) and torch.isfinite(b["depth"]).all()
        assert float(b["depth"].min()) >= 0.0
        assert b["t5_text_embeddings"].shape == (1, 512, 4096)
        assert bool(b["is_preprocessed"]) is True
        # Chunk-grid invariants: generation starts on the first turn frame, on the 1+12k lattice.
        assert sum(legs) == n, (legs, n)
        assert gen_start == legs[0], (gen_start, legs)
        assert (gen_start - 1) % 12 == 0, f"gen_start={gen_start} is off the 1+12k chunk grid"
        n_chunks = int(b["n_chunks"][0])
        tail = (n - gen_start) - 12 * n_chunks
        print(
            f"ep {i}: key={b['__key__'][0]} seed={int(b['seed'][0])} legs={legs} trim={trim} "
            f"clip={n} gen[{gen_start},{gen_start + 12 * n_chunks}) chunks={n_chunks} tail={tail} "
            f"fx_pix={fx:.2f} depth[{b['depth'].min():.2f},{b['depth'].max():.2f}]"
        )

    # ---- Pitch check: the reversal shards' build config carries no `depth:` block, so the
    # renderer's own default (CAMERA_PITCH_DEG = 5.71) was used. Confirm empirically rather than
    # trusting it: unproject GT depth and measure how sharply the reconstructed points pile up on
    # the floor plane. The maze is axis-aligned, so a correct pose gives a spike at -eye_height and
    # few points outside [floor, wall_top]; a wrong pitch hinges every frame and smears both.
    b = next(iter(build_reversal_dataloader(MazeReversalDataset(args.root))))
    n = int(b["num_frames"][0])
    probes = list(range(0, n, max(1, n // 12)))[:12]

    def _floor_stats(pitch_deg: float) -> tuple[float, float]:
        ds_p = MazeReversalDataset(args.root, camera_pitch_deg=pitch_deg)
        bp = next(iter(build_reversal_dataloader(ds_p)))
        w2c, K, dep = bp["camera_w2c"][0], bp["intrinsics"][0], bp["depth"][0]
        # The dataset rebases so c2w[0] = I, i.e. world axes ARE frame 0's camera axes; with a
        # tilted mount, true world-up in that (right, down, forward) basis is (0,-cos p,-sin p).
        p = math.radians(pitch_deg)
        up = torch.tensor([0.0, -math.cos(p), -math.sin(p)])
        hs = []
        for t in probes:
            pts = unproject_points(
                depth=dep[t].unsqueeze(0),
                w2c=w2c[t].unsqueeze(0),
                intrinsic=K[t].unsqueeze(0),
                is_depth=True,
                is_ftheta=False,
                mask=(dep[t].unsqueeze(0) > 0),
                return_sparse=True,
            )[0]
            if pts.numel():
                hs.append(pts @ up)
        h = torch.cat(hs)
        spike = float(((h + EYE_HEIGHT).abs() < 0.03).float().mean())
        out = float(((h < -EYE_HEIGHT - 0.03) | (h > WALL_HEIGHT - EYE_HEIGHT + 0.03)).float().mean())
        return spike, out

    spike_p, out_p = _floor_stats(TRUE_PITCH_DEG)
    spike_0, out_0 = _floor_stats(0.0)
    print(
        f"\npitch check over {len(probes)} frames (floor_spike higher is better, frac_out lower):\n"
        f"  pitch={TRUE_PITCH_DEG:.2f}\tfloor_spike={spike_p:.4f}\tfrac_out={out_p:.4f}\n"
        f"  pitch=0.00\tfloor_spike={spike_0:.4f}\tfrac_out={out_0:.4f}"
    )
    assert spike_p > spike_0, (
        f"level poses reconstruct the floor better than {TRUE_PITCH_DEG} deg "
        f"({spike_0:.4f} vs {spike_p:.4f}) -- these shards may have been rendered level"
    )
    print(f"PASS: reversal shards carry the {TRUE_PITCH_DEG} deg mount tilt.")

    print("PASS: maze reversal loader yields well-formed Lyra2 batches.")
