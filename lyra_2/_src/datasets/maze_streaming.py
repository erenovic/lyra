"""Streaming Memory-Maze dataset emitting Lyra2 training batches.

Streams the raw Memory-Maze WebDataset shards (one ~1001-frame 64x64 episode per sample,
written by mmz-convert-zip: flat members ``<key>.image.npy`` / ``.depth.npy`` /
``.agent_pos.npy`` / ``.agent_dir.npy`` / ``.maze_layout.npy`` / ``.json``) and emits the
per-sample dict Lyra2 training consumes -- the ``InfiniteCommonDataset`` schema of
``depth_warp_dataloader.py`` (``video`` in [-1,1], pixel-space ``intrinsics``,
``camera_w2c``, metric ``depth``, constant T5 text embedding, ...). The model VAE-encodes
pixels internally and builds Plucker rays from ``intrinsics`` + ``camera_w2c`` itself, so
no latents/plucker are produced here.

Mirrors ``src/data/dataset_maze_streaming.py`` in the host repo (wds ``split_by_node`` +
``split_by_worker``, resampled infinite train stream, per-worker seeds) instead of the
upstream map-style ``Radym`` + megatron DP sharding. The geometry helpers are ported
verbatim from ``src/data/dataset_maze.py`` (source of truth; keep byte-identical).

Text conditioning: Memory Maze has no captions, so a single fixed UMT5 embedding
(``checkpoints/text_encoder/negative_prompt.pt``) is injected as ``t5_text_embeddings``
directly -- the optional ``t5_chunk_*`` branch (lyra2_model.py, guarded by
``"t5_chunk_keys" in data_batch``) is intentionally skipped. Requires the conditioner's
text ``dropout_rate=0.0`` (the snapshot ships no ``empty_string_umt5.pt``).

Smoke test:
  python -m lyra_2._src.datasets.maze_streaming --root /cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# src/external/lyra; the checkpoints symlink lives at its root (staged per job).
_LYRA_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_T5_PATH = str(_LYRA_ROOT / "checkpoints" / "text_encoder" / "negative_prompt.pt")

# Flat per-field members the converter writes.
_RAW_SUFFIXES = ("image.npy", "depth.npy", "agent_pos.npy", "agent_dir.npy", "maze_layout.npy", "json")


# ---------------------------------------------------------------------------
# Geometry helpers, ported VERBATIM from src/data/dataset_maze.py (host repo).
# Do not edit here without syncing the source of truth (parity-checked in __main__).
# ---------------------------------------------------------------------------


def _build_intrinsics(num_frames: int, height: int, width: int, fov_vertical_deg: float) -> torch.Tensor:
    """(T, 3, 3) NORMALIZED intrinsics shared across frames (FoV-only).

    fx/W, fy/H, cx/W, cy/H; for square images fx_norm = fy_norm = 0.5 / tan(fov_vert/2).
    """
    fov_rad = math.radians(fov_vertical_deg)
    fy_pixel = (height / 2.0) / math.tan(fov_rad / 2.0)
    fx_pixel = fy_pixel  # square pixels: vertical FoV determines both
    K = torch.tensor(
        [
            [fx_pixel / width, 0.0, 0.5],
            [0.0, fy_pixel / height, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return K.unsqueeze(0).expand(num_frames, 3, 3).contiguous()


def _agent_pose_to_c2w(
    agent_pos: np.ndarray, agent_dir: np.ndarray, eye_height: float, pitch_deg: float = 0.0
) -> torch.Tensor:
    """Build (T, 4, 4) c2w from Memory Maze agent state.

    Maze (x, y) floor -> y-up world (x, eye_height, -y); camera basis (right, down,
    forward); frame-0 rebased so c2w[0] = I. See src/data/dataset_maze.py for the
    full derivation and handedness rationale.

    Args:
        pitch_deg: Constant downward tilt of the camera, in degrees. The Memory-Maze
            egocentric camera is rigidly mounted 5.71 deg (= atan(0.1)) below horizontal
            (``zaxis = (0, -0.995, 0.0995)``, ``camera_control=False``), and the depth
            renderer bakes that tilt into the depth maps. A level pose (0.0) therefore
            disagrees with the data and hinges every unprojected point cloud upward by
            this angle. Defaults to 0.0 so the level construction is still reachable
            (and so the parity check against ``src.data.camera_geometry`` still holds);
            :class:`MazeLyraDataset` passes the real tilt.
    """
    T = agent_pos.shape[0]

    eye = torch.zeros(T, 3, dtype=torch.float32)
    eye[:, 0] = torch.from_numpy(np.asarray(agent_pos[:, 0])).float()
    eye[:, 1] = float(eye_height)
    eye[:, 2] = -torch.from_numpy(np.asarray(agent_pos[:, 1])).float()

    forward = torch.zeros(T, 3, dtype=torch.float32)
    forward[:, 0] = torch.from_numpy(np.asarray(agent_dir[:, 0])).float()
    forward[:, 2] = -torch.from_numpy(np.asarray(agent_dir[:, 1])).float()
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    up = torch.tensor([0.0, 1.0, 0.0]).expand(T, 3)
    right = torch.cross(forward, up, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    down = torch.cross(forward, right, dim=-1)  # = -up for a level camera

    if pitch_deg:
        # Rotate the optical axis down about the camera's own right axis, matching the depth
        # renderer's basis
        cp = math.cos(math.radians(float(pitch_deg)))
        sp = math.sin(math.radians(float(pitch_deg)))
        forward, down = cp * forward + sp * down, -sp * forward + cp * down

    R_c2w = torch.stack([right, down, forward], dim=-1)  # (T, 3, 3)
    c2w = torch.zeros(T, 4, 4, dtype=torch.float32)
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3] = eye
    c2w[:, 3, 3] = 1.0

    c2w_inv_0 = torch.linalg.inv(c2w[0:1])
    c2w = c2w_inv_0 @ c2w
    return c2w


def _sanitize_maze(sample: dict, suffixes=_RAW_SUFFIXES) -> dict:
    """Remap WebDataset-split keys (``key.field``) back to bare ``field`` names."""
    clean = {}
    for k, v in sample.items():
        if k.startswith("__"):
            clean[k] = v
            continue
        for s in suffixes:
            if k.endswith(s):
                clean[s] = v
                break
        else:
            clean[k] = v
    return clean


def _dict_collation_fn(samples):
    """Collate sample dicts into a batched dict (copied from depth_warp_dataloader.py:81
    -- importing it would drag megatron/transformer_engine into every dataloader worker)."""
    batched = {key: [s[key] for s in samples] for key in samples[0]}
    result = {}
    for key, vals in batched.items():
        if isinstance(vals[0], bool):
            result[key] = vals[0]
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.from_numpy(np.array(vals))
        elif isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


def _worker_seed(base_seed: int) -> int:
    """Distinct per (rank, dataloader-worker) seed so windows/shuffle differ."""
    wi = get_worker_info()
    worker_id = wi.id if wi is not None else 0
    rank = int(os.environ.get("RANK", 0))
    return int(base_seed) + worker_id + rank * 10_000


class MazeLyraDataset(IterableDataset):
    """Streams raw Memory-Maze episodes as Lyra2 training dicts (one window per episode)."""

    def __init__(
        self,
        roots,
        num_frames: int = 201,
        height: int = 64,
        width: int = 64,
        fov_vertical_deg: float = 80.0,
        eye_height: float = 0.45,
        camera_pitch_deg: float = 5.71,
        t5_embedding_path: str = _DEFAULT_T5_PATH,
        fps: int = 24,
        shuffle_buffer: int = 50,
        resampled: bool = True,
        stage: str = "train",
        seed: int = 0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.height, self.width = int(height), int(width)
        self.fov_vertical_deg = float(fov_vertical_deg)
        self.eye_height = float(eye_height)

        # 5.71 (= atan(0.1)) is the camera's TRUE downward mount tilt and matches the depth
        # renderer (REPORT.md section 10; GT reprojection inliers 0.99 vs 0.78 level). Pass 0.0
        # only when serving checkpoints that were fitted on the old level poses.
        self.camera_pitch_deg = float(camera_pitch_deg)

        self.t5_embedding_path = str(t5_embedding_path)
        self.fps = int(fps)
        self.shuffle_buffer = int(shuffle_buffer)
        self.resampled = bool(resampled)
        self.stage = str(stage)
        self.seed = int(seed)
        self.urls = [str(p) for root in roots for p in sorted(Path(root).glob("shard-*.tar"))]
        if not self.urls:
            raise FileNotFoundError(f"No shard-*.tar found in any of: {list(roots)}")

        self._t5: torch.Tensor | None = None  # lazy per-worker load

    def _t5_embedding(self) -> torch.Tensor:
        if self._t5 is None:
            d = torch.load(self.t5_embedding_path, map_location="cpu", weights_only=False)
            emb = d["t5_text_embeddings"] if isinstance(d, dict) else d
            # bf16: the net runs bfloat16 and text_embedding has no autocast (wan2pt1_lyra2.py:968).
            self._t5 = emb.reshape(-1, emb.shape[-1])[:512].to(torch.bfloat16)  # (512, 4096)
        return self._t5

    def _decode(self, rng):
        H, W, N = self.height, self.width, self.num_frames
        K_norm = _build_intrinsics(N, H, W, self.fov_vertical_deg)  # (N,3,3) normalized
        # Lyra2 consumes PIXEL-space K (ray_condition / forward_warp use fx,fy,cx,cy in px).
        K_pix = K_norm.clone()
        K_pix[:, 0, :] *= W
        K_pix[:, 1, :] *= H

        def decode(sample):
            sample = _sanitize_maze(sample)
            img = np.load(io.BytesIO(sample["image.npy"]))  # (T,H,W,3) uint8
            pos = np.load(io.BytesIO(sample["agent_pos.npy"]))  # (T,2) f32
            head = np.load(io.BytesIO(sample["agent_dir.npy"]))  # (T,2) f32
            dep = np.load(io.BytesIO(sample["depth.npy"]))  # (T,H,W) f16, sky=inf
            T = img.shape[0]
            if T < N:
                return None

            start = int(rng.integers(0, T - N + 1))
            sl = slice(start, start + N)

            video = torch.from_numpy(img[sl].copy()).permute(3, 0, 1, 2).contiguous()  # (3,N,H,W)
            video = video.float().div_(127.5).sub_(1.0)  # [-1, 1]

            c2w = _agent_pose_to_c2w(
                pos[sl], head[sl], eye_height=self.eye_height, pitch_deg=self.camera_pitch_deg
            )  # (N,4,4)
            w2c = torch.linalg.inv(c2w)

            d = dep[sl].astype(np.float32)
            d[~np.isfinite(d)] = 0.0  # sky/inf -> invalid; units = maze cells (match c2w translation)
            depth = torch.from_numpy(d).unsqueeze(1)  # (N,1,H,W)

            emb = self._t5_embedding()
            return {
                "video": video,
                "camera_w2c": w2c.float().contiguous(),
                "intrinsics": K_pix.clone(),
                "depth": depth,
                # Raw agent poses (windowed) for the top-down map renderer; the model itself only
                # consumes camera_w2c / intrinsics.
                "agent_pos": torch.from_numpy(pos[sl].astype(np.float32).copy()),  # (N, 2) maze-cell xy
                "agent_dir": torch.from_numpy(head[sl].astype(np.float32).copy()),  # (N, 2) unit heading
                # Wall grid (episode-constant), so the top-down map can draw walls
                "maze_layout": torch.from_numpy(
                    np.load(io.BytesIO(sample["maze_layout.npy"])).astype(np.uint8).copy()
                ),  # (15, 15)
                "is_preprocessed": True,
                "t5_text_embeddings": emb.clone(),  # (512, 4096) constant (no captions)
                "t5_text_mask": torch.ones(512),
                "sample_frame_indices": torch.arange(start, start + N, dtype=torch.long),
                "num_frames": N,
                "fps": self.fps,  # conditioning scalar only; maze steps are not real fps
                "image_size": torch.tensor([H, W], dtype=torch.long),
                "padding_mask": torch.zeros(1, H, W),
                "__key__": sample["__key__"],
                "clip_name": f"maze-{sample['__key__']}-s{start:04d}",
            }

        return decode

    def __iter__(self):
        rng = np.random.default_rng(_worker_seed(self.seed))
        is_resampled = self.resampled and self.stage == "train"
        do_split = not is_resampled
        do_shuffle = self.stage == "train"

        dataset = wds.WebDataset(
            self.urls,
            nodesplitter=wds.split_by_node if do_split else None,
            workersplitter=wds.split_by_worker if do_split else None,
            shardshuffle=do_shuffle if not is_resampled else False,
            resampled=is_resampled,
            empty_check=False,
        )
        if do_shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.map(self._decode(rng)).select(lambda s: s is not None)
        yield from iter(dataset)


def build_maze_dataloader(
    dataset: MazeLyraDataset,
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    """DataLoader over the streaming maze dataset with Lyra2's dict collation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_dict_collation_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke test the Lyra2 maze streaming loader.")
    ap.add_argument("--root", default="/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval")
    ap.add_argument("--num-frames", type=int, default=201)
    ap.add_argument("--batches", type=int, default=2)
    args = ap.parse_args()

    ds = MazeLyraDataset([args.root], num_frames=args.num_frames, stage="eval", resampled=False)
    print(f"shards: {len(ds.urls)}")
    dl = build_maze_dataloader(ds, batch_size=1, num_workers=0)

    it = iter(dl)
    for i in range(args.batches):
        b = next(it)
        v = b["video"]
        N = args.num_frames
        assert v.shape == (1, 3, N, 64, 64), v.shape
        assert -1.0001 <= float(v.min()) and float(v.max()) <= 1.0001, (v.min(), v.max())
        assert b["camera_w2c"].shape == (1, N, 4, 4)
        # frame-0 c2w = I  =>  frame-0 w2c = I.
        assert torch.allclose(b["camera_w2c"][0, 0], torch.eye(4), atol=1e-4), "w2c[0] != I"
        fx = float(b["intrinsics"][0, 0, 0, 0])
        assert abs(fx - (0.5 / math.tan(math.radians(80.0) / 2.0)) * 64) < 1e-2, f"pixel fx off: {fx}"
        assert b["depth"].shape == (1, N, 1, 64, 64) and torch.isfinite(b["depth"]).all()
        assert float(b["depth"].min()) >= 0.0
        assert b["t5_text_embeddings"].shape == (1, 512, 4096)
        assert bool(b["is_preprocessed"]) is True
        print(
            f"batch {i}: video {tuple(v.shape)} [{v.min():.2f},{v.max():.2f}] fx_pix={fx:.2f} "
            f"depth[{b['depth'].min():.2f},{b['depth'].max():.2f}] key={b['__key__']}"
        )

    # Geometry parity vs the host repo's source-of-truth helpers (run from repo root).
    try:
        import sys

        sys.path.insert(0, "/cluster/scratch/ecetin/MemoryKrea")
        from src.data.camera_geometry import agent_pose_to_c2w as ref_pose
        from src.data.camera_geometry import build_intrinsics as ref_K

        pos = np.random.default_rng(0).uniform(0, 15, (8, 2)).astype(np.float32)
        ang = np.random.default_rng(1).uniform(-np.pi, np.pi, 8)
        head = np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32)
        assert torch.equal(ref_K(8, 64, 64, 80.0), _build_intrinsics(8, 64, 64, 80.0))
        # Parity holds only at pitch_deg=0: the host repo's helper still builds a LEVEL camera,
        # which disagrees with the depth renderer's 5.71 deg downward mount tilt (REPORT.md
        # section 10). This loader corrects it; src/data/camera_geometry.py has not been changed.
        assert torch.equal(ref_pose(pos, head, eye_height=0.45), _agent_pose_to_c2w(pos, head, eye_height=0.45))
        pitched = _agent_pose_to_c2w(pos, head, eye_height=0.45, pitch_deg=5.71)
        level = _agent_pose_to_c2w(pos, head, eye_height=0.45)
        assert not torch.allclose(pitched, level, atol=1e-3), "pitch_deg had no effect"
        # The tilt must be a pure rotation about the camera's own right axis, so the basis stays
        # orthonormal and right-handed and the eye position is untouched.
        R = pitched[:, :3, :3]
        assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(R.shape[0], 3, 3), atol=1e-5)
        assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[0]), atol=1e-5)
        # Eye positions are rebased into frame 0's (now tilted) camera frame, so the translation
        # vectors rotate; what must not change is their length -- the tilt is a rotation about the
        # eye, not a displacement of it.
        assert torch.allclose(
            pitched[:, :3, 3].norm(dim=-1), level[:, :3, 3].norm(dim=-1), atol=1e-4
        ), "pitch changed camera positions"
        print("PASS: geometry parity with src.data.dataset_maze (at pitch_deg=0) + pitch sanity")
    except ImportError as e:
        print(f"parity check skipped (host repo not importable): {e}")

    print("PASS: maze streaming loader yields well-formed Lyra2 batches.")
