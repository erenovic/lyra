"""Memory-Maze small-scale from-scratch training configs for Lyra 2.0.

Auto-imported by ``make_config()`` (``import_all_modules_from_package``). Registers:
  * nets: ``wan2pt1_240M_i2v_lyra2`` (main) and ``wan2pt1_65M_i2v_lyra2`` (smoke);
  * data: ``maze_streaming_train`` / ``maze_streaming_val`` (LazyCall -- built at
    ``instantiate`` time, unlike the eager upstream dataloader nodes);
  * experiments: ``lyra2_maze_small`` (240M, production-shaped framepack with 5 spatial
    slots, g=5) and ``lyra2_maze_smoke`` (65M, temporal-only framepack, 50 iters).

From-scratch specifics vs the upstream ``lyra2`` finetune experiment: DDP (not FSDP),
``framepack_trainable_modules=None`` (train everything), ``init_framepack_weights=False``
(no Wan init to copy from), self-aug and spatial-region corruption off, higher LR, and
``conditioner.text.dropout_rate=0.0`` -- the released snapshot ships no
``empty_string_umt5.pt``, so text dropout would crash; the dataset injects a constant
``negative_prompt.pt`` embedding instead (see ``datasets/maze_streaming.py``).
"""

from hydra.core.config_store import ConfigStore

from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._src.callbacks.maze_training_logger import MazeTrainingLoggerCallback
from lyra_2._ext.imaginaire.lazy_config import LazyDict
from lyra_2._src.datasets.maze_streaming import MazeLyraDataset, build_maze_dataloader
from lyra_2._src.modules.selective_activation_checkpoint import SACConfig
from lyra_2._src.models.lyra2_model import Lyra2T2VConfig
from lyra_2._src.models.maze_lyra2_model import MazeLyra2Model
from lyra_2._src.networks.wan2pt1_lyra2 import Lyra2WanModel

cs = ConfigStore.instance()

_MAZE_TRAIN_ROOT = "/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/train"
_MAZE_EVAL_ROOT = "/cluster/scratch/ecetin/MemoryKrea/data/memory-maze-15x15/eval"

WAN2PT1_240M_I2V_LYRA2: LazyDict = L(Lyra2WanModel)(
    # Grown to ~250M params (cross-attn pruned) to use the idle 64x64 VRAM (~3.7GB at 113M).
    dim=1024,
    eps=1e-06,
    ffn_dim=4096,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=8,  # head_dim=1024/8=128, matching Wan2.1 (1.3B 1536/12, 14B 5120/40)
    num_layers=16,
    out_dim=16,
    text_len=512,
    patch_size=(1, 1, 1),  # 64x64 -> 8x8 latent; keep all 8x8 tokens (no 2x2 spatial patch)
    disable_cross_attn=True,  # maze has no text/CLIP signal -> drop all cross-attention
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="none"),  # ~152-token seq: SAC recompute saves no VRAM, only costs time
    postpone_checkpoint=False,
)

WAN2PT1_65M_I2V_LYRA2: LazyDict = L(Lyra2WanModel)(
    dim=512,
    eps=1e-06,
    ffn_dim=2048,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=8,
    num_layers=8,
    out_dim=16,
    text_len=512,
    patch_size=(1, 1, 1),  # 64x64 -> 8x8 latent; keep all 8x8 tokens (no 2x2 spatial patch)
    disable_cross_attn=True,  # maze has no text/CLIP signal -> drop all cross-attention
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="none"),  # ~152-token seq: SAC recompute saves no VRAM, only costs time
    postpone_checkpoint=False,
)


DDP_MAZE_LYRA2_SPATIAL = dict(
    trainer=dict(distributed_parallelism="ddp"),
    model=L(MazeLyra2Model)(
        config=Lyra2T2VConfig(state_t=20),
        _recursive_=False,
    ),
)


def register_maze_model():
    cs.store(group="model", package="_global_", name="ddp_maze_lyra2_spatial", node=DDP_MAZE_LYRA2_SPATIAL)


def register_maze_nets():
    cs.store(group="net", package="model.config.net", name="wan2pt1_240M_i2v_lyra2", node=WAN2PT1_240M_I2V_LYRA2)
    cs.store(group="net", package="model.config.net", name="wan2pt1_65M_i2v_lyra2", node=WAN2PT1_65M_I2V_LYRA2)


def register_maze_dataloaders():
    train_node = L(build_maze_dataloader)(
        dataset=L(MazeLyraDataset)(
            roots=[_MAZE_TRAIN_ROOT],
            num_frames=401,
            stage="train",
            resampled=True,
        ),
        batch_size=8,  # per-rank B (net is batch-generic); uses idle VRAM, larger effective batch
        num_workers=4,
        prefetch_factor=2,
    )
    val_node = L(build_maze_dataloader)(
        dataset=L(MazeLyraDataset)(
            roots=[_MAZE_EVAL_ROOT],
            num_frames=401,
            stage="eval",
            resampled=False,
        ),
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
    )
    cs.store(group="data_train", package="dataloader_train", name="maze_streaming_train", node=train_node)
    cs.store(group="data_val", package="dataloader_val", name="maze_streaming_val", node=val_node)


# Trained 243M base checkpoint (job 7931520/8130376, iter 50000); the self-aug stage finetunes it.
_TRAINED_SMALL_CKPT = (
    "/cluster/scratch/ecetin/MemoryKrea/src/external/lyra/outputs/lyra2_v2/"
    "lyra_2/maze/lyra2_maze_small/checkpoints/iter_000050000"
)


def _maze_small_experiment(name: str, self_aug_enabled: bool = False, load_path: str = "",
                           batch_size: int | None = None) -> dict:
    """Build the 240M maze experiment config. ``self_aug_enabled`` + ``load_path`` distinguish the
    from-scratch base run from the self-augmentation finetune (loads base weights only, iter 0).
    ``batch_size`` overrides the dataloader batch (self-aug's denoise loop is B=1-only)."""
    cfg = dict(
        defaults=[
            {"override /model": "ddp_maze_lyra2_spatial"},
            {"override /net": "wan2pt1_240M_i2v_lyra2"},
            {"override /conditioner": "lyra2_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /data_train": "maze_streaming_train"},
            {"override /data_val": "maze_streaming_val"},
            "_self_",
        ],
        job=dict(project="lyra_2", group="maze", name=name),
        model=dict(
            config=dict(
                ema=dict(enabled=False),
                # 20 temporal-history latents + 5 spatial slots, generate 5 latents (20 px frames).
                framepack_type="f1k1f4s2f1s1f16k4f2k2f1k1_g5",
                state_t=5,
                max_segments=20,  # with num_frames=401: windows up to ~401f -> full-length AR rollout
                starting_frame_ratio=0.0,
                init_framepack_weights=False,  # from scratch: no Wan patch-embed to copy
                framepack_trainable_modules=None,  # train the whole net
                self_aug_enabled=self_aug_enabled,  # Stage-A self-augmentation (exposure-bias reduction)
                apply_corruption_to_spatial_region="none",
                spatial_memory_use_image=True,
                spatial_memory_stride=8,
                spatial_memory_skip_recent=16,
                spatial_memory_downsample=1,  # full-res depth point cloud (4 too coarse at 64x64)
                warp_chunk_size=16,
                # 0.05 (upstream) discards ~90%+ of 64x64 maze warp points; 0.45 keeps more coverage.
                warp_continuity_ratio_thresh=0.45,
                fsdp_shard_size=1,
                precision="bfloat16",
                conditioner=dict(text=dict(dropout_rate=0.0), wanclip=dict(disable_clip_forward=True)),  # no empty_string_umt5.pt in snapshot
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        optimizer=dict(lr=1e-4, weight_decay=1e-3),
        scheduler=dict(warm_up_steps=[2000]),
        checkpoint=dict(
            save_iter=500,
            # load_path (base weights) is used only on the first submit; requeues auto-resume this
            # run's own checkpoints. load_training_state=False -> weights only, fresh optimizer/iter.
            load_path=load_path,
            load_training_state=False,
            strict_resume=False,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=50000,
            logging_iter=10,
            run_validation=True,
            validation_iter=2000,
            max_val_iter=2,
            grad_accum_iter=4,
            callbacks=dict(iter_logger=L(MazeTrainingLoggerCallback)()),
            ddp=dict(find_unused_parameters=True, static_graph=False),
        ),
    )
    if batch_size is not None:
        cfg["dataloader_train"] = dict(batch_size=batch_size)
    return cfg


def register_lyra2_maze_small():
    """240M from-scratch run: production-shaped framepack (5 spatial slots), 20-frame chunks."""
    cs.store(group="experiment", package="_global_", name="lyra2_maze_small",
             node=_maze_small_experiment("lyra2_maze_small"))


def register_lyra2_maze_selfaug():
    """Self-augmentation finetune: same config with self_aug_enabled=True, weights initialized from
    the trained base checkpoint (50k). Trains a fresh 50k steps."""
    cs.store(group="experiment", package="_global_", name="lyra2_maze_selfaug",
             node=_maze_small_experiment("lyra2_maze_selfaug", self_aug_enabled=True,
                                         load_path=_TRAINED_SMALL_CKPT, batch_size=1))


def register_lyra2_maze_smoke():
    """65M pipeline smoke: temporal-only framepack, short windows, 50 iters."""
    experiment_config = dict(
        defaults=[
            {"override /model": "ddp_maze_lyra2_spatial"},
            {"override /net": "wan2pt1_65M_i2v_lyra2"},
            {"override /conditioner": "lyra2_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /data_train": "maze_streaming_train"},
            {"override /data_val": "maze_streaming_val"},
            "_self_",
        ],
        job=dict(project="lyra_2", group="maze", name="lyra2_maze_smoke"),
        model=dict(
            config=dict(
                ema=dict(enabled=False),
                # Same production-shaped layout as the main run: the model hard-forces
                # use_correspondence=True (lyra2_model.py:268), so a temporal-only framepack
                # with spatial memory off is not a supported path.
                framepack_type="f1k1f4s2f1s1f16k4f2k2f1k1_g5",
                state_t=5,
                max_segments=4,
                starting_frame_ratio=0.0,
                init_framepack_weights=False,
                framepack_trainable_modules=None,
                self_aug_enabled=False,
                apply_corruption_to_spatial_region="none",
                spatial_memory_use_image=True,
                spatial_memory_stride=8,
                spatial_memory_skip_recent=16,
                spatial_memory_downsample=1,  # full-res depth point cloud (4 too coarse at 64x64)
                warp_chunk_size=16,
                # 0.05 (upstream) discards ~90%+ of 64x64 maze warp points; 0.35 keeps ~6-8x more
                # while depth stays exact at short offsets (see eval_warp_geometry.py sweep).
                warp_continuity_ratio_thresh=0.45,
                fsdp_shard_size=1,
                precision="bfloat16",
                conditioner=dict(text=dict(dropout_rate=0.0), wanclip=dict(disable_clip_forward=True)),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        optimizer=dict(lr=1e-4, weight_decay=1e-3),
        scheduler=dict(warm_up_steps=[10]),
        checkpoint=dict(
            save_iter=20,
            load_path="",
            load_training_state=False,
            strict_resume=False,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=50,
            logging_iter=5,
            run_validation=True,
            validation_iter=25,
            max_val_iter=1,
            grad_accum_iter=1,
            callbacks=dict(iter_logger=L(MazeTrainingLoggerCallback)()),
            ddp=dict(find_unused_parameters=True, static_graph=False),
        ),
        dataloader_train=dict(dataset=dict(num_frames=101)),
    )
    cs.store(group="experiment", package="_global_", name="lyra2_maze_smoke", node=experiment_config)


register_maze_model()
register_maze_nets()
register_maze_dataloaders()
register_lyra2_maze_small()
register_lyra2_maze_selfaug()
register_lyra2_maze_smoke()
