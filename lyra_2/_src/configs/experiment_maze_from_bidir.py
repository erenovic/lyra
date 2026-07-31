"""Maze finetune bootstrapped from a pretrained bidirectional DiT -- SELF-CONTAINED.

Reproduces upstream's ``lyra2`` recipe (``experiment.py``) at OUR model size instead of 14B. The
pretrained weights come from ``RavenDiTBidir`` (a Wan2.1-backbone bidirectional video DiT trained
in the parent repo) converted by ``src/utils/export_dit_to_lyra.py``. Both stacks share the Wan2.1
RoPE exactly -- ``[t|h|w] = 44/42/42``, ``theta=10000`` per sub-table, GPT-J interleaved pairing,
``(t h w)`` token order -- so the attention weights mean the same thing here.

Deliberately imports nothing from ``experiment_maze.py``: that module's ``_maze_small_experiment``
is shaped for a FROM-SCRATCH run (self-aug off, corruption off, train-everything, lr 1e-4), and
inheriting it meant silently picking up all of that while overriding four keys. Everything is
spelled out below. The ``defaults`` list still NAMES groups registered by ``experiment_maze.py``
(``ddp_maze_lyra2_spatial``, ``maze_streaming_train`` / ``_val``) -- that is a string reference
resolved through the ConfigStore, not a code dependency.

Registers ``lyra2_maze_from_bidir`` (50k) and ``lyra2_maze_from_bidir_smoke`` (50 iters).

From upstream ``lyra2``: self-augmentation + spatial-region corruption from step 0, the trainable
whitelist, lr 3e-5, save_iter 100. From the maze side: the 64x64 / 401-frame framepack geometry,
GT-depth spatial memory, bf16, DDP, and the constant-negative-prompt conditioner.

Launch via ``scripts/euler/train_lyra_from_bidir.sh`` in the parent repo, which supplies
``checkpoint.load_path`` and a fresh output dir.
"""

import os
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._ext.imaginaire.lazy_config import LazyDict
from lyra_2._src.callbacks.maze_training_logger import MazeTrainingLoggerCallback
from lyra_2._src.modules.selective_activation_checkpoint import SACConfig
from lyra_2._src.networks.wan2pt1_lyra2 import Lyra2WanModel

cs = ConfigStore.instance()

# Trainable set from experiment.py:52. Applied by MazeLyra2Model.__init__ (upstream's own pass in
# build_net is undone by wan_t2v_model.py:204). "patch_embedding" already covers
# clean_patch_embeddings and patch_embedding_buffer by substring.
_TRAINABLE = "cam_encoder,buffer_encoder,self_attn,clean_patch_embeddings,patch_embedding"

# MUST match the converted checkpoint exactly -- non_strict_load_model POPS shape mismatches
# SILENTLY. Keep in lockstep with the parent repo's configs/model/dit/raven_dit.yaml and
# configs/utils/export_dit_to_lyra.yaml. Lyra adds per-block cam_/buffer_encoder on top, so the
# net is ~270.4M here vs the DiT's 250.16M (measured at 3 spatial slots -> buffer_in_dim 768).
WAN2PT1_280M_I2V_LYRA2_BIDIR: LazyDict = L(Lyra2WanModel)(
    dim=1024,
    eps=1e-06,
    ffn_dim=6400,
    freq_dim=256,
    in_dim=36,  # [ latent(16) | mask(4) | cond(16) ]
    model_type="i2v",
    num_heads=8,  # head_dim = 1024/8 = 128, as Wan2.1 at every scale
    num_layers=14,
    out_dim=16,
    text_len=512,
    patch_size=(1, 1, 1),  # 64x64 -> 8x8 latent; keep all 8x8 tokens
    disable_cross_attn=True,  # maze has no text/CLIP signal
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="none"),
    postpone_checkpoint=False,
    # NOTE: buffer_* / use_correspondence / use_plucker_condition / inject_kq_only are NOT set --
    # build_net overwrites all seven at lyra2_model.py:272-282, deriving buffer_in_dim from the
    # framepack spatial-slot count.
)


# ---- finetuned Wan-VAE ------------------------------------------------------------------------
# Read from the PARENT REPO's config rather than copied: the weights and their whitening constants
# are one unit (latents written with one pair only decode with that pair), and a second transcript
# of the numbers is exactly how the two drift apart.
_LYRA_ROOT = Path(__file__).resolve().parents[3]  # .../src/external/lyra
_REPO = _LYRA_ROOT.parents[2]  # .../MemoryKrea
_FT_VAE_YAML = _REPO / "configs" / "model" / "vae" / "finetuned_maze64.yaml"


def _load_ft_vae() -> tuple[str, list[float], list[float]]:
    """Read ``(vae_path, latent_mean, latent_std)`` from the parent repo's vae config.

    The yaml interpolates ``${oc.env:PWD}``, which Hydra resolves against the CWD -- and Lyra runs
    from the LYRA root, so it must be pinned to the parent repo for the duration of the resolve or
    the path silently points into the wrong tree.

    Raises:
        FileNotFoundError: If the config is missing; failing loudly beats falling back to the stock
            VAE under a config named ``ftvae``.
    """
    if not _FT_VAE_YAML.is_file():
        raise FileNotFoundError(
            f"finetuned-VAE config not found at {_FT_VAE_YAML}; the ftvae experiments read their "
            "weights + whitening constants from the parent repo"
        )

    prev = os.environ.get("PWD")
    os.environ["PWD"] = str(_REPO)
    try:
        cfg = OmegaConf.load(_FT_VAE_YAML)
        OmegaConf.resolve(cfg)
        return str(cfg.vae_path), list(cfg.latent_mean), list(cfg.latent_std)
    finally:
        if prev is None:
            os.environ.pop("PWD", None)
        else:
            os.environ["PWD"] = prev


def _from_bidir_experiment(job_name: str, smoke: bool = False, ft_vae: bool = False) -> dict:
    """Build the from-bidir experiment. ``smoke`` shortens everything for a pipeline check.

    ``job_name`` is the OUTPUT-FOLDER leaf, not the Hydra selector: the run lands in
    ``<IMAGINAIRE_OUTPUT_ROOT>/<project>/<group>/<job_name>`` (imaginaire/config.py:189-195). The
    launcher exports an empty ``OUT_DIR`` so the root is plain ``outputs/`` and this job block owns
    the whole path -- i.e. ``outputs/lyra2_from_bidir/memorymaze/<job_name>``.
    """
    exp = dict(
        defaults=[
            {"override /model": "ddp_maze_lyra2_spatial"},
            {"override /net": "wan2pt1_280M_i2v_lyra2_bidir"},
            {"override /conditioner": "lyra2_conditioner"},
            # ckpt_type MUST be overridden: the base default is `dummy` (DummyCheckpointer), which
            # silently never writes a checkpoint.
            {"override /ckpt_type": "dcp"},
            {"override /data_train": "maze_streaming_train"},
            {"override /data_val": "maze_streaming_val"},
            "_self_",  # must stay LAST so these keys merge OVER the group nodes
        ],
        job=dict(project="lyra2_from_bidir", group="memorymaze", name=job_name),
        model=dict(
            config=dict(
                # Base default is `power` with enabled=True, which builds a SECOND full net.
                ema=dict(enabled=False),
                # Match the bidir init's wan_shift=1.0 pretraining; eval with sample_maze --shift 1.0.
                shift=1,
                # ---- framepack geometry: matched to the in-house AR model for comparison ----
                # anchor + 6 temporal context + 3 spatial slots + 3 generated latents, ALL at
                # kernel size 1 (uncompressed). That is the point: our model's context frames are
                # full-res 8x8 latents, so upstream's f4s2/f16k4 compression (5 spatial slots
                # costing 2 frames' worth of tokens) would not be a like-for-like comparison.
                #
                # The leading f1k1 is the ANCHOR slot. It is not structurally required -- the
                # anchor lands wherever seg_idx==0 and j==0 (lyra2_model.py _compose_selected_indices)
                # -- but writing it separately keeps the count honest: a bare "f6k1" would put the
                # anchor INSIDE those 6, giving only 5 recent context frames. f1k1f6k1 == f7k1.
                # Tokens at 64/latent frame: (1 anchor + 6 temporal + 3 spatial + 3 gen)*64 = 832.
                # g3 -> 3 new latents = 12 px frames per segment, so at num_frames=401 the data
                # allows (401-1)//12 = 33 segments; self_aug costs one (lyra2_model.py:177-178).
                framepack_type="f1k1f3s1f6k1_g3",
                state_t=3,
                max_segments=4 if smoke else 33,
                starting_frame_ratio=0.0,
                # ---- the bootstrap ----
                # Tiles clean_patch_embeddings from the loaded patch_embedding. Runs in
                # training_step at iteration==0 (lyra2_model.py:1221-1223), i.e. AFTER the
                # checkpoint load, so it copies the PRETRAINED embed, not the random init.
                init_framepack_weights=True,
                framepack_trainable_modules=_TRAINABLE,
                # ---- upstream lyra2: self-augmentation (exposure-bias reduction) ----
                # Forces batch_size=1: the Stage-A denoise loop is B=1-only
                # (sigmasA.squeeze().item() at lyra2_model.py:1329, latents_A[0] at :1343).
                self_aug_enabled=True,
                self_aug_steps=1,
                self_aug_guidance=1.0,  # ==1.0 skips the uncond pass
                self_aug_scheduler_shift=1.0,
                self_aug_every_k=2,
                self_aug_prob=1.0,
                self_aug_max_T=500,
                self_aug_copy_chunk=True,
                self_aug_encode_gt_with_clean_history=True,
                # ---- upstream lyra2: spatial-region corruption ----
                apply_corruption_to_spatial_region="noise_with_sigma",
                augment_sigma_sample_p_mean=-3.0,
                augment_sigma_sample_p_std=2.0,
                augment_sigma_sample_multiplier=1.0,
                # ---- spatial memory (maze tuning) ----
                spatial_memory_use_image=True,  # asserted True at lyra2_model.py:2064
                spatial_memory_stride=4,
                spatial_memory_skip_recent=16,
                spatial_memory_downsample=1,  # full-res depth point cloud (4 too coarse at 64x64)
                warp_chunk_size=16,
                # 0.05 (upstream) discards ~90%+ of 64x64 maze warp points; 0.45 keeps more.
                warp_continuity_ratio_thresh=0.45,
                fsdp_shard_size=1,  # DDP
                precision="bfloat16",
                # Carried from experiment.py for fidelity, but INERT here: use_mp_policy_fsdp only
                # builds mp_policy kwargs for fully_shard (lyra2_model.py:213-221), unreachable at
                # fsdp_shard_size=1; keep_original_net_dtype is declared at wan_t2v_model.py:117
                # and never read anywhere in the tree.
                use_mp_policy_fsdp=True,
                keep_original_net_dtype=True,
                # The released snapshot ships no empty_string_umt5.pt, so text dropout would crash;
                # the dataset injects a constant negative_prompt.pt embedding instead.
                conditioner=dict(text=dict(dropout_rate=0.0), wanclip=dict(disable_clip_forward=True)),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        # optimizer=dict(lr=3e-5),  # upstream's finetune LR, not 1e-4
        optimizer=dict(lr=1e-4),
        scheduler=dict(warm_up_steps=[10] if smoke else [2000]),
        checkpoint=dict(
            save_iter=20 if smoke else 100,
            # load_path (the converted .pth) is supplied by the launcher and is used only on the
            # first submit; requeues auto-resume from this run's own latest_checkpoint.txt, which
            # beats load_path unconditionally (dcp.py:704-712).
            load_path="",
            load_training_state=False,  # weights only: fresh optimizer/iteration
            strict_resume=False,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=50 if smoke else 50000,
            logging_iter=5 if smoke else 10,
            run_validation=True,
            validation_iter=25 if smoke else 2000,
            max_val_iter=1 if smoke else 2,
            # batch_size=1 is forced by self-aug, so effective batch comes from accumulation:
            # 1 x 16 ranks x 4 = 64. Measured 1.43s per micro-step on one 4090 at num_frames=401
            # with the f6k1f3s1_g3 geometry, so an optimizer iteration is ~5.7s and the 50k run
            # lands near 3.3 days (the anchor slot added since then makes it slightly longer).
            # This value is authoritative -- the launcher passes no trainer overrides.
            grad_accum_iter=1 if smoke else 4,
            callbacks=dict(iter_logger=L(MazeTrainingLoggerCallback)()),
            ddp=dict(find_unused_parameters=True, static_graph=False),
        ),
        dataloader_train=dict(
            batch_size=1,  # mandatory under self_aug_enabled
            dataset=dict(num_frames=101 if smoke else 401),
        ),
    )
    if ft_vae:
        # The tokenizer group is registered at package `model.config.tokenizer`
        # (configs/defaults/common/tokenizer.py), NOT at the config root -- an override placed at
        # the top level is silently dropped and the stock VAE loads instead.
        vae_pth, latent_mean, latent_std = _load_ft_vae()
        exp["model"]["config"]["tokenizer"] = dict(
            vae_pth=vae_pth, latent_mean=latent_mean, latent_std=latent_std
        )

    return exp


def register_from_bidir_net():
    cs.store(
        group="net",
        package="model.config.net",
        name="wan2pt1_280M_i2v_lyra2_bidir",
        node=WAN2PT1_280M_I2V_LYRA2_BIDIR,
    )


def register_lyra2_from_bidir():
    """270M finetune of the Stage-A bidirectional DiT, upstream lyra2 recipe.

    Hydra selector ``maze_small``; output folder leaf ``finetuned``.
    """
    cs.store(
        group="experiment",
        package="_global_",
        name="maze_small",
        node=_from_bidir_experiment("finetuned"),
    )


def register_lyra2_from_bidir_smoke():
    """Pipeline smoke: same net + recipe, 50 iters, short windows. Proves the .pth actually loads
    (Lyra's loader only logs on failure) and that the trainable whitelist really applies."""
    cs.store(
        group="experiment",
        package="_global_",
        name="maze_small_smoke",
        node=_from_bidir_experiment("finetuned_smoke", smoke=True),
    )


def register_lyra2_from_bidir_ftvae():
    """Same recipe on the FINETUNED Wan-VAE (weights + its measured whitening constants).

    Hydra selector ``maze_small_ftvae``; output folder leaf ``finetuned_ftvae``. Pair this with an
    init checkpoint exported from a DiT trained on finetuned-VAE latents -- a DiT fitted to the
    stock latent space would be reading a different distribution.
    """
    cs.store(
        group="experiment",
        package="_global_",
        name="maze_small_ftvae",
        node=_from_bidir_experiment("finetuned_ftvae", ft_vae=True),
    )


def register_lyra2_from_bidir_ftvae_smoke():
    """Pipeline smoke for the finetuned-VAE variant: 50 iters, short windows."""
    cs.store(
        group="experiment",
        package="_global_",
        name="maze_small_ftvae_smoke",
        node=_from_bidir_experiment("finetuned_ftvae_smoke", smoke=True, ft_vae=True),
    )


register_from_bidir_net()
register_lyra2_from_bidir()
register_lyra2_from_bidir_smoke()
register_lyra2_from_bidir_ftvae()
register_lyra2_from_bidir_ftvae_smoke()
