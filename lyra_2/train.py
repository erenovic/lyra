"""Training entry point for Lyra 2.0 (the released snapshot ships none).

Wires ``make_config`` -> experiment override -> ``ImaginaireTrainer.train`` following the
known-good pattern of ``_src/utils/model_loader.py``. The trainer is constructed BEFORE
the model/dataloaders: its ``__init__`` runs ``distributed.init()`` + megatron
``parallel_state.initialize_model_parallel`` (single-GPU-safe), which the model relies on.

Usage (CWD must be the lyra repo root so ./checkpoints/... resolves):
  torchrun --standalone --nproc_per_node=1 -m lyra_2.train -- -- experiment=lyra2_maze_smoke
Extra hydra-style overrides append after the experiment, e.g. trainer.max_iter=100.
"""

import argparse
import importlib
import os

from lyra_2._ext.imaginaire.lazy_config import instantiate
from lyra_2._ext.imaginaire.utils import log
from lyra_2._ext.imaginaire.utils.config_helper import get_config_module, override


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Lyra 2.0.")
    ap.add_argument("--config", default="lyra_2/_src/configs/config.py", help="config module path")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help='overrides: -- "experiment=<name>" [k=v ...]')
    args = ap.parse_args()

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    config.validate()
    config.freeze()

    # Where MazeLyra2Model.validation_step writes its GT-vs-generated sample grids.
    os.environ.setdefault("LYRA_VAL_SAMPLE_DIR", os.path.join(config.job.path_local, "val_samples"))

    # Trainer first: inits torch.distributed + megatron parallel_state (model needs both).
    trainer = config.trainer.type(config)

    log.info("Instantiating model...")
    model = instantiate(config.model)
    log.info("Instantiating dataloaders...")
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)

    trainer.train(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()
