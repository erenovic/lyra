"""Rich per-iteration training logger for maze runs.

The stock ``IterationLoggerCallback`` only logs iter-time + total loss. This superset also logs
the learning rate and the (read-only, pre-clip) global gradient norm -- the minimum needed to
judge training health at a glance. Logging only: it never modifies gradients or the optimizer.

Wire it in place of ``iter_logger`` in the experiment's ``trainer.callbacks``.
"""

import time

import torch

from lyra_2._ext.imaginaire.utils import distributed, log
from lyra_2._ext.imaginaire.utils.callback import Callback


class MazeTrainingLoggerCallback(Callback):
    """Log ``iter | loss | lr | grad_norm | s/it`` every ``logging_iter`` (rank 0 only)."""

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0) -> None:
        self._start = time.time()
        self._elapsed = 0.0
        self._lr = 0.0
        self._grad_norm = 0.0

    @distributed.rank0_only
    def on_training_step_start(self, model, data, iteration: int = 0) -> None:
        self._start = time.time()

    @distributed.rank0_only
    def on_before_optimizer_step(self, model_ddp, optimizer, scheduler, grad_scaler, iteration: int = 0) -> None:
        # Captured here because the optimizer/scheduler aren't exposed to on_training_step_end.
        # Grads are already all-reduced (post-backward); read-only, no clipping. bf16 -> scale~1.
        self._lr = float(optimizer.param_groups[0]["lr"])
        grads = [p.grad for p in model_ddp.parameters() if p.grad is not None]
        if grads:
            self._grad_norm = float(torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2))

    @distributed.rank0_only
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0) -> None:
        self._elapsed += time.time() - self._start
        if iteration % self.config.trainer.logging_iter == 0:
            avg = self._elapsed / self.config.trainer.logging_iter
            log.info(
                f"iter {iteration} | loss {loss.item():.4f} | lr {self._lr:.3e} | "
                f"grad_norm {self._grad_norm:.3f} | {avg:.3f}s/it"
            )
            self._elapsed = 0.0
