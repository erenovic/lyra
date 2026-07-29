"""Dump the trained Lyra 2.0 net's parameters (name, shape, count) to a text file.

Loads the maze checkpoint through the experiment config and enumerates
``model.net.named_parameters()`` -- the trained DiT (the frozen Wan VAE / T5 encoders
are NOT part of the trained model and are excluded). Writes a per-parameter table plus
totals to ``--out`` (default ``params.txt``).

Usage (CWD = lyra repo root; GPU node):
  python -m lyra_2.count_params \
      --checkpoint outputs/lyra2_v2/lyra_2/maze/lyra2_maze_selfaug/checkpoints/iter_000041500
"""

import argparse

from lyra_2._src.utils.model_loader import load_model_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump Lyra2 trained-net parameters to a text file.")
    ap.add_argument("--experiment", default="lyra2_maze_selfaug")
    ap.add_argument(
        "--checkpoint",
        default="outputs/lyra2_v2/lyra_2/maze/lyra2_maze_selfaug/checkpoints/iter_000041500",
    )
    ap.add_argument("--out", default="params.txt")
    args = ap.parse_args()

    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint,
        config_file="lyra_2/_src/configs/config.py",
        instantiate_ema=False,
        strict=True,
        experiment_opts=["model.config.self_aug_enabled=False"],
    )
    net = model.net

    rows = []
    total = trainable = 0
    for name, p in net.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        rows.append((name, tuple(p.shape), n, "train" if p.requires_grad else "frozen"))

    name_w = max((len(r[0]) for r in rows), default=4)
    with open(args.out, "w") as f:
        f.write(f"# Lyra2.0 trained net: {type(net).__name__}\n")
        f.write(f"# checkpoint: {args.checkpoint}\n")
        f.write(f"# total parameters:     {total:,}\n")
        f.write(f"# trainable parameters: {trainable:,}\n")
        f.write(f"# frozen parameters:    {total - trainable:,}\n")
        f.write(f"# num tensors:          {len(rows)}\n")
        f.write("#\n")
        f.write(f"# {'name'.ljust(name_w)}\tshape\tnumel\tstate\n")
        for name, shape, n, state in rows:
            f.write(f"{name.ljust(name_w)}\t{shape}\t{n}\t{state}\n")

    print(f"[count_params] {type(net).__name__}: total={total:,} trainable={trainable:,} -> {args.out}")


if __name__ == "__main__":
    main()
