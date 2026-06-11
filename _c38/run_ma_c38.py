"""Launch MA_TRADES multi-asset training on cluster38 without mutating the
WIP configuration.py. Controlled by env vars:
  SMOKE=1   -> tiny run (1 epoch, few batches) just to prove it trains
  NDEV=1|2  -> number of GPUs (Lightning devices)
  EPOCHS=N  -> override epochs for a real run (SMOKE=0)
"""
from __future__ import annotations
import os
from types import SimpleNamespace
import lightning as L

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import train, HP_DICT_MODEL


def main():
    smoke = os.environ.get("SMOKE", "1") == "1"
    ndev = int(os.environ.get("NDEV", "1"))

    # Optional fixed seed so graph-ON vs no-graph A/B share identical trunk
    # init + data order (the only difference is the disable_graph fuse path).
    seed = os.environ.get("SEED")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)
        print(f"[seed] seed_everything({int(seed)})")

    model_name = os.environ.get("MODEL", "MA_TRADES")
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models[model_name]
    cfg.MULTI_ASSET = (cfg.CHOSEN_MODEL == cst.Models.MA_TRADES)
    cfg.IS_DATA_PREPROCESSED = True
    cfg.IS_WANDB = False
    cfg.IS_TRAINING = True
    cfg.IS_AUGMENTATION = True
    # assets: MA_TRADES needs >=2; single-asset TRADES/CGAN uses the first.
    assets = [a.strip() for a in os.environ.get("ASSETS", "INTC,SYNTH").split(",") if a.strip()]
    cfg.CHOSEN_STOCK = [SimpleNamespace(name=a) for a in assets]
    cfg.ASSET_UNIVERSE = None  # built from CHOSEN_STOCK inside train()
    cfg.DISABLE_GRAPH = os.environ.get("DISABLE_GRAPH", "0") == "1"
    cfg.DISABLE_ARB_GUIDANCE = os.environ.get("DISABLE_ARB_GUIDANCE", "0") == "1"
    cfg.FREEZE_EDGE_WEIGHTS = os.environ.get("FREEZE_EDGE_WEIGHTS", "0") == "1"
    # Tiny LR for the graph-coupling branch (gate gamma + edge/message/fusion
    # MLPs) so the residual gate opens slowly instead of running away from 0.
    cfg.GRAPH_LR = float(os.environ.get("GRAPH_LR", "1e-5"))

    # apply the fixed hyperparameters exactly like run() does
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    suffix = os.environ.get("CKPT_SUFFIX", "graph" if not cfg.DISABLE_GRAPH else "nograph")
    cfg.FILENAME_CKPT = f"MA_{'_'.join(assets)}_c38_{suffix}"

    epochs = int(os.environ.get("EPOCHS", "1" if smoke else "10"))
    cfg.HYPER_PARAMETERS[LHP.EPOCHS] = epochs

    # Optional LR override (training is seed-unstable at the default 1e-3;
    # testing whether a lower base LR stabilizes the stuck seeds).
    if os.environ.get("LR"):
        cfg.HYPER_PARAMETERS[LHP.LEARNING_RATE] = float(os.environ["LR"])
        print(f"[lr] LEARNING_RATE={cfg.HYPER_PARAMETERS[LHP.LEARNING_RATE]}")

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=ndev,
        precision=cst.PRECISION,
        max_epochs=epochs,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        check_val_every_n_epoch=1,
        gradient_clip_val=float(os.environ.get("GRAD_CLIP", "1.0")),
    )
    if smoke:
        trainer_kwargs.update(limit_train_batches=30, limit_val_batches=5, val_check_interval=1.0)
    else:
        trainer_kwargs.update(val_check_interval=0.5)
    if ndev > 1:
        # Graph-coupling params (gamma, edge weights) are not used on every
        # step, so DDP must allow unused parameters.
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"

    print(f"[launch] SMOKE={smoke} NDEV={ndev} EPOCHS={epochs} DEVICE={cst.DEVICE}")
    trainer = L.Trainer(**trainer_kwargs)
    train(cfg, trainer)
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
