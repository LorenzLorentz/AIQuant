"""Task bucket 3.2 (training stability) — env-gated, reversible patches.

Applies two changes to the deployed cluster38 WIP multi-asset stack:

  (a) MSE_REDUCE  (ma_gaussian_diffusion._mse_loss_per_asset)
      default 'norm' = current torch.norm(p=2) (baseline UNCHANGED);
      'mean' = mean-of-squares per asset -> loss scale ~sqrt(K*F) -> O(1),
      cooling the hot early gradient suspected of seed-sensitivity.

  (b) LR_SCHED   (ma_diffusion_engine.configure_optimizers / on_validation_epoch_end)
      default 'plateau' = current manual halve-on-plateau (baseline UNCHANGED);
      'cosine' = linear warmup (WARMUP_FRAC, default 0.05 of total steps) then
      cosine decay to LR_MIN_FRAC (default 0.0); manual halving is skipped so
      the scheduler owns the LR. Goal: allow LR>=1e-3 without seed divergence.

Both default to the existing behaviour, so every prior ckpt/experiment is
reproducible. Backups written to <file>.bak_stab. Idempotent (re-run safe).
"""
from __future__ import annotations
import os
import shutil
import sys

ROOT = os.environ.get("AIQUANT_ROOT", "/DATA/DATANAS1/wpj24/AIQuant")
GAUSS = os.path.join(ROOT, "DeepMarket/models/diffusers/multi_asset/ma_gaussian_diffusion.py")
ENGINE = os.path.join(ROOT, "DeepMarket/models/diffusers/multi_asset/ma_diffusion_engine.py")


def backup(path):
    bak = path + ".bak_stab"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"[backup] {bak}")
    else:
        print(f"[backup] exists, kept: {bak}")


# ---- (a) MSE_REDUCE in _mse_loss_per_asset ----------------------------------
GAUSS_OLD = """        if tw != 1.0:
            w = torch.ones(diff.shape[-1], device=diff.device, dtype=diff.dtype)
            w[0] = tw ** 0.5  # squared inside L2 norm -> effective weight tw
            diff = diff * w
        return torch.norm(diff, p=2, dim=[2, 3])"""

GAUSS_NEW = """        if tw != 1.0:
            w = torch.ones(diff.shape[-1], device=diff.device, dtype=diff.dtype)
            w[0] = tw ** 0.5  # squared inside L2 norm -> effective weight tw
            diff = diff * w
        # MSE_REDUCE='mean' (task 3.2): mean-of-squares per asset instead of the
        # L2 norm. Norm scale ~sqrt(K*F) makes the early epsilon gradient hot ->
        # seed-sensitivity; mean keeps it O(1). Default 'norm' = baseline.
        if os.environ.get("MSE_REDUCE", "norm") == "mean":
            return (diff ** 2).mean(dim=[2, 3])
        return torch.norm(diff, p=2, dim=[2, 3])"""


# ---- (b) LR_SCHED: configure_optimizers returns warmup+cosine ---------------
ENGINE_OPT_OLD = """            )
        return self.optimizer

    def _define_log_metrics(self):"""

ENGINE_OPT_NEW = """            )
        # LR_SCHED='cosine' (task 3.2): linear warmup then cosine decay, stepped
        # per optimizer step. Lets LR>=1e-3 train without the seed-divergence
        # seen under the flat-LR + manual-halve recipe. Default 'plateau' keeps
        # the bare optimizer + on_validation_epoch_end halving (baseline).
        if os.environ.get("LR_SCHED", "plateau") == "cosine":
            total_steps = max(1, int(self.trainer.estimated_stepping_batches))
            warmup_frac = float(os.environ.get("WARMUP_FRAC", "0.05"))
            min_frac = float(os.environ.get("LR_MIN_FRAC", "0.0"))
            warmup_steps = max(1, int(warmup_frac * total_steps))

            def _lr_lambda(step, _t=total_steps, _w=warmup_steps, _m=min_frac):
                if step < _w:
                    return float(step) / float(_w)
                prog = float(step - _w) / float(max(1, _t - _w))
                prog = min(1.0, prog)
                cos = 0.5 * (1.0 + np.cos(np.pi * prog))
                return _m + (1.0 - _m) * cos

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
            print(f"[sched] cosine warmup={warmup_steps}/{total_steps} steps "
                  f"min_frac={min_frac}")
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return self.optimizer

    def _define_log_metrics(self):"""

ENGINE_VAL_OLD = """        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        if loss_ema < self.min_loss_ema:
            if loss_ema - self.min_loss_ema > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2
            self.min_loss_ema = loss_ema
            self.model_checkpointing(loss_ema)
        else:
            self.optimizer.param_groups[0]["lr"] /= 2"""

ENGINE_VAL_NEW = """        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)
        # Under LR_SCHED='cosine' the scheduler owns the LR; skip the manual
        # halve so the two don't fight (task 3.2). Checkpointing stays.
        sched_cosine = os.environ.get("LR_SCHED", "plateau") == "cosine"
        if loss_ema < self.min_loss_ema:
            if (not sched_cosine) and loss_ema - self.min_loss_ema > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2
            self.min_loss_ema = loss_ema
            self.model_checkpointing(loss_ema)
        elif not sched_cosine:
            self.optimizer.param_groups[0]["lr"] /= 2"""


def main():
    for p in (GAUSS, ENGINE):
        if not os.path.exists(p):
            print(f"[ERROR] missing {p}")
            sys.exit(1)
        backup(p)

    # (a)
    with open(GAUSS) as f:
        g = f.read()
    if "MSE_REDUCE" in g:
        print("[skip] MSE_REDUCE already present")
    else:
        assert g.count(GAUSS_OLD) == 1, g.count(GAUSS_OLD)
        g = g.replace(GAUSS_OLD, GAUSS_NEW, 1)
        with open(GAUSS, "w") as f:
            f.write(g)
        print("[patched] MSE_REDUCE")

    # (b)
    with open(ENGINE) as f:
        e = f.read()
    if 'os.environ.get("LR_SCHED"' in e:
        print("[skip] LR_SCHED already present")
    else:
        assert e.count(ENGINE_OPT_OLD) == 1, ("opt anchor", e.count(ENGINE_OPT_OLD))
        assert e.count(ENGINE_VAL_OLD) == 1, ("val anchor", e.count(ENGINE_VAL_OLD))
        e = e.replace(ENGINE_OPT_OLD, ENGINE_OPT_NEW, 1)
        e = e.replace(ENGINE_VAL_OLD, ENGINE_VAL_NEW, 1)
        with open(ENGINE, "w") as f:
            f.write(e)
        print("[patched] LR_SCHED")

    # syntax check
    import py_compile
    for p in (GAUSS, ENGINE):
        py_compile.compile(p, doraise=True)
        print(f"[ok] compiles: {os.path.basename(p)}")
    print("PATCH_DONE")


if __name__ == "__main__":
    main()
