"""Gradient probe: does the graph backward inject a different/large gradient
into the SHARED TRUNK, and does it inflate the global grad-norm so clipping
scales the trunk gradient down?

On the SAME seeded training batch, with graph-ON vs graph-OFF (disable_graph),
run forward(is_train=True)+loss+backward and measure:
  * ||trunk_grad|| and cosine(trunk_grad_ON, trunk_grad_OFF)   -> hyp (a)
  * ||graph_branch_grad||                                       -> hyp (b)
  * global grad-norm and clip factor min(1, 1.0/global_norm)    -> hyp (b)

Two states: a FRESH seeded init (gamma=0 exactly -> ON/OFF must match: sanity)
and the DIVERGED checkpoint (gamma~9e-4 -> shows the live difference).
"""
from __future__ import annotations
import os
import numpy as np
import torch

import constants as cst
from constants import LearningHyperParameter as LHP
from configuration import Configuration
from run import HP_DICT_MODEL
from models.diffusers.multi_asset.ma_diffusion_engine import MultiAssetDiffusionEngine
from preprocessing.AssetUniverse import AssetUniverse

CKPT = os.environ.get("CKPT", "")
DATAZ = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket/data"
ASSETS = ["BTCUSDT", "ETHUSDT"]
B = 64
SEED = 0
CLIP = 1.0


def build_engine(disable_graph, ckpt):
    cfg = Configuration()
    cfg.CHOSEN_MODEL = cst.Models.MA_TRADES
    cfg.MULTI_ASSET = True
    cfg.IS_WANDB = False
    cfg.IS_TRAINING = True
    cfg.IS_AUGMENTATION = True
    cfg.DISABLE_GRAPH = disable_graph
    cfg.DISABLE_ARB_GUIDANCE = True
    cfg.FILENAME_CKPT = "diag"
    fixed = HP_DICT_MODEL[cfg.CHOSEN_MODEL].fixed
    for p in cst.LearningHyperParameter:
        if p.value in fixed:
            cfg.HYPER_PARAMETERS[p] = fixed[p.value]
    torch.manual_seed(SEED); np.random.seed(SEED)
    universe = AssetUniverse(assets=ASSETS, relation_types={(0, 1): 0, (1, 0): 1})
    eng = MultiAssetDiffusionEngine(cfg, universe).to(cst.DEVICE)
    if ckpt:
        sd = torch.load(ckpt, map_location=cst.DEVICE)
        eng.load_state_dict(sd.get("state_dict", sd), strict=False)
    eng.train()
    return eng, cfg


def make_batch(cfg):
    seq = cfg.HYPER_PARAMETERS[LHP.SEQ_SIZE]
    kgen = cfg.HYPER_PARAMETERS[LHP.MASKED_SEQ_SIZE]
    kcond = seq - kgen
    z = {a: np.load(f"{DATAZ}/{a}/train.npy").astype(np.float32) for a in ASSETS}
    L = min(z[a].shape[0] for a in ASSETS)
    rng = np.random.default_rng(SEED)
    idxs = rng.integers(kcond, L - kgen - 1, size=B)
    lob = {}
    for a in ASSETS:
        lo = z[a][:, cst.LEN_ORDER:].copy(); lo = np.roll(lo, 1, axis=0); lo[0] = 0
        lob[a] = lo
    cond_o, x0, cond_l = [], [], []
    for i in idxs:
        cond_o.append(np.stack([z[a][i - kcond:i, :cst.LEN_ORDER] for a in ASSETS]))
        x0.append(np.stack([z[a][i:i + kgen, :cst.LEN_ORDER] for a in ASSETS]))
        cond_l.append(np.stack([lob[a][i - kcond:i + kgen] for a in ASSETS]))
    to = lambda arr: torch.from_numpy(np.stack(arr)).float().to(cst.DEVICE)
    return to(cond_o), to(x0), to(cond_l)


def trunk_and_graph_params(eng):
    trunk = list(eng.diffuser.NN.parameters())
    gmods = [getattr(eng.diffuser, a) for a in
             ("relation_embedding", "edge_weight_net", "message_fn", "aggregator", "noise_fusion")]
    graph = [p for m in gmods for p in m.parameters()]
    return trunk, graph


def grads(eng, cfg, batch):
    cond_o, x0, cond_l = batch
    eng.zero_grad(set_to_none=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    eng.diffuser.init_losses()
    cl = cond_l.clone() if cfg.COND_TYPE == "full" else None
    _ = eng.forward(cond_o.clone(), x0.clone(), cl, is_train=True, batch_idx=0)
    Lh, Ls, Lv = eng.loss()
    loss = Lh.mean()
    loss.backward()
    trunk, graph = trunk_and_graph_params(eng)
    tg = torch.cat([p.grad.flatten() for p in trunk if p.grad is not None])
    gg = torch.cat([p.grad.flatten() for p in graph if p.grad is not None]) \
        if any(p.grad is not None for p in graph) else torch.zeros(1, device=tg.device)
    glob = torch.cat([p.grad.flatten() for p in eng.parameters() if p.grad is not None])
    return dict(loss=float(loss), trunk_grad=tg, trunk_norm=float(tg.norm()),
                graph_norm=float(gg.norm()), global_norm=float(glob.norm()),
                gamma=float(eng.diffuser.graph_gamma.item()))


def probe(tag, ckpt, batch):
    on, cfg = build_engine(False, ckpt)
    r_on = grads(on, cfg, batch)
    off, cfg2 = build_engine(True, ckpt)
    r_off = grads(off, cfg2, batch)
    a, b = r_on["trunk_grad"], r_off["trunk_grad"]
    cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    ratio = r_on["trunk_norm"] / (r_off["trunk_norm"] + 1e-12)
    clip_on = min(1.0, CLIP / (r_on["global_norm"] + 1e-12))
    clip_off = min(1.0, CLIP / (r_off["global_norm"] + 1e-12))
    print(f"\n===== {tag} (gamma_on={r_on['gamma']:.6f}) =====")
    print(f"  loss            ON={r_on['loss']:.4f}   OFF={r_off['loss']:.4f}")
    print(f"  trunk_grad_norm ON={r_on['trunk_norm']:.4f}   OFF={r_off['trunk_norm']:.4f}   ratio_ON/OFF={ratio:.4f}")
    print(f"  cosine(trunk_grad_ON, trunk_grad_OFF) = {cos:.6f}")
    print(f"  graph_branch_grad_norm ON={r_on['graph_norm']:.4f}")
    print(f"  global_grad_norm ON={r_on['global_norm']:.3f} OFF={r_off['global_norm']:.3f}  "
          f"-> clip_factor(@{CLIP}) ON={clip_on:.4f} OFF={clip_off:.4f}")
    if cos < 0.99:
        print(f"  >> trunk gradient DIRECTION differs (cos {cos:.3f}) -> graph backward perturbs trunk grad (hyp a)")
    if clip_on < 0.9 * clip_off:
        print(f"  >> grad-clip scales trunk DOWN more in ON ({clip_on:.3f} vs {clip_off:.3f}) -> clipping mechanism (hyp b)")


def main():
    base = build_engine(False, "")[1]
    batch = make_batch(base)
    print(f"[batch] B={B}  cond_o={tuple(batch[0].shape)} x0={tuple(batch[1].shape)} cond_l={tuple(batch[2].shape)}")
    probe("FRESH INIT (gamma=0, sanity: ON==OFF expected)", "", batch)
    if CKPT:
        probe("DIVERGED CKPT", CKPT, batch)


if __name__ == "__main__":
    main()
