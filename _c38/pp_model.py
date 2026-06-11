"""Task 2.2 option (c): a conditional neural temporal point-process model for
order inter-arrival times, used to REPLACE the diffusion's time channel at AR
rollout (the diffusion still generates etype/size/price/dir). Tests whether
modeling event timing as a proper point process fixes lob_bench inter_arrival
(options a/b -- MSE time-upweight and log-time reparam -- both failed at ~0.9).

Model: GRU over the recent (log1p dt, event_type) history -> parameters of a
log-normal MIXTURE for the next dt (Shchur et al. intensity-free TPP style).
Trained by mixture NLL on the real dt+etype stream (from RAW adapter npy:
col0 = raw inter-arrival, col1 = event_type in {0,1,2}).

Usage:
  train:  python pp_model.py BTCUSDT ETHUSDT     -> data/checkpoints/pp/<asset>_pp.pt
  infer:  from pp_model import PointProcess; pp = PointProcess.load(path); pp.sample(hist)
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn

ROOT = "/DATA/DATANAS1/wpj24/AIQuant/DeepMarket"
RAW = f"{ROOT}/data/_adapter_raw"
CKDIR = f"{ROOT}/data/checkpoints/pp"
SPLIT = 0.85
K = 32          # history length
NMIX = 3
HID = 64
EMB = 8


class PPNet(nn.Module):
    def __init__(self, nmix=NMIX, hid=HID, emb=EMB):
        super().__init__()
        self.nmix = nmix
        self.etype_emb = nn.Embedding(3, emb)
        self.gru = nn.GRU(1 + emb, hid, batch_first=True)
        self.head = nn.Linear(hid, nmix * 3)

    def forward(self, logdt, etype):
        # logdt (B,K,1) float, etype (B,K) long
        x = torch.cat([logdt, self.etype_emb(etype)], dim=-1)
        _, h = self.gru(x)                  # h (1,B,hid)
        o = self.head(h[0]).view(-1, self.nmix, 3)
        logits = o[..., 0]
        mu = o[..., 1]
        log_sigma = o[..., 2].clamp(-5.0, 3.0)
        return logits, mu, log_sigma


def _mixture_nll(logits, mu, log_sigma, target_logdt):
    # target_logdt (B,) ; returns mean NLL of log-normal mixture on dt
    lw = torch.log_softmax(logits, dim=-1)                       # (B,M)
    sigma = log_sigma.exp()
    t = target_logdt.unsqueeze(-1)                               # (B,1)
    log_n = -0.5 * ((t - mu) / sigma) ** 2 - log_sigma - 0.5 * np.log(2 * np.pi)
    log_p_logdt = torch.logsumexp(lw + log_n, dim=-1)           # density in log-dt space
    # change of variables to dt: p(dt)=p(logdt)/dt -> add -target_logdt; constant in params
    return -(log_p_logdt - target_logdt).mean()


def _load_stream(asset):
    arr = np.load(f"{RAW}/{asset}.npy").astype(np.float64)
    dt = np.clip(arr[:, 0], 1e-6, None)
    et = np.clip(np.round(arr[:, 1]), 0, 2).astype(np.int64)
    n = int(len(arr) * SPLIT)
    return np.log1p(dt[:n]), et[:n], np.log1p(dt[n:]), et[n:]


def _windows(logdt, et, device, bs=4096):
    n = len(logdt) - K
    idx = np.arange(n)
    for b0 in range(0, n, bs):
        bi = idx[b0:b0 + bs]
        H_ld = np.stack([logdt[i:i + K] for i in bi])[:, :, None].astype(np.float32)
        H_et = np.stack([et[i:i + K] for i in bi]).astype(np.int64)
        tgt = logdt[bi + K].astype(np.float32)
        yield (torch.from_numpy(H_ld).to(device), torch.from_numpy(H_et).to(device),
               torch.from_numpy(tgt).to(device))


def train_asset(asset, epochs=4, device="cuda"):
    ld_tr, et_tr, ld_va, et_va = _load_stream(asset)
    net = PPNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(ld_tr) - K)
        # shuffle by reindexing the stream-window starts
        losses = []
        for b0 in range(0, len(perm), 4096):
            bi = perm[b0:b0 + 4096]
            H_ld = np.stack([ld_tr[i:i + K] for i in bi])[:, :, None].astype(np.float32)
            H_et = np.stack([et_tr[i:i + K] for i in bi]).astype(np.int64)
            tgt = ld_tr[bi + K].astype(np.float32)
            H_ld = torch.from_numpy(H_ld).to(device); H_et = torch.from_numpy(H_et).to(device)
            tgt = torch.from_numpy(tgt).to(device)
            opt.zero_grad()
            loss = _mixture_nll(*net(H_ld, H_et), tgt)
            loss.backward(); opt.step()
            losses.append(loss.item())
        # val
        net.eval()
        with torch.no_grad():
            vl = [ _mixture_nll(*net(a, b), c).item() for a, b, c in _windows(ld_va, et_va, device) ]
        print(f"[{asset}] ep{ep} train_nll={np.mean(losses):.4f} val_nll={np.mean(vl):.4f}")
    os.makedirs(CKDIR, exist_ok=True)
    path = f"{CKDIR}/{asset}_pp.pt"
    torch.save({"state_dict": net.state_dict(), "K": K, "nmix": NMIX, "hid": HID, "emb": EMB}, path)
    print(f"PP_SAVED {path}")
    return path


class PointProcess:
    """Inference wrapper for AR rollout. Maintains per-stream history of
    (log1p dt, etype) and samples the next dt."""
    def __init__(self, net, K, device):
        self.net = net.eval(); self.K = K; self.device = device

    @classmethod
    def load(cls, path, device="cuda"):
        ck = torch.load(path, map_location=device)
        net = PPNet(nmix=ck["nmix"], hid=ck["hid"], emb=ck["emb"]).to(device)
        net.load_state_dict(ck["state_dict"])
        return cls(net, ck["K"], device)

    @torch.no_grad()
    def sample(self, hist_logdt, hist_et):
        """hist_logdt: list/array of last >=K log1p(dt); hist_et: last >=K etype.
        Returns a sampled raw dt (float)."""
        ld = np.asarray(hist_logdt[-self.K:], np.float32)[None, :, None]
        et = np.asarray(hist_et[-self.K:], np.int64)[None, :]
        ld = torch.from_numpy(ld).to(self.device); et = torch.from_numpy(et).to(self.device)
        logits, mu, log_sigma = self.net(ld, et)
        w = torch.softmax(logits, dim=-1)[0]
        m = int(torch.multinomial(w, 1).item())
        g = mu[0, m].item() + log_sigma[0, m].exp().item() * float(np.random.randn())
        return float(np.expm1(min(g, 50.0)))


if __name__ == "__main__":
    assets = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for a in assets:
        train_asset(a, device=dev)
    print("PP_TRAIN_DONE")
