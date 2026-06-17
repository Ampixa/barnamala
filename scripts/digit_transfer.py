"""DHCD -> CMATERdb 3.2.1 Devanagari-digit distribution-shift transfer (experiment #2).

The compact student is trained only on DHCD (Acharya & Pant, UCI 389). CMATERdb
3.2.1 (Das et al. 2012, Jadavpur Univ.) is an independently-collected set of the
same 10 Devanagari numerals. We test how the DHCD digit head transfers to it:

  1. Provenance guard: perceptual-hash (DCT pHash, 64-bit) dedup of every CMATERdb
     image against all DHCD digit images. Few/no near-duplicates => the gap is a
     real distribution shift, not leakage.
  2. Zero-shot: feed CMATERdb test through the DHCD-trained student, read off the
     10 digit logits (DHCD classes 36..45). Reported for the distilled student
     (mean +/- std over 5 seeds) and the supervised control.
  3. Light adaptation: linear probe (frozen backbone, fresh 10-way head) and a
     short full fine-tune on CMATERdb train (2500 imgs), eval on test (500).

Polarity: CMATERdb is black-on-white, DHCD white-on-black; we auto-invert
CMATERdb to match DHCD (reported), the only domain-alignment step applied.

Usage: .venv/bin/python scripts/digit_transfer.py [--epochs-probe 40 --epochs-ft 30]
Produces Table 4 (transfer rows) of the paper.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dctn
from torch.utils.data import DataLoader, TensorDataset

from devnet.data import load_split
from devnet.evaluate import predict
from devnet.models.student import DevNet
from devnet.stats import wilson_ci
from devnet.train import resolve_device

DHCD_ROOT = "data/extracted/DevanagariHandwrittenCharacterDataset"
CMATER = "data/cmaterdb/devanagari-numerals"
DIGIT_OFFSET = 36  # DHCD class index of digit_0 (sorted folder order)
DISTILLED = [f"results/student_distilled_seed{s}/best.pth" for s in range(5)]
SUPERVISED = [f"results/student_supervised_seed{s}/best.pth" for s in range(5)]


# --------------------------------------------------------------------------- data
def to_gray(rgb_uint8):
    """RGB uint8 [N,32,32,3] -> luminosity grayscale uint8 [N,32,32]."""
    w = np.array([0.299, 0.587, 0.114])
    return (rgb_uint8.astype(np.float64) @ w).round().clip(0, 255).astype(np.uint8)


def load_cmaterdb(split):
    d = np.load(f"{CMATER}/{split}-images.npz")
    return to_gray(d["images"]), d["labels"].astype(np.int64)


def align_polarity(gray, ref_border_median):
    """Invert CMATERdb if its background polarity is opposite to DHCD's."""
    border = np.concatenate(
        [gray[:, 0, :], gray[:, -1, :], gray[:, :, 0], gray[:, :, -1]], axis=1
    )
    inverted = np.median(border) > 127 != (ref_border_median > 127)
    return (255 - gray if inverted else gray), bool(inverted)


# --------------------------------------------------------------- provenance guard
def phash(gray_batch):
    """64-bit DCT perceptual hash per image -> packed uint64 array [N]."""
    x = gray_batch.astype(np.float64)
    coeffs = dctn(x, axes=(1, 2), norm="ortho")[:, :8, :8].reshape(len(x), 64)
    med = np.median(coeffs[:, 1:], axis=1, keepdims=True)  # exclude DC term
    bits = (coeffs > med).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    return (bits * weights).sum(axis=1, dtype=np.uint64)


def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101) >> np.uint64(56)) & np.uint64(0xFF)


def min_hamming(query_hashes, ref_hashes):
    """For each query hash, min Hamming distance to any reference hash."""
    out = np.empty(len(query_hashes), dtype=np.int64)
    for i, q in enumerate(query_hashes):
        out[i] = int(_popcount64(q ^ ref_hashes).min())
    return out


# ------------------------------------------------------------------------- models
def load_student(path, device):
    c = torch.load(path, map_location=device, weights_only=False)
    m = DevNet(tuple(c["config"]["widths"]), tuple(c["config"]["depths"]),
               46, c["config"]["dropout"]).to(device)
    m.load_state_dict(c["model_state_dict"])
    return m, c["mean"], c["std"]


def normalize(gray, mean, std):
    x = torch.from_numpy(gray).float().unsqueeze(1) / 255.0
    return (x - mean) / std


@torch.no_grad()
def zero_shot_acc(model, x, labels, device):
    model.eval()
    logits = model(x.to(device))[:, DIGIT_OFFSET:DIGIT_OFFSET + 10]
    pred = logits.argmax(1).cpu().numpy()
    return float((pred == labels).mean())


def features(model, x, device, bs=512):
    """160-d penultimate features (post global-avg-pool), backbone frozen."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            b = x[i:i + bs].to(device)
            f = F.relu(model.bn(model.blocks(model.stem(b)))).mean(dim=(2, 3))
            outs.append(f.cpu())
    return torch.cat(outs)


def train_head(feat_tr, y_tr, feat_te, y_te, in_dim, device, epochs, lr=1e-2):
    """Linear probe: logistic regression on frozen features."""
    head = nn.Linear(in_dim, 10).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ft, yt = feat_tr.to(device), torch.from_numpy(y_tr).to(device)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(ft), yt)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(feat_te.to(device)).argmax(1).cpu().numpy()
    return float((pred == y_te).mean())


def full_finetune(model, x_tr, y_tr, x_te, y_te, device, epochs, lr=3e-4, bs=128):
    """Short full fine-tune with a fresh 10-way head."""
    model.fc = nn.Linear(model.fc.in_features, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    ds = TensorDataset(x_tr, torch.from_numpy(y_tr))
    loader = DataLoader(ds, bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(x_te), 512):
            pred.append(model(x_te[i:i + 512].to(device)).argmax(1).cpu())
        pred = torch.cat(pred).numpy()
    return float((pred == y_te).mean())


def acc_with_ci(acc, n):
    lo, hi = wilson_ci(int(round(acc * n)), n)
    return {"acc": round(acc, 4), "n": n,
            "wilson_ci": [round(lo, 4), round(hi, 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-probe", type=int, default=40)
    ap.add_argument("--epochs-ft", type=int, default=30)
    ap.add_argument("--out", default="results/digit_transfer.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = resolve_device("auto")

    # DHCD digit reference (train+test) for polarity + dedup
    dhcd_imgs, dhcd_lbl, _ = load_split(DHCD_ROOT, "Test")
    dhcd_tr_imgs, dhcd_tr_lbl, _ = load_split(DHCD_ROOT, "Train")
    dhcd_digits = np.concatenate([dhcd_imgs[dhcd_lbl >= DIGIT_OFFSET],
                                  dhcd_tr_imgs[dhcd_tr_lbl >= DIGIT_OFFSET]])
    db = np.concatenate([dhcd_digits[:, 0, :], dhcd_digits[:, -1, :],
                         dhcd_digits[:, :, 0], dhcd_digits[:, :, -1]], axis=1)
    dhcd_border_median = float(np.median(db))

    g_tr, lbl_tr = load_cmaterdb("training")
    g_te, lbl_te = load_cmaterdb("testing")
    g_tr, inv_tr = align_polarity(g_tr, dhcd_border_median)
    g_te, inv_te = align_polarity(g_te, dhcd_border_median)

    # 1. provenance guard
    ref_hashes = phash(dhcd_digits)
    cmater_hashes = phash(np.concatenate([g_tr, g_te]))
    dists = min_hamming(cmater_hashes, ref_hashes)
    dedup = {
        "dhcd_digit_refs": int(len(dhcd_digits)),
        "cmaterdb_images": int(len(cmater_hashes)),
        "min_hamming_min": int(dists.min()),
        "min_hamming_mean": round(float(dists.mean()), 2),
        "near_dupes_le5bits": int((dists <= 5).sum()),
        "near_dupes_le10bits": int((dists <= 10).sum()),
    }

    # 2. zero-shot
    def zs_over(paths):
        accs = []
        for p in paths:
            m, mean, std = load_student(p, device)
            x = normalize(g_te, mean, std)
            accs.append(zero_shot_acc(m, x, lbl_te, device))
        return accs

    zs_distilled = zs_over(DISTILLED)
    zs_supervised = zs_over(SUPERVISED)

    # 3. adaptation on the distilled seed0 student
    m, mean, std = load_student(DISTILLED[0], device)
    x_tr, x_te = normalize(g_tr, mean, std), normalize(g_te, mean, std)
    feat_tr = features(m, x_tr, device)
    feat_te = features(m, x_te, device)
    probe_acc = train_head(feat_tr, lbl_tr, feat_te, lbl_te,
                           m.fc.in_features, device, args.epochs_probe)
    m_ft, _, _ = load_student(DISTILLED[0], device)
    ft_acc = full_finetune(m_ft, x_tr, lbl_tr, x_te, lbl_te,
                           device, args.epochs_ft)

    n = len(lbl_te)
    result = {
        "dataset": "CMATERdb 3.2.1 Devanagari numerals (Das et al. 2012)",
        "n_train": int(len(lbl_tr)), "n_test": n, "classes": 10,
        "polarity_inverted_to_match_dhcd": {"train": inv_tr, "test": inv_te},
        "provenance_guard_phash": dedup,
        "zero_shot": {
            "distilled_seeds": [round(a, 4) for a in zs_distilled],
            "distilled_mean": round(float(np.mean(zs_distilled)), 4),
            "distilled_std": round(float(np.std(zs_distilled)), 4),
            "distilled_seed0_ci": acc_with_ci(zs_distilled[0], n),
            "supervised_seeds": [round(a, 4) for a in zs_supervised],
            "supervised_mean": round(float(np.mean(zs_supervised)), 4),
        },
        "adaptation_distilled_seed0": {
            "linear_probe": acc_with_ci(probe_acc, n),
            "full_finetune": acc_with_ci(ft_acc, n),
            "epochs_probe": args.epochs_probe, "epochs_ft": args.epochs_ft,
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
