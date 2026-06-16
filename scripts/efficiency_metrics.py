"""Efficiency metrics for the compact student vs the MallaNet baseline.

Reports, for each model, parameters, multiply-accumulate ops (MACs) for a single
32x32 grayscale image, fp32 model size, peak activation memory (estimated), and
CPU inference latency (batch=1, single-thread edge proxy) + batch throughput.

Architecture-only: params/MACs/latency do not depend on trained weights, so this
runs without any checkpoint. Produces Table 2 / Figure 2 of the paper.

Usage: python scripts/efficiency_metrics.py [--runs 100] [--out results/efficiency_metrics.json]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from devnet.models.student import DevNet
from devnet.models.mallanet_baseline import EnhancedBMCNNwHFCs

INPUT = (1, 1, 32, 32)  # (B, C, H, W) — DHCD grayscale 32x32


def count_macs(model: nn.Module, input_shape=INPUT):
    """Count Conv2d + Linear MACs for one forward pass via hooks.

    Conv2d and Linear dominate both models; the MallaNet HFC layers do only
    elementwise multiplies (~num_classes*D_b ~= 5e4 each), negligible vs the
    conv MACs (~1e8), so a Conv2d+Linear count matches standard FLOP reporting.
    """
    macs = {"total": 0}
    handles = []

    def conv_hook(module, inp, out):
        out_elems = out.shape[1] * out.shape[2] * out.shape[3]  # per sample
        k = module.kernel_size[0] * module.kernel_size[1]
        macs["total"] += out_elems * (module.in_channels // module.groups) * k

    def linear_hook(module, inp, out):
        macs["total"] += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return macs["total"]


def peak_activation_bytes(model: nn.Module, input_shape=INPUT):
    """Estimate peak activation memory as the sum of all module output tensor
    sizes for one forward pass (fp32). A coarse upper-bound proxy, reported as
    a rough indicator, not an exact runtime measurement."""
    total = {"bytes": 0}
    handles = []

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            total["bytes"] += out.numel() * 4

    for m in model.modules():
        if len(list(m.children())) == 0:  # leaf modules only
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return total["bytes"]


def latency_ms(model: nn.Module, batch=1, runs=100, warmup=5):
    """Mean+/-std single-forward latency on CPU, single-threaded (edge proxy)."""
    model.eval()
    x = torch.zeros((batch, *INPUT[1:]))
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times), statistics.pstdev(times)


def params(model):
    """Trainable parameter count (excludes BatchNorm running buffers), the
    standard reporting convention. Counted identically for both models, so the
    ratio is apples-to-apples. NB: the student's saved state_dict is 1,111,449
    because it also stores BN buffers; the trainable count is 1,109,116."""
    return sum(p.numel() for p in model.parameters())


def measure(name, model, runs):
    torch.set_num_threads(1)  # single-thread edge proxy; same for both models
    p = params(model)
    macs = count_macs(model)
    act = peak_activation_bytes(model)
    lat1_mean, lat1_std = latency_ms(model, batch=1, runs=runs)
    tput_batch = 128
    lat_b_mean, _ = latency_ms(model, batch=tput_batch, runs=5, warmup=2)
    throughput = tput_batch / (lat_b_mean / 1000.0)
    return {
        "model": name,
        "params": p,
        "params_millions": round(p / 1e6, 4),
        "model_size_mb_fp32": round(p * 4 / 1e6, 2),
        "macs": macs,
        "macs_millions": round(macs / 1e6, 2),
        "peak_activation_mb_est": round(act / 1e6, 2),
        "cpu_latency_ms_b1_mean": round(lat1_mean, 3),
        "cpu_latency_ms_b1_std": round(lat1_std, 3),
        "throughput_img_per_s": round(throughput, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", default="results/efficiency_metrics.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    models = [
        ("student (compact)", DevNet()),
        ("MallaNet (baseline)", EnhancedBMCNNwHFCs(num_classes=46)),
    ]
    rows = [measure(n, m, args.runs) for n, m in models]

    # derived comparison
    s, b = rows[0], rows[1]
    comparison = {
        "param_reduction_x": round(b["params"] / s["params"], 1),
        "mac_reduction_x": round(b["macs"] / s["macs"], 1),
        "speedup_b1_x": round(b["cpu_latency_ms_b1_mean"] / s["cpu_latency_ms_b1_mean"], 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": rows, "comparison": comparison,
                               "settings": {"threads": 1, "input": list(INPUT),
                                            "runs": args.runs}}, indent=2))

    w = 22
    cols = ["params_millions", "macs_millions", "model_size_mb_fp32",
            "peak_activation_mb_est", "cpu_latency_ms_b1_mean",
            "throughput_img_per_s"]
    print(f"{'metric':<28}" + "".join(f"{r['model']:>{w}}" for r in rows))
    for c in cols:
        print(f"{c:<28}" + "".join(f"{str(r[c]):>{w}}" for r in rows))
    print("\ncomparison (MallaNet / student):", json.dumps(comparison))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
