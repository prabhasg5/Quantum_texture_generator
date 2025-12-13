from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _robust_ybins(values: np.ndarray, bins: int = 120):
    v = np.asarray(values, dtype=float)
    lo = np.nanpercentile(v, 0.5)
    hi = np.nanpercentile(v, 99.5)
    if not np.isfinite(lo):
        lo = np.nanmin(v)
    if not np.isfinite(hi):
        hi = np.nanmax(v)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return np.linspace(lo, hi, bins)


def _ema(x: np.ndarray, beta: float = 0.98) -> np.ndarray:
    acc = []
    m = 0.0
    for t, v in enumerate(x, 1):
        m = beta * m + (1 - beta) * float(v)
        acc.append(m / (1 - beta ** t))
    return np.asarray(acc)


def _plot_one(ax, losses: np.ndarray, title: str, color: str, bins_x: int, bins_y: int):
    steps = np.arange(len(losses), dtype=float)
    x_edges = np.linspace(steps.min(), steps.max() + 1e-9, bins_x)
    y_edges = _robust_ybins(losses, bins=bins_y)

    H, x_e, y_e = np.histogram2d(steps, losses, bins=[x_edges, y_edges])
    H = H.T  # y,x for pcolormesh

    # Avoid empty bins hiding everything; use logarithmic color scaling
    H_plot = np.where(H > 0, H, np.nan)

    mesh = ax.pcolormesh(
        x_e, y_e, H_plot,
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=np.nanmax(H_plot) if np.isfinite(np.nanmax(H_plot)) else 1),
        shading="auto",
    )
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="sample density")

    # Overlay EMA trend to guide the eye
    ema = _ema(losses, beta=0.985)
    ax.plot(steps, ema, color=color, lw=1.5, alpha=0.9, label="EMA")

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, ls=":", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)


def plot_loss_histogram(pkl_path: Path, out_path: Path | None = None, bins_x: int = 220, bins_y: int = 140):
    with open(pkl_path, "rb") as f:
        hist: Dict[str, List[float]] = pickle.load(f)

    d = np.asarray(hist.get("discriminator", []), dtype=float)
    g = np.asarray(hist.get("generator", []), dtype=float)

    if d.size == 0 and g.size == 0:
        raise ValueError("loss_history.pkl has no 'discriminator' or 'generator' entries.")

    if out_path is None:
        out_path = pkl_path.parent / "loss_curve_hist.png"

    plt.figure(figsize=(12, 7))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25)

    ax1 = plt.subplot(gs[0])
    if d.size:
        _plot_one(ax1, d, "Discriminator Loss (2D histogram + EMA)", color="#19d1ff", bins_x=bins_x, bins_y=bins_y)
    else:
        ax1.set_visible(False)

    ax2 = plt.subplot(gs[1], sharex=ax1 if d.size else None)
    if g.size:
        _plot_one(ax2, g, "Generator Loss (2D histogram + EMA)", color="#ff6ad5", bins_x=bins_x, bins_y=bins_y)
    else:
        ax2.set_visible(False)

    plt.suptitle("Training Curves as Density Histograms", y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot loss_history.pkl as overlap-free 2D histogram curves.")
    ap.add_argument("--pkl", required=True, type=Path, help="Path to reports/<run>/loss_history.pkl")
    ap.add_argument("--out", type=Path, default=None, help="Output image path (PNG).")
    ap.add_argument("--bins-x", type=int, default=220, help="Bins along iteration axis.")
    ap.add_argument("--bins-y", type=int, default=140, help="Bins along loss axis.")
    args = ap.parse_args()
    path = plot_loss_histogram(args.pkl, args.out, args.bins_x, args.bins_y)
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
