from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def _ema(x: np.ndarray, beta: float = 0.98) -> np.ndarray:
    acc = []
    m = 0.0
    for t, v in enumerate(x, 1):
        m = beta * m + (1 - beta) * float(v)
        acc.append(m / (1 - beta ** t))
    return np.asarray(acc)


def plot_line(
    pkl_path: Path,
    out_path: Path | None = None,
    beta: float = 0.985,
    show_raw: bool = False,
    split_axes: bool = False,
    twin_axes: bool = False,
):
    with open(pkl_path, "rb") as f:
        hist: Dict[str, List[float]] = pickle.load(f)

    d = np.asarray(hist.get("discriminator", []), dtype=float)
    g = np.asarray(hist.get("generator", []), dtype=float)

    if d.size == 0 and g.size == 0:
        raise ValueError("loss_history.pkl has no 'discriminator' or 'generator' entries.")

    if out_path is None:
        out_path = pkl_path.parent / "loss_curve_line_smooth.png"

    d_ema = _ema(d, beta=beta) if d.size else None
    g_ema = _ema(g, beta=beta) if g.size else None

    if split_axes:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        steps_d = np.arange(len(d)) if d.size else None
        steps_g = np.arange(len(g)) if g.size else None
        if d.size:
            if show_raw:
                ax1.plot(steps_d, d, color="#7be6ff", lw=0.8, alpha=0.5, label="Discriminator (raw)")
            ax1.plot(steps_d, d_ema, color="#19d1ff", lw=2.0, alpha=0.95, label="Discriminator (EMA)")
            ax1.set_ylabel("Loss")
            ax1.set_title("Discriminator Loss")
            ax1.grid(True, ls=":", alpha=0.25)
            ax1.legend(loc="upper right", frameon=False)
        if g.size:
            if show_raw:
                ax2.plot(steps_g, g, color="#ffb3e7", lw=0.8, alpha=0.5, label="Generator (raw)")
            ax2.plot(steps_g, g_ema, color="#ff6ad5", lw=2.0, alpha=0.95, label="Generator (EMA)")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Loss")
            ax2.set_title("Generator Loss")
            ax2.grid(True, ls=":", alpha=0.25)
            ax2.legend(loc="upper right", frameon=False)
        plt.suptitle("Training Loss (Smoothed)", y=0.98)
        fig.tight_layout()
        out = out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out

    # Single axes with both lines
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    steps = np.arange(max(len(d), len(g)))

    if not twin_axes:
        if d.size:
            if show_raw:
                ax.plot(np.arange(len(d)), d, color="#7be6ff", lw=0.8, alpha=0.35, label="Discriminator (raw)")
            ax.plot(np.arange(len(d)), d_ema, color="#19d1ff", lw=2.0, alpha=0.95, label="Discriminator (EMA)")
        if g.size:
            if show_raw:
                ax.plot(np.arange(len(g)), g, color="#ffb3e7", lw=0.8, alpha=0.35, label="Generator (raw)")
            ax.plot(np.arange(len(g)), g_ema, color="#ff6ad5", lw=2.0, alpha=0.95, label="Generator (EMA)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss (Smoothed, Single Axes)")
        ax.grid(True, ls=":", alpha=0.25)
        ax.legend(loc="best", frameon=False)
    else:
        # Optional twin y-axis if dynamic ranges differ greatly
        ax2 = ax.twinx()
        if d.size:
            if show_raw:
                ax.plot(np.arange(len(d)), d, color="#7be6ff", lw=0.8, alpha=0.35, label="Discriminator (raw)")
            l1, = ax.plot(np.arange(len(d)), d_ema, color="#19d1ff", lw=2.0, alpha=0.95, label="Discriminator (EMA)")
        if g.size:
            if show_raw:
                ax2.plot(np.arange(len(g)), g, color="#ffb3e7", lw=0.8, alpha=0.35, label="Generator (raw)")
            l2, = ax2.plot(np.arange(len(g)), g_ema, color="#ff6ad5", lw=2.0, alpha=0.95, label="Generator (EMA)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Discriminator Loss")
        ax2.set_ylabel("Generator Loss")
        ax.set_title("Training Loss (Smoothed, Twin Axes)")
        ax.grid(True, ls=":", alpha=0.25)
        lines = []
        labels = []
        for a in (ax, ax2):
            h, l = a.get_legend_handles_labels()
            lines += h
            labels += l
        fig.legend(lines, labels, loc="upper right", frameon=False)

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    desc = "Plot generator and discriminator losses as smoothed EMA lines; single axes by default."
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--pkl", required=True, type=Path, help="Path to reports/<run>/loss_history.pkl")
    ap.add_argument("--out", type=Path, default=None, help="Output image path (PNG). Defaults beside the .pkl")
    ap.add_argument("--ema-beta", type=float, default=0.99, help="EMA smoothing factor (closer to 1 = smoother)")
    ap.add_argument("--show-raw", action="store_true", help="Also plot raw lines (faint)")
    ap.add_argument("--split", action="store_true", help="Use two stacked axes instead of one")
    ap.add_argument("--twin", action="store_true", help="Use twin y-axes for different scales on one plot")
    args = ap.parse_args()

    out_default = None
    if args.out is None:
        out_default = Path(args.pkl).parent / ("loss_curve_line_smooth_split.png" if args.split else "loss_curve_line_smooth.png")

    out = plot_line(
        args.pkl,
        args.out or out_default,
        beta=args.ema_beta,
        show_raw=args.show_raw,
        split_axes=args.split,
        twin_axes=args.twin,
    )
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
