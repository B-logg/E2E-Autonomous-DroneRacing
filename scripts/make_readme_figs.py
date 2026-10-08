#!/usr/bin/env python3
"""Plots for the README: learning curve and success against gate window.

    python scripts/make_readme_figs.py --metrics <run>/metrics.jsonl \
        --evals 1.2M=<dir>/evaluation.json ... --out docs/results
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Success against the half-width of the gate window, 16.99M checkpoint, paper start,
# window disabled in the simulator and applied afterwards to the recorded crossings
# (scripts/diagnose_start.py; N = 256 episodes x 5 laps).
WINDOW = {
    "m": [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5],
    "all gates": [91.0, 89.8, 82.4, 75.4, 64.8, 37.1, 13.3],
    "real gates only": [91.8, 91.0, 86.3, 81.6, 74.6, 52.7, 28.9],
}


def rolling(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=pathlib.Path)
    ap.add_argument("--evals", nargs="+", required=True, help="label=path/to/evaluation.json")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    rows = [json.loads(l) for l in a.metrics.read_text().splitlines() if '"episode/length"' in l]
    step = np.array([r["step"] for r in rows]) / 1e6
    length = np.array([r["episode/length"] for r in rows])
    ev = []
    for item in a.evals:
        lab, p = item.split("=", 1)
        d = json.loads(pathlib.Path(p).read_text())
        n = d["episodes"]
        s = d["success_rate"]
        ev.append((float(lab.rstrip("M")), s * 100, 1.96 * (s * (1 - s) / n) ** 0.5 * 100,
                   d["mean_gates_passed"]))
    ev.sort()

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    w = 400
    ax[0].plot(step[w - 1:], rolling(length, w), color="#1f6f8b", lw=1.4)
    ax[0].set_xlabel("training step [M]")
    ax[0].set_ylabel("episode length [steps]")
    ax[0].set_title("Training: episode length (rolling mean)")
    ax[0].axhline(2000, color="grey", lw=0.8, ls="--")
    ax[0].text(0.3, 2030, "2000-step limit", color="grey", fontsize=8)
    x = [e[0] for e in ev]
    ax[1].errorbar(x, [e[1] for e in ev], yerr=[e[2] for e in ev], marker="o", color="#b5483a", capsize=3)
    ax[1].set_ylim(0, 100)
    ax[1].set_xlabel("training step [M]")
    ax[1].set_ylabel("5-lap success [%]")
    ax[1].set_title("Evaluation: paper protocol (95% CI)")
    ax[2].plot(x, [e[3] for e in ev], marker="o", color="#4a6b3a")
    ax[2].set_ylim(0, 15.5)
    ax[2].axhline(15, color="grey", lw=0.8, ls="--")
    ax[2].set_xlabel("training step [M]")
    ax[2].set_ylabel("gates passed (of 15)")
    ax[2].set_title("Evaluation: mean gates passed")
    fig.tight_layout()
    fig.savefig(a.out / "learning_curve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(WINDOW["m"], WINDOW["all gates"], marker="o", color="#b5483a", label="all gates")
    ax.plot(WINDOW["m"], WINDOW["real gates only"], marker="s", color="#1f6f8b", label="real gates only")
    for xv, lab in ((1.0, "paper eval d_g"), (0.8, "paper train d_g"), (0.75, "gate half-width")):
        ax.axvline(xv, color="grey", lw=0.8, ls=":")
        ax.text(xv, 3, lab, rotation=90, ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_xlabel("gate window half-width d_g [m]")
    ax.set_ylabel("5-lap success [%]")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Success against gate window (16.99M)")
    fig.tight_layout()
    fig.savefig(a.out / "window_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
