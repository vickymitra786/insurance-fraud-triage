"""Four paper figures from results/results.json and results/test_scores.csv. Author: Vivek Mitra"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import cum_caught, net_saving

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper" / "figures"
MODELS = [("score_gb", "gradient_boosting", "#1f4e8c", "Gradient boosting"), ("score_lr", "logistic_regression", "#d97a1e", "Logistic regression")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads((HERE / "results" / "results.json").read_text())
    t = pd.read_csv(HERE / "results" / "test_scores.csv")
    y, share = t.label.to_numpy(), np.arange(len(t) + 1) / len(t) * 100

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for col, _, colour, label in MODELS:
        ax.plot(share, cum_caught(y, t[col].to_numpy()) / y.sum() * 100, color=colour, lw=1.8, label=label)
    ax.plot([0, 100], [0, 100], color="#8a93a0", ls="--", lw=1, label="Random order")
    ax.set(xlim=(0, 60), ylim=(0, 100), xlabel="Claims checked, highest score first (%)", ylabel="Frauds found (%)")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "capture.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    for _, key, colour, label in MODELS:
        b = r["calibration"][key]["bins"]
        ax.plot([x["mean_pred"] for x in b], [x["observed"] for x in b], "o-", color=colour, ms=3.5, lw=1.2, label=label)
    ax.plot([0, 0.35], [0, 0.35], color="#8a93a0", ls="--", lw=1)
    ax.set(xlabel="Mean predicted fraud probability (tenth)", ylabel="Observed fraud rate")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for col, _, colour, label in MODELS:
        ax.plot(share, net_saving(y, t[col].to_numpy(), r["cost"]["assumptions"]) / 1000, color=colour, lw=1.6, label=label)
    depth = r["cost"][r["decision"]["primary"]]["depth_chosen_on_1995"]["share_checked"] * 100
    ax.axvline(depth, color="#1f4e8c", ls=":", lw=1)
    ax.axhline(0, color="#8a93a0", lw=0.6)
    ax.set(xlabel="Claims checked (%)", ylabel="Net saving, $ thousand (assumed money)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(OUT / "net_saving.png"); plt.close(fig)

    imp = r["importance"][:10][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.barh([x["field"] for x in imp], [x["mean_abs_push"] for x in imp], color="#1f4e8c")
    ax.set_xlabel("Mean size of push on 1996 scores (log-odds)")
    fig.tight_layout(); fig.savefig(OUT / "importance.png"); plt.close(fig)
    print("  wrote 4 figures to paper/figures")


if __name__ == "__main__":
    main()
