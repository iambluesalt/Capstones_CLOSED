"""Figures for the V3 report. Reads the CSVs the other scripts wrote plus the raw file (fig 1). Style follows the dataviz skill:
one hue per role, hairline solid grid, headline titles, selective direct labels, text in ink (never the series colour)."""
import sys, textwrap
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw

T = ROOT / "reports" / "tables"; F = ROOT / "reports" / "figures"; F.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": next((f for f in ["Segoe UI", "Helvetica Neue", "DejaVu Sans"] if f in fams), "sans-serif"),
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 9.5,
})


def frame(title, sub, size=(8.2, 4.6)):
    lines = textwrap.wrap(sub, int(size[0] * 13.2))
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(top=0.80 - 0.05 * (len(lines) - 1), left=0.09, right=0.97, bottom=0.14)
    fig.text(0.03, 0.955, title, fontsize=13.5, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.03, 0.885, "\n".join(lines), fontsize=9.5, ha="left", va="top", color=INK2, linespacing=1.4)
    return fig, ax


def save(fig, name):
    fig.savefig(F / name, dpi=160); plt.close(fig); print("wrote", name)


# 1 ---- TransactionKey leak -----------------------------------------------------------------------------------------
raw = load_raw()
dec = pd.qcut(raw.TransactionKey, 10, labels=False)
rate = raw.groupby(dec).Fraud.mean() * 100
fig, ax = frame("Every fraud sits in the top decile of TransactionKey",
                "Fraud rate by TransactionKey decile. The key alone scores AUC = 1.000, so it is a dataset-construction artifact, not a signal.")
ax.bar(np.arange(1, 11), rate.values, width=0.6, color=BLUE)
ax.set_xticks(np.arange(1, 11)); ax.set_xlabel("TransactionKey decile (1 = lowest keys)"); ax.set_ylabel("Fraud rate (%)")
ax.set_ylim(0, 33); ax.grid(axis="x", visible=False)
ax.annotate(f"{rate.values[-1]:.1f}%  (all 3,963 frauds)", (10, rate.values[-1]), xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=9.5, color=INK)
ax.text(5, 4.5, "0 frauds in 124,877 rows", ha="center", fontsize=9.5, color=INK2)
save(fig, "fig1_key_leak.png")

# 2 ---- univariate AUC ----------------------------------------------------------------------------------------------
nu = pd.read_csv(T / "numeric_univariate.csv").sort_values("auc").reset_index(drop=True)
n1, n0 = int(raw.Fraud.sum()), int((1 - raw.Fraud).sum()); se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
fig, ax = frame("No single feature separates fraud from non-fraud beyond noise",
                f"Univariate AUC of {len(nu)} numeric features, sorted. Shaded band = 95% range expected by chance (0.5 ± {1.96*se:.3f}). "
                f"{int(((nu.auc < 0.5 - 1.96*se) | (nu.auc > 0.5 + 1.96*se)).sum())} of {len(nu)} fall outside it; chance alone gives ~{0.05*len(nu):.0f}, and none survive FDR correction.")
ax.axhspan(0.5 - 1.96 * se, 0.5 + 1.96 * se, color=GRID, alpha=0.9, lw=0)
ax.axhline(0.5, color=AXIS, lw=0.8)
ax.scatter(np.arange(len(nu)), nu.auc, s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
ax.set_xlabel("Features, ranked by AUC"); ax.set_ylabel("AUC (0.5 = no signal)"); ax.set_ylim(0.47, 0.53); ax.set_xticks([]); ax.grid(axis="x", visible=False)
for i, va, dy in [(len(nu) - 1, "bottom", 7), (0, "top", -7)]:
    r = nu.iloc[i]; ax.annotate(f"{r.feature}  {r.auc:.3f}", (i, r.auc), xytext=(-4 if i else 4, dy), textcoords="offset points",
                                ha="right" if i else "left", va=va, fontsize=9, color=INK)
save(fig, "fig2_univariate_auc.png")

# 3 ---- p-value histogram -------------------------------------------------------------------------------------------
pv = np.concatenate([nu.p.to_numpy(), pd.read_csv(T / "categorical_levels.csv").p.to_numpy()])
cnt, edges = np.histogram(pv, bins=10, range=(0, 1))
fig, ax = frame("190 significance tests look like pure chance",
                f"P-values of every numeric feature (56) and every categorical level with n ≥ 200 (134). {int((pv < .05).sum())} have p < 0.05 "
                f"(chance alone gives ~{0.05*len(pv):.0f}); none survive FDR correction.")
ax.bar(edges[:-1] + 0.05, cnt, width=0.085, color=BLUE)
ax.axhline(len(pv) / 10, color=INK2, lw=1.0); ax.text(0.99, 28.5, "line = count expected if there is no signal", ha="right", va="top", fontsize=9, color=INK2)
ax.set_ylim(0, 29)
ax.set_xlabel("p-value"); ax.set_ylabel("Number of tests"); ax.grid(axis="x", visible=False)
save(fig, "fig3_pvalues.png")

# 4 ---- protocols ---------------------------------------------------------------------------------------------------
lk = pd.read_csv(T / "leakage_demo.csv")
leaky = lk.protocol.str[:2].str.strip().isin(["A", "B", "E"]) & ~lk.protocol.str.startswith("E'")
lk = lk.assign(leaky=leaky).iloc[::-1].reset_index(drop=True)
fig, ax = frame("V1/V2's near-perfect scores were leakage; done correctly, AUC is 0.49",
                "Same data and same XGBoost, different evaluation protocol. Test AUC (0.5 = chance).", size=(8.6, 4.8))
fig.subplots_adjust(left=0.36)
ax.barh(np.arange(len(lk)), lk.auc, height=0.55, color=[ORANGE if l else BLUE for l in lk.leaky])
ax.set_yticks(np.arange(len(lk))); ax.set_yticklabels(lk.protocol, fontsize=9, color=INK2); ax.set_xlim(0, 1.1)
ax.axvline(0.5, color=INK2, lw=0.8); ax.text(0.503, len(lk) - 0.45, "chance", fontsize=8.5, color=INK2, va="bottom")
ax.grid(axis="y", visible=False); ax.set_xlabel("Test AUC")
for i, r in lk.iterrows():
    ax.text(max(r.auc, 0.5) + 0.012, i, f"{r.auc:.3f}", va="center", fontsize=9, color=INK)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
          labels=["Leaky protocol", "Correct protocol"], loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
save(fig, "fig4_protocols.png")

# 5 ---- power check -------------------------------------------------------------------------------------------------
pw = pd.read_csv(T / "power_check.csv")
fig, ax = frame("Planted signal is recovered once it reaches AUC ~0.55; the real labels show none",
                "Signals of known strength planted in the real features: recovered CV AUC vs the best achievable (oracle) AUC. Grey = range CV AUC takes by chance on shuffled labels (±2 SD of 20 shuffles).")
ax.plot([0.5, 0.62], [0.5, 0.62], color=AXIS, lw=1.0)
ax.plot(pw.oracle_auc, pw.lgbm_cv_auc, color=BLUE, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.axhspan(0.5 - 0.0155, 0.5 + 0.0155, color=GRID, alpha=0.9, lw=0)  # null SD of CV AUC measured at 0.0072-0.0080 (02_signal_test.py)
ax.axhspan(0.4987, 0.5022, color=ORANGE, alpha=0.55, lw=0)
ax.set_xlabel("Oracle AUC of planted signal"); ax.set_ylabel("LightGBM 5-fold CV AUC"); ax.set_xlim(0.495, 0.62); ax.set_ylim(0.495, 0.62)
ax.text(0.618, 0.5195, "real fraud labels: 0.499–0.502 (orange)", ha="right", va="bottom", fontsize=9, color=INK)
ax.text(0.618, 0.5075, "chance range", ha="right", va="bottom", fontsize=9, color=INK2)
ax.text(0.600, 0.550, "LightGBM recovered", fontsize=9, color=INK, ha="center", va="top")
ax.text(0.575, 0.586, "perfect recovery", fontsize=9, color=INK2, ha="right", va="bottom")
save(fig, "fig5_power.png")

# 6 ---- tuning noise ------------------------------------------------------------------------------------------------
if (T / "tuning_noise.csv").exists():
    tn = pd.read_csv(T / "tuning_noise.csv").set_index("labels")
    fig, ax = frame("Tuning does not create signal: real and shuffled labels score alike",
                    f"Optuna ({int(tn.n_trials.iloc[0])} trials) on real vs randomly shuffled labels, scored on a locked test split (AUC standard error ≈ 0.010).")
    x = np.arange(2); off = 0.09
    for lab, color, dx, name in [("real", BLUE, -off, "Real labels"), ("permuted", ORANGE, off, "Shuffled labels (no signal by construction)")]:
        ys = [tn.loc[lab, "best_cv_auc"], tn.loc[lab, "locked_test_auc"]]
        ax.scatter(x + dx, ys, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
        for xi, v in zip(x + dx, ys):
            ax.text(xi + (-0.05 if dx < 0 else 0.05), v, f"{v:.3f}", ha="right" if dx < 0 else "left", va="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["Best CV AUC found by tuning", "Locked test AUC of the tuned model"]); ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.48, 0.53); ax.axhline(0.5, color=AXIS, lw=0.8); ax.grid(axis="x", visible=False); ax.set_ylabel("AUC (0.5 = chance)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    save(fig, "fig6_tuning_noise.png")
