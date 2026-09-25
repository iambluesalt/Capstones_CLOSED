"""Deeper 'is there signal?' tests: entity clustering (permutation), categorical levels (chi2 + BH-FDR),
numeric univariate (Mann-Whitney + BH-FDR). Writes CSVs to reports/tables."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw
from fd.features import build_features, CAT_COLS

OUT = ROOT / "reports" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


raw = load_raw()
X, y = build_features(raw)
rng = np.random.default_rng(0)

# ---- 1. entity clustering: are frauds over-represented in the same entity? -------------------------------------
print("== 1. fraud co-occurrence within entities (stat = #fraud pairs sharing a key; permutation null, 2000 perms)")
keys = {
    "AddressUpdateDate": raw["AddressUpdateDate"],
    "EmailUpdateDate": raw["EmailUpdateDate"],
    "Address+Email": raw["AddressUpdateDate"].astype(str) + raw["EmailUpdateDate"].astype(str),
    "TransactionDateTime(sec)": raw["TransactionDateTime"],
    "Last lat/long": raw["LastLat"].astype(str) + "," + raw["LastLong"].astype(str),
    "Current long": raw["CurrentLong"],
    "ConnectionOrg(raw)": raw["ConnectionOrg"],
    "AreaCode+V4+V5": raw["AreaCode"].astype(str) + raw["V4CF"].astype(str) + raw["V5CF"].astype(str),
    "MainEntityUse": raw["MainEntityUse"],
}
rows = []
# (label, key, size cap): the capped variants drop giant groups (e.g. a 9,820-row shared AddressUpdateDate that looks like
# a system default) so only plausible same-account clusters of 2-50 rows are counted.
variants = [(n, k, None) for n, k in keys.items()] + [
    (f"{n} [groups 2-50]", keys[n], 50) for n in
    ["AddressUpdateDate", "EmailUpdateDate", "Last lat/long", "AreaCode+V4+V5", "TransactionDateTime(sec)"]
]
for name, k, cap in variants:
    codes, uniq = pd.factorize(k)
    miss = codes < 0  # missing keys become singleton groups so they cannot create fake co-occurrence
    codes[miss] = len(uniq) + np.arange(miss.sum())
    size = np.bincount(codes)
    multi = size[codes] >= 2
    if cap:
        multi = multi & (size[codes] <= cap)
    w = multi.astype(float)
    stat = lambda yy: (lambda f: (f * (f - 1) / 2).sum())(np.bincount(codes, weights=yy * w))
    obs = stat(y)
    null = np.array([stat(rng.permutation(y)) for _ in range(2000)])
    rows.append(dict(key=name, n_groups=len(size), rows_in_multi=int(multi.sum()), obs_fraud_pairs=obs,
                     null_mean=null.mean(), null_sd=null.std(), z=(obs - null.mean()) / null.std(),
                     p=(1 + (null >= obs).sum()) / 2001))
ent = pd.DataFrame(rows); print(ent.round(3).to_string(index=False)); ent.to_csv(OUT / "entity_clustering.csv", index=False)

# ---- 2. categorical levels -------------------------------------------------------------------------------------
print("\n== 2. categorical association with Fraud (chi2 per column; per-level binomial vs base rate, BH-FDR over levels n>=200)")
base = y.mean(); crow = []; lrow = []
for c in CAT_COLS:
    ct = pd.crosstab(X[c], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct.loc[ct.sum(axis=1) >= 20], correction=False)
    crow.append(dict(col=c, levels=len(ct), chi2=chi2, dof=dof, p=p))
    for lvl, (n0, n1) in ct.iterrows():
        n = n0 + n1
        if n >= 200:
            lo, hi = wilson(n1, n)
            lrow.append(dict(col=c, level=lvl, n=n, frauds=n1, rate=n1 / n, lift=n1 / n / base, ci_lo=lo, ci_hi=hi,
                             p=stats.binomtest(int(n1), int(n), base).pvalue))
cdf = pd.DataFrame(crow); cdf["q_bh"] = bh(cdf.p); print(cdf.round(4).to_string(index=False))
ldf = pd.DataFrame(lrow); ldf["q_bh"] = bh(ldf.p); ldf = ldf.sort_values("p")
print(f"\nlevels tested: {len(ldf)}; q<0.05: {(ldf.q_bh<0.05).sum()}; nominal p<0.05: {(ldf.p<0.05).sum()} (expect ~{0.05*len(ldf):.0f} by chance)")
print(ldf.head(12).round(4).to_string(index=False))
cdf.to_csv(OUT / "categorical_chi2.csv", index=False); ldf.to_csv(OUT / "categorical_levels.csv", index=False)

# ---- 3. numeric univariate ------------------------------------------------------------------------------------
print("\n== 3. numeric univariate: AUC via Mann-Whitney (BH-FDR)")
nrow = []
num = [c for c in X.columns if c not in CAT_COLS]
n1, n0 = y.sum(), (1 - y).sum()
for c in num:
    x = X[c].astype(float); m = x.notna().to_numpy()
    if x[m].nunique() < 2: continue
    u, p = stats.mannwhitneyu(x[m][y[m] == 1], x[m][y[m] == 0], alternative="two-sided")
    nrow.append(dict(feature=c, n=int(m.sum()), auc=u / (y[m].sum() * (1 - y[m]).sum()), p=p))
ndf = pd.DataFrame(nrow); ndf["q_bh"] = bh(ndf.p); ndf["abs_dev"] = (ndf.auc - .5).abs()
ndf = ndf.sort_values("p"); print(f"features: {len(ndf)}; q<0.05: {(ndf.q_bh<0.05).sum()}; nominal p<0.05: {(ndf.p<0.05).sum()} (expect ~{0.05*len(ndf):.1f})")
print(ndf.head(10).round(4).to_string(index=False)); ndf.to_csv(OUT / "numeric_univariate.csv", index=False)
se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
print(f"\nnull SE of AUC = {se:.4f}; min detectable |AUC-0.5| at alpha=.05 ~ {1.96*se:.4f} (80% power ~ {(1.96+0.84)*se:.4f})")
