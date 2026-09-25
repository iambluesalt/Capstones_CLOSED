"""Power check: plant a known weak signal in real X and confirm the CV pipeline recovers it.
If it does, a null AUC on the real labels is a property of the data, not of the pipeline."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw
from fd.features import build_features

raw = load_raw(); X, y_real = build_features(raw)
rng = np.random.default_rng(7)


def z(s):
    s = s.astype(float); s = s.fillna(s.median()); return ((s - s.mean()) / s.std()).to_numpy()


# planted score mixes a numeric, a geo, a time and a categorical feature
score = 0.5 * z(X["V2CF"]) + 0.4 * z(np.log1p(X["geo_dist_km"])) + 0.4 * z(X["hour_sin"]) + 0.5 * z((X["channel"] == "KOL").astype(float)) + 0.4 * z(X["addr_age_days"])
score = (score - score.mean()) / score.std()
TARGET = y_real.mean()
params = dict(n_estimators=200, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, n_jobs=20, verbose=-1, random_state=0)


def cv_auc(X, y):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
        oof[va] = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return roc_auc_score(y, oof)


rows = []
for lam in [0.0, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4]:
    b = brentq(lambda b: (1 / (1 + np.exp(-(b + lam * score)))).mean() - TARGET, -12, 0)
    p_true = 1 / (1 + np.exp(-(b + lam * score)))
    res = []
    for rep in range(3):
        y = (np.random.default_rng(100 + rep).random(len(p_true)) < p_true).astype(int)
        res.append((roc_auc_score(y, p_true), cv_auc(X, y)))
    oracle, got = np.mean([r[0] for r in res]), np.mean([r[1] for r in res])
    rows.append(dict(planted_lambda=lam, oracle_auc=oracle, lgbm_cv_auc=got, recovered_frac=(got - .5) / max(oracle - .5, 1e-9) if lam else np.nan))
    print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(ROOT / "reports" / "tables" / "power_check.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
