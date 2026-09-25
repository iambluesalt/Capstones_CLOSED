"""Is there ANY detectable signal? Stratified-CV LightGBM AUC/PR-AUC vs a label-permutation null."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd.data import load_raw
from fd.features import build_features

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ROUNDS = [25, 50, 100, 200, 400]
PARAMS = dict(n_estimators=max(ROUNDS), learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=20, verbose=-1, random_state=0)


def cv_scores(X, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = {r: np.zeros(len(y)) for r in ROUNDS}
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr])
        for r in ROUNDS:
            oof[r][va] = m.predict_proba(X.iloc[va], num_iteration=r)[:, 1]
    return {r: (roc_auc_score(y, oof[r]), average_precision_score(y, oof[r])) for r in ROUNDS}, oof


if __name__ == "__main__":
    raw = load_raw()
    X, y = build_features(raw)
    print("X", X.shape, "pos", y.sum(), "base rate", y.mean().round(4))
    t0 = time.time()
    real, _ = cv_scores(X, y)
    print("REAL (rounds: AUC, PR-AUC):", {r: tuple(round(v, 4) for v in s) for r, s in real.items()}, f"{time.time()-t0:.0f}s")
    if N_PERM:
        rng = np.random.default_rng(0)
        null = {r: [] for r in ROUNDS}
        for i in range(N_PERM):
            yp = rng.permutation(y)
            s, _ = cv_scores(X, yp, seed=i)
            for r in ROUNDS:
                null[r].append(s[r])
            print(f"perm {i+1}/{N_PERM}", {r: round(null[r][-1][0], 4) for r in ROUNDS}, flush=True)
        for r in ROUNDS:
            a = np.array([v[0] for v in null[r]]); p = np.array([v[1] for v in null[r]])
            print(f"rounds={r}: real AUC {real[r][0]:.4f} | null AUC mean {a.mean():.4f} sd {a.std():.4f} max {a.max():.4f} | "
                  f"p={(1+(a>=real[r][0]).sum())/(1+len(a)):.3f} | real PR {real[r][1]:.4f} null PR {p.mean():.4f}")
