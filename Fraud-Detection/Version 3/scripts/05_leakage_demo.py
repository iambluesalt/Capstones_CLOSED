"""Reproduce V1/V2's headline numbers with their (flawed) protocols, then redo it correctly on the same data.

A  V2 protocol : SMOTE on ALL rows, then train_test_split          -> synthetic neighbours of test rows sit in train
B  V1 protocol : RandomOverSampler on ALL rows, test = original 20% -> exact copies of test rows sit in train
C  correct     : split first, SMOTE on train only, score untouched test
D  correct     : split first, no resampling (scale_pos_weight), score untouched test
E  key leak    : same as D but TransactionKey left in as a feature
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw

V2_CSV = ROOT.parent / "Version 2" / "FD-Dataset-2.csv"
d = pd.read_csv(V2_CSV)
X, y = d.drop(columns="Fraud"), d["Fraud"].to_numpy()
print("V2 dataset", X.shape, "fraud rate", y.mean().round(4))


def xgb(**kw):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=20, random_state=0, tree_method="hist", **kw)


def report(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
    r = dict(protocol=name, test_n=len(yte), test_pos=int(yte.sum()), auc=roc_auc_score(yte, p), pr_auc=average_precision_score(yte, p),
             precision=precision_score(yte, pred, zero_division=0), recall=recall_score(yte, pred), f1=f1_score(yte, pred))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r


res = []
# A. V2-style
Xs, ys = SMOTE(random_state=0).fit_resample(X, y)
Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.3, random_state=0)
res.append(report("A  V2: SMOTE(all) -> split", xgb(), Xtr, ytr, Xte, yte))

# B. V1-style
Xtr0, Xte0, ytr0, yte0 = train_test_split(X, y, test_size=0.2, random_state=42)
Xr, yr = RandomOverSampler(random_state=42).fit_resample(X, y)
res.append(report("B  V1: ROS(all) -> test on original 20%", xgb(), Xr, yr, Xte0, yte0))

# C. correct + SMOTE on train only
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xs, ys = SMOTE(random_state=0).fit_resample(Xtr, ytr)
res.append(report("C  correct: split -> SMOTE(train)", xgb(), Xs, ys, Xte, yte))

# D. correct, class weighting only
spw = (ytr == 0).sum() / (ytr == 1).sum()
res.append(report("D  correct: split -> scale_pos_weight", xgb(scale_pos_weight=spw), Xtr, ytr, Xte, yte))

# E. TransactionKey left in
raw = load_raw()
Xk = d.drop(columns="Fraud").copy()
# V2 rows are the raw rows minus 380 dropped rows; realign on the shared row order via TransactionKey-free index is not
# possible, so use the raw file directly with a minimal numeric feature set + the key.
num = raw[["V1CF", "V2CF", "V3CF", "V4CF", "V5CF", "TransactionKey"]].copy()
yr_ = raw["Fraud"].to_numpy()
a, b, c, e = train_test_split(num, yr_, test_size=0.2, random_state=42, stratify=yr_)
res.append(report("E  key leak: + TransactionKey", xgb(scale_pos_weight=(c == 0).sum() / (c == 1).sum()), a, c, b, e))
a2, b2 = a.drop(columns="TransactionKey"), b.drop(columns="TransactionKey")
res.append(report("E' same, key removed", xgb(scale_pos_weight=(c == 0).sum() / (c == 1).sum()), a2, c, b2, e))

out = pd.DataFrame(res); out.to_csv(ROOT / "reports" / "tables" / "leakage_demo.csv", index=False)
print(out.round(4).to_string(index=False))
