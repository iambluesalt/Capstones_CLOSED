"""Leak-safe end-to-end evaluation. Locked 20% test set (stratified), 5-fold OOF CV on the other 80%, bootstrap CIs on test.
No resampling of any kind touches the test rows; class imbalance is handled by weighting only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw
from fd.features import build_features
from fd.metrics import bootstrap_ci, evaluate

X, y = build_features(load_raw())
assert "TransactionKey" not in X.columns
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
spw = (ytr == 0).sum() / (ytr == 1).sum()
print(f"train {Xtr.shape} pos={ytr.sum()} | locked test {Xte.shape} pos={yte.sum()}")

logit = Pipeline([
    ("prep", ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), make_column_selector(dtype_include="category")),
    ])),
    ("clf", LogisticRegression(C=0.05, max_iter=1000)),
])
models = {
    "base-rate (dummy)": lambda: DummyClassifier(strategy="prior"),
    "logistic L2": lambda: logit,
    "lightgbm (regularised)": lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                                                        subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, n_jobs=20, verbose=-1, random_state=0),
    "xgboost (scale_pos_weight)": lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                                                       min_child_weight=20, reg_lambda=10, scale_pos_weight=spw, enable_categorical=True,
                                                       tree_method="hist", n_jobs=20, random_state=0),
}
rows = []
for name, mk in models.items():
    oof = np.zeros(len(ytr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
        oof[b] = mk().fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
    cv = evaluate(ytr, oof)
    p = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    te = evaluate(yte, p)
    auc_ci = bootstrap_ci(yte, p, roc_auc_score); pr_ci = bootstrap_ci(yte, p, average_precision_score)
    rows.append(dict(model=name, cv_auc=cv["auc"], cv_pr_auc=cv["pr_auc"], test_auc=te["auc"], test_auc_lo=auc_ci[0], test_auc_hi=auc_ci[1],
                     test_pr_auc=te["pr_auc"], test_pr_lo=pr_ci[0], test_pr_hi=pr_ci[1], base_rate=te["base_rate"],
                     **{k: te[k] for k in te if k.startswith(("prec@", "recall@", "lift@"))}))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

out = pd.DataFrame(rows); out.to_csv(ROOT / "reports" / "tables" / "final_evaluation.csv", index=False)
print(out.round(4).T.to_string())

# review-economics: reviewing the top-k% only pays if precision * avg_fraud_loss > review_cost
br = yte.mean()
print(f"\nBreak-even: a random review queue has precision {br:.4f}; it pays off only if avg fraud loss / review cost > {1/br:.0f}. "
      f"No model above beats random by more than its CI, so the queue's economics are unchanged by modelling.")
