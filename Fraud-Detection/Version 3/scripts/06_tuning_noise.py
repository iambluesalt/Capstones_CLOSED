"""Does hyper-parameter tuning find real signal? Tune LightGBM with Optuna on the real labels and on PERMUTED labels.
If the 'best CV AUC' looks the same in both, tuning gains are selection bias. Final score is on a locked, untouched test split."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw
from fd.features import build_features
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
X, y_real = build_features(load_raw())


def run(label, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    skf = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xtr, ytr))

    def make(t):
        return lgb.LGBMClassifier(
            n_estimators=t.suggest_int("n_estimators", 50, 400), learning_rate=t.suggest_float("lr", 0.005, 0.2, log=True),
            num_leaves=t.suggest_int("num_leaves", 4, 256, log=True), min_child_samples=t.suggest_int("mcs", 5, 400, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float("colsample", 0.3, 1.0),
            reg_alpha=t.suggest_float("alpha", 1e-8, 10, log=True), reg_lambda=t.suggest_float("lambda", 1e-8, 10, log=True),
            n_jobs=20, verbose=-1, random_state=0)

    def objective(t):
        m = make(t); oof = np.zeros(len(ytr))
        for a, b in skf:
            oof[b] = lgb.LGBMClassifier(**m.get_params()).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
        return roc_auc_score(ytr, oof)

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=N_TRIALS)
    best = make(optuna.trial.FixedTrial(st.best_params)).fit(Xtr, ytr)
    p = best.predict_proba(Xte)[:, 1]
    r = dict(labels=label, n_trials=N_TRIALS, best_cv_auc=st.best_value, median_trial_cv_auc=float(np.median([t.value for t in st.trials])),
             locked_test_auc=roc_auc_score(yte, p), locked_test_pr_auc=average_precision_score(yte, p), test_base_rate=float(yte.mean()))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r


rows = [run("real", y_real), run("permuted", np.random.default_rng(0).permutation(y_real))]
out = pd.DataFrame(rows); out.to_csv(ROOT / "reports" / "tables" / "tuning_noise.csv", index=False)
print(out.round(4).to_string(index=False))
