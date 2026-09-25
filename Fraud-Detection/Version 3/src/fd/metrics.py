"""Metrics that mean something at a 2.9% base rate. Accuracy is intentionally absent."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score


def top_k(y, p, frac):
    """(precision, recall, lift) when reviewing the top `frac` of transactions by score."""
    k = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-p, kind="stable")[:k]
    tp = y[idx].sum()
    prec = tp / k
    return prec, tp / max(y.sum(), 1), prec / y.mean()


def recall_at_precision(y, p, min_precision):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def best_f1(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(f1.max())


def evaluate(y, p):
    out = dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), base_rate=float(np.mean(y)),
               best_f1=best_f1(y, p), brier=brier_score_loss(y, np.clip(p, 0, 1)))
    for f in (0.01, 0.05):
        pr, rc, lift = top_k(y, p, f)
        out.update({f"prec@top{int(f*100)}%": pr, f"recall@top{int(f*100)}%": rc, f"lift@top{int(f*100)}%": lift})
    return out


def bootstrap_ci(y, p, fn=roc_auc_score, n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
