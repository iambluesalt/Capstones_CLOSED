"""Raw-data audit: label leak, missingness, sentinels, structure. Prints a report and saves reports/tables/audit.txt."""
import io, sys, contextlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fd.data import load_raw

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pd.set_option("display.width", 200)
    d = load_raw()
    print(f"rows={len(d):,} cols={d.shape[1]} fraud={d.Fraud.sum():,} ({d.Fraud.mean():.2%})  duplicate rows={d.duplicated().sum()}  duplicate keys={d.TransactionKey.duplicated().sum()}")

    print("\n[1] TransactionKey is a perfect label leak")
    fk, nk = d.TransactionKey[d.Fraud == 1], d.TransactionKey[d.Fraud == 0]
    print(f"fraud keys {fk.min():,}..{fk.max():,} | non-fraud keys {nk.min():,}..{nk.max():,} | overlap={fk.min() <= nk.max()} | AUC(key)={roc_auc_score(d.Fraud, d.TransactionKey):.4f}")
    print(f"row order is NOT informative: corr(row index, key)={np.corrcoef(np.arange(len(d)), d.TransactionKey)[0,1]:.4f}")

    print("\n[2] Missing values and whether missingness carries signal")
    rows = []
    for c in d.columns[d.isna().any()]:
        m = d[c].isna(); rows.append((c, int(m.sum()), f"{m.mean():.1%}", f"{d.Fraud[m].mean():.4f}", f"{d.Fraud[~m].mean():.4f}"))
    print(pd.DataFrame(rows, columns=["column", "n_missing", "pct", "fraud_rate|missing", "fraud_rate|present"]).to_string(index=False))
    print(f"CurrentLong present but CurrentLat missing: {(d.CurrentLong.notna() & d.CurrentLat.isna()).sum():,}  (geo columns are inconsistently populated)")

    print("\n[3] Sentinels / impossible values")
    print(f"TimeZone==999: {(d.TimeZone == 999).sum()} | TimeZone range excluding 999: {d.TimeZone[d.TimeZone < 100].min()}..{d.TimeZone[d.TimeZone < 100].max()}")
    print(f"WebSessionRetail distinct values (non-null): {sorted(d.WebSessionRetail.dropna().unique())}  -> zero-variance column")
    print(f"AddressUpdateDate after TransactionDateTime: {(d.AddressUpdateDate > d.TransactionDateTime).sum()} | EmailUpdateDate after txn: {(d.EmailUpdateDate > d.TransactionDateTime).sum()} | EmailUpdateDate unparseable: {d.EmailUpdateDate.isna().sum()}")
    print(f"AddressUpdateDate earliest: {d.AddressUpdateDate.min()}")
    vc = d.AddressUpdateDate.value_counts(); print(f"most common AddressUpdateDate ({vc.index[0]}) appears {vc.iloc[0]:,}x -> looks like a system default, not an account")

    print("\n[4] Time structure")
    t = d.TransactionDateTime; tail = t >= "2013-06-03"
    print(f"span {t.min()} .. {t.max()} | main burst 2013-05-22..2013-06-02 holds {(~tail).sum():,} rows; {tail.sum():,} stragglers spread over 5 months")
    print(f"fraud rate main={d.Fraud[~tail].mean():.4f} tail={d.Fraud[tail].mean():.4f} (tail n_fraud={d.Fraud[tail].sum()}) -> a temporal hold-out is not meaningful here ({(~tail).mean():.1%} of rows fall in 12 days)")
    print(f"distinct ConnectionOrg strings: {d.ConnectionOrg.nunique():,} | distinct ConnectionRegion: {d.ConnectionRegion.nunique()}")
out = buf.getvalue(); print(out)
(ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "reports" / "tables" / "audit.txt").write_text(out, encoding="utf-8")
