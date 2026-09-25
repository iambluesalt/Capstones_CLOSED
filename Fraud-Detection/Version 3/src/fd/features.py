"""Feature engineering that never touches the label, so it is safe to run before any split.

TransactionKey is deliberately NOT a feature: in MyBank.csv every fraud row has key >= 134790 and every
non-fraud row has key <= 134789 (AUC = 1.0). It is a dataset-construction artifact, not a signal.
"""
import numpy as np
import pandas as pd

CAT_COLS = [
    "ConnectionOrg", "ConnectionType", "ConnectionSpeed", "V6CF", "channel",
    "webSessOS", "webSessWebBrowser", "Region", "State", "Country",
]
NAN_FLAG_COLS = ["AreaCode", "LastLong", "LastLat", "CurrentLat", "IsOldDevice", "WebSessionRetail", "MainEntityUse"]
EARTH_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def _lump(s: pd.Series, min_count: int) -> pd.Series:
    vc = s.value_counts()
    keep = vc[vc >= min_count].index
    return s.where(s.isin(keep), "other")


def build_features(raw: pd.DataFrame, org_min_count: int = 300):
    """Return (X, y). X has category dtype for CAT_COLS; every other column is numeric (NaN allowed)."""
    d = raw.copy()
    y = d["Fraud"].astype(int).to_numpy()
    X = pd.DataFrame(index=d.index)

    # --- raw numerics -------------------------------------------------------------------------------
    for c in ["V1CF", "V2CF", "V3CF", "V4CF", "V5CF", "AreaCode", "MainEntityUse", "IsOldDevice"]:
        X[c] = d[c].astype(float)
    tz = d["TimeZone"].where(d["TimeZone"].abs() <= 14)  # 999 is a sentinel
    X["TimeZone"] = tz
    X["tz_sentinel"] = (d["TimeZone"].abs() > 14).astype(int)

    # --- missingness --------------------------------------------------------------------------------
    for c in NAN_FLAG_COLS:
        X[f"{c}_isna"] = d[c].isna().astype(int)
    X["n_missing"] = d[NAN_FLAG_COLS].isna().sum(axis=1)
    X["partial_geo"] = d[["LastLong", "LastLat", "CurrentLong", "CurrentLat"]].isna().sum(axis=1)

    # --- geography ----------------------------------------------------------------------------------
    for c in ["LastLong", "LastLat", "CurrentLong", "CurrentLat"]:
        X[c] = d[c]
    X["geo_dist_km"] = _haversine_km(d["LastLat"], d["LastLong"], d["CurrentLat"], d["CurrentLong"])
    X["geo_moved"] = (X["geo_dist_km"] > 1).astype(float).where(X["geo_dist_km"].notna())
    X["tz_long_gap"] = (d["CurrentLong"] / 15 - tz).abs()  # local clock vs. longitude-implied offset
    X["areacode_eq_v4"] = (d["AreaCode"] == d["V4CF"]).astype(float).where(d["AreaCode"].notna())
    X["v5_zero"] = (d["V5CF"] == 0).astype(int)
    X["v4_zero"] = (d["V4CF"] == 0).astype(int)

    # --- time ---------------------------------------------------------------------------------------
    t = d["TransactionDateTime"]
    X["hour"] = t.dt.hour
    X["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    X["dow"] = t.dt.dayofweek
    X["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    X["days_since_start"] = (t - t.min()).dt.total_seconds() / 86400
    X["in_tail_period"] = (t >= "2013-06-03").astype(int)  # ~3.6K stragglers after the main 12-day burst
    for name, col in [("addr", "AddressUpdateDate"), ("email", "EmailUpdateDate")]:
        u = d[col]
        X[f"{name}_age_days"] = (t - u).dt.total_seconds() / 86400
        X[f"{name}_update_after_txn"] = (u > t).astype(int)
        X[f"{name}_update_hour"] = u.dt.hour
        X[f"{name}_update_dow"] = u.dt.dayofweek
        X[f"{name}_update_year"] = u.dt.year
        X[f"{name}_age_lt_1d"] = ((t - u).dt.total_seconds() < 86400).astype(int)
    X["addr_email_gap_days"] = (d["AddressUpdateDate"] - d["EmailUpdateDate"]).dt.total_seconds() / 86400
    X["email_date_isna"] = d["EmailUpdateDate"].isna().astype(int)

    # --- entity / velocity counts (label-free) ------------------------------------------------------
    addr_key = d["AddressUpdateDate"].astype("int64")
    email_key = d["EmailUpdateDate"].astype("int64")
    X["addr_n"] = addr_key.map(addr_key.value_counts())
    X["email_n"] = email_key.map(email_key.value_counts())
    pair = addr_key.astype(str) + "|" + email_key.astype(str)
    X["pair_n"] = pair.map(pair.value_counts())
    X["txn_same_sec_n"] = t.map(t.value_counts())
    org_norm = d["ConnectionOrg"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    X["org_n"] = org_norm.map(org_norm.value_counts())
    X["addr_org_n"] = (addr_key.astype(str) + org_norm).map((addr_key.astype(str) + org_norm).value_counts())

    # --- categoricals -------------------------------------------------------------------------------
    reg = d["ConnectionRegion"].str.split("@", expand=True)
    cats = pd.DataFrame({
        "ConnectionOrg": _lump(org_norm, org_min_count),
        "ConnectionType": d["ConnectionType"],
        "ConnectionSpeed": d["ConnectionSpeed"],
        "V6CF": _lump(d["V6CF"].str.strip().str.lower().fillna("na"), 100),
        "channel": d["channel"],
        "webSessOS": d["webSessOS"],
        "webSessWebBrowser": _lump(d["webSessWebBrowser"], 100),
        "Region": reg[0],
        "State": reg[1],
        "Country": _lump(reg[2], 100),
    })
    for c in CAT_COLS:
        X[c] = cats[c].fillna("na").astype("category")
    return X, y
