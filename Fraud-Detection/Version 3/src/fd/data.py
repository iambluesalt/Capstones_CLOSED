"""Raw data loading. The label and the leaky TransactionKey are returned separately from features."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT.parent / "Version 1" / "MyBank.csv"
RAW_PATH = Path(os.environ.get("FD_RAW_CSV", DEFAULT_RAW))

DATE_FMT = "%d/%m/%Y %H:%M:%S:%f"
DATE_COLS = ["TransactionDateTime", "AddressUpdateDate", "EmailUpdateDate"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load MyBank.csv with 'NA' as missing and the three date columns parsed."""
    d = pd.read_csv(path, na_values=["NA"])
    for c in DATE_COLS:
        d[c] = pd.to_datetime(d[c], format=DATE_FMT, errors="coerce")
    return d
