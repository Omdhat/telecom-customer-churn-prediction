"""This is a shared feature code. used by model.py, eda.py and app.py, so the exact same logic runs everywhere."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CHARGE_COLS = ["totaldaycharge", "totalevecharge", "totalnightcharge", "totalintlcharge"]
CHARGE_RATES = {"day": 0.17, "eve": 0.085, "night": 0.045, "intl": 0.27}

INPUT_COLUMNS = [
    "accountlength", "internationalplan", "voicemailplan", "numbervmailmessages",
    "totaldayminutes", "totaldaycalls", "totaleveminutes", "totalevecalls",
    "totalnightminutes", "totalnightcalls", "totalintlminutes", "totalintlcalls",
    "numbercustomerservicecalls",
]

def to_binary(series):
    if series.dtype == object or str(series.dtype).startswith("str"):
        return series.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).astype(int)
    return series.astype(int)

def basic_features(X):
    X = X.copy()
    for col in ["internationalplan", "voicemailplan"]:
        X[col] = to_binary(X[col])
    return X

def add_features(X):
    X = basic_features(X).drop(columns=CHARGE_COLS, errors="ignore")
    
    X["total_minutes"] = (X["totaldayminutes"] + X["totaleveminutes"]
                          + X["totalnightminutes"] + X["totalintlminutes"])

    X["total_charge"] = (X["totaldayminutes"] * CHARGE_RATES["day"]
                         + X["totaleveminutes"] * CHARGE_RATES["eve"]
                         + X["totalnightminutes"] * CHARGE_RATES["night"]
                         + X["totalintlminutes"] * CHARGE_RATES["intl"])

    X["high_service_calls"] = (X["numbercustomerservicecalls"] >= 4).astype(int)
    return X

def load_data(path="churn_data.csv"):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = raw.columns.str.strip()
    id_cols = [c for c in raw.columns if c.lower() == "customerid"]

    if not id_cols:
        shutil.copyfile(path, "churn_data_original.csv")      # one-time backup
        raw.insert(0, "CustomerID", [f"CUST{i:05d}" for i in range(1, len(raw) + 1)])
        raw.to_csv(path, index=False)
    elif raw[id_cols[0]].duplicated().any() or (raw[id_cols[0]].str.strip() == "").any():
        raise ValueError("CustomerID column has duplicate or empty values.")

    df = pd.read_csv(path)
    df.columns = ["CustomerID" if c.strip().lower() == "customerid" else c.strip().lower()
                  for c in df.columns]
    return df