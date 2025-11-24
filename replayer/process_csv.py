from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw_data"         # only one CSV here
CHUNKSIZE = 200_000

RACE_NUMBER = int(os.getenv("RACE_NUMBER", "1"))
TRACK_NAME = os.getenv("TRACK_NAME") or "unknown_track"

OUT_FINAL = BASE_DIR / "replayer_data" / f"replay_ready_r{RACE_NUMBER}_{TRACK_NAME}.csv"

# -------------------------
# CSV setup
# -------------------------

READ_KW = dict(
    chunksize=CHUNKSIZE,
    engine="c",
    low_memory=False,
    dtype={
        "lap": "Int32",
        "outing": "string",
        "telemetry_name": "string",
        "telemetry_value": "float64",
        "vehicle_id": "string",
        "vehicle_number": "string",
    },
)

USECOLS_RAW = [
    "lap",
    "outing",
    "telemetry_name",
    "telemetry_value",
    "timestamp",
    "vehicle_id",
]


def get_single_csv() -> Path:
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR}")
    if len(csvs) > 1:
        raise RuntimeError(f"Expected one CSV, found {len(csvs)}")
    return csvs[0]


# -------------------------
# Normalization
# -------------------------

def _clean_timestamp_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(">", ":", regex=False)
    s = s.str.replace("::", ":", regex=False)
    s = s.str.replace(r"[^0-9T:\+\-\. Z]", "", regex=True)
    return pd.to_datetime(s, errors="coerce", utc=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    for c in USECOLS_RAW:
        if c not in df:
            df[c] = pd.NA

    for c in ["telemetry_name", "vehicle_id", "outing"]:
        df[c] = df[c].astype("string").str.strip()
        df.loc[df[c].isin(["", "NA", "NaN", "null", "None"]), c] = pd.NA

    if not is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = _clean_timestamp_series(df["timestamp"])
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    df["lap"] = pd.to_numeric(df["lap"], errors="coerce").astype("Int64")
    df["outing"] = pd.to_numeric(df["outing"], errors="coerce").astype("Int16")

    good = (
        df["timestamp"].notna()
        & df["telemetry_name"].notna()
        & df["vehicle_id"].notna()
    )
    return df.loc[good, USECOLS_RAW].copy()


def iter_normalized_chunks(csv_path: Path):
    for chunk in pd.read_csv(csv_path, **READ_KW):
        chunk = normalize(chunk)
        if not chunk.empty:
            yield chunk


# -------------------------
# Pass 1 — find per-vehicle first timestamps
# -------------------------

def compute_first_ts(csv_path: Path) -> Dict[str, pd.Timestamp]:
    first_ts = {}
    for chunk in iter_normalized_chunks(csv_path):
        grouped = chunk.groupby("vehicle_id")["timestamp"].min()
        for veh, ts in grouped.items():
            if veh not in first_ts or ts < first_ts[veh]:
                first_ts[veh] = ts
    return first_ts


# -------------------------
# Pass 2 — load entire CSV, compute relative times, sort
# -------------------------

def build_replay_csv(csv_path: Path, out_path: Path):
    first_ts = compute_first_ts(csv_path)

    print("Loading full CSV into memory for sorting…")
    df_full = pd.concat(iter_normalized_chunks(csv_path), ignore_index=True)
    print(f"Loaded {len(df_full):,} rows")

    df_full["first_ts"] = df_full["vehicle_id"].map(first_ts)
    df_full["relative_time_s"] = (
        df_full["timestamp"] - df_full["first_ts"]
    ).dt.total_seconds()

    df_full["race_number"] = RACE_NUMBER
    df_full["track"] = TRACK_NAME

    df_full.sort_values(
        ["relative_time_s", "vehicle_id", "timestamp"],
        inplace=True,
        kind="mergesort",
    )

    df_full.to_csv(out_path, index=False)
    print(f"Replay CSV written: {out_path}")


def main():
    csv_path = get_single_csv()
    print(f"Using raw telemetry file: {csv_path}")

    build_replay_csv(csv_path, OUT_FINAL)


if __name__ == "__main__":
    main()
