from __future__ import annotations
import os, io, glob, logging, traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


PG_URL = "postgresql+psycopg2://racer:changeme@localhost:5432/racing"
DATA_DIR = Path(__file__).parent / "data"
CHUNKSIZE = 200_000
FAIL_SAMPLE_ROWS = 1000
NUKE_OLD = False

SCHEMA = "telem"
FACT = f"{SCHEMA}.stream_fast"  # partitioned parent
STAGE = f"{SCHEMA}.stream_fast_stage"  # staging (unpartitioned)

# Quiet Postgres client notices
os.environ.setdefault("PGOPTIONS", "-c client_min_messages=WARNING")

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

READ_CSV_KW = dict(
    chunksize=CHUNKSIZE,
    engine="c",
    low_memory=False,
    dtype={
        "lap": "Int32",
        "meta_event": "string",
        "meta_session": "string",
        "meta_source": "string",
        "meta_time": "string",
        "original_vehicle_id": "string",
        "outing": "string",
        "telemetry_name": "string",
        "telemetry_value": "float64",
        "vehicle_id": "string",
        "vehicle_number": "string",
    },
)

# We keep only columns needed for fast queries + a few helpfuls
USECOLS_RAW = [
    "lap", "outing", "telemetry_name", "telemetry_value",
    "timestamp", "vehicle_id"
]

REQUIRED = ["timestamp", "telemetry_name", "vehicle_id"]

# =========================
# LOGGING
# =========================
logger = logging.getLogger("ingest_fast")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

fh = RotatingFileHandler("ingest_fast.log", maxBytes=5_000_000, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(fh)

# =========================
# SQL INIT
# =========================
INIT_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

-- Dimension tables (dedupe strings -> small ints)
CREATE TABLE IF NOT EXISTS {SCHEMA}.event (
  id           SERIAL PRIMARY KEY,
  race_number  INTEGER NOT NULL,
  track_name   TEXT NOT NULL,
  UNIQUE (race_number, track_name)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.vehicle (
  id    SERIAL PRIMARY KEY,
  code  TEXT NOT NULL UNIQUE   -- e.g., "GR86-002-2"
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.tname (
  id    SERIAL PRIMARY KEY,
  name  TEXT NOT NULL UNIQUE   -- e.g., "accx_can"
);

-- Partitioned fact table (by event)
CREATE TABLE IF NOT EXISTS {FACT} (
  event_id     INTEGER NOT NULL REFERENCES {SCHEMA}.event(id),
  vehicle_id   INTEGER NOT NULL REFERENCES {SCHEMA}.vehicle(id),
  name_id      INTEGER NOT NULL REFERENCES {SCHEMA}.tname(id),
  "timestamp"  timestamptz NOT NULL,
  value        double precision NULL,
  lap          integer NULL,
  outing       smallint NULL,
  relative_time_s double precision NOT NULL
) PARTITION BY LIST (event_id);

-- Staging table for COPY (unpartitioned)
CREATE TABLE IF NOT EXISTS {STAGE} (
  event_id     INTEGER NOT NULL,
  vehicle_id   INTEGER NOT NULL,
  name_id      INTEGER NOT NULL,
  "timestamp"  timestamptz NOT NULL,
  value        double precision NULL,
  lap          integer NULL,
  outing       smallint NULL
);
"""

DROP_OLD_SQL = f"""
-- Drop old wide tables (irreversible)
DROP TABLE IF EXISTS {SCHEMA}.stream_stage CASCADE;
DROP TABLE IF EXISTS {SCHEMA}.stream CASCADE;
"""


def ensure_schema(engine):
    with engine.begin() as conn:
        if NUKE_OLD:
            conn.execute(text(DROP_OLD_SQL))
            logger.info("Dropped old telem.stream/stream_stage.")
        conn.execute(text(INIT_SQL))
    logger.info("Schema & dim tables ready.")


# =========================
# DIM UTILS
# =========================
def get_or_create_event_id(conn, race_number: int, track_name: str) -> int:
    r = conn.execute(text(f"""
        INSERT INTO {SCHEMA}.event (race_number, track_name)
        VALUES (:r, :t)
        ON CONFLICT (race_number, track_name) DO NOTHING
        RETURNING id;
    """), {"r": race_number, "t": track_name}).fetchone()
    if r:
        return r[0]
    r2 = conn.execute(text(f"""
        SELECT id FROM {SCHEMA}.event
        WHERE race_number=:r AND track_name=:t;
    """), {"r": race_number, "t": track_name}).fetchone()
    return r2[0]


def bulk_get_or_create_map(conn, table: str, keycol: str, vals: Iterable[str]) -> Dict[str, int]:
    """Return {value -> id} map for {vehicle.code} or {tname.name}."""
    vals = [v for v in vals if v]  # drop None/empty
    if not vals:
        return {}

    # INSERT missing (use ANY(:vals) instead of UNNEST)
    conn.execute(
        text(f"""
            INSERT INTO {table} ({keycol})
            SELECT DISTINCT v FROM (SELECT unnest(:vals) AS v) AS tmp
            ON CONFLICT ({keycol}) DO NOTHING;
        """),
        {"vals": vals},
    )

    # FETCH ids
    rows = conn.execute(
        text(f"SELECT {keycol}, id FROM {table} WHERE {keycol} = ANY(:vals)"),
        {"vals": vals},
    ).fetchall()

    return {k: i for (k, i) in rows}


# =========================
# PARTITION MGMT
# =========================
def ensure_event_partition(conn, event_id: int):
    part = f"{SCHEMA}.stream_fast_e{event_id}"
    # Create partition if not exists
    conn.execute(text(f"""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = '{SCHEMA}' AND c.relname = 'stream_fast_e{event_id}'
          ) THEN
            EXECUTE 'CREATE TABLE {part} PARTITION OF {FACT} FOR VALUES IN ({event_id})';
          END IF;
        END$$;
    """))
    # Create per-partition index for the hot path
    conn.execute(text(f"""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = '{SCHEMA}' AND c.relname = 'idx_stream_fast_e{event_id}_veh_name_ts'
          ) THEN
            EXECUTE 'CREATE INDEX idx_stream_fast_e{event_id}_veh_name_ts
                     ON {part} (vehicle_id, name_id, "timestamp")';
          END IF;
        END$$;
    """))
    conn.execute(text(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = '{SCHEMA}' AND c.relname = 'idx_stream_fast_e{event_id}_veh_name_rtime'
            ) THEN
                EXECUTE 'CREATE INDEX idx_stream_fast_e{event_id}_veh_name_rtime
                        ON {part} (vehicle_id, name_id, relative_time_s)';
            END IF;
        END$$;
    """))


# =========================
# PARSING / CLEANUP
# =========================
def parse_race_track_from_name(path: Path) -> tuple[int, str]:
    n = path.name.lower()
    if "r1" in n:
        race = 1
    elif "r2" in n:
        race = 2
    else:
        raise ValueError(f"could not find race number in {n}")
    if "indianapolis" in n:
        track = "indianapolis"
    elif "sonoma" in n:
        track = "sonoma"
    elif "cota" in n:
        track = "cota"
    elif "sebring" in n:
        track = "sebring"
    elif "vir" in n:
        track = "vir"
    elif "barber" in n:
        track = "barber"
    else:
        raise ValueError(f"could not find track name in {n}")
    return race, track


def _clean_timestamp_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(">", ":", regex=False)
    s = s.str.replace("::", ":", regex=False)
    s = s.str.replace(r"[^0-9T:\+\-\. Z]", "", regex=True)
    return pd.to_datetime(s, errors="coerce", utc=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    for col in USECOLS_RAW:
        if col not in df.columns:
            df[col] = pd.NA

    # Text sanitization
    for col in ["telemetry_name", "vehicle_id", "outing"]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "NA", "NaN", "null", "None"]), col] = pd.NA

    # Timestamp
    if not is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = _clean_timestamp_series(df["timestamp"])
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Numerics
    df["lap"] = pd.to_numeric(df["lap"], errors="coerce").astype("Int64")
    # outing -> smallint
    df["outing"] = pd.to_numeric(df["outing"], errors="coerce").astype("Int16")

    # Required rows only
    good = df["timestamp"].notna() & df["telemetry_name"].notna() & df["vehicle_id"].notna()
    dropped = (~good).sum()
    if dropped:
        logger.warning(f"dropped {dropped} rows missing required fields")
    df = df.loc[good, USECOLS_RAW].copy()

    return df


# =========================
# COPY/MERGE
# =========================
def copy_stage(conn, df_ids: pd.DataFrame):
    """df_ids has int IDs + timestamp/value/lap/outing"""
    if df_ids.empty:
        return
    buf = io.StringIO()
    df_ids.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.connection.cursor() as cur:
        cur.copy_expert(
            f"COPY {STAGE} (event_id, vehicle_id, name_id, \"timestamp\", value, lap, outing) "
            f"FROM STDIN WITH (FORMAT CSV)",
            buf
        )


def finalize_event_relative_time(conn, event_id: int):
    conn.execute(text(f"""
        WITH mins AS (
          SELECT vehicle_id, MIN("timestamp") AS first_ts
          FROM {FACT}
          WHERE event_id = :e
          GROUP BY vehicle_id
        )
        UPDATE {FACT} f
        SET relative_time_s = EXTRACT(EPOCH FROM (f."timestamp" - m.first_ts))
        FROM mins m
        WHERE f.event_id = :e AND f.vehicle_id = m.vehicle_id;
    """), {"e": event_id})


def merge_stage(conn):
    # Compute per-(event_id, vehicle_id) first_ts using FACT ∪ STAGE, then insert with relative_time_s
    conn.execute(text(f"""
        WITH firsts AS (
          SELECT event_id, vehicle_id, MIN(min_ts) AS first_ts
          FROM (
            SELECT event_id, vehicle_id, MIN("timestamp") AS min_ts
            FROM {FACT}
            GROUP BY event_id, vehicle_id
            UNION ALL
            SELECT event_id, vehicle_id, MIN("timestamp") AS min_ts
            FROM {STAGE}
            GROUP BY event_id, vehicle_id
          ) u
          GROUP BY event_id, vehicle_id
        )
        INSERT INTO {FACT} (event_id, vehicle_id, name_id, "timestamp", value, lap, outing, relative_time_s)
        SELECT s.event_id, s.vehicle_id, s.name_id, s."timestamp", s.value, s.lap, s.outing,
               EXTRACT(EPOCH FROM (s."timestamp" - f.first_ts)) AS relative_time_s
        FROM {STAGE} s
        JOIN firsts f
          ON f.event_id = s.event_id AND f.vehicle_id = s.vehicle_id;

        TRUNCATE {STAGE};
    """))


# =========================
# MAIN INGEST
# =========================
def ingest_file(engine, path: Path):
    race, track = parse_race_track_from_name(path)
    with engine.begin() as conn:
        event_id = get_or_create_event_id(conn, race, track)
        ensure_event_partition(conn, event_id)

    logger.info(f"START {path.name} -> event_id={event_id}")

    for idx, chunk in enumerate(pd.read_csv(path, **READ_CSV_KW), start=1):
        chunk = normalize_columns(chunk)
        if chunk.empty:
            logger.info(f"SKIP {path.name} [chunk {idx}] rows=0")
            continue

        # build ID maps (vehicles + telemetry names)
        veh_codes = set(chunk["vehicle_id"].dropna().unique().tolist())
        names = set(chunk["telemetry_name"].dropna().unique().tolist())
        with engine.begin() as conn:
            veh_map = bulk_get_or_create_map(conn, f"{SCHEMA}.vehicle", "code", veh_codes)
            name_map = bulk_get_or_create_map(conn, f"{SCHEMA}.tname", "name", names)

        # map to integer IDs
        df_ids = pd.DataFrame({
            "event_id": event_id,
            "vehicle_id": chunk["vehicle_id"].map(veh_map).astype("Int64"),
            "name_id": chunk["telemetry_name"].map(name_map).astype("Int64"),
            "timestamp": chunk["timestamp"],
            "value": pd.to_numeric(chunk["telemetry_value"], errors="coerce"),
            "lap": chunk["lap"].astype("Int64"),
            "outing": chunk["outing"].astype("Int16"),
        })

        # Any rows that somehow failed mapping? drop them.
        ok = df_ids["vehicle_id"].notna() & df_ids["name_id"].notna() & df_ids["timestamp"].notna()
        dropped = (~ok).sum()
        if dropped:
            logger.warning(f"dropped {dropped} rows after ID mapping")
        df_ids = df_ids.loc[ok].copy()

        try:
            with engine.begin() as conn:
                copy_stage(conn, df_ids)
            with engine.begin() as conn:
                merge_stage(conn)
            logger.info(f"OK {path.name} [chunk {idx}] rows={len(df_ids)}")
        except SQLAlchemyError as e:
            pgcode = getattr(getattr(e, "orig", None), "pgcode", None)
            logger.error(f"DB ERROR file={path.name} chunk={idx} pgcode={pgcode}")
            # save sample
            out = f"_failed_fast_{path.stem}_chunk{idx}.csv"
            df_ids.head(FAIL_SAMPLE_ROWS).to_csv(out, index=False)
            logger.error(f"Saved failing sample to {out}")
            logger.debug("".join(traceback.format_exception(e)))
            # continue or raise:
            # raise

    with engine.begin() as conn:
        finalize_event_relative_time(conn, event_id)

    logger.info(f"DONE  {path.name}")


def discover_telemetry_files(base: Path) -> list[Path]:
    paths = [Path(p) for p in glob.glob(str(base / "**/*.csv"), recursive=True)]
    return [p for p in paths if "telemetry" in p.name.lower()]


def main():
    engine = create_engine(PG_URL, echo=False, pool_pre_ping=True, future=True)
    ensure_schema(engine)
    files = sorted(discover_telemetry_files(DATA_DIR))
    if not files:
        logger.warning(f"No telemetry CSVs found under {DATA_DIR}")
        return
    for p in files:
        ingest_file(engine, p)


if __name__ == "__main__":
    main()
