import os
import time
import json
import datetime as dt
import decimal
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from confluent_kafka import Producer

load_dotenv()

# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Stream telemetry data from replay CSV.")
parser.add_argument(
    "--skip-prompt",
    action="store_true",
    help="Skip the continue prompt and start streaming immediately.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------
# ENV / CONFIG
# ---------------------------------------------------------------------
BROKER = os.getenv("BROKER")
TOPIC = os.getenv("TOPIC")

RACE_NUMBER = int(os.getenv("RACE_NUMBER", "1"))
TRACK_NAME = os.getenv("TRACK_NAME") or "unknown_track"
SPEED = float(os.getenv("SPEED_MULTIPLIER", "1.0"))

# NOTE: these are now *string* vehicle IDs (codes) matching the CSV, e.g. "GR86-022-13"
VEHICLE_ID_ENV = os.getenv("VEHICLE_ID")
VEHICLE_ID = VEHICLE_ID_ENV if VEHICLE_ID_ENV not in (None, "", "None") else None

EXCLUDE_VEHICLE_IDS = [
    s for s in os.getenv("EXCLUDE_VEHICLE_IDS", "").replace(" ", "").split(",")
    if s.strip()
]
if EXCLUDE_VEHICLE_IDS:
    print(f"[filter] excluding vehicle_ids (codes): {EXCLUDE_VEHICLE_IDS}")

# Where the replay CSV lives (same folder as this script)
BASE_DIR = Path(__file__).parent
REPLAY_CSV = BASE_DIR / "replayer_data" / f"replay_ready_r{RACE_NUMBER}_{TRACK_NAME}.csv"

CHUNKSIZE = 200_000

# ---------------------------------------------------------------------
# Kafka producer
# ---------------------------------------------------------------------
def delivery_report(err, msg):
    if err:
        print(f"❌ Delivery failed: {err}")


producer_conf = {
    "bootstrap.servers": BROKER,
    "linger.ms": 10,
    "batch.num.messages": 10000,
    "compression.type": "zstd",
    "acks": "1",
}
producer = Producer(producer_conf)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def to_json_safe(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


def get_time_bounds(csv_path: Path):
    """
    Scan the replay CSV (in chunks) and compute:
      - min relative_time_s
      - max relative_time_s
      - row count
    with VEHICLE_ID / EXCLUDE_VEHICLE_IDS filters applied.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Replay CSV not found: {csv_path}")

    tmin = None
    tmax = None
    nrows = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        # Filter by vehicle if requested
        if VEHICLE_ID is not None:
            chunk = chunk[chunk["vehicle_id"] == VEHICLE_ID]

        if EXCLUDE_VEHICLE_IDS:
            chunk = chunk[~chunk["vehicle_id"].isin(EXCLUDE_VEHICLE_IDS)]

        if chunk.empty:
            continue

        rel = chunk["relative_time_s"].astype(float)
        cmin = rel.min()
        cmax = rel.max()

        if tmin is None or cmin < tmin:
            tmin = cmin
        if tmax is None or cmax > tmax:
            tmax = cmax

        nrows += len(chunk)

    return tmin, tmax, nrows


def stream_rows(csv_path: Path):
    """
    Yield rows from the replay CSV in order.
    The replay CSV is already globally sorted by relative_time_s.
    """
    for chunk in pd.read_csv(
        csv_path,
        chunksize=CHUNKSIZE,
        parse_dates=["timestamp"],  # parse timestamp to datetime
    ):
        # Filter by vehicle
        if VEHICLE_ID is not None:
            chunk = chunk[chunk["vehicle_id"] == VEHICLE_ID]

        if EXCLUDE_VEHICLE_IDS:
            chunk = chunk[~chunk["vehicle_id"].isin(EXCLUDE_VEHICLE_IDS)]

        if chunk.empty:
            continue

        # Ensure sorted within chunk (should already be, but cheap safety)
        chunk = chunk.sort_values(["relative_time_s", "vehicle_id", "timestamp"])

        for row in chunk.itertuples(index=False):
            # Row fields: vehicle_id, lap, outing, telemetry_name, telemetry_value,
            #             timestamp, first_ts (optional), relative_time_s,
            #             race_number, track
            yield row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not REPLAY_CSV.exists():
        raise FileNotFoundError(
            f"Replay CSV not found: {REPLAY_CSV}\n"
            f"Make sure you ran the CSV processor to create it."
        )

    tmin, tmax, nrows = get_time_bounds(REPLAY_CSV)
    if nrows == 0 or tmin is None or tmax is None:
        print("No rows found after applying filters.")
        return

    print(f"Using replay CSV: {REPLAY_CSV}")
    print(f"Rows: {nrows}")
    print(f"First relative time: {tmin:.6f} s")
    print(f"Last  relative time: {tmax:.6f} s (duration)")

    if VEHICLE_ID is not None:
        scope = f"vehicle_id={VEHICLE_ID}"
    else:
        scope = "ALL vehicles"

    print(f"Race #{RACE_NUMBER} @ {TRACK_NAME} ({scope})")

    if not args.skip_prompt:
        input("Continue? ")

    start = dt.datetime.now(dt.timezone.utc)
    print(f"Streaming {nrows} rows (speed×{SPEED}) …")

    for r in stream_rows(REPLAY_CSV):
        # unpack row object (namedtuple-like)
        vehicle_id = r.vehicle_id
        telemetry_name = r.telemetry_name
        telemetry_value = r.telemetry_value
        ts = r.timestamp
        rel_time = float(r.relative_time_s)
        lap = r.lap if hasattr(r, "lap") else None
        track = getattr(r, "track", TRACK_NAME)
        race_number = getattr(r, "race_number", RACE_NUMBER)

        # pacing: align so relative_time_s = elapsed time * SPEED
        target_elapsed = rel_time / max(SPEED, 1e-9)
        now_elapsed = (dt.datetime.now(dt.timezone.utc) - start).total_seconds()
        delay = target_elapsed - now_elapsed
        if delay > 0:
            time.sleep(delay)

        payload = {
            "vehicle_id": vehicle_id,
            "race_number": race_number,
            "telemetry_name": telemetry_name,
            "telemetry_value": to_json_safe(telemetry_value),
            "ts": to_json_safe(ts),
            "relative_time_s": rel_time,
            "track": track,
            # We don't have a separate 'vehicle_number' column in the replay CSV;
            # reuse vehicle_id for now. If you add a vehicle_number column in the
            # CSV builder, you can swap it in here.
            "vehicle_number": vehicle_id,
            "lap": int(lap) if lap is not None and not pd.isna(lap) else None,
        }

        producer.produce(
            topic=TOPIC,
            key=str(vehicle_id),  # preserves per-vehicle ordering
            value=json.dumps(payload),
            on_delivery=delivery_report,
        )
        producer.poll(0)

    producer.flush()
    print("Replay finished.")


if __name__ == "__main__":
    main()