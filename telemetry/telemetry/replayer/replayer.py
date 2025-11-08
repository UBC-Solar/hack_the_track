import os, time, json, datetime as dt, decimal
from confluent_kafka import Producer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

BROKER = os.getenv("BROKER")
TOPIC  = os.getenv("TOPIC")
DATABASE_URL = os.environ["DATABASE_URL"]
VEHICLE_ID = int(os.getenv("VEHICLE_ID"))
RACE_NUMBER = int(os.getenv("RACE_NUMBER"))
TRACK_NAME = os.getenv("TRACK_NAME")  # required for partition pruning
SPEED = float(os.getenv("SPEED_MULTIPLIER"))

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

def load_rows():
    if not TRACK_NAME:
        raise RuntimeError("TRACK_NAME env var must be set (e.g. TRACK_NAME='circuit-of-the-americas').")

    engine = create_engine(DATABASE_URL, future=True)
    with engine.connect() as con:
        # Query parent table; partition pruning will hit the right child.
        sql = text("""
            SELECT
                f.vehicle_id,
                e.race_number,
                n.name        AS telemetry_name,
                f.value       AS telemetry_value,
                f.timestamp   AS ts,
                e.track_name  AS track,
                v.code        AS vehicle_number
            FROM telem.stream_fast f
            JOIN telem.event   e ON e.id = f.event_id
            JOIN telem.tname   n ON n.id = f.name_id
            JOIN telem.vehicle v ON v.id = f.vehicle_id
            WHERE f.vehicle_id   = :v
              AND e.race_number  = :r
              AND e.track_name   = :t
            ORDER BY f.timestamp ASC
        """)
        rows = con.execute(sql, {"v": VEHICLE_ID, "r": int(RACE_NUMBER), "t": TRACK_NAME}).mappings().all()
    return rows

def to_json_safe(v):
    if isinstance(v, (dt.datetime, dt.date)): return v.isoformat()
    if isinstance(v, decimal.Decimal): return float(v)
    return v

def main():
    rows = load_rows()
    if not rows:
        print("No rows found.")
        return

    first_ts = rows[0]["ts"]
    if isinstance(first_ts, str):
        first_ts = dt.datetime.fromisoformat(first_ts)
    start = dt.datetime.now(dt.timezone.utc)

    print(f"Streaming {len(rows)} rows (speed×{SPEED}) …")

    for r in rows:
        ts = r["ts"]
        if isinstance(ts, str):
            ts = dt.datetime.fromisoformat(ts)

        # pacing
        orig_elapsed = (ts - first_ts).total_seconds()
        target_elapsed = orig_elapsed / max(SPEED, 1e-9)
        now_elapsed = (dt.datetime.now(dt.timezone.utc) - start).total_seconds()
        delay = target_elapsed - now_elapsed
        if delay > 0:
            time.sleep(delay)

        payload = {
            "vehicle_id": r["vehicle_id"],
            "race_number": r["race_number"],
            "telemetry_name": r["telemetry_name"],
            "telemetry_value": to_json_safe(r["telemetry_value"]),
            "ts": to_json_safe(ts),
            "track": r["track"],
            "vehicle_number": r["vehicle_number"],
        }

        print(f"Sending to Kafka: {json.dumps(payload)}")

        producer.produce(
            topic=TOPIC,
            key=str(r["vehicle_id"]),            # keeps per-vehicle ordering
            value=json.dumps(payload),           # JSON payload
            on_delivery=delivery_report,
        )
        producer.poll(0)

    producer.flush()
    print("✅ Replay finished.")

if __name__ == "__main__":
    main()
