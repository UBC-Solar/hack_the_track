import os, time, json, datetime as dt
from typing import Dict, Set, Tuple, Optional
from confluent_kafka import Consumer, KafkaException
import psycopg2
from psycopg2 import sql

BROKER     = os.getenv("BROKER", "redpanda:9092")
TOPIC      = os.getenv("TOPIC", "telem.stream_fast.raw")
GROUP_ID   = os.getenv("GROUP_ID", "tick-snapshots")
PG_DSN     = os.getenv("PG_DSN", "postgresql://telemetry:telemetry@tickdb:5434/telemetry")
TICK_TABLE = os.getenv("TICK_TABLE", "telem_tick")
TICK_SECS  = float(os.getenv("TICK_SECS", "0.1"))
NAMES_ALLOWLIST = {s.strip() for s in os.getenv("NAMES_ALLOWLIST", "").split(",") if s.strip()}

# Holds the latest value by (vehicle_id -> {name: value})
latest_by_vehicle: Dict[int, Dict[str, Optional[float]]] = {}
# Set of all telemetry names observed (filtered by allowlist if provided)
observed_names: Set[str] = set()

def to_float_or_none(v):
    # Incoming payload is JSON from your producer; telemetry_value often numeric
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None  # non-numerics are skipped; adjust if you want TEXT columns

def ensure_columns(conn, names: Set[str]):
    """Add columns to wide table if they don't exist. Columns are double precision."""
    if not names:
        return
    with conn.cursor() as cur:
        # Fetch existing column names
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = current_schema()
        """, (TICK_TABLE,))
        existing = {r[0] for r in cur.fetchall()}

        to_add = [n for n in names if n not in existing and n not in ("ts","vehicle_id")]
        for n in to_add:
            # safe identifier quoting with psycopg2.sql
            stmt = sql.SQL("ALTER TABLE {} ADD COLUMN {} double precision NULL").format(
                sql.Identifier(TICK_TABLE),
                sql.Identifier(n)
            )
            cur.execute(stmt)
            print(f"[schema] Added column {n}")
    conn.commit()

def upsert_tick(conn, ts: dt.datetime, vehicle_id: int, row: Dict[str, Optional[float]]):
    """
    Insert/Upsert one row: (ts, vehicle_id, <telemetry columns...>).
    Unknown names are ignored unless ensure_columns() added them first.
    """
    # Ensure base row exists; then update the columns we have values for
    with conn.cursor() as cur:
        # Insert base (ts, vehicle_id) if not exists
        cur.execute(
            sql.SQL("INSERT INTO {} (ts, vehicle_id) VALUES (%s, %s) ON CONFLICT (ts, vehicle_id) DO NOTHING")
               .format(sql.Identifier(TICK_TABLE)),
            (ts, vehicle_id)
        )

        if row:
            cols = [sql.Identifier(k) for k in row.keys()]
            sets = [sql.SQL("{} = %s").format(c) for c in cols]
            values = [row[k] for k in row.keys()]
            q = sql.SQL("UPDATE {} SET ").format(sql.Identifier(TICK_TABLE)) + \
                sql.SQL(", ").join(sets) + \
                sql.SQL(" WHERE ts = %s AND vehicle_id = %s")
            cur.execute(q, (*values, ts, vehicle_id))
    conn.commit()

def run():
    # Kafka consumer
    conf = {
        "bootstrap.servers": BROKER,
        "group.id": GROUP_ID,
        "enable.auto.commit": True,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])

    # Postgres connection
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False

    tick_next = time.monotonic()
    print(f"[tick] interval = {TICK_SECS}s, writing to table {TICK_TABLE}")

    try:
        while True:
            # Poll Kafka quickly to keep latency low
            msg = consumer.poll(timeout=0.01)
            if msg is not None:
                if msg.error():
                    raise KafkaException(msg.error())
                try:
                    payload = json.loads(msg.value())
                    vid = int(payload.get("vehicle_id"))
                    name = str(payload.get("telemetry_name"))
                    val  = to_float_or_none(payload.get("telemetry_value"))

                    if NAMES_ALLOWLIST and name not in NAMES_ALLOWLIST:
                        pass  # ignore
                    else:
                        latest_by_vehicle.setdefault(vid, {})[name] = val
                        observed_names.add(name)
                except Exception as e:
                    print(f"[warn] bad message: {e}")

            # Tick?
            now = time.monotonic()
            if now >= tick_next:
                wall_ts = dt.datetime.now(dt.timezone.utc)

                # Make sure table has required columns before writing
                ensure_columns(conn, observed_names)

                # For each vehicle, write one row with current latest values
                for vid, m in latest_by_vehicle.items():
                    row = {k: v for k, v in m.items() if v is not None}
                    upsert_tick(conn, wall_ts, vid, row)

                tick_next += TICK_SECS

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        conn.close()

if __name__ == "__main__":
    run()