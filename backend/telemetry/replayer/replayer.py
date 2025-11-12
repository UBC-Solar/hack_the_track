import os, time, json, datetime as dt, decimal
from dotenv import load_dotenv
from confluent_kafka import Producer
from sqlalchemy import create_engine, text, bindparam  # add bindparam

load_dotenv()

EXCLUDE_VEHICLE_IDS = [
    int(s) for s in os.getenv("EXCLUDE_VEHICLE_IDS", "").replace(" ", "").split(",")
    if s.strip()
]
if EXCLUDE_VEHICLE_IDS:
    print(f"[filter] excluding vehicle_ids: {EXCLUDE_VEHICLE_IDS}")

BROKER = os.getenv("BROKER")
TOPIC = os.getenv("TOPIC")
DATABASE_URL = os.environ["DATABASE_URL"]
RACE_NUMBER = int(os.getenv("RACE_NUMBER"))
TRACK_NAME = os.getenv("TRACK_NAME")  # required
SPEED = float(os.getenv("SPEED_MULTIPLIER"))

# Optional: if you set VEHICLE_ID, we'll filter to that one; otherwise ALL vehicles
VEHICLE_ID_ENV = os.getenv("VEHICLE_ID")
VEHICLE_ID     = int(VEHICLE_ID_ENV) if VEHICLE_ID_ENV not in (None, "", "None") else None

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

def get_time_bounds():
    if not TRACK_NAME:
        raise RuntimeError("TRACK_NAME must be set (e.g. TRACK_NAME='circuit-of-the-americas').")

    engine = create_engine(DATABASE_URL, future=True)
    with engine.connect() as con:
        params = {"r": RACE_NUMBER, "t": TRACK_NAME}
        clauses = ["e.race_number = :r", "e.track_name = :t"]

        if VEHICLE_ID is not None:
            clauses.append("f.vehicle_id = :v")
            params["v"] = VEHICLE_ID

        excl_bind = None
        if EXCLUDE_VEHICLE_IDS:
            excl_bind = bindparam("excl_ids", expanding=True)
            clauses.append("f.vehicle_id NOT IN :excl_ids")

        where_sql = " AND ".join(clauses)
        sql = f"""
            WITH base AS (
              SELECT
                COALESCE(
                  f.relative_time_s,
                  EXTRACT(EPOCH FROM (f."timestamp" - MIN(f."timestamp") OVER (PARTITION BY f.event_id, f.vehicle_id)))
                ) AS rtime
              FROM telem.stream_fast f
              JOIN telem.event e ON e.id = f.event_id
              WHERE {where_sql}
            )
            SELECT MIN(rtime) AS tmin, MAX(rtime) AS tmax, COUNT(*) AS n FROM base;
        """
        stmt = text(sql)
        if excl_bind is not None:
            stmt = stmt.bindparams(excl_bind)

        exec_params = {**params}
        if EXCLUDE_VEHICLE_IDS:
            exec_params["excl_ids"] = EXCLUDE_VEHICLE_IDS

        row = con.execute(stmt, exec_params).mappings().first()
        return row["tmin"], row["tmax"], row["n"]


def stream_rows():
    """Yield rows ordered by per-vehicle relative time (seconds since that vehicle's first ts)."""
    engine = create_engine(DATABASE_URL, future=True)
    with engine.connect().execution_options(stream_results=True, yield_per=50_000) as con:
        params = {"r": RACE_NUMBER, "t": TRACK_NAME}
        clauses = ["e.race_number = :r", "e.track_name  = :t"]

        if VEHICLE_ID is not None:
            clauses.append("f.vehicle_id = :v")
            params["v"] = VEHICLE_ID

        excl_bind = None
        if EXCLUDE_VEHICLE_IDS:
            excl_bind = bindparam("excl_ids", expanding=True)
            clauses.append("f.vehicle_id NOT IN :excl_ids")

        where_sql = " AND ".join(clauses)
        # rtime: use stored relative_time_s; fallback to windowed min(ts) if any legacy rows exist
        sql_txt = f"""
            SELECT
                f.vehicle_id,
                e.race_number,
                n.name        AS telemetry_name,
                f.value       AS telemetry_value,
                f."timestamp" AS ts,
                e.track_name  AS track,
                v.code        AS vehicle_number,
                f.lap         AS lap,
                COALESCE(
                  f.relative_time_s,
                  EXTRACT(EPOCH FROM (f."timestamp" - MIN(f."timestamp") OVER (PARTITION BY f.event_id, f.vehicle_id)))
                ) AS rtime
            FROM telem.stream_fast f
            JOIN telem.event   e ON e.id = f.event_id
            JOIN telem.tname   n ON n.id = f.name_id
            JOIN telem.vehicle v ON v.id = f.vehicle_id
            WHERE {where_sql}
            ORDER BY rtime ASC, f.vehicle_id ASC, f."timestamp" ASC
        """

        stmt = text(sql_txt)
        if excl_bind is not None:
            stmt = stmt.bindparams(excl_bind)

        exec_params = {**params}
        if EXCLUDE_VEHICLE_IDS:
            exec_params["excl_ids"] = EXCLUDE_VEHICLE_IDS

        result = con.execute(stmt, exec_params)
        for r in result.mappings():
            yield r


def to_json_safe(v):
    if isinstance(v, (dt.datetime, dt.date)): return v.isoformat()
    if isinstance(v, decimal.Decimal): return float(v)
    return v


def main():
    tmin, tmax, nrows = get_time_bounds()
    if not (tmax is not None) or nrows == 0:
        print("No rows found.")
        return

    print(f"Rows: {nrows}")
    # These are relative seconds
    print(f"First relative time: {tmin:.6f} s")
    print(f"Last  relative time: {tmax:.6f} s (duration)")
    input("Continue? ")

    start = dt.datetime.now(dt.timezone.utc)

    scope = f"ALL vehicles" if VEHICLE_ID is None else f"vehicle_id={VEHICLE_ID}"
    print(f"Streaming {nrows} rows for race #{RACE_NUMBER} @ {TRACK_NAME} ({scope}) (speed×{SPEED}) …")

    for r in stream_rows():
        rtime = float(r["rtime"])  # seconds since that vehicle's first timestamp

        # pacing (align all vehicles to start together at t=0)
        target_elapsed = rtime / max(SPEED, 1e-9)
        now_elapsed = (dt.datetime.now(dt.timezone.utc) - start).total_seconds()
        delay = target_elapsed - now_elapsed
        if delay > 0:
            time.sleep(delay)

        payload = {
            "vehicle_id": r["vehicle_id"],
            "race_number": r["race_number"],
            "telemetry_name": r["telemetry_name"],
            "telemetry_value": to_json_safe(r["telemetry_value"]),
            "ts": to_json_safe(r["ts"]),                  # original timestamp (for reference)
            "relative_time_s": rtime,                     # NEW: explicit in payload
            "track": r["track"],
            "vehicle_number": r["vehicle_number"],
            "lap": int(r["lap"]) if r["lap"] is not None else None,
        }

        producer.produce(
            topic=TOPIC,
            key=str(r["vehicle_id"]),     # preserves per-vehicle ordering
            value=json.dumps(payload),
            on_delivery=delivery_report,
        )
        producer.poll(0)

    producer.flush()
    print("Replay finished.")


if __name__ == "__main__":
    main()
