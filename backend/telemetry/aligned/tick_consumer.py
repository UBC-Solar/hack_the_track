import os
import time
import json
import datetime as dt
from typing import Dict, Set, Optional

from confluent_kafka import Consumer, KafkaException, KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic
import psycopg2
from psycopg2 import sql

# --- Env/config ---------------------------------------------------------------
BROKER = os.getenv("BROKER", "redpanda:9092")
TOPIC = os.getenv("TOPIC", "telem.stream_fast.raw")
GROUP_ID = os.getenv("GROUP_ID", "tick-snapshots")
# Data topic creation config (for the main telemetry topic)
DATA_TOPIC_CREATE = os.getenv("DATA_TOPIC_CREATE", "true").lower() == "true"
DATA_TOPIC_PARTITIONS = int(os.getenv("DATA_TOPIC_PARTITIONS", "8"))
DATA_TOPIC_REPLICATION = int(os.getenv("DATA_TOPIC_REPLICATION", "1"))

PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://telemetry:telemetry@tickdb:5434/telemetry",
)
TICK_TABLE = os.getenv("TICK_TABLE", "telem_tick")

TICK_SECS = float(os.getenv("TICK_SECS", "0.05"))
NAMES_ALLOWLIST = {
    s.strip()
    for s in os.getenv("NAMES_ALLOWLIST", "").split(",")
    if s.strip()
}

CONTROL_TOPIC = os.getenv("CONTROL_TOPIC", "tick.control")
CONTROL_TOPIC_CREATE = (
    os.getenv("CONTROL_TOPIC_CREATE", "true").lower() == "true"
)
CONTROL_TOPIC_PARTITIONS = int(os.getenv("CONTROL_TOPIC_PARTITIONS", "1"))
CONTROL_TOPIC_REPLICATION = int(os.getenv("CONTROL_TOPIC_REPLICATION", "1"))
CONTROL_TOPIC_COMPACT = (
    os.getenv("CONTROL_TOPIC_COMPACT", "false").lower() == "true"
)
BROKER_READY_TIMEOUT_SECS = float(
    os.getenv("BROKER_READY_TIMEOUT_SECS", "45")
)

WRITE_ENABLED = False

WIPE_ON_START = os.getenv("WIPE_ON_START", "false").lower() == "true"
WIPE_TABLES = [
    t.strip()
    for t in os.getenv("WIPE_TABLES", "").split(",")
    if t.strip()
]

# columns we add for training-style reconstruction
LAP_COLUMN = "lap"
ORIG_TS_COLUMN = "orig_ts"
RTIME_COLUMN = "rtime"

# caches: latest values per vehicle
latest_by_vehicle: Dict[str, Dict[str, Optional[float]]] = {}
latest_meta_by_vehicle: Dict[str, Dict[str, Optional[float]]] = {}
latest_lap_by_vehicle: Dict[str, Optional[int]] = {}
# signal names observed
observed_names: Set[str] = set()


# --- Helpers ------------------------------------------------------------------
def to_float_or_none(v):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def to_int_or_none(v):
    try:
        return int(float(v))
    except Exception:
        return None


def wait_for_broker(brokers: str, timeout_s: float = 30.0) -> None:
    """Poll metadata until the broker responds or timeout."""
    start = time.monotonic()
    p = Producer({"bootstrap.servers": brokers})
    while True:
        try:
            md = p.list_topics(timeout=2.0)
            if md.brokers:
                print(f"[broker] connected to {brokers}")
                return
        except Exception:
            pass

        if time.monotonic() - start > timeout_s:
            raise RuntimeError(
                f"Broker not reachable at {brokers} within {timeout_s}s"
            )
        time.sleep(0.5)


def ensure_topic(
    brokers: str,
    name: str,
    partitions: int,
    replication: int,
    compact: bool,
) -> None:
    """Create topic if missing; ignore if it already exists."""
    admin = AdminClient({"bootstrap.servers": brokers})
    cfg = {"cleanup.policy": "compact"} if compact else {}
    newt = NewTopic(
        topic=name,
        num_partitions=partitions,
        replication_factor=replication,
        config=cfg,
    )

    # quick check
    try:
        md = admin.list_topics(timeout=5.0)
        if name in md.topics and md.topics[name].error is None:
            print(f"[control] topic {name} already exists")
            return
    except Exception:
        pass

    fs = admin.create_topics([newt])
    try:
        fs[name].result()
        print(
            f"[control] created topic {name} "
            f"(p={partitions}, r={replication}, compact={compact})"
        )
    except KafkaException as e:
        err = e.args[0]
        if hasattr(err, "code") and err.code() == KafkaError.TOPIC_ALREADY_EXISTS:
            print(f"[control] topic {name} already exists (race)")
            return
        print(f"[control] topic create warning for {name}: {e}")


def ensure_schema(conn):
    """Ensure lap/orig_ts/rtime columns exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = current_schema()
        """,
            (TICK_TABLE,),
        )
        existing = {r[0] for r in cur.fetchall()}

        needed = {
            ORIG_TS_COLUMN: "timestamptz",
            RTIME_COLUMN: "double precision",
            LAP_COLUMN: "integer",
        }

        for col, ctype in needed.items():
            if col not in existing:
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} ADD COLUMN {} {} NULL"
                    ).format(
                        sql.Identifier(TICK_TABLE),
                        sql.Identifier(col),
                        sql.SQL(ctype),
                    )
                )
                print(f"[schema] added column {col} ({ctype})")

    conn.commit()


def ensure_signal_columns(conn, names: Set[str]):
    """Add columns for observed signal names, double precision."""
    if not names:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = current_schema()
        """,
            (TICK_TABLE,),
        )
        existing = {r[0] for r in cur.fetchall()}

        for n in names:
            if n not in existing and n not in ("ts", "vehicle_id"):
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} ADD COLUMN {} double precision NULL"
                    ).format(
                        sql.Identifier(TICK_TABLE),
                        sql.Identifier(n),
                    )
                )
                print(f"[schema] added column {n}")

    conn.commit()


def upsert_tick(
    conn,
    tick_ts: dt.datetime,
    vehicle_id: str,
    row: Dict[str, Optional[float]],
):
    """Insert/Upsert one row: (ts, vehicle_id, <telemetry columns...>)."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "INSERT INTO {} (ts, vehicle_id) VALUES (%s, %s) "
                "ON CONFLICT (ts, vehicle_id) DO NOTHING"
            ).format(sql.Identifier(TICK_TABLE)),
            (tick_ts, vehicle_id),
        )

        if row:
            cols = [sql.Identifier(k) for k in row.keys()]
            sets = [sql.SQL("{} = %s").format(c) for c in cols]
            values = [row[k] for k in row.keys()]

            q = (
                sql.SQL("UPDATE {} SET ").format(sql.Identifier(TICK_TABLE))
                + sql.SQL(", ").join(sets)
                + sql.SQL(" WHERE ts = %s AND vehicle_id = %s")
            )
            cur.execute(q, (*values, tick_ts, vehicle_id))

    conn.commit()


# --- Kafka consumers ----------------------------------------------------------
def make_consumer(group_id: str, reset="latest") -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": BROKER,
            "group.id": group_id,
            "enable.auto.commit": True,
            "auto.offset.reset": reset,
        }
    )


def poll_control(ctrl_consumer) -> None:
    """
    Poll control topic and toggle WRITE_ENABLED.

    Supports:
      - {"write_enabled": true/false}
      - {"cmd": "start" | "resume" | "on" | "pause" | "stop" | "off"}
    """
    global WRITE_ENABLED
    msg = ctrl_consumer.poll(timeout=0.0)
    if not msg:
        return
    if msg.error():
        # don't spam, but log once in a while if needed
        if msg.error().code() != KafkaError._PARTITION_EOF:
            print(f"[control] error: {msg.error()}")
        return

    try:
        payload = json.loads(msg.value())
        print(f"[control] received: {payload}")

        keys = {k.lower(): k for k in payload.keys()}

        # new style: write_enabled
        w_key = keys.get("write_enabled")
        if w_key is not None and isinstance(payload[w_key], bool):
            WRITE_ENABLED = payload[w_key]
            state = "enabled" if WRITE_ENABLED else "disabled"
            print(f"[control] write {state} (via write_enabled={WRITE_ENABLED})")
            return

        # legacy style: cmd
        cmd_key = keys.get("cmd")
        cmd = str(payload[cmd_key]).lower() if cmd_key is not None else ""

        if cmd in ("pause", "stop", "off"):
            WRITE_ENABLED = False
            print("[control] write disabled (via cmd)")
        elif cmd in ("resume", "start", "on"):
            WRITE_ENABLED = True
            print("[control] write enabled (via cmd)")
        else:
            print(f"[control] unknown control payload: {payload}")

    except Exception as e:
        print(f"[control] bad control message: {e}")


# --- Main loop ----------------------------------------------------------------
def run():
    print(f"[config] BROKER={BROKER}")
    print(f"[config] DATA TOPIC={TOPIC}")
    print(f"[config] CONTROL TOPIC={CONTROL_TOPIC}")
    print(f"[config] PG_DSN={PG_DSN}")
    print(f"[config] TICK_TABLE={TICK_TABLE}")
    print(f"[config] TICK_SECS={TICK_SECS}")

    # 0) Ensure broker is ready and control topic exists
    # 0) Ensure broker is ready and topics exist
    wait_for_broker(BROKER, timeout_s=BROKER_READY_TIMEOUT_SECS)

    if CONTROL_TOPIC_CREATE:
        ensure_topic(
            brokers=BROKER,
            name=CONTROL_TOPIC,
            partitions=CONTROL_TOPIC_PARTITIONS,
            replication=CONTROL_TOPIC_REPLICATION,
            compact=CONTROL_TOPIC_COMPACT,
        )

    # NEW: ensure the data topic exists as well (non-compacted)
    if DATA_TOPIC_CREATE:
        ensure_topic(
            brokers=BROKER,
            name=TOPIC,
            partitions=DATA_TOPIC_PARTITIONS,
            replication=DATA_TOPIC_REPLICATION,
            compact=False,
        )

    # 1) Data & control consumers
    data_consumer = make_consumer(GROUP_ID, reset="earliest")
    data_consumer.subscribe([TOPIC])

    print(f"[data] subscribed to {TOPIC} (group {GROUP_ID})")

    ctrl_consumer = make_consumer(GROUP_ID + "-control", reset="latest")
    ctrl_consumer.subscribe([CONTROL_TOPIC])
    print(f"[control] subscribed to {CONTROL_TOPIC} (group {GROUP_ID}-control)")

    # 2) Postgres
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False

    if WIPE_ON_START:
        with conn.cursor() as cur:
            for t in WIPE_TABLES:
                cur.execute(f'TRUNCATE TABLE "{t}" RESTART IDENTITY')
                print(f"[wipe] truncated {t}")
        conn.commit()

    ensure_schema(conn)

    tick_next = time.monotonic()
    print(f"[tick] interval = {TICK_SECS}s, writing to table {TICK_TABLE}")

    try:
        while True:
            # control first
            poll_control(ctrl_consumer)

            # data poll
            msg = data_consumer.poll(timeout=0.01)
            if msg is not None:
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        print(f"[tick] data error: {msg.error()}")
                    time.sleep(0.1)
                else:
                    try:
                        payload = json.loads(msg.value())

                        vid_raw = payload.get("vehicle_id")
                        if vid_raw is None:
                            # no vehicle_id -> nothing to do
                            continue

                        # vehicle_id comes as a string code now (e.g. "GR86-022-13"),
                        # but also handle older numeric IDs just in case:
                        vid = str(vid_raw)

                        name = str(payload.get("telemetry_name"))
                        val = to_float_or_none(payload.get("telemetry_value"))

                        rtime = to_float_or_none(
                            payload.get("relative_time_s")
                        )
                        orig_ts_str = payload.get("ts")

                        if orig_ts_str:
                            try:
                                orig_ts = dt.datetime.fromisoformat(orig_ts_str)
                            except Exception:
                                orig_ts = None
                        else:
                            orig_ts = None

                        # filter by name allowlist if provided
                        if NAMES_ALLOWLIST and name not in NAMES_ALLOWLIST:
                            # still track lap & meta, but skip this signal
                            pass
                        else:
                            observed_names.add(name)
                            latest_by_vehicle.setdefault(vid, {})[name] = val

                        meta = latest_meta_by_vehicle.setdefault(vid, {})
                        if rtime is not None:
                            meta[RTIME_COLUMN] = rtime
                        if orig_ts is not None:
                            meta[ORIG_TS_COLUMN] = orig_ts

                        lap = payload.get("lap")
                        lap_i = to_int_or_none(lap)
                        if lap_i is not None:
                            latest_lap_by_vehicle[vid] = lap_i

                    except Exception as e:
                        print(f"[warn] bad data message: {e}")

            # tick?
            now = time.monotonic()
            if now >= tick_next:
                tick_ts = dt.datetime.now(dt.timezone.utc)

                ensure_signal_columns(conn, observed_names)

                if WRITE_ENABLED:
                    if not latest_by_vehicle:
                        print("[tick] WRITE_ENABLED but no telemetry buffered yet")
                    for vid, sigvals in latest_by_vehicle.items():
                        row = {k: v for k, v in sigvals.items() if v is not None}

                        meta = latest_meta_by_vehicle.get(vid, {})
                        row.update(meta)

                        lapval = latest_lap_by_vehicle.get(vid)
                        if lapval is not None:
                            row[LAP_COLUMN] = lapval

                        # print(
                        #     f"[tick] writing row for vehicle {vid} "
                        #     f"with {len(row)} fields"
                        # )
                        upsert_tick(conn, tick_ts, vid, row)
                else:
                    # uncomment if you want spammy visibility:
                    # print("[tick] WRITE_ENABLED = False; skipping writes")
                    pass

                tick_next += TICK_SECS

    except KeyboardInterrupt:
        print("[tick] interrupted, shutting down")
    finally:
        data_consumer.close()
        ctrl_consumer.close()
        conn.close()


if __name__ == "__main__":
    run()
