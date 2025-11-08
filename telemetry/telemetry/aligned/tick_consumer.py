import os, time, json, datetime as dt
from typing import Dict, Set, Optional
from confluent_kafka import Consumer, KafkaException, KafkaError, Producer
from confluent_kafka.admin import AdminClient, NewTopic
import psycopg2
from psycopg2 import sql

# --- Env/config ---------------------------------------------------------------
BROKER = os.getenv("BROKER", "redpanda:9092")
TOPIC = os.getenv("TOPIC", "telem.stream_fast.raw")
GROUP_ID = os.getenv("GROUP_ID", "tick-snapshots")
PG_DSN = os.getenv("PG_DSN", "postgresql://telemetry:telemetry@tickdb:5434/telemetry")
TICK_TABLE = os.getenv("TICK_TABLE", "telem_tick")
TICK_SECS = float(os.getenv("TICK_SECS", "0.1"))
NAMES_ALLOWLIST = {s.strip() for s in os.getenv("NAMES_ALLOWLIST", "").split(",") if s.strip()}

CONTROL_TOPIC = os.getenv("CONTROL_TOPIC", "tick.control")
# auto-create control topic? (recommended)
CONTROL_TOPIC_CREATE = os.getenv("CONTROL_TOPIC_CREATE", "true").lower() == "true"
CONTROL_TOPIC_PARTITIONS = int(os.getenv("CONTROL_TOPIC_PARTITIONS", "1"))
CONTROL_TOPIC_REPLICATION = int(os.getenv("CONTROL_TOPIC_REPLICATION", "1"))
CONTROL_TOPIC_COMPACT = os.getenv("CONTROL_TOPIC_COMPACT", "false").lower() == "true"
BROKER_READY_TIMEOUT_SECS = float(os.getenv("BROKER_READY_TIMEOUT_SECS", "45"))

WRITE_ENABLED = True   # default: writing on

WIPE_ON_START = os.getenv("WIPE_ON_START", "false").lower() == "true"
WIPE_TABLES = [t.strip() for t in os.getenv("WIPE_TABLES","").split(",") if t.strip()]

# --- Helpers: broker/topic bootstrap -----------------------------------------
def wait_for_broker(brokers: str, timeout_s: float = 30.0) -> None:
    """Poll metadata until the broker responds or timeout."""
    start = time.monotonic()
    p = Producer({"bootstrap.servers": brokers})
    while True:
        try:
            md = p.list_topics(timeout=2.0)
            if md.brokers:
                return
        except Exception:
            pass
        if time.monotonic() - start > timeout_s:
            raise RuntimeError(f"Broker not reachable at {brokers} within {timeout_s}s")
        time.sleep(0.5)

def ensure_topic(brokers: str, name: str, partitions: int, replication: int, compact: bool) -> None:
    """Create topic if missing; ignore if it already exists."""
    admin = AdminClient({"bootstrap.servers": brokers})
    cfg = {"cleanup.policy": "compact"} if compact else {}
    newt = NewTopic(topic=name, num_partitions=partitions, replication_factor=replication, config=cfg)

    # quick check to avoid noisy logs if present
    try:
        md = admin.list_topics(timeout=5.0)
        if name in md.topics and md.topics[name].error is None:
            return
    except Exception:
        # if metadata fails here, we’ll still attempt creation
        pass

    fs = admin.create_topics([newt])
    try:
        fs[name].result()
        print(f"[control] created topic {name} (p={partitions}, r={replication}, compact={compact})")
    except KafkaException as e:
        err = e.args[0]
        if hasattr(err, "code") and err.code() == KafkaError.TOPIC_ALREADY_EXISTS:
            return
        print(f"[control] topic create warning for {name}: {e}")

# --- DB helpers ---------------------------------------------------------------
def wipe_tables(conn):
    if not WIPE_TABLES:
        return
    with conn.cursor() as cur:
        for t in WIPE_TABLES:
            cur.execute(f'TRUNCATE TABLE "{t}" RESTART IDENTITY')
            print(f"[wipe] truncated {t}")
    conn.commit()

def to_float_or_none(v):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None

def ensure_columns(conn, names: Set[str]):
    """Add columns to wide table if they don't exist. Columns are double precision."""
    if not names:
        return
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = current_schema()
        """, (TICK_TABLE,))
        existing = {r[0] for r in cur.fetchall()}

        to_add = [n for n in names if n not in existing and n not in ("ts","vehicle_id")]
        for n in to_add:
            stmt = sql.SQL("ALTER TABLE {} ADD COLUMN {} double precision NULL").format(
                sql.Identifier(TICK_TABLE),
                sql.Identifier(n)
            )
            cur.execute(stmt)
            print(f"[schema] Added column {n}")
    conn.commit()

def upsert_tick(conn, ts: dt.datetime, vehicle_id: int, row: Dict[str, Optional[float]]):
    """Insert/Upsert one row: (ts, vehicle_id, <telemetry columns...>)."""
    with conn.cursor() as cur:
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

# --- Kafka consumers ----------------------------------------------------------
def make_consumer(group_id: str, topics: list[str], reset="latest") -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKER,
        "group.id": group_id,
        "enable.auto.commit": True,
        "auto.offset.reset": reset,
    })

def poll_control(ctrl_consumer) -> None:
    """Poll control topic non-blocking and toggle WRITE_ENABLED."""
    global WRITE_ENABLED
    msg = ctrl_consumer.poll(timeout=0.0)
    if not msg or msg.error():
        return
    try:
        payload = json.loads(msg.value())
        cmd = str(payload.get("cmd", "")).lower()
        if cmd in ("pause", "stop", "off"):
            WRITE_ENABLED = False
            print("[control] write disabled")
        elif cmd in ("resume", "start", "on"):
            WRITE_ENABLED = True
            print("[control] write enabled")
    except Exception as e:
        print(f"[control] bad control message: {e}")

# Holds the latest value by (vehicle_id -> {name: value})
latest_by_vehicle: Dict[int, Dict[str, Optional[float]]] = {}
# Set of all telemetry names observed (filtered by allowlist if provided)
observed_names: Set[str] = set()

# --- Main loop ----------------------------------------------------------------
def run():
    # 0) Ensure broker is ready and control topic exists
    wait_for_broker(BROKER, timeout_s=BROKER_READY_TIMEOUT_SECS)
    if CONTROL_TOPIC_CREATE:
        ensure_topic(
            brokers=BROKER,
            name=CONTROL_TOPIC,
            partitions=CONTROL_TOPIC_PARTITIONS,
            replication=CONTROL_TOPIC_REPLICATION,
            compact=CONTROL_TOPIC_COMPACT,
        )

    # 1) Data & control consumers
    data_consumer = make_consumer(GROUP_ID, [TOPIC], reset="earliest")
    data_consumer.subscribe([TOPIC])

    ctrl_consumer = make_consumer(GROUP_ID + "-control", [CONTROL_TOPIC], reset="latest")
    ctrl_consumer.subscribe([CONTROL_TOPIC])

    # 2) Postgres
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    if WIPE_ON_START:
        wipe_tables(conn)

    tick_next = time.monotonic()
    print(f"[tick] interval = {TICK_SECS}s, writing to table {TICK_TABLE}")

    try:
        while True:
            # control first (non-blocking)
            poll_control(ctrl_consumer)

            # data poll (low-latency)
            msg = data_consumer.poll(timeout=0.01)
            if msg is not None:
                if msg.error():
                    raise KafkaException(msg.error())
                try:
                    payload = json.loads(msg.value())
                    vid = int(payload.get("vehicle_id"))
                    name = str(payload.get("telemetry_name"))
                    val  = to_float_or_none(payload.get("telemetry_value"))

                    if not (NAMES_ALLOWLIST and name not in NAMES_ALLOWLIST):
                        latest_by_vehicle.setdefault(vid, {})[name] = val
                        observed_names.add(name)
                except Exception as e:
                    print(f"[warn] bad message: {e}")

            # tick?
            now = time.monotonic()
            if now >= tick_next:
                wall_ts = dt.datetime.now(dt.timezone.utc)

                # ensure schema columns exist for observed signals
                ensure_columns(conn, observed_names)

                # write per-vehicle row (only if enabled)
                if WRITE_ENABLED:
                    for vid, m in latest_by_vehicle.items():
                        row = {k: v for k, v in m.items() if v is not None}
                        upsert_tick(conn, wall_ts, vid, row)

                # always advance the tick, even if paused
                tick_next += TICK_SECS

    except KeyboardInterrupt:
        pass
    finally:
        data_consumer.close()
        ctrl_consumer.close()
        conn.close()

if __name__ == "__main__":
    run()