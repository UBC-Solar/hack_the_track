# backend/main.py

"""
Telemetry Backend Service
-------------------------
Provides telemetry data endpoints, GPS lap extraction,
per-vehicle queries, and fake vehicle telemetry generation.
"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from datetime import datetime, timezone
import json
import os
from math import cos, sin, radians, pi

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, MetaData, Table, select, desc, text

from confluent_kafka import Producer

from load_gps_data import get_lap_gps_data, data_path


# -------------------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------------------
app = FastAPI(title="Telemetry Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # NOTE: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Configuration / Environment
# -------------------------------------------------------------
PG_DSN = os.getenv("PG_DSN", "postgresql://telemetry:telemetry@localhost:5432/telemetry")
TICK_TABLE = os.getenv("TICK_TABLE", "telem_tick")

BROKER = os.getenv("BROKER", "localhost:9092")
CONTROL_TOPIC = os.getenv("CONTROL_TOPIC", "tick.control")

# -------------------------------------------------------------
# Database Setup
# -------------------------------------------------------------
engine = create_engine(PG_DSN, future=True)
metadata = MetaData()
tick_table = Table(TICK_TABLE, metadata, autoload_with=engine)

# Kafka producer (initialized in startup event)
producer: Producer | None = None


# -------------------------------------------------------------
# Routes: Basic
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello World!"}


# -------------------------------------------------------------
# Routes: GPS Lap Data
# -------------------------------------------------------------
@app.get("/laps/")
def get_example_lap(lapNumber: int, samplePeriod: int = 1):
    """
    Returns down-sampled GPS data for a specific lap.
    """
    barber_path = (
        data_path
        / "barber-motorsports-park"
        / "barber"
        / "Race 1"
        / "R1_barber_telemetry_data.csv"
    )

    lat_df, lon_df = get_lap_gps_data(barber_path, lapNumber, chunksize=100000)

    lat_vals = lat_df["telemetry_value"][::samplePeriod]
    lon_vals = lon_df["telemetry_value"][::samplePeriod]

    return {"lat_vals": list(lat_vals), "lon_vals": list(lon_vals)}


# -------------------------------------------------------------
# Routes: Telemetry Queries
# -------------------------------------------------------------
@app.get("/currentLaps")
def get_current_laps(vehicle_ID: str):
    """
    Placeholder for returning lap information for a vehicle.
    Replace when real lap logic is implemented.
    """
    try:
        # Mocked data
        result = [
            {"number": 1, "time": 92.5},
            {"number": 2, "time": 90.2},
            {"number": 3, "time": 95.1},
        ]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vehicles")
def list_all_vehicles():
    """
    Returns all vehicle IDs present in the telemetry table.
    """
    try:
        with engine.connect() as conn:
            stmt = text("SELECT DISTINCT vehicle_id FROM telem_tick")
            rows = conn.execute(stmt).mappings().fetchall()

            if not rows:
                raise HTTPException(status_code=404, detail="No telemetry data found")

            return [row["vehicle_id"] for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest")
def get_latest_row():
    """
    Returns the latest telemetry tick entry.
    """
    try:
        with engine.connect() as conn:
            stmt = select(tick_table).order_by(desc(tick_table.c.ts)).limit(1)
            row = conn.execute(stmt).mappings().fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="No telemetry data found")

            return dict(row)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latestAll")
def get_latest_all():
    """
    Returns the most recent telemetry (lat/lon) for each distinct vehicle.
    Only includes data from the last second.
    """
    try:
        with engine.connect() as conn:
            stmt = text("""
                SELECT DISTINCT ON (vehicle_id)
                    ts,
                    vehicle_id,
                    "VBOX_Lat_Min",
                    "VBOX_Long_Minutes"
                FROM telem_tick
                WHERE ts > now() - interval '1 second'
                ORDER BY vehicle_id, ts DESC;
            """)

            rows = conn.execute(stmt).mappings().fetchall()

            if not rows:
                raise HTTPException(status_code=404, detail="No telemetry data found")

            return {
                row["vehicle_id"]: (row["VBOX_Lat_Min"], row["VBOX_Long_Minutes"])
                for row in rows
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# Routes: Fake Telemetry Generator
# -------------------------------------------------------------
@app.get("/latestAllFake")
def get_latest_all_fake(
    num_cars: int = 12,
    radius_m: float = 100,
    period_s: float = 5,
    center_coords: tuple[float, float] | None = None,
):
    """
    Returns fake vehicle telemetry for testing.
    Cars drive in a circle around the track center.
    """
    earth_circ = 40075.017 * 1000  # meters
    radius_deg = radius_m / earth_circ * 360

    if center_coords is None:
        center_coords = (33.5325017, -86.6215766)

    results = {}
    angle_step = 360 / num_cars

    now = datetime.now().timestamp()

    for i in range(num_cars):
        base_angle = radians(angle_step * i)
        phase = (now % period_s) * (2 * pi / period_s)
        angle = base_angle + phase

        d_lon = sin(angle) * radius_deg
        d_lat = cos(angle) * radius_deg

        results[i] = (center_coords[0] + d_lat, center_coords[1] + d_lon)

    return results


# -------------------------------------------------------------
# Kafka Events
# -------------------------------------------------------------
@app.on_event("startup")
def init_kafka():
    """Initialize Kafka producer."""
    global producer
    try:
        producer = Producer({"bootstrap.servers": BROKER})
        print("[INFO] Kafka producer initialized")
    except Exception as e:
        print(f"[WARN] Kafka producer init failed: {e}")
        producer = None


@app.on_event("shutdown")
def flush_kafka():
    """Flush Kafka producer on shutdown."""
    if producer:
        try:
            producer.flush(5)
        except Exception:
            pass


class TogglePayload(BaseModel):
    enable: bool


@app.post("/control/tick-consumer")
def toggle_tick_consumer(payload: TogglePayload):
    """
    Controls tick-consumer via Kafka message.
    """
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka producer not initialized")

    message = {
        "command": "toggle_write",
        "write_enabled": payload.enable,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        producer.produce(
            topic=CONTROL_TOPIC,
            value=json.dumps(message).encode("utf-8"),
        )
        producer.poll(0)
    except BufferError:
        producer.flush(1)
        producer.produce(
            topic=CONTROL_TOPIC,
            value=json.dumps(message).encode("utf-8"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kafka produce failed: {e}")

    return {"ok": True, "sent": message}


# -------------------------------------------------------------
# Debug Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    print(get_latest_all_fake())
