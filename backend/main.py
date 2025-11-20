# backend/main.py
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, MetaData, Table, select, desc, text
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from confluent_kafka import Producer
from math import cos, sin, radians, pi
from datetime import datetime
import json
import os

from load_gps_data import get_lap_gps_data, data_path


app = FastAPI()

# Allow frontend to call backend (important for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you like, e.g. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/laps/")
def get_example_lap(lapNumber: int, samplePeriod: int = 1):
    barber_tel_path = data_path / "barber-motorsports-park" / "barber" / "Race 1" / "R1_barber_telemetry_data.csv"
    lat_df, lon_df = get_lap_gps_data(barber_tel_path, lapNumber, chunksize=100000, chunk_limit=None)
    lat_vals = lat_df['telemetry_value'][::samplePeriod]
    lon_vals = lon_df['telemetry_value'][::samplePeriod]
    return {"lat_vals": list(lat_vals), "lon_vals": list(lon_vals)}

PG_DSN = os.getenv("PG_DSN", "postgresql://telemetry:telemetry@localhost:5432/telemetry")
TICK_TABLE = os.getenv("TICK_TABLE", "telem_tick")

# Create SQLAlchemy engine
engine = create_engine(PG_DSN, future=True)
metadata = MetaData()
tick_table = Table(TICK_TABLE, metadata, autoload_with=engine)

@app.get("/latest")
def get_latest_row():
    try:
        with engine.connect() as conn:
            stmt = select(tick_table).order_by(desc(tick_table.c.ts)).limit(1)
            result = conn.execute(stmt).mappings().fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="No telemetry data found")
            return dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latestAll")
def get_latest_all():
    try:
        with engine.connect() as conn:

            # Get a table with rows containing the most recent lat/lon for each distinct vehicle
            sql_txt = f"""
                SELECT DISTINCT ON (vehicle_id)
                    ts,
                    vehicle_id,
                    "VBOX_Lat_Min",
                    "VBOX_Long_Minutes"
                FROM telem_tick
                WHERE ts > now() - interval '1 second'
                ORDER BY vehicle_id, ts DESC;
            """
            stmt = text(sql_txt)

            result = conn.execute(stmt).mappings().fetchall()

            if not result:
                raise HTTPException(status_code=404, detail="No telemetry data found")

            # Create a mapping from vehicle ids to most recent position

            veh_locations: dict[int, tuple[float, float]] = {
                veh_row['vehicle_id']: (veh_row['VBOX_Lat_Min'], veh_row['VBOX_Long_Minutes']) for veh_row in result
            }

            return veh_locations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latestAllFake")
def get_latest_all_fake(num_cars: int = 12, radius_m: float = 100, period_s: float = 5, center_coords = None):
    """Return a dictionary of fake car GPS data

    Will return 12 evenly spaced cars driving at constant speed in a circle at the center of the track.
    """
    earth_circumference = 40075.017 * 1000  # m
    radius_deg = radius_m / earth_circumference * 360
    center_coords = center_coords if center_coords is not None else (33.5325017, -86.6215766)

    positions: dict[int, tuple[float, float]] = {}

    deg_per_car = 360 / num_cars
    for car_i in range(num_cars):

        car_relative_angle = radians(deg_per_car * car_i)
        current_time = datetime.now().timestamp() # seconds
        phase = (current_time % period_s) * (2 * pi / period_s)
        car_angle = car_relative_angle + phase

        delta_lon = sin(car_angle) * radius_deg
        delta_lat = cos(car_angle) * radius_deg

        positions[car_i] = (center_coords[0] + delta_lat, center_coords[1] + delta_lon)

    return positions

# -------- Kafka control plumbing --------
BROKER = os.getenv("BROKER", "localhost:9092")
CONTROL_TOPIC = os.getenv("CONTROL_TOPIC", "tick.control")

producer: Producer | None = None

@app.on_event("startup")
def _init_kafka():
    global producer
    try:
        producer = Producer({"bootstrap.servers": BROKER})
    except Exception as e:
        # Don't crash the app; just make it clear controls won't work
        print(f"[WARN] Failed to create Kafka producer: {e}")
        producer = None

@app.on_event("shutdown")
def _flush_kafka():
    try:
        if producer is not None:
            producer.flush(5)
    except Exception:
        pass

class TogglePayload(BaseModel):
    enable: bool

@app.post("/control/tick-consumer")
def toggle_tick_consumer(payload: TogglePayload):
    """
    Publish a control message to enable/disable the tick-consumer.
    Message format (JSON):
      {"command":"toggle_write","write_enabled":<bool>,"ts":"<iso8601>"}
    """
    if producer is None:
        raise HTTPException(status_code=503, detail="Kafka producer not initialized")

    message = {
        "command": "toggle_write",
        "write_enabled": payload.enable,
    }

    try:
        producer.produce(
            topic=CONTROL_TOPIC,
            value=json.dumps(message).encode("utf-8"),
        )
        # Let librdkafka handle batching; flush lightly here if you want immediate delivery:
        producer.poll(0)
    except BufferError:
        # If queue is full, try a quick flush and retry once
        producer.flush(1)
        producer.produce(topic=CONTROL_TOPIC, value=json.dumps(message).encode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kafka produce failed: {e}")

    return {"ok": True, "sent": message}


if __name__ == "__main__":
    print(get_latest_all_fake())
