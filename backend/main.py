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
from random import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, MetaData, Table, select, desc, text

from confluent_kafka import Producer

from load_gps_data import get_lap_gps_data, data_path

from inference.insights import get_insights, ControlModification, make_predictor

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


predictor = make_predictor()

control_modifications = [
    ControlModification(
        name="baseline",
        apply=lambda df: df.copy(),
    ),
    ControlModification(
        name="Brake 25% Less",
        apply=lambda df: df.assign(
            pbrake_f=df["pbrake_f"] * 0.75,
            pbrake_r=df["pbrake_r"] * 0.75,
        ),
    ),
    ControlModification(
        name="Gear Up",
        apply=lambda df: df.assign(gear=df["gear"] + 1),
    ),
    ControlModification(
        name="Gear Down",
        apply=lambda df: df.assign(gear=df["gear"] - 1),
    ),
    ControlModification(
        name="Brake 25% More",
        apply=lambda df: df.assign(
            pbrake_f=df["pbrake_f"] * 1.25,
            pbrake_r=df["pbrake_r"] * 1.25,
        ),
    ),
    ControlModification(
        name="Double Steering Angle",
        apply=lambda df: df.assign(steering_angle=df["steering_angle"] * 2.0),
    ),
    ControlModification(
        name="Steer 5deg Left",
        apply=lambda df: df.assign(steering_angle=df["steering_angle"] + 5.0),
    ),
    ControlModification(
        name="Steer 5deg Right",
        apply=lambda df: df.assign(steering_angle=df["steering_angle"] - 5.0),
    )
]

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
def get_current_laps(vehicleID: str):
    """
    Get lap times for a specified vehicle, excluding the current lap.
    """
    try:
        with engine.connect() as conn:
            # First, we query to get the current lap (the one with the most recent timestamp)
            current_lap_stmt = text("""
                SELECT lap
                FROM telem_tick
                WHERE vehicle_id = :vehicle_id
                ORDER BY ts DESC
                LIMIT 1;
            """)

            # Execute the query to get the current lap number
            current_lap_result = conn.execute(current_lap_stmt, {"vehicle_id": vehicleID}).mappings().fetchone()

            # If no laps are found for the vehicle, raise an error
            if not current_lap_result:
                raise HTTPException(status_code=404, detail="No telemetry data found for this vehicle.")

            current_lap = current_lap_result["lap"]

            # Query to get the first and last timestamps for each lap, excluding the current lap
            stmt = text("""
                SELECT
                    lap,
                    MIN(ts) AS first_timestamp,
                    MAX(ts) AS last_timestamp
                FROM
                    telem_tick
                WHERE
                    vehicle_id = :vehicle_id
                    AND lap != :current_lap  -- Exclude the current lap
                GROUP BY
                    lap
                ORDER BY
                    lap;
            """)

            # Execute the query and fetch the results
            result = conn.execute(stmt, {"vehicle_id": vehicleID, "current_lap": current_lap}).mappings().fetchall()

            # If no laps are found (after excluding current lap), raise an error
            if not result:
                raise HTTPException(status_code=404, detail="No lap data found for this vehicle.")

            # Calculate lap times (difference between first and last timestamp)
            lap_times = []
            for row in result:
                lap_number = row["lap"]
                first_timestamp = row["first_timestamp"]
                last_timestamp = row["last_timestamp"]
                if first_timestamp and last_timestamp:
                    # Calculate the time difference in seconds
                    lap_time = (last_timestamp - first_timestamp).total_seconds()
                    lap_times.append({"number": lap_number, "time": lap_time})

            return lap_times

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/currentLap")
def get_current_lap(vehicleID: str):
    """
    Returns the current lap for the selected vehicle.
    This is determined by the most recent telemetry data for that vehicle.
    """
    try:
        with engine.connect() as conn:
            # Query to get the most recent telemetry data for the given vehicle
            current_lap_stmt = text("""
                SELECT lap
                FROM telem_tick
                WHERE vehicle_id = :vehicle_id
                ORDER BY ts DESC
                LIMIT 1;
            """)

            # Execute the query to get the current lap number
            current_lap_result = conn.execute(current_lap_stmt, {"vehicle_id": vehicleID}).mappings().fetchone()

            if not current_lap_result:
                raise HTTPException(status_code=404, detail="No telemetry data found for this vehicle.")

            # Return the current lap number
            return {"currentLap": current_lap_result["lap"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/currentLapTime")
def get_current_lap_time(vehicleID: str):
    """
    Returns the current lap time for the selected vehicle.
    This is calculated by finding the difference between the current time and the first timestamp
    of the vehicle's current lap.
    """
    try:
        with engine.connect() as conn:
            # Query to get the current lap number and time (based on most recent telemetry)
            current_lap_stmt = text("""
                SELECT lap, ts
                FROM telem_tick
                WHERE vehicle_id = :vehicle_id
                ORDER BY ts DESC
                LIMIT 1;
            """)

            current_lap_result = conn.execute(current_lap_stmt, {"vehicle_id": vehicleID}).mappings().fetchone()

            if not current_lap_result:
                raise HTTPException(status_code=404, detail="No telemetry data found for this vehicle.")

            current_lap = current_lap_result["lap"]
            current_ts = current_lap_result["ts"]

            # Query to get the first timestamp for the current lap
            first_timestamp_stmt = text("""
                SELECT MIN(ts) AS first_timestamp
                FROM telem_tick
                WHERE vehicle_id = :vehicle_id
                AND lap = :current_lap;
            """)

            first_timestamp_result = conn.execute(first_timestamp_stmt, {"vehicle_id": vehicleID, "current_lap": current_lap}).mappings().fetchone()

            if not first_timestamp_result or not first_timestamp_result["first_timestamp"]:
                raise HTTPException(status_code=404, detail="No telemetry data found for the current lap.")

            first_timestamp = first_timestamp_result["first_timestamp"]

            # Calculate the difference between current time and first timestamp of the lap
            lap_time_seconds = (current_ts - first_timestamp).total_seconds()

            return {"currentLapTime": lap_time_seconds}

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
# Dummy Functions
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

@app.get("/driverInsightFake")
def get_insight_fake(
    vehicleID: str,
):
    """
    Returns fake driver insights for testing
    """

    try:
        all_positions = get_latest_all()
        lat = all_positions[vehicleID][0]
        lon = all_positions[vehicleID][1]
    except:
        lat = 33.5297157 + random() * (33.5348805 - 33.5297157)
        lon = -86.6153219 + random() * (-86.6238813 - -86.6153219)

    rand = random()

    if rand < 0.25: insight = "➡️ Steer Right +10%" 
    elif rand < 0.5: insight = "⬅️ Steer Left +10%" 
    elif rand < 0.75: insight = "⏩ Increase Acceleration +10%" 
    else: insight = "⏪ Decrease Acceleration +10%" 

    return {"startLat": lat, "startLon": lon, "driverInsight": insight}

# -------------------------------------------------------------
# Insights
# -------------------------------------------------------------

@app.get("/driverInsight")
def get_insight(
    vehicleID: str,
):
    """
    Returns fake driver insights for testing
    """
    duration_s = 5.0

    try:
        all_positions = get_latest_all()
        lat = all_positions[vehicleID][0]
        lon = all_positions[vehicleID][1]
    except:
        lat = 33.5297157 + random() * (33.5348805 - 33.5297157)
        lon = -86.6153219 + random() * (-86.6238813 - -86.6153219)

    (
        lat_true,
        lon_true,
        pred_results,
        gates_with_intersections,
        best_controls,
        best_improvement_s,
    ) = get_insights(
        vehicleID,
        duration_s,
        control_modifications,
        predictor=predictor,
        engine=engine,
    )

    # Print best control(s) and time improvement
    if best_improvement_s is not None and best_controls:
        if len(best_controls) == 1:
            insight = best_controls[0]
            print(
                f"\nBest control modification: {best_controls[0]} "
                f"(Δt = {best_improvement_s:.3f} s vs baseline)"
            )
        else:
            combo_str = " And ".join(best_controls)
            insight = combo_str

            print(
                f"\nBest combined controls: {insight} "
                f"(Δt = {best_improvement_s:.3f} s vs baseline)"
            )
    else:
        insight = None
        best_improvement_s = None
        print("\nNo beneficial control modifications found.")

    if best_improvement_s:
        best_improvement = best_improvement_s / duration_s

    return {"startLat": lat, "startLon": lon, "driverInsight": insight, "improvement": best_improvement}


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
    print(get_current_laps(36))
