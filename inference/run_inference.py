import os
import datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
from inference.prediction import PathPredictor


# 1. Connect to tickdb
TICKDB_URL = "postgresql+psycopg2://user:pass@host:5432/tickdb"
engine = create_engine(TICKDB_URL)


def load_tick_window(
    engine,
    vehicle_code: str,
    start_time: dt.datetime,
    duration_s: float = 20.0,
) -> pd.DataFrame:
    """
    Load ~`duration_s` seconds of aligned tick data for one car.
    Assumes:
      - table: tickdb.aligned (change if different)
      - timestamp column: ts
      - vehicle identifier column: vehicle_code
      - signals: accx_can, accy_can, speed, gear, aps, nmot,
                 pbrake_f, pbrake_r, VBOX_Lat_Min, VBOX_Long_Minutes
    """
    start_time = 0
    end_time = start_time + duration_s

    query = text(
        """
        SELECT
            ts AS timestamp,
            accx_can,
            accy_can,
            speed,
            gear,
            aps,
            nmot,
            pbrake_f,
            pbrake_r,
            "VBOX_Lat_Min",
            "VBOX_Long_Minutes"
        FROM tickdb.aligned   -- TODO: adjust schema/table name if needed
        WHERE vehicle_code = :vehicle_code
          AND ts >= :start_time
          AND ts <  :end_time
        ORDER BY ts
        """
    )

    df = pd.read_sql(
        query,
        engine,
        params={
            "vehicle_code": vehicle_code,
            "start_time": start_time,
            "end_time": end_time,
        },
    )

    # Ensure time index is monotonic
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


# 2. Example usage: load window and run PathPredictor

vehicle_code = "GR86-022-13"
start_time = dt.datetime(2025, 11, 21, 14, 0, 0, tzinfo=dt.timezone.utc)  # example

df_window = load_tick_window(engine, vehicle_code, start_time, duration_s=20.0)

predictor = PathPredictor()
pred_states = predictor.predict(df_window)  # or pred_states, origin_rad = ...

# pred_states is a NumPy array with columns in STATE_COLS order

lat_true = df_window["VBOX_Lat_Min"].to_numpy()
lon_true = df_window["VBOX_Long_Minutes"].to_numpy()

# 2. Predicted GPS from PathPredictor
lat_pred, lon_pred = predictor.predict(df_window)  # (lat, lon)

# 3. Align lengths if needed
n = min(len(lat_true), len(lat_pred))
lat_true = lat_true[:n]
lon_true = lon_true[:n]
lat_pred = lat_pred[:n]
lon_pred = lon_pred[:n]

# 4. Plot map-style lat/lon
plt.figure(figsize=(8, 8))

plt.plot(lon_true, lat_true, label="True", linewidth=2)
plt.plot(lon_pred, lat_pred, label="Predicted", linestyle="--", linewidth=2)

# Optional: mark start/end points
plt.scatter(lon_true[0], lat_true[0], c="green", marker="o", s=60, label="Start (true)")
plt.scatter(lon_true[-1], lat_true[-1], c="red", marker="x", s=60, label="End (true)")

plt.scatter(lon_pred[0], lat_pred[0], c="green", marker="o", s=30, alpha=0.6)
plt.scatter(lon_pred[-1], lat_pred[-1], c="red", marker="x", s=30, alpha=0.6)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted vs True Trajectory")
plt.legend()
plt.gca().set_aspect("equal", "box")   # keep geometry realistic
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()