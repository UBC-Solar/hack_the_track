import datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
from inference.prediction import prepare_tickdb_dataframe_for_model, CarTrajectoryPredictor
from inference.models import MODEL_PATH
from inference.constants import state, control


# 1. Connect to tickdb
TICKDB_URL = "postgresql+psycopg2://telemetry:telemetry@localhost:5432/telemetry"
engine = create_engine(TICKDB_URL)


def load_tick_window(
    engine,
    vehicle_id: int,
    duration_s: float = 5.0,
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
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(seconds=duration_s)

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
        FROM telem_tick
        WHERE vehicle_id = :vehicle_id
          AND ts >= :start_time
          AND ts <  :end_time
        ORDER BY ts
        """
    )

    df = pd.read_sql(
        query,
        engine,
        params={
            "vehicle_id": vehicle_id,
            "start_time": start_time,
            "end_time": end_time,
        },
    )

    # Ensure time index is monotonic
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


# 2. Example usage: load window and run PathPredictor

vehicle_id = 36

df_window: pd.DataFrame = load_tick_window(engine, vehicle_id, duration_s=20.0)
print(df_window.head())
print(len(df_window))
print(df_window.index.to_series().diff().describe())
print("TickDB head (post-rename):")
print(df_window.head())

print("\nBasic stats:")
print(df_window.describe())

# pred_states is a NumPy array with columns in STATE_COLS order

lat_true = df_window["VBOX_Lat_Min"].to_numpy()
lon_true = df_window["VBOX_Long_Minutes"].to_numpy()

df_model = prepare_tickdb_dataframe_for_model(df_window, state, control)

predictor = CarTrajectoryPredictor(
    state_cols=state,
    control_cols=control,
    model_path=str(MODEL_PATH / "car13_multistep_model.pt"),
    scaler_path=str(MODEL_PATH / "car13_multistep_scaler.pkl"),
    seq_len=10,
    scale=50.0,
)

true_lat, true_lon, lat_pred, lon_pred = predictor.predict(df_model)

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
