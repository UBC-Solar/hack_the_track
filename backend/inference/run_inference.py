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
            "VBOX_Long_Minutes",
            "Steering_Angle"
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
df_model = prepare_tickdb_dataframe_for_model(df_window, state, control)

# Utility: run prediction and return (lat_pred, lon_pred)
predictor = CarTrajectoryPredictor(
    state_cols=state,
    control_cols=control,
    model_path=str(MODEL_PATH / "new_multicar_multistep_model.pt"),
    scaler_path=str(MODEL_PATH / "new_multicar_multistep_scaler.pkl"),
    seq_len=10,
    scale=100.0,
)

def run_prediction_with_mod(df_mod: pd.DataFrame):
    true_lat, true_lon, lat_pred, lon_pred = predictor.predict(df_mod)
    return lat_pred, lon_pred

# ---- Create modified variants ----

df_variants = {}

# 1. Baseline (already done, but add for consistency)
df_variants["baseline"] = df_model.copy()

# 2. Double the steering angle
if "steering_angle" in df_model.columns:
    df_mod = df_model.copy()
    df_mod["steering_angle"] = df_mod["steering_angle"] * 2.0
    df_variants["steering_x2"] = df_mod
else:
    print("WARNING: steering_angle not found in df_model columns!")

# 3. Decrease both brake pressures by 25%
df_mod = df_model.copy()
df_mod["pbrake_f"] = df_mod["pbrake_f"] * 0.75
df_mod["pbrake_r"] = df_mod["pbrake_r"] * 0.75
df_variants["brakes_minus25pct"] = df_mod

# 4. Gear +1
df_mod = df_model.copy()
df_mod["gear"] = df_mod["gear"] + 1
df_variants["gear_plus1"] = df_mod

# 5. Gear -1
df_mod = df_model.copy()
df_mod["gear"] = df_mod["gear"] - 1
df_variants["gear_minus1"] = df_mod


# ---- Run predictions ----

pred_results = {}  # name -> (lat_pred, lon_pred)

for name, df_mod in df_variants.items():
    print(f"Running variant: {name}")
    lat_pred, lon_pred = run_prediction_with_mod(df_mod)
    pred_results[name] = (lat_pred, lon_pred)

lat_true, lon_true, _, _ = predictor.predict(df_model)

# 4. Plot map-style lat/lon for each variant
plt.figure(figsize=(8, 8))

plt.plot(lon_true, lat_true, label="True", linewidth=2, color="black")

# Plot each variant
for name, (lat_pred, lon_pred) in pred_results.items():
    plt.plot(lon_pred, lat_pred, label=name, linestyle="--", linewidth=1.5)

plt.scatter(lon_true[0], lat_true[0], c="green", marker="o", s=60, label="Start (true)")
plt.scatter(lon_true[-1], lat_true[-1], c="red", marker="x", s=60, label="End (true)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted vs True Trajectory (All Control Variants)")
plt.legend()
plt.gca().set_aspect("equal", "box")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
