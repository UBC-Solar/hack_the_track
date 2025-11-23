import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from raw.TelemetryDB import TelemetryDB
from inference.prediction import index, combine_dfs_car

telemetry_names = [
    "accx", "accy", "speed", "gear", "aps",
    "nmot", "pbrake_f", "pbrake_r", "latitude", "longitude", "steering_angle"
]

def data_each_car(db: TelemetryDB, vehicle_id: str, race_number: int = 2):
    """
    Fetch raw telemetry DataFrames for one car.
    Returns a list of DataFrames in a fixed order matching telemetry_names.
    """
    car = db.get_car_race(track="barber", race_number=race_number, vehicle_code=vehicle_id)

    if not car:
        return []

    df_accx      = car.get_telemetry("accx_can")
    df_accy      = car.get_telemetry("accy_can")
    df_speed     = car.get_telemetry("speed")
    df_ath       = car.get_telemetry("ath")          # currently unused
    df_gear      = car.get_telemetry("gear")
    df_aps       = car.get_telemetry("aps")
    df_nmotor    = car.get_telemetry("nmot")
    df_latitude  = car.get_telemetry("VBOX_Lat_Min")
    df_longitude = car.get_telemetry("VBOX_Long_Minutes")
    df_pbrake_f  = car.get_telemetry("pbrake_f")
    df_pbrake_r  = car.get_telemetry("pbrake_r")
    df_steering  = car.get_telemetry("Steering_Angle")

    # order must match telemetry_names
    list_all_dfs = [
        df_accx, df_accy, df_speed, df_gear, df_aps,
        df_nmotor, df_pbrake_f, df_pbrake_r, df_latitude, df_longitude, df_steering
    ]
    return list_all_dfs

def build_car_dataframe(db: TelemetryDB, car_name: str, race_number: int = 2):
    """
    High-level helper:
      - fetch telemetry for a car
      - align on common index
      - combine into a single DataFrame
      - compute local x/y from lat/lon

    Returns:
      df_xy      : DataFrame with states/controls + x/y
      origin_rad : (lat0, lon0) in radians
    """
    telemetry_list = data_each_car(db, car_name, race_number)
    if not telemetry_list:
        raise RuntimeError(f"No telemetry found for car {car_name!r}")

    common_index, list_dfs = index(telemetry_list)
    final_df_car = combine_dfs_car(telemetry_names, common_index, list_dfs)

    return final_df_car


DB_TELEM = "postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing"
DB_TICK  = "postgresql+psycopg2://telemetry:telemetry@localhost:5432/telemetry"

car_name   = "GR86-022-13"
race_number = 2
vehicle_id_tick = 36  # whatever matches this car in telem_tick

# --- 1) load training-style df from TelemetryDB ---
db = TelemetryDB(DB_TELEM)
df_car = build_car_dataframe(db, car_name, race_number=race_number)  # same helper used for training
df_car = df_car.copy()
df_car.index = pd.to_datetime(df_car.index)

# we only need a slice similar to what you use for inference
df_car_slice = df_car.iloc[2000:2000+2000]  # tweak indices as you like

# --- 2) load tickdb window covering ~same time span, in *wall time* ---
engine_tick = create_engine(DB_TICK)

# we don't know wall_ts matching that slice, so just take a fresh window:
df_tick = pd.read_sql(
    """
    SELECT ts AS timestamp,
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
    WHERE vehicle_id = %(vid)s
    ORDER BY ts
    LIMIT 2000
    """,
    engine_tick,
    params={"vid": vehicle_id_tick},
)

df_tick = df_tick.sort_values("timestamp").set_index("timestamp")

# rename to match training df
rename_map = {
    "accx_can": "accx",
    "accy_can": "accy",
    "speed": "speed",
    "gear": "gear",
    "aps": "aps",
    "nmot": "nmot",
    "pbrake_f": "pbrake_f",
    "pbrake_r": "pbrake_r",
    "VBOX_Lat_Min": "latitude",
    "VBOX_Long_Minutes": "longitude",
}
df_tick = df_tick.rename(columns=rename_map)

# --- 3) align them by nearest-in-time index ---
aligned = pd.merge_asof(
    df_tick.sort_index(),
    df_car_slice.sort_index(),
    left_index=True,
    right_index=True,
    suffixes=("_tick", "_train"),
)

cols = ["accx", "accy", "speed", "gear", "aps", "nmot", "pbrake_f", "pbrake_r", "latitude", "longitude"]

print("Mean abs diff per column (tickdb vs TelemetryDB training df):")
for c in cols:
    diff = aligned[f"{c}_tick"] - aligned[f"{c}_train"]
    print(f"{c:10s}: {np.nanmean(np.abs(diff)):.6f}")

print("\nStd of diff per column:")
for c in cols:
    diff = aligned[f"{c}_tick"] - aligned[f"{c}_train"]
    print(f"{c:10s}: {np.nanstd(diff):.6f}")