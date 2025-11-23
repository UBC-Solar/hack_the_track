import pandas as pd


def index(list_dfs):
    """
    Standardize timestamp column, rename 'telemetry_value' -> 'value',
    and compute a global common index at 50ms.
    """
    for i, df in enumerate(list_dfs):
        list_dfs[i] = df.copy()
        list_dfs[i]["timestamp"] = pd.to_datetime(
            list_dfs[i]["timestamp"], unit="ns"
        )
        if "telemetry_value" in list_dfs[i].columns:
            list_dfs[i].rename(columns={"telemetry_value": "value"}, inplace=True)

    start_time = min(df["timestamp"].min() for df in list_dfs)
    end_time   = max(df["timestamp"].max() for df in list_dfs)
    common_index = pd.date_range(start=start_time, end=end_time, freq="50ms")
    return common_index, list_dfs


def resample(df, common_index):
    """
    Resample one telemetry DataFrame onto the common index and interpolate in time.
    """
    df_resampled = df.copy()
    df = df[~df["timestamp"].duplicated()]
    df_new = df.set_index("timestamp", inplace=False)
    df_resampled["value"] = pd.to_numeric(df_resampled["value"], errors="coerce")

    df_resampled = df_new.reindex(common_index).interpolate(method="time")
    df_resampled["value"] = df_resampled["value"].ffill().bfill()
    df_resampled.drop(columns=["name"], inplace=True, errors="ignore")

    return df_resampled


def combine_dfs_car(telemetry_names, common_index, all_dfs):
    """
    Resample each telemetry DataFrame and combine them into a single wide DataFrame.
    """
    combined_df = pd.DataFrame(index=common_index)

    for name, df in zip(telemetry_names, all_dfs):
        df_interp = resample(df, common_index)
        combined_df[name] = pd.to_numeric(df_interp["value"], errors="coerce").values

    return combined_df


def prepare_tickdb_dataframe_for_model(df_tick, state_cols, control_cols):
    """
    Convert a dataframe pulled from tickdb into the unified dataframe format
    used to train the RNN model.

    - Renames tickdb signal names → model names
    - Adds x/y from latitude/longitude
    - Returns (df_ready, origin_rad)

    Assumes:
      - df_tick has a proper datetime index
      - already at correct sampling rate
      - no resampling required
    """

    # --- rename tickdb fields → model fields ---
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

    df = df_tick.copy()
    df.rename(columns=rename_map, inplace=True)

    df = df_tick.copy()
    df.rename(columns=rename_map, inplace=True)

    # basic forward-fill/backward-fill on the raw columns we care about
    fill_cols = list(rename_map.values())
    df[fill_cols] = df[fill_cols].ffill().bfill()

    # --- required *raw* fields (exclude x,y because we create them) ---
    base_state_cols = [c for c in state_cols if c not in ("x", "y")]
    required = set(base_state_cols + control_cols + ["latitude", "longitude"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tick DB dataframe missing fields: {missing}")

    return df
