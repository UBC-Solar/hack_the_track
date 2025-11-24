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
    Prepare tickdb dataframe for RNN inference.
    Produces a dataframe identical in shape/semantics to the training input.
    """

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
        "Steering_Angle": "steering_angle",
    }

    df = df_tick.copy()
    df.rename(columns=rename_map, inplace=True)

    # ---- drop duplicates & sort (training pipeline did this) ----
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # ---- fill numeric columns like training did (minus interpolate) ----
    numeric_cols = [c for c in rename_map.values() if c in df.columns]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # forward/back fill
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # ---- ensure required fields exist ----
    base_state_cols = [c for c in state_cols if c not in ("x", "y")]
    required = set(base_state_cols + control_cols + ["latitude", "longitude"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tick DB dataframe missing fields: {missing}")

    # ---- final column ordering matches scaler ----
    ordered_cols = state_cols + control_cols
    df = df[ordered_cols]

    return df