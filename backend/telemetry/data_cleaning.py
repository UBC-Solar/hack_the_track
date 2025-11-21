
from os import rename

import pandas
from sqlalchemy.testing.util import total_size

from telemetry import VehicleRaceRecord
from telemetry.raw.TelemetryDB import TelemetryDB
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np




class data_clean:
    def __init__(self, db):
        self.db = db


    def data_each_car(self, db, vehicle_id):
        # ignoring ath for now
        list_all_dfs = []
        car = db.get_car_race(track="barber", race_number=2, vehicle_code=vehicle_id)

        if car:
            df_accx = car.get_telemetry("accx_can")
            df_accy = car.get_telemetry("accy_can")
            df_speed = car.get_telemetry("speed")
            df_ath = car.get_telemetry("ath")
            df_gear = car.get_telemetry("gear")
            df_aps = car.get_telemetry("aps")
            df_nmotor = car.get_telemetry("nmot")

            df_pbrake_f = car.get_telemetry("pbrake_f")
            df_pbrake_r = car.get_telemetry("pbrake_r")
            list_all_dfs = [df_accx, df_accy, df_speed, df_gear, df_aps, df_nmotor, df_pbrake_f, df_pbrake_r]
        return list_all_dfs


    def combine_dfs_car(self, telemetry_names, index_common, all_dfs):
        combined_df = pd.DataFrame(index=index_common)

        for name, df in zip(telemetry_names, all_dfs):
            df_interp = self.resample(df, index_common)
            combined_df[name] = pd.to_numeric(df_interp['value'], errors='coerce').values

        return combined_df

    # gets common index, ensures timestamps are in datetime format.

    def index(self, list_dfs):
        for i, df in enumerate(list_dfs):
            list_dfs[i] = df.copy()
            list_dfs[i]['timestamp'] = pd.to_datetime(list_dfs[i]['timestamp'], unit='ns')
            if 'telemetry_value' in list_dfs[i].columns:
                list_dfs[i].rename(columns={'telemetry_value': 'value'},
                                   inplace=True)  # rename everything to values for easier access

        start_time = min(df['timestamp'].min() for df in list_dfs)
        end_time = max(df['timestamp'].max() for df in list_dfs)
        common_index = pd.date_range(start=start_time, end=end_time, freq='1ms')
        return common_index, list_dfs

    # resample and interpolate data
    def resample(self, df, common_index):
        df_resampled = df.copy()
        df = df[~df['timestamp'].duplicated()]
        df_new = df.set_index('timestamp', inplace=False)
        df_resampled['value'] = pd.to_numeric(df_resampled['value'], errors='coerce')

        df_resampled = df_new.reindex(common_index).interpolate(
            method='time')  # time’: Works on daily and higher resolution data to interpolate given length of interval.
        df_resampled['value'] = df_resampled['value'].ffill().bfill()
        df_resampled.drop(columns=['name'], inplace=True, errors='ignore')

        return df_resampled

    def extract_10s(self, df, start_ts, ts_col="timestamp", sample_count=None):

        df = df.copy()
        start_ts = pd.to_datetime(start_ts, utc=True)

        # determine timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            timestamps = df[ts_col]
        else:
            raise ValueError("No datetime index or timestamp column found.")

        # slice by nearest timestamp
        if sample_count is not None:

            diffs = np.abs((timestamps - start_ts).total_seconds())
            nearest_idx = diffs.argmin()
            end_idx = nearest_idx + sample_count
            df_slice = df.iloc[nearest_idx:end_idx]
        else:
            end_ts = start_ts + pd.Timedelta(seconds=10)
            if isinstance(df.index, pd.DatetimeIndex):
                df_slice = df.loc[start_ts:end_ts]
            else:
                df_slice = df[(timestamps >= start_ts) & (timestamps < end_ts)]

        return df_slice.reset_index(drop=False)









