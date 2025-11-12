
from os import rename

import pandas
from sqlalchemy.testing.util import total_size

from telemetry import VehicleRaceRecord
from telemetry.raw.TelemetryDB import TelemetryDB
from matplotlib import pyplot as plt
import pandas as pd


class data_clean:
    def __init__(self, db):
        self.db = db


    def data_each_car(db, vehicle_id):
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

    # gets common index, ensures timestamps are in datetime format.

    def index(list_dfs):
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
    def resample(df, common_index):
        df_resampled = df.copy()
        df = df[~df['timestamp'].duplicated()]
        df_new = df.set_index('timestamp', inplace=False)
        df_resampled['value'] = pd.to_numeric(df_resampled['value'], errors='coerce')

        df_resampled = df_new.reindex(common_index).interpolate(
            method='time')  # time’: Works on daily and higher resolution data to interpolate given length of interval.
        df_resampled['value'] = df_resampled['value'].ffill().bfill()
        df_resampled.drop(columns=['name'], inplace=True, errors='ignore')

        return df_resampled


