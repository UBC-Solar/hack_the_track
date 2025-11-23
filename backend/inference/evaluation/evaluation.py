import numpy as np
from inference.evaluation import StatePosition
import pandas as pd


def trajectory_to_state_positions(lat: np.ndarray,
                                  lon: np.ndarray,
                                  timestamps: pd.DatetimeIndex) -> list[StatePosition]:
    """
    Convert (lat, lon, timestamps) into a list[StatePosition].
    We treat lon as x, lat as y.
    Time is seconds since the first timestamp.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Align length – in case your model outputs fewer points than df_window
    n = min(len(lat), len(lon), len(timestamps))
    lat = lat[:n]
    lon = lon[:n]
    ts = timestamps[:n]

    t0 = ts[0]
    times = (ts - t0).total_seconds()

    return [StatePosition(x=float(lon[i]),
                          y=float(lat[i]),
                          time=float(times[i]))
            for i in range(n)]