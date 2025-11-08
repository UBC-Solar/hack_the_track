# Functions for querying files directly from a local CSV
# Will not work in Docker; this is only for development/testing

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

data_path = Path(r"C:\Users\sanar\PycharmProjects\hack_the_track\backend\R1_barber_telemetry_data.csv")

def get_lap_gps_data(path: Path, lap: int, chunksize: int = 1000, chunk_limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        reader = pd.read_csv(r"C:\Users\sanar\PycharmProjects\hack_the_track\backend\R1_barber_telemetry_data.csv", chunksize=chunksize)

        lat_df = None
        long_df = None
        reached_lap = False
        for i, chunk in enumerate(reader):

            if chunk_limit is not None and i >= chunk_limit:
                break

            if not reached_lap and lap in chunk['lap']:
                # Reached data for the desired lap for the first time
                reached_lap = True

            if reached_lap and lap not in chunk['lap']:
                # Finished going through data for the desired lap
                break

            lap_data = chunk[chunk['lap'] == lap]

            # Get GPS data rows
            to_add_lat = lap_data[lap_data['telemetry_name'] == 'VBOX_Lat_Min']
            to_add_long = lap_data[lap_data['telemetry_name'] == 'VBOX_Long_Minutes']

            # Add this chunk's data
            lat_df = pd.concat([lat_df, to_add_lat]) if lat_df is not None else to_add_lat
            long_df = pd.concat([long_df, to_add_long]) if long_df is not None else to_add_long

        return lat_df, long_df

if __name__ == "__main__":

    barber_tel_path = data_path / "barber-motorsports-park" / "barber" / "Race 1" / "R1_barber_telemetry_data.csv"
#
#     # NOTE: I haven't resampled timestamps properly for this plot; it is assumed that the frequency and spacing of lat/lon
#     #       data points is identical. This seems to be a decent assumption, by looking at the plots.
#
    # Create the figure and a 3x5 grid of subplots
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Flatten the 2D array of axes to make indexing easier
    axes = axes.flatten()

    # Loop through the first 15 laps
    for i, lap_number in enumerate(range(2, 17)):
        # Get the GPS data for this lap
        lat_df, lon_df = get_lap_gps_data(barber_tel_path, lap_number, chunksize=100000, chunk_limit=None)

        # Validate the data
        # assert lat_df['telemetry_name'].unique() == ['VBOX_Lat_Min']
        # assert lon_df['telemetry_name'].unique() == ['VBOX_Long_Minutes']
        # assert lat_df['lap'].unique() == [lap_number]
        # assert lon_df['lap'].unique() == [lap_number]
        # assert len(lat_df['vehicle_number'].unique()) == 1
        # assert len(lon_df['vehicle_number'].unique()) == 1

        # Get the latitude and longitude values
        lat_vals = lat_df['telemetry_value']
        lon_vals = lon_df['telemetry_value']

        # Select the axis for the current lap
        ax = axes[i]

        # Plot on the current axis
        ax.scatter(lat_vals, lon_vals, c=range(len(lat_vals)))
        ax.set_title(f"Lap {lap_number}")
        ax.set_xlabel("Latitude (Min)")
        ax.set_ylabel("Longitude (Min)")

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


