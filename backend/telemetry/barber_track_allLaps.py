import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from telemetry.raw.TelemetryDB import TelemetryDB

db = TelemetryDB("postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing")

data_path = Path(r"C:\Users\sanar\PycharmProjects\hack_the_track\backend\R1_barber_telemetry_data.csv")


def get_lap_data(path: Path, lap, vehicle, telemetry):
    df = pd.read_csv(path)

    return df[
        (df['lap'] == lap) &
        (df['original_vehicle_id'] == vehicle) &
        (df['telemetry_name'] == telemetry)
        ]


# question if getTelemetry is separated by lap??

if __name__ == "__main__":
    barber_tel_path = data_path / "barber-motorsports-park" / "barber" / "Race 1" / "R1_barber_telemetry_data.csv"

    # Create the figure and a 3x5 grid of subplots
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Flatten the 2D array of axes to make indexing easier
    axes = axes.flatten()

    vehicle = "GR86-002-000"

    # event_id = r["event_id"],
    # race_number = r["race_number"],
    # track_name = r["track_name"],
    # vehicle_id = r["vehicle_id"],
    # vehicle_code = r["vehicle_code"],
    df_all_vehicles = pd.DataFrame()
    for car_race in db.list_car_races():
        df_all_vehicles = car_race.vehicle_code

    i = 0
    vehicle = df_all_vehicles[i]

#plot all

    for vehicle in df_all_vehicles:
        for i, lap_number in enumerate(range(2, 30)):
            df_long = get_lap_data(barber_tel_path, lap_number, vehicle, "VBOX_Long_Minutes")
            df_lat = get_lap_data(barber_tel_path, lap_number, vehicle, "VBOX_Lat_Minutes")
            ax = axes[i]

            # Plot on the current axis: latitude vs longitude
            ax.scatter(df_lat, df_long, c=range(len(df_lat)))
            ax.set_title(f"Lap {lap_number}")
            ax.set_xlabel("Latitude (Min)")
            ax.set_ylabel("Longitude (Min)")

        # Adjust layout to prevent overlapping


plt.tight_layout()
plt.show()
