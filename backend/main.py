# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from load_gps_data import get_lap_gps_data, data_path

app = FastAPI()

# Allow frontend to call backend (important for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.get("/laps/")
def get_example_lap(lapNumber: int, samplePeriod: int = 1):

    lat_df, lon_df = get_lap_gps_data(data_path, lapNumber, chunksize=100000, chunk_limit=None)

    # Validate the data
    # assert lat_df['telemetry_name'].unique() == ['VBOX_Lat_Min']
    # assert lon_df['telemetry_name'].unique() == ['VBOX_Long_Minutes']
    # assert lat_df['lap'].unique() == [lap_number]
    # assert lon_df['lap'].unique() == [lap_number]
    # assert len(lat_df['vehicle_number'].unique()) == 1
    # assert len(lon_df['vehicle_number'].unique()) == 1

    # Get the latitude and longitude values
    lat_vals = lat_df['telemetry_value'][::samplePeriod]
    lon_vals = lon_df['telemetry_value'][::samplePeriod]

    return {"lat_vals": list(lat_vals), "lon_vals": list(lon_vals)}

if __name__ == "__main__":
    print(get_example_lap(4))