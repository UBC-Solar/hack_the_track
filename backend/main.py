# backend/main.py
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, MetaData, Table, select, desc
from fastapi.middleware.cors import CORSMiddleware
import os

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

    barber_tel_path = data_path / "barber-motorsports-park" / "barber" / "Race 1" / "R1_barber_telemetry_data.csv"

    lat_df, lon_df = get_lap_gps_data(barber_tel_path, lapNumber, chunksize=100000, chunk_limit=None)

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

PG_DSN = os.getenv("PG_DSN", "postgresql://telemetry:telemetry@localhost:5432/telemetry")
TICK_TABLE = os.getenv("TICK_TABLE", "telem_tick")

# Create SQLAlchemy engine
engine = create_engine(PG_DSN, future=True)

# Reflect table metadata (automatically load columns)
metadata = MetaData()
tick_table = Table(TICK_TABLE, metadata, autoload_with=engine)

@app.get("/latest")
def get_latest_row():
    """
    Fetch the most recent row from the telemetry tick table.
    """
    try:
        with engine.connect() as conn:
            stmt = (
                select(tick_table)
                .order_by(desc(tick_table.c.ts))
                .limit(1)
            )
            result = conn.execute(stmt).mappings().fetchone()

            if not result:
                raise HTTPException(status_code=404, detail="No telemetry data found")

            return dict(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(get_example_lap(4))
