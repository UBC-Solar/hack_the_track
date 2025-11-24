from __future__ import annotations
from typing import Optional, List
from sqlalchemy import create_engine, text
import pandas as pd
from backend.telemetry.raw.VehicleRaceRecord import VehicleRaceRecord
import datetime as dt

class TelemetryDB:
    """Database accessor for telemetry events and car race data."""

    def __init__(self, url: str):
        self.engine = create_engine(url, future=True)

    # ------------------------------
    # Listing / lookup
    # ------------------------------
    def list_car_races(self) -> List[VehicleRaceRecord]:
        """List all combinations of race, track, and vehicle."""
        q = text("""
            SELECT
            e.id          AS event_id,
            e.race_number,
            e.track_name,
            v.id          AS vehicle_id,
            v.code        AS vehicle_code
            FROM telem.event e
            JOIN LATERAL (
            SELECT DISTINCT f.vehicle_id
            FROM telem.stream_fast f
            WHERE f.event_id = e.id
            ) ev ON true
            JOIN telem.vehicle v ON v.id = ev.vehicle_id
            ORDER BY e.race_number, e.track_name, v.code;
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(q).mappings().all()
        return [
            VehicleRaceRecord(
                event_id=r["event_id"],
                race_number=r["race_number"],
                track_name=r["track_name"],
                vehicle_id=r["vehicle_id"],
                vehicle_code=r["vehicle_code"],
                db=self,
            )
            for r in rows
        ]

    def get_car_race(
        self,
        track: str,
        race_number: int,
        vehicle_code: str,
    ) -> Optional[VehicleRaceRecord]:
        """Retrieve a CarRaceData object for a specific car/race/track."""
        q = text("""
            SELECT e.id AS event_id, e.race_number, e.track_name,
                   v.id AS vehicle_id, v.code AS vehicle_code
            FROM telem.event e
            JOIN telem.vehicle v ON TRUE
            WHERE e.race_number = :r AND e.track_name = :t AND v.code = :v
        """)
        with self.engine.begin() as conn:
            r = conn.execute(q, {"r": race_number, "t": track, "v": vehicle_code}).mappings().first()
        if not r:
            return None
        return VehicleRaceRecord(
            event_id=r["event_id"],
            race_number=r["race_number"],
            track_name=r["track_name"],
            vehicle_id=r["vehicle_id"],
            vehicle_code=r["vehicle_code"],
            db=self,
        )

    def load_tick_window(
            self,
            race: VehicleRaceRecord,
            vehicle_id: int,
            duration_s: float = 5.0,
    ) -> pd.DataFrame:
        """
        Load ~`duration_s` seconds of fast-stream telemetry for one vehicle in one event.
        Assumes columns:
          - sample_time -> timestamp
          - accx, accy, speed, gear, aps, nmot, pbrake_f, pbrake_r, vbox_lat, vbox_lon
        """
        # Define time window
        end_time = dt.datetime.now(dt.timezone.utc)
        start_time = end_time - dt.timedelta(seconds=duration_s)

        query = text(
            """
            SELECT 
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
            WHERE vehicle_id = :vehicle_id
              AND ts >= :start_time
              AND ts
                < :end_time
            ORDER BY ts
            """
        )

        df = pd.read_sql(
            query,
            engine,
            params={
                "vehicle_id": vehicle_id,
                "start_time": start_time,
                "end_time": end_time,
            },
        )

        # Ensure time index is monotonic
        df = df.sort_values("timestamp").set_index("timestamp")
        return df

