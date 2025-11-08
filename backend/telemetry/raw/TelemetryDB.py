from __future__ import annotations
from typing import Optional, List
from sqlalchemy import create_engine, text
import pandas as pd
from telemetry.raw import VehicleRaceRecord


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
