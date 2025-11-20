from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Any
from sqlalchemy import text
import pandas as pd


@dataclass
class VehicleRaceRecord:
    """Represents telemetry data for one vehicle in one race at one track."""
    event_id: int
    race_number: int
    track_name: str
    vehicle_id: int
    vehicle_code: str
    db: Any

    # ------------------------------
    # Utility methods
    # ------------------------------
    def list_telemetry_names(self) -> List[str]:
        """Return all available telemetry signal names for this car/race."""
        q = text("""
            SELECT DISTINCT n.name
            FROM telem.stream_fast f
            JOIN telem.tname n ON n.id = f.name_id
            WHERE f.event_id = :eid AND f.vehicle_id = :vid
            ORDER BY n.name
        """)
        with self.db.engine.begin() as conn:
            return [r[0] for r in conn.execute(q, {"eid": self.event_id, "vid": self.vehicle_id})]

    def get_telemetry(
        self,
        name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a telemetry signal (and timestamp) for this car/race."""
        params = {"eid": self.event_id, "vid": self.vehicle_id, "name": name}
        cond = ""
        if start:
            cond += " AND f.timestamp >= :start"
            params["start"] = start
        if end:
            cond += " AND f.timestamp <= :end"
            params["end"] = end

        q = text(f"""
            SELECT f.timestamp, f.value, n.name
            FROM telem.stream_fast f
            JOIN telem.tname n ON n.id = f.name_id
            WHERE f.event_id = :eid AND f.vehicle_id = :vid AND n.name = :name
            {cond}
            ORDER BY f.timestamp
        """)
        with self.db.engine.begin() as conn:
            df = pd.read_sql(q, conn, params=params, parse_dates=["timestamp"])
        return df

    def get_telemetry_10s(
            self,
            name: str,
            start: Optional[str] = None,
            end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch telemetry signal within a 10-second window, keep original frequency."""

        params = {"eid": self.event_id, "vid": self.vehicle_id, "name": name}
        cond = ""
        if start:
            cond += " AND f.timestamp >= :start"
            params["start"] = start
            # Automatically set end = start + 10s if end is None
            if not end:
                cond += " AND f.timestamp < :end"
                params["end"] = pd.to_datetime(start) + pd.Timedelta(seconds=10)
        elif end:
            cond += " AND f.timestamp <= :end"
            params["end"] = end

        q = text(f"""
            SELECT f.timestamp, f.value, n.name
            FROM telem.stream_fast f
            JOIN telem.tname n ON n.id = f.name_id
            WHERE f.event_id = :eid
              AND f.vehicle_id = :vid
              AND n.name = :name
              {cond}
            ORDER BY f.timestamp
        """)
        with self.db.engine.begin() as conn:
            df = pd.read_sql(q, conn, params=params, parse_dates=["timestamp"])
        return df

