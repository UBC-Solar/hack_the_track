-- Database objects for time-bucketed snapshots
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- optional, handy later

-- Wide table: one row per tick per vehicle, columns for telemetry names will be added at runtime
CREATE TABLE IF NOT EXISTS telem_tick (
  ts          timestamptz NOT NULL,
  vehicle_id  integer     NOT NULL,
  PRIMARY KEY (ts, vehicle_id)
);

-- Fast time-range queries
CREATE INDEX IF NOT EXISTS idx_telem_tick_ts ON telem_tick (ts);

-- Fast “by vehicle in time range”
CREATE INDEX IF NOT EXISTS idx_telem_tick_vid_ts ON telem_tick (vehicle_id, ts);