// src/App.tsx
import { useState } from 'react';
import { MapContainer, TileLayer, Polyline } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import FloatingForm from './components/FloatingForm';

const position: LatLngTuple = [33.5325017, -86.6215766];

export default function App() {
  const [lapNumber, setLapNumber] = useState<number>(1);
  const [samplePeriod, setSamplePeriod] = useState<number>(1);
  const [gpsData, setGpsData] = useState<{ lat_vals: number[]; lon_vals: number[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // NEW: tick-consumer toggle state (optimistic)
  const [writeEnabled, setWriteEnabled] = useState<boolean>(true);
  const [toggling, setToggling] = useState<boolean>(false);
  const [toggleError, setToggleError] = useState<string | null>(null);

  const handleLapNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const cleanedValue = value.replace(/^0+/, '') || '0';
    setLapNumber(parseInt(cleanedValue, 10));
  };

  const handleSamplePeriodChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSamplePeriod(Number(value));
  };

  const fetchGpsData = async () => {
    try {
      const response = await fetch(`http://localhost:8000/laps/?lapNumber=${lapNumber}&samplePeriod=${samplePeriod}`);
      if (!response.ok) throw new Error(`Failed to fetch data. Status: ${response.status}`);
      const data = await response.json();
      if (!data.lat_vals || !data.lon_vals) throw new Error('Invalid data structure: Missing lat_vals or lon_vals');
      setGpsData(data);
      setError(null);
    } catch (error: any) {
      console.error('Error fetching GPS data:', error);
      setError(error.message);
      setGpsData(null);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchGpsData();
  };

  const polylineCoordinates = gpsData
    ? gpsData.lat_vals.map((lat, i) => [lat, gpsData.lon_vals[i]] as LatLngTuple)
    : [];

  // NEW: backend call for toggling the consumer
  const sendToggle = async (enable: boolean) => {
    setToggling(true);
    setToggleError(null);
    const prev = writeEnabled;
    setWriteEnabled(enable); // optimistic

    try {
      const res = await fetch('http://localhost:8000/control/tick-consumer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Toggle failed (${res.status}): ${text}`);
      }
      // success: nothing else to do
    } catch (err: any) {
      // rollback optimistic UI
      setWriteEnabled(prev);
      setToggleError(err.message ?? 'Failed to toggle tick-consumer');
    } finally {
      setToggling(false);
    }
  };

  return (
    <div>
      <FloatingForm
        lapNumber={lapNumber}
        samplePeriod={samplePeriod}
        error={error}
        onSubmit={handleSubmit}
        onLapNumberChange={handleLapNumberChange}
        onSamplePeriodChange={handleSamplePeriodChange}
      />

      {/* NEW: simple overlay for the toggle */}
      <div
        style={{
          position: 'fixed',
          top: 12,
          right: 12,
          zIndex: 1000,
          background: 'rgba(255,255,255,0.95)',
          padding: '10px 12px',
          borderRadius: 8,
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          minWidth: 220,
          fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <label htmlFor="toggle-write" style={{ fontWeight: 600 }}>
            Tick consumer
          </label>
          <input
            id="toggle-write"
            type="checkbox"
            checked={writeEnabled}
            disabled={toggling}
            onChange={(e) => sendToggle(e.target.checked)}
            style={{ transform: 'scale(1.25)', cursor: toggling ? 'not-allowed' : 'pointer' }}
            aria-label="Enable or disable tick consumer"
          />
          <span style={{ fontSize: 12, opacity: 0.75 }}>
            {toggling ? 'Updating…' : writeEnabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
        {toggleError && (
          <div style={{ marginTop: 6, color: '#b00020', fontSize: 12 }}>
            {toggleError}
          </div>
        )}
      </div>

      <MapContainer
        center={position}
        zoom={16}
        scrollWheelZoom={true}
        style={{ height: '100vh', width: '100vw' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {gpsData && <Polyline positions={polylineCoordinates} color="blue" />}
      </MapContainer>
    </div>
  );
}