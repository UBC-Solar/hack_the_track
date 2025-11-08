// src/App.tsx
import { useState } from 'react';
import { MapContainer, TileLayer, Polyline } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import FloatingForm from './components/FloatingForm';
import TickConsumerToggle from './components/TickConsumerToggle';

const position: LatLngTuple = [33.5325017, -86.6215766];

export default function App() {
  const [lapNumber, setLapNumber] = useState<number>(1);
  const [samplePeriod, setSamplePeriod] = useState<number>(1);
  const [gpsData, setGpsData] = useState<{ lat_vals: number[]; lon_vals: number[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

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

      <TickConsumerToggle backendUrl="http://localhost:8000" />

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