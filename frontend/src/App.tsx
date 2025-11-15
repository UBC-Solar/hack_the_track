// src/App.tsx
import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Polyline, CircleMarker } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import FloatingForm from './components/FloatingForm';
import TickConsumerToggle from './components/TickConsumerToggle';
import LapDisplay from './components/LapsDisplay';

const initialPosition: LatLngTuple = [33.5325017, -86.6215766];

export default function App() {
  const [lapNumber, setLapNumber] = useState<number>(1);
  const [samplePeriod, setSamplePeriod] = useState<number>(1);
  const [error, setError] = useState<string | null>(null);
  const [latestPosition, setLatestPosition] = useState<LatLngTuple | null>(null); // Store the latest position

  // Mocked values for back end 
  const LapTime = 10;
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [laps, setLaps] = useState<Array<{ number: number; time: number }>>([
    { number: 1, time: 92.5 },
    { number: 2, time: 90.2 },
    { number: 3, time: 95.1 },
  ]);
  const lapAppended = useRef(false); 

  const handleLapNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const cleanedValue = value.replace(/^0+/, '') || '0';
    setLapNumber(parseInt(cleanedValue, 10));
  };

  const handleSamplePeriodChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSamplePeriod(Number(value));
  };

  const fetchLatestPosition = async () => {
    try {
      const response = await fetch('http://localhost:8000/latest/');
      if (!response.ok) throw new Error(`Failed to fetch latest position. Status: ${response.status}`);
      const data = await response.json();
      if (data.VBOX_Lat_Min === null || data.VBOX_Long_Minutes === null) throw new Error('Invalid position data');
      setLatestPosition([data.VBOX_Lat_Min, data.VBOX_Long_Minutes]);
    } catch (error: any) {
      console.error('Error fetching latest position:', error);
      setError(error.message);
    }
  };

  // Polling function for fetching latest GPS position
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchLatestPosition();
    }, 100); // Poll period ms

    // Fetch the first position right away
    fetchLatestPosition();

    // Cleanup polling on component unmount
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array to run the effect only once on mount

// Increments the timer (THIS WILL BE DELETED LATER !!!)
  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentTime(prev => parseFloat((prev + 0.1).toFixed(1)));
    }, 100);

    return () => clearInterval(intervalId);
  }, []);

  // Watch currentTime and append lap when threshold is crossed
  useEffect(() => {
    if (currentTime >= LapTime && !lapAppended.current) {
      setLaps(prev => [...prev, { number: prev.length + 1, time: currentTime }]);
      setCurrentTime(0); // reset timer
      lapAppended.current = true; // mark as appended
    }

    if (currentTime < LapTime) {
      lapAppended.current = false; // reset flag for next lap
    }
  }, [currentTime]);


  return (
    <div>
      <TickConsumerToggle backendUrl="http://localhost:8000" />

      <LapDisplay
        currentLap={4} // Locked for now, should come from backend
        currentTime={currentTime} // Current time for the lap is being mocked
        laps={laps} // Mocked laps data until its added
      />

      <MapContainer
        center={initialPosition}
        zoom={16}
        scrollWheelZoom={true}
        style={{ height: '100vh', width: '100vw' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {latestPosition && (
          <CircleMarker
            center={latestPosition}
            radius={10}
            fillColor="red"
            color="white"
            fillOpacity={0.8}
          />
        )}
      </MapContainer>
    </div>
  );
}
