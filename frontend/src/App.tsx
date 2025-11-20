import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import TickConsumerToggle from './components/TickConsumerToggle';
import VehicleMarkers from './components/VehicleMarkers';
import LapDisplay from './components/LapsDisplay';

const initialPosition: LatLngTuple = [33.5325017, -86.6215766];

// Map car IDs to their most recent position
export interface LatestPositions {
  [index: number]: LatLngTuple;
}

export default function App() {
  const [latestPositions, setLatestPositions] = useState<LatestPositions>({}); // Store the latest position

  // Mocked values for back end 
  const LapTime = 10;
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [laps, setLaps] = useState<Array<{ number: number; time: number }>>([]);
  const lapAppended = useRef(false); 

  const fetchLatestPosition = async () => {
    try {
      // Use /latestAll for actual data, or latestAllFake for mocked data
      const response = await fetch('http://localhost:8000/latestAll/');
      if (!response.ok) throw new Error(`Failed to fetch latest position. Status: ${response.status}`);
      const data = await response.json();
      if (data == null) throw new Error('Invalid position data');
      const positions: LatestPositions = data;
      setLatestPositions(positions);
    } catch (error: any) {
      console.error('Error fetching latest position:', error);
    }
  };

  const fetchLatestLaps = async () => {
    const response = await fetch('http://localhost:8000/currentLaps/');
    
    const data = await response.json();
    setLaps(data);
  }

  // Polling function for fetching latest GPS position
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchLatestPosition();
      fetchLatestLaps();
    }, 50); // Poll period

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
        {/* Use Google Maps Satellite tiles */}
        <TileLayer
          attribution='&copy; <a href="https://www.esri.com/en-us/arcgis/products/arcgis-online">Esri</a> contributors'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />

        {/* Display only the newest position as a marker */}
        {latestPositions && (
          <VehicleMarkers positions={latestPositions}/>
        )}
      </MapContainer>
    </div>
  );
}
