import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, CircleMarker } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import TickConsumerToggle from './components/TickConsumerToggle';
import VehicleMarkers from './components/VehicleMarkers';

const initialPosition: LatLngTuple = [33.5325017, -86.6215766];

// Interface to store position and its timestamp
interface PositionWithTimestamp {
  position: LatLngTuple;
  timestamp: number;
}

// Map car IDs to their most recent position
export interface LatestPositions {
  [index: number]: LatLngTuple;
}

export default function App() {
  const [latestPositions, setLatestPositions] = useState<LatestPositions>({}); // Store the latest position
  const [positionHistory, setPositionHistory] = useState<PositionWithTimestamp[]>([]); // Store position history

  const fetchLatestPosition = async () => {
    try {
      const response = await fetch('http://localhost:8000/latest/');
      if (!response.ok) throw new Error(`Failed to fetch latest position. Status: ${response.status}`);
      const data = await response.json();
      console.log(data)
      if (data.VBOX_Lat_Min === null || data.VBOX_Long_Minutes === null) throw new Error('Invalid position data');
      setLatestPosition([data.VBOX_Lat_Min, data.VBOX_Long_Minutes]);
    } catch (error: any) {
      console.error('Error fetching latest position:', error);
    }
  };

  // Polling function for fetching latest GPS position
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchLatestPosition();
    }, 100); // Poll period

    // Fetch the first position right away
    fetchLatestPosition();

    // Cleanup polling on component unmount
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array to run the effect only once on mount

  // Update position history when new position is fetched
  // useEffect(() => {
  //   if (latestPositions) {
  //     const timestamp = Date.now();
  //     const newPosition: PositionWithTimestamp = {
  //       position: latestPosition,
  //       timestamp,
  //     };

  //     // Filter out positions older than 60 seconds
  //     const updatedHistory = [
  //       ...positionHistory.filter((p) => timestamp - p.timestamp <= 60000),
  //       newPosition,
  //     ];

  //     setPositionHistory(updatedHistory);
  //   }
  // }, [latestPosition]);

  return (
    <div>
      <TickConsumerToggle backendUrl="http://localhost:8000" />

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

        {/* Polyline for the path (full trail of recent positions) */}
        {/* {positionHistory.length > 0 && (
          <Polyline
            positions={positionHistory.map((p) => p.position)}
            color="blue"
            weight={2}
            opacity={0.5}
          />
        )} */}

        {/* Display only the newest position as a marker */}
        {latestPositions && (
          <VehicleMarkers positions={latestPositions}}/>
        )}
      </MapContainer>
    </div>
  );
}
