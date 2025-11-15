import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, CircleMarker } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import TickConsumerToggle from './components/TickConsumerToggle';
import VehicleMarkers from './components/VehicleMarkers';

const initialPosition: LatLngTuple = [33.5325017, -86.6215766];

// Map car IDs to their most recent position
export interface LatestPositions {
  [index: number]: LatLngTuple;
}

export default function App() {
  const [latestPositions, setLatestPositions] = useState<LatestPositions>({}); // Store the latest position

  const fetchLatestPosition = async () => {
    try {
      const response = await fetch('http://localhost:8000/latestAllFake/');
      if (!response.ok) throw new Error(`Failed to fetch latest position. Status: ${response.status}`);
      const data = await response.json();
      if (data == null) throw new Error('Invalid position data');
      const positions: LatestPositions = data;
      setLatestPositions(positions);
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

  return (
    <div>
      <TickConsumerToggle backendUrl="http://localhost:8000" />

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
