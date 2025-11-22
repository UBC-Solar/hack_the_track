import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import TickConsumerToggle from './components/TickConsumerToggle';
import VehicleMarkers from './components/VehicleMarkers';
import LapDisplay from './components/LapsDisplay';
import VehicleSelector from './components/VehicleSelector';


// ================ CONSTANTS ================

const initialPosition: LatLngTuple = [33.5325017, -86.6215766];

// Map car IDs to their most recent position
export interface LatestPositions {
  [index: number]: LatLngTuple;
}

export default function App() {

  // ================ STATE ================

  // Latest vehicle positions
  const [latestPositions, setLatestPositions] = useState<LatestPositions>({});

  // Laps numbers and times
  const [currentLap, setCurrentLap] = useState<number>(1);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [laps, setLaps] = useState<Array<{ number: number; time: number }>>([]);

  // Vehicle selection
  const [vehicles, setVehicles] = useState<Array<number>>([]);
  const [selectedVehicleID, setSelectedVehicleID] = useState<number | null>(null);
  const [showOption, setShowOption] = useState<'all' | 'primary'>('all');


  // ================ QUERY BACKEND ================

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
    try {
      const response = await fetch(`http://localhost:8000/currentLaps?vehicleID=${selectedVehicleID}`);

      // Handle non-2xx status codes
      if (!response.ok) {
        console.error(`Lap fetch failed: ${response.status} ${response.statusText}`);
        setLaps([]);    // fallback
        return;
      }

      const data = await response.json();

      // Validate array shape
      if (Array.isArray(data)) {
        setLaps(data);
      } else {
        console.error("Lap fetch returned non-array:", data);
        setLaps([]);    // fallback
      }

    } catch (error) {
      // Handles network errors, server down, CORS failures, JSON fail, etc.
      console.error('Error fetching laps:', error);
      setLaps([]);      // fallback
    }
  };

  const fetchCurrentLap = async () => {
    try {
      const response = await fetch(`http://localhost:8000/currentLap?vehicleID=${selectedVehicleID}`);

      // Handle non-2xx status codes
      if (!response.ok) {
        console.error(`Lap fetch failed: ${response.status} ${response.statusText}`);
        return;
      }

      const data = await response.json();

      setCurrentLap(data["currentLap"]);

    } catch (error) {
      // Handles network errors, server down, CORS failures, JSON fail, etc.
      console.error('Error fetching laps:', error);
    }
  };

    const fetchCurrentTime = async () => {
    try {
      const response = await fetch(`http://localhost:8000/currentLapTime?vehicleID=${selectedVehicleID}`);

      // Handle non-2xx status codes
      if (!response.ok) {
        console.error(`Lap fetch failed: ${response.status} ${response.statusText}`);
        return;
      }

      const data = await response.json();

      // Validate array shape
      setCurrentTime(data["currentLapTime"]);

    } catch (error) {
      // Handles network errors, server down, CORS failures, JSON fail, etc.
      console.error('Error fetching laps:', error);
    }
  };

  const fetchVehicles = async () => {
    try {
      const response = await fetch('http://localhost:8000/vehicles'); // Your endpoint
      if (!response.ok) {
        throw new Error(`Failed to fetch vehicles. Status: ${response.status}`);
      }

      const data = await response.json();

      // Validate that the data is an array of numbers (vehicle IDs)
      if (Array.isArray(data) && data.every((item: any) => typeof item === 'number')) {
        setVehicles(data); // return the array of vehicle IDs
      } else {
        throw new Error('Invalid vehicle data');
      }

    } catch (error: any) {
      console.error('Error fetching vehicles:', error);
      setVehicles([]); // Return an empty array in case of error
    }
  };

  // ================ POLLING LOOPS ================

  // Poll latest position of all cars
  const positionPollMs = 50;
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchLatestPosition();
      fetchVehicles();

      // Only fetch latest laps if selectedVehicleID is not null
      if (selectedVehicleID !== null) {
        fetchLatestLaps();
        fetchCurrentLap();
        fetchCurrentTime();
      }

    }, positionPollMs);

    // Fetch the first position right away
    fetchLatestPosition();

    // Cleanup polling on component unmount
    return () => clearInterval(intervalId);
  }, [selectedVehicleID]); // Add selectedVehicleID as a dependency

  // ================ RETURN ================

  return (
    <div>
      <TickConsumerToggle backendUrl="http://localhost:8000" />

      {selectedVehicleID && (
        <LapDisplay
          currentLap={currentLap} // Locked for now, should come from backend
          currentTime={currentTime} // Current time for the lap is being mocked
          laps={laps} // Mocked laps data until its added
        />
      )}

      <VehicleSelector
        vehicleID={selectedVehicleID}
        showOption={showOption}
        vehicles={vehicles}
        setVehicleID={setSelectedVehicleID}
        setShowOption={setShowOption}
      />

      <MapContainer
        center={initialPosition}
        zoom={16}
        zoomControl={false}
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
          <VehicleMarkers positions={latestPositions} showOption={showOption} selectedVehicleID={selectedVehicleID}/>
        )}
      </MapContainer>
    </div>
  );
}
