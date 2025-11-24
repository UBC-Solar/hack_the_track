import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
import type { LatLngTuple } from 'leaflet';
import TickConsumerToggle from './components/TickConsumerToggle';
import VehicleMarkers from './components/VehicleMarkers';
import LapDisplay from './components/LapsDisplay';
import VehicleSelector from './components/VehicleSelector';
import DriverInsightsList from './components/DriverInsightsList';

// ================ CONSTANTS ================

const initialPosition: LatLngTuple = [33.5325017, -86.6215766];

// Map car IDs to their most recent position
export interface LatestPositions {
  [index: number]: LatLngTuple;
}

export interface DriverInsight {
  startPosition: LatLngTuple;
  insight: string;
}

export default function App() {
  // ================ STATE ================
  const [latestPositions, setLatestPositions] = useState<LatestPositions>({});
  const [currentLap, setCurrentLap] = useState<number>(1);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [laps, setLaps] = useState<Array<{ number: number; time: number }>>([]);
  const [vehicles, setVehicles] = useState<Array<string>>([]);
  const [selectedVehicleID, setSelectedVehicleID] = useState<string | null>(null);
  const [showOption, setShowOption] = useState<'all' | 'primary'>('all');
  const [latestInsight, setLatestInsight] = useState<DriverInsight | null>(null);
  const [driverInsightList, setDriverInsightList] = useState<Array<[string, number]>>([]);

  // ================ POPUP STATE ================
  const [showPopup, setShowPopup] = useState(true); // Popup visibility state

  const closePopup = () => {
    setShowPopup(false); // Close the popup
  };

  // ================ QUERY BACKEND ================

  const fetchLatestPosition = async () => {
    try {
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
      if (!response.ok) {
        console.error(`Lap fetch failed: ${response.status} ${response.statusText}`);
        setLaps([]); 
        return;
      }
      const data = await response.json();
      if (Array.isArray(data)) {
        setLaps(data);
      } else {
        console.error("Lap fetch returned non-array:", data);
        setLaps([]);
      }
    } catch (error) {
      console.error('Error fetching laps:', error);
      setLaps([]); 
    }
  };

  const fetchVehicles = async () => {
    try {
      const response = await fetch('http://localhost:8000/vehicles');
      if (!response.ok) {
        throw new Error(`Failed to fetch vehicles. Status: ${response.status}`);
      }
      const data = await response.json();
      if (Array.isArray(data) && data.every((item: any) => typeof item === 'string')) {
        setVehicles(data);
      } else {
        throw new Error('Invalid vehicle data');
      }
    } catch (error: any) {
      console.error('Error fetching vehicles:', error);
      setVehicles([]);
    }
  };

  const fetchInsight = async () => {
    try {
      const response = await fetch(`http://localhost:8000/driverInsight?vehicleID=${selectedVehicleID}`);
      if (!response.ok) {
        console.error(`Lap fetch failed: ${response.status} ${response.statusText}`);
        return;
      }
      const data = await response.json();
      const sentence = `${data["driverInsight"]} 5s ago to save ${data["improvement"].toFixed(2)}s`;
      const insight: DriverInsight = {
        startPosition: [data["startLat"], data["startLon"]],
        insight: sentence
      };
      setLatestInsight(insight);
      setDriverInsightList(data.total_improvement_list ?? []);
    } catch (error) {
      console.error('Error fetching insights:', error);
    }
  };

  // ================ POLLING LOOPS ================
  const positionPollMs = 100;
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchLatestPosition();
      fetchVehicles();
      if (selectedVehicleID !== null) {
        fetchLatestLaps();
      }
    }, positionPollMs);
    return () => clearInterval(intervalId);
  }, [selectedVehicleID]);

  const insightPollMs = 5000;
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (selectedVehicleID !== null) {
        fetchInsight();
      }
    }, insightPollMs);
    return () => clearInterval(intervalId);
  }, [selectedVehicleID]);

  // ================ RETURN ================
  return (
    <div>
      <TickConsumerToggle backendUrl="http://localhost:8000" />
      
      {selectedVehicleID && (
        <LapDisplay
          currentLap={currentLap}
          currentTime={currentTime}
          laps={laps}
        />
      )}

      {selectedVehicleID && (
        <DriverInsightsList insights={driverInsightList} />
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
        <TileLayer
          attribution='&copy; <a href="https://www.esri.com/en-us/arcgis/products/arcgis-online">Esri</a> contributors'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />

        {latestPositions && (
          <VehicleMarkers positions={latestPositions} showOption={showOption} selectedVehicleID={selectedVehicleID}/>
        )}

        {latestInsight && selectedVehicleID && (
          <CircleMarker
            center={latestInsight.startPosition}
            radius={0}
            weight={0}
          >
            <Tooltip permanent>{latestInsight.insight}</Tooltip>
          </CircleMarker>
        )}

      </MapContainer>

      {/* Popup Component */}
      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-container">
            <h2>UBC Solar x Toyota Hack The Track DEMO</h2>
            <p>Data shown is from <strong>Barber Motorsports Park Race 1</strong>, and is being replayed on a loop.</p>
            <button className="close-button" onClick={closePopup}>Let's Go!</button>
          </div>
        </div>
      )}
    </div>
  );
}
