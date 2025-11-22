import type { LatestPositions } from '../App.tsx';
import { CircleMarker, Tooltip } from 'react-leaflet';

interface VehicleMarkersProps {
  positions: LatestPositions;
  showOption: 'primary' | 'all';
  selectedVehicleID: number | null;
}

// Generate a consistent color from the vehicle ID
function colorFromId(id: string): string {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = id.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 80%, 50%)`; // bright, distinct colors
}

export default function VehicleMarkers({
  positions = {},
  showOption,
  selectedVehicleID,
}: VehicleMarkersProps) {
  // Create separate arrays for selected and non-selected vehicles
  const nonSelectedVehicles = Object.entries(positions).filter(
    ([vehicleID]) => showOption === 'all' && vehicleID !== String(selectedVehicleID)
  );
  const selectedVehicles = Object.entries(positions).filter(
    ([vehicleID]) => vehicleID === String(selectedVehicleID)
  );

  return (
    <>
      {/* Render non-selected vehicles */}
      {nonSelectedVehicles.map(([vehicleID, position]) => (
        <CircleMarker
          key={vehicleID}
          center={position}
          radius={10}  // Radius for non-selected vehicles
          fillColor={colorFromId(vehicleID)}  // Color based on vehicle ID
          color={'white'}  // Border color for non-selected vehicles
          weight={2}
          fillOpacity={0.5}  // Opacity for non-selected vehicles
        >
          <Tooltip direction="top" offset={[0, -10]}>
            <span>{vehicleID}</span>
          </Tooltip>
        </CircleMarker>
      ))}

      {/* Render selected vehicle(s) */}
      {selectedVehicles.map(([vehicleID, position]) => (
        <CircleMarker
          key={vehicleID}
          center={position}
          radius={10}  // Larger radius for selected vehicles
          fillColor={'red'}  // Color based on vehicle ID
          color={'black'}  // Border color for selected vehicle(s)
          weight={2}
          fillOpacity={1}  // Full opacity for selected vehicles
        >
          <Tooltip direction="top" offset={[0, -10]}>
            <span>{vehicleID}</span>
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  );
}
