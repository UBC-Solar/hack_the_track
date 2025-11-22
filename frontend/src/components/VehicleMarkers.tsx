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

export default function VehicleMarkers({ positions = {}, showOption, selectedVehicleID }: VehicleMarkersProps) {
  return (
    <>
      {Object.entries(positions).map(([vehicleID, position]) => {
        // If showOption is 'primary', only render the marker for the selected vehicle
        if (showOption === 'primary' && vehicleID !== String(selectedVehicleID)) {
          return null;
        }

        return (
          <CircleMarker
            key={vehicleID}
            center={position}
            radius={10}
            fillColor={colorFromId(vehicleID)}
            color="white"
            weight={2}
            fillOpacity={0.8}
          >
            <Tooltip direction="top" offset={[0, -10]}>
              <span>{vehicleID}</span>
            </Tooltip>
          </CircleMarker>
        );
      })}
    </>
  );
}
