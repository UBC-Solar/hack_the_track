// src/components/VehicleMarkers.tsx
import type { LatestPositions } from '../App.tsx'
import { CircleMarker } from 'react-leaflet';

interface VehicleMarkersProps {
  positions: LatestPositions;
}
export default function VehicleMarkers({positions = {}}: VehicleMarkersProps) {
  Object.entries(positions).map(
    ([vehicleID, position]) =>
      <CircleMarker
        center={position}
        radius={10}
        fillColor="red"
        color="white"
        fillOpacity={0.8}
      >
        <p>{vehicleID}</p>
      </CircleMarker>
  );
}