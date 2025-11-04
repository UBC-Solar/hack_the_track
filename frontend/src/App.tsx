import type { LatLngTuple } from 'leaflet'
import { MapContainer, TileLayer } from 'react-leaflet'

const position: LatLngTuple = [33.5325017,-86.6215766]

export default function App(){
  return (
    <MapContainer
      center={position}
      zoom={16}
      scrollWheelZoom={true}
      style={{height: "100vh", width: "100vw"}}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
    </MapContainer>
  )
}