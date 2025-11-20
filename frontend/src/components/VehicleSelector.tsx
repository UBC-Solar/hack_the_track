import React, { useState } from 'react';

interface Vehicle {
  id: number;
  name: string;
}

const VehicleSelector: React.FC = () => {
  // Example list of vehicles
  const vehicles: Vehicle[] = [
    { id: 1, name: 'Car 1' },
    { id: 2, name: 'Car 2' },
    { id: 3, name: 'Truck 1' },
    { id: 4, name: 'Bike 1' },
  ];

  // State to manage the selected primary vehicle
  const [primaryVehicle, setPrimaryVehicle] = useState<Vehicle | null>(null);
  // State to manage the visibility option
  const [showOption, setShowOption] = useState<'all' | 'primary'>('all');

  // Handle changes in primary vehicle selection
  const handlePrimaryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedVehicleId = parseInt(event.target.value, 10);
    const selectedVehicle = vehicles.find(vehicle => vehicle.id === selectedVehicleId) || null;
    setPrimaryVehicle(selectedVehicle);
  };

  // Handle changes in visibility option
  const handleVisibilityChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setShowOption(event.target.value as 'all' | 'primary');
  };

  // Filter vehicles based on the visibility option
  const filteredVehicles = showOption === 'all' ? vehicles : primaryVehicle ? [primaryVehicle] : [];

  return (
    <div className="vehicle-selector">
      <div className="dropdown ui-box">

        <div>
          <label htmlFor="primary-vehicle">Select Vehicle: </label>
          <select id="primary-vehicle" onChange={handlePrimaryChange} value={primaryVehicle?.id || ''}>
            <option value="">-- Select Vehicle --</option>
            {vehicles.map(vehicle => (
              <option key={vehicle.id} value={vehicle.id}>
                {vehicle.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="show-option">Show: </label>
          <select id="show-option" onChange={handleVisibilityChange} value={showOption}>
            <option value="all">All</option>
            <option value="primary">Only Selected</option>
          </select>
        </div>

      </div>
    </div>
  );
};

export default VehicleSelector;
