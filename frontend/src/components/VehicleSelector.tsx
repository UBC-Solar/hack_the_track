import React from 'react';

interface VehicleSelectorProps {
  vehicleID: string | null;
  showOption: 'all' | 'primary';
  vehicles: Array<string>;
  setVehicleID: (id: string | null) => void;
  setShowOption: (show: 'all' | 'primary') => void;
}

const VehicleSelector: React.FC<VehicleSelectorProps> = ({ vehicleID, showOption, vehicles, setVehicleID, setShowOption}) => {

  const handlePrimaryChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedId = event.target.value || null;  // keep as string
    setVehicleID(selectedId);
  };

  const handleVisibilityChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const visibility: 'all' | 'primary' =  event.target.value == 'all' ? 'all' : 'primary';
    setShowOption(visibility); // <-- Mutates parent state
  };

  return (
    <div className="vehicle-selector">
      <div className="dropdown ui-box">

        <div>
          <label htmlFor="primary-vehicle">Select Vehicle: </label>
          <select
            id="primary-vehicle"
            onChange={handlePrimaryChange}
            value={vehicleID ?? ''}
          >
            <option value="">-- Select Vehicle --</option>
            {vehicles.map(vehicle => (
              <option key={vehicle} value={vehicle}>
                {vehicle}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="show-option">Show: </label>
          <select
            id="show-option"
            onChange={handleVisibilityChange}
            value={showOption}
          >
            <option value="all">All</option>
            <option value="primary">Only Selected</option>
          </select>
        </div>

      </div>
    </div>
  );
};

export default VehicleSelector;
