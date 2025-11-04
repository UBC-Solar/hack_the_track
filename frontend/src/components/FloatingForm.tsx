// src/components/FloatingForm.tsx
import React from 'react';

interface FloatingFormProps {
  lapNumber: number;
  samplePeriod: number;
  error: string | null;
  onSubmit: (e: React.FormEvent) => void;
  onLapNumberChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSamplePeriodChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const FloatingForm: React.FC<FloatingFormProps> = ({
  lapNumber,
  samplePeriod,
  error,
  onSubmit,
  onLapNumberChange,
  onSamplePeriodChange,
}) => {
  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      right: '20px',
      padding: '20px',
      backgroundColor: '#333',  // Dark gray background
      borderRadius: '15px',
      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
      zIndex: 1000,
      width: 'auto',
      maxWidth: '300px',
    }}>
      <form onSubmit={onSubmit}>
        <label>
          <strong style={{ color: 'white' }}>Lap Number: </strong>
          <input
            type="text"
            value={lapNumber}
            onChange={onLapNumberChange}
            pattern="\d*"
            required
            style={{
              width: '120px',  // Reduced width for input fields
              padding: '8px',
              marginTop: '5px',
              borderRadius: '8px',
              border: '1px solid #ccc',
              backgroundColor: '#444',  // Darker input background
              color: 'white',
            }}
          />
        </label>
        <br />
        <label style={{ marginTop: '10px' }}>
          <strong style={{ color: 'white' }}>Sample Period: </strong>
          <input
            type="number"
            value={samplePeriod}
            onChange={onSamplePeriodChange}
            min={1}
            required
            style={{
              width: '120px',  // Reduced width for input fields
              padding: '8px',
              marginTop: '5px',
              borderRadius: '8px',
              border: '1px solid #ccc',
              backgroundColor: '#444',  // Darker input background
              color: 'white',
            }}
          />
        </label>
        <br />
        <button type="submit" style={{
          marginTop: '10px',
          padding: '10px 15px',
          borderRadius: '8px',
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          cursor: 'pointer',
          width: '100%',
        }}>Fetch GPS Data</button>
      </form>

      {/* Display error if any */}
      {error && <div style={{ color: 'red', padding: '10px 0' }}>Error: {error}</div>}
    </div>
  );
};

export default FloatingForm;
