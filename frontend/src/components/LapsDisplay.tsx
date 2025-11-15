// src/components/LapDisplay.tsx
import React from 'react';

interface Lap {
    number: number;
    time: number;
}

interface LapDisplayProps {
    currentLap: number;
    currentTime: number;
    laps: Lap[];
}

const LapDisplay: React.FC<LapDisplayProps> = ({ currentLap, currentTime, laps }) => {
    // Determine the best lap automatically
    const bestLapObj = laps.length > 0 ? laps.reduce((prev, curr) => (curr.time < prev.time ? curr : prev)) : null;
    const bestLapTime = bestLapObj?.time ?? 0;
    const bestLapNum = bestLapObj?.number ?? null;

    return (
        <div
            style={{
                position: 'absolute',
                bottom: '20px',
                right: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '15px',
                zIndex: 900,
            }}
        >
            {/* Current Lap */}
            <div style={boxStyle}>
                <h4 style={headerStyle}>Current Lap</h4>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={textStyle}>Lap {currentLap}</span>
                    <span style={timeStyle}>{currentTime.toFixed(1)}s</span>
                </div>
            </div>

            {/* Best Lap */}
            <div style={boxStyle}>
                <h4 style={headerStyle}>Best Lap</h4>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={textStyle}>Lap {bestLapNum ?? '—'}</span>
                    <span style={timeStyle}>{bestLapTime.toFixed(1)}s</span>
                </div>
            </div>

            {/* Previous Laps */}
            <div style={{ ...boxStyle, maxHeight: '150px'}}>
                <h4 style={headerStyle}>Previous Laps</h4>
                <div style={{maxHeight: '130px', overflowY: 'auto'}}>
                    {[...laps].reverse().map((lap, index, array) => {
                        // Get the previous lap object. It will be undefined for the first lap.
                        const prevLap = array[index + 1];
                        
                        // Check if a previous lap exists before comparing times
                        const diff = prevLap ? lap.time - prevLap.time : 0; 
                        
                        const diffColor = diff === 0 ? '#ffee56ff' : diff < 0 ? '#4CAF50' : '#FF5252';
                        return (
                            <div key={lap.number} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                                <span style={textStyle}>Lap {lap.number}</span>
                                <span style={{ ...timeStyle, color: diffColor }}>
                                    {lap.time.toFixed(1)}s {`(${diff > 0 ? '+' : ''}${diff.toFixed(1)})`}
                                </span>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

const boxStyle: React.CSSProperties = {
    backgroundColor: '#333',
    color: 'white',
    padding: '15px',
    borderRadius: '15px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
    width: '250px',
};

const headerStyle: React.CSSProperties = {
    margin: '0 0 8px 0',
    color: '#4CAF50',
};

const textStyle: React.CSSProperties = {
    margin: 0,
    fontSize: '12px',
};

const timeStyle: React.CSSProperties = {
    fontSize: '12px',
    fontWeight: 'bold',
};

export default LapDisplay;
