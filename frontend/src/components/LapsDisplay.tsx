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
        <div className="lap-display">
            {/* Current Lap */}
            <div className="lap-box">
                <h4 className="lap-header">Current Lap</h4>
                <div className="lap-info">
                    <span className="lap-text">Lap {currentLap}</span>
                    <span className="lap-time">{currentTime.toFixed(1)}s</span>
                </div>
            </div>

            {/* Best Lap */}
            <div className="lap-box">
                <h4 className="lap-header">Best Lap</h4>
                <div className="lap-info">
                    <span className="lap-text">Lap {bestLapNum ?? '—'}</span>
                    <span className="lap-time">{bestLapTime.toFixed(1)}s</span>
                </div>
            </div>

            {/* Previous Laps */}
            <div className="lap-box previous-laps">
                <h4 className="lap-header">Previous Laps</h4>
                <div className="lap-scroll">
                    {[...laps].reverse().map((lap, index, array) => {
                        // Get the previous lap object. It will be undefined for the first lap.
                        const prevLap = array[index + 1];

                        // Check if a previous lap exists before comparing times
                        const diff = prevLap ? lap.time - prevLap.time : 0; 

                        const diffColor = diff === 0 ? '#ffee56ff' : diff < 0 ? '#4CAF50' : '#FF5252';
                        return (
                            <div key={lap.number} className="lap-item">
                                <span className="lap-text">Lap {lap.number}</span>
                                <span className="lap-time" style={{ color: diffColor }}>
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

export default LapDisplay;
