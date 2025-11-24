// src/components/DriverInsightsList.tsx
import React from 'react';

interface DriverInsightsListProps {
  insights: Array<[string, number]>; // [driverInsight, improvement]
}

export default function DriverInsightsList({ insights }: DriverInsightsListProps) {
  return (
    <div
      style={{
        position: 'fixed',
        top: '20%',
        right: 10,
        width: '17vw',
        maxHeight: '60vh',
        overflowY: 'auto',
        backgroundColor: '#333',
        color: 'white',
        padding: '15px',
        borderRadius: '10px',
        fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif',
        fontSize: '14px',
        lineHeight: 1.4,
        zIndex: 2000,
      }}
    >
      <h3 style={{ marginTop: 0 }}>Driver Insights (Over 5s Intervals)</h3>
        <ul style={{ paddingLeft: '1em' }}>
        {insights.slice().reverse().map(([text, improvement], idx) => (
            <li key={idx}>
            {/* Format to match: "<driverInsight> 5s ago to save <improvement>s" */}
            {`${text} 5s ago to save ${improvement.toFixed(2)}s`}
            </li>
        ))}
        </ul>
    </div>
  );    
}
