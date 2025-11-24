// src/components/DriverInsightsList.tsx
import React from 'react';

interface DriverInsightsListProps {
  insights: Array<[string, number]>; // [driverInsight, improvement]
}

export default function DriverInsightsList({ insights }: DriverInsightsListProps) {
  return (
    <div className="driver-insights">
      <div className="ui-box">
        <h4 className="ui-header">Driver Insights (Over 5s Intervals)</h4>

        {/* Scrollable content area, header stays pinned */}
        <div className="insights-scroll">
          <ul style={{ paddingLeft: '1em', margin: 0 }}>
            {insights
              .slice()
              .reverse()
              .map(([text, improvement], idx) => (
                <li key={idx}>
                  {`${text} 5s ago to save ${improvement.toFixed(2)}s`}
                </li>
              ))}
          </ul>
        </div>
      </div>
    </div>
  );
}