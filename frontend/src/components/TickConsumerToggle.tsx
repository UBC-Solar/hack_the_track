// src/components/TickConsumerToggle.tsx
import { useState } from 'react';
import { BACKEND_URL } from "../config.tsx";

interface TickConsumerToggleProps {
  backendUrl?: string; // optional override (defaults to localhost)
}

export default function TickConsumerToggle({ backendUrl = `${BACKEND_URL}` }: TickConsumerToggleProps) {
  const [writeEnabled, setWriteEnabled] = useState<boolean>(false); // ⬅ default: disabled
  const [toggling, setToggling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const sendToggle = async (enable: boolean) => {
    setToggling(true);
    setError(null);
    const prev = writeEnabled;
    setWriteEnabled(enable); // optimistic UI update

    try {
      const res = await fetch(`${backendUrl}/control/tick-consumer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Toggle failed (${res.status}): ${text}`);
      }
    } catch (err: any) {
      setWriteEnabled(prev);
      setError(err.message ?? 'Failed to toggle tick-consumer');
    } finally {
      setToggling(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 20,
        left: 20,
        zIndex: 1000,
        backgroundColor: '#333',
        color: 'white',
        padding: '15px',
        borderRadius: '15px',
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
        minWidth: 220,
        fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <label htmlFor="toggle-write" style={{ fontWeight: 600 }}>
          Data Collection
        </label>
        <input
          id="toggle-write"
          type="checkbox"
          checked={writeEnabled}
          disabled={toggling}
          onChange={(e) => sendToggle(e.target.checked)}
          style={{ transform: 'scale(1.25)', cursor: toggling ? 'not-allowed' : 'pointer' }}
          aria-label="Enable or disable tick consumer"
        />
        <span style={{ fontSize: 12, opacity: 0.75 }}>
          {toggling ? 'Updating…' : writeEnabled ? 'Enabled' : 'Disabled'}
        </span>
      </div>
      {error && (
        <div style={{ marginTop: 6, color: '#FF5252', fontSize: 12 }}>
          {error}
        </div>
      )}
    </div>
  );
}