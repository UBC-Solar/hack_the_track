// src/components/TickConsumerToggle.tsx
import { useState } from 'react';

interface TickConsumerToggleProps {
  backendUrl?: string; // optional override (defaults to localhost)
}

export default function TickConsumerToggle({ backendUrl = 'http://localhost:8000' }: TickConsumerToggleProps) {
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
        top: 12,
        right: 12,
        zIndex: 1000,
        background: 'rgba(255,255,255,0.95)',
        padding: '10px 12px',
        borderRadius: 8,
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
        minWidth: 220,
        fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <label htmlFor="toggle-write" style={{ fontWeight: 600 }}>
          Tick consumer
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
        <div style={{ marginTop: 6, color: '#b00020', fontSize: 12 }}>
          {error}
        </div>
      )}
    </div>
  );
}