export default function InsightsHeader() {
  return (
    <div
      style={{
        position: 'fixed',
        top: 20,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 5000,

        backgroundColor: '#333',
        color: 'white',
        padding: '10px 10px 10px 10px',
        borderRadius: '15px',
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',

        fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif',
        fontSize: '24px',
        fontWeight: 700,
        whiteSpace: 'nowrap',
      }}
    >
      Insights Are Over <span style={{ color: '#00ff88' }}>5s Intervals</span>
    </div>
  );
}
