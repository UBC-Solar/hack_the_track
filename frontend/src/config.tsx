export const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL ??
  (window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : `${window.location.protocol}//${window.location.hostname}:8000`);