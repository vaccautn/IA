// Mirrored video policies; the Python contract test verifies parity without an endpoint.
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
export const SAMPLE_FPS = 5;
export const MAX_FRAMES = 300;
export const POLL_INTERVAL_MS = 2000;
export const PLAYBACK_INTERVAL_MS = 400;
// Distinct from POLL_INTERVAL_MS so track refreshes never share the history poll timer.
export const TRACKS_POLL_INTERVAL_MS = 3000;
