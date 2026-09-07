/**
 * Client for the NeuroPace inference service.
 *
 * Every call resolves to a tagged result instead of throwing, so a service
 * that is down renders as a readable state rather than an unhandled rejection
 * in the middle of a render.
 */

export const API_BASE =
  process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8080';

async function request(path, options) {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    const body = await response.json().catch(() => null);

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        detail: body?.detail ?? `The service returned ${response.status}.`,
      };
    }
    return { ok: true, status: response.status, body };
  } catch {
    return {
      ok: false,
      status: 0,
      detail: `Can't reach the service at ${API_BASE}.`,
    };
  }
}

export const getHealth = () => request('/health');
export const getLabels = () => request('/labels');
export const postPredict = (window) =>
  request('/predict/', { method: 'POST', body: JSON.stringify({ data: window }) });
