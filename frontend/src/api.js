const API_URL = import.meta.env.DEV
  ? '/api'
  : 'https://log-classification-system.onrender.com';

export async function checkApi() {
  try {
    const r = await fetch(`${API_URL}/`, { signal: AbortSignal.timeout(60000) });
    if (r.ok) return { ok: true, data: await r.json() };
    return { ok: false, data: {} };
  } catch {
    return { ok: false, data: {} };
  }
}

export async function getModelVersions() {
  try {
    const r = await fetch(`${API_URL}/model/versions`);
    if (r.ok) {
      const data = await r.json();
      return data.versions || [];
    }
  } catch {}
  return [];
}

export async function getModelInfo() {
  try {
    const r = await fetch(`${API_URL}/model/info`);
    if (r.ok) return await r.json();
  } catch {}
  return null;
}

export async function trainModel(file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API_URL}/train`, { method: 'POST', body: form, signal: AbortSignal.timeout(30000) });
  return { status: r.status, data: await r.json() };
}

export async function getTrainingStatus() {
  try {
    const r = await fetch(`${API_URL}/train/status`);
    if (r.ok) return await r.json();
  } catch {}
  return null;
}

export async function predictSingle(logMessage, version = 'main') {
  const r = await fetch(`${API_URL}/predict?version=${version}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ log_message: logMessage }),
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

export async function predictBatch(file, version = 'main') {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API_URL}/predict/batch?version=${version}`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}
