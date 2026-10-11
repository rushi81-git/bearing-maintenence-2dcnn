/**
 * API Service for Backend Communication
 */

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`);
  return res.json();
}

export async function fetchMachines() {
  const res = await fetch(`${API_BASE}/machines`);
  return res.json();
}

export async function createMachine(machineData) {
  const res = await fetch(`${API_BASE}/machines`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(machineData)
  });
  return res.json();
}

export async function updateMachine(id, machineData) {
  const res = await fetch(`${API_BASE}/machines/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(machineData)
  });
  return res.json();
}

export async function deleteMachine(id) {
  const res = await fetch(`${API_BASE}/machines/${id}`, {
    method: 'DELETE'
  });
  return res.json();
}

export async function fetchSchedule() {
  const res = await fetch(`${API_BASE}/schedule`);
  return res.json();
}

export async function updateScheduleStatus(id, status) {
  const res = await fetch(`${API_BASE}/schedule/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status })
  });
  return res.json();
}

export async function createScheduleEntry(entryData) {
  const res = await fetch(`${API_BASE}/schedule`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entryData)
  });
  return res.json();
}

export async function analyzeBearingSignal({
  signal,
  image_base64,
  crop_box,
  input_type,
  machine_id,
  sampling_rate_hz = 48000,
  signal_unit = 'g',
  source_type = 'csv',
  source_filename = 'vibration.csv',
  model_mode = 'auto',
  use_classical_ml = false,
  bearing_location = 'Drive End (DE)'
}) {
  const res = await fetch(`${API_BASE}/bearing-analysis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      signal,
      image_base64,
      crop_box,
      input_type,
      machine_id,
      sampling_rate_hz,
      signal_unit,
      source_type,
      source_filename,
      model_mode,
      use_classical_ml,
      bearing_location
    })
  });
  return res.json();
}

export async function fetchBearingHistory(filters = {}) {
  const params = new URLSearchParams();
  if (filters.machine_id) params.append('machine_id', filters.machine_id);
  if (filters.severity) params.append('severity', filters.severity);
  if (filters.fault_type) params.append('fault_type', filters.fault_type);
  if (filters.search) params.append('search', filters.search);

  const res = await fetch(`${API_BASE}/bearing-history?${params.toString()}`);
  return res.json();
}

export async function fetchDemoSamples() {
  const res = await fetch(`${API_BASE}/demo-samples`);
  return res.json();
}

export async function fetchBearingModels() {
  const res = await fetch(`${API_BASE}/bearing-models`);
  return res.json();
}
