import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000, // 60s
});

export const getHealth = () => api.get('/health');
export const getDatasetInfo = () => api.get('/dataset/info');
export const getDashboard = () => api.get('/dashboard');
export const getActivity = () => api.get('/dashboard/activity');

// Paginated & filtered signals history
export const getSignals = (params = {}) => {
  const query = typeof params === 'number' ? { limit: params } : params;
  return api.get('/signals', { params: query });
};

export const getSignal = (id) => api.get(`/signals/${id}`);
export const getPrediction = (id, modelName = null) => {
  const params = modelName ? { model_name: modelName } : {};
  return api.get(`/signals/${id}/prediction`, { params });
};
export const getWaveform = (id) => api.get(`/signals/${id}/waveform`);
export const getSpectrum = (id) => api.get(`/signals/${id}/spectrum`);
export const getSpectrogram = (id) => api.get(`/signals/${id}/spectrogram`);

export const analyzeSignal = (datasetIndex, modelName = 'ResNet1D') =>
  api.post('/signals/analyze', { dataset_index: datasetIndex, model_name: modelName });

export const getRandomSample = (modelName = 'ResNet1D') => {
  const params = modelName ? { model_name: modelName } : {};
  return api.get('/sample/random', { params });
};

export const getActiveSample = () => api.get('/sample/active');
export const setActiveSample = (datasetIndex, signalId = null) =>
  api.post('/sample/active', { dataset_index: datasetIndex, signal_id: signalId });

export const getModels = () => api.get('/models');

export const searchDataset = (modulation, minSnr, modelName = 'ResNet1D') => {
  const params = {};
  if (modulation) params.modulation = modulation;
  if (minSnr) params.min_snr = minSnr;
  if (modelName) params.model_name = modelName;
  return api.get('/sample/search', { params });
};

// Simulation lab
export const simulateInterference = (data) => api.post('/interference/simulate', data);

export const getInterferenceHistory = (params = {}) => {
  const query = typeof params === 'number' ? { limit: params } : params;
  return api.get('/interference/history', { params: query });
};

export const getSimulationDetail = (id) => api.get(`/interference/simulations/${id}`);

// Reports & Exports
export const getSignalPdfUrl = (id) => `${BASE_URL}/reports/signal/${id}/pdf`;
export const getSimulationPdfUrl = (id) => `${BASE_URL}/reports/simulation/${id}/pdf`;
export const getSignalsCsvUrl = (params = {}) => {
  const q = new URLSearchParams(params).toString();
  return `${BASE_URL}/reports/signals/csv${q ? `?${q}` : ''}`;
};
export const getSimulationsCsvUrl = (params = {}) => {
  const q = new URLSearchParams(params).toString();
  return `${BASE_URL}/reports/simulations/csv${q ? `?${q}` : ''}`;
};
export const verifyReport = (refCode) => api.get(`/reports/verify/${encodeURIComponent(refCode)}`);
