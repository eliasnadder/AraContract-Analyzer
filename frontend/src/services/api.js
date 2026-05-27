import axios from 'axios';

// Create an axios instance with base URL (defaults to proxy in package.json)
const API = axios.create({
  timeout: 60000, // 60s timeout for large contract processing on CPU
});

/**
 * Upload and analyze a contract file.
 * Returns the unified AnalysisResponse.
 */
export const analyzeContract = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await API.post('/api/contract/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

/**
 * Compare two contract files.
 * Returns the ComparisonResponse.
 */
export const compareContracts = async (file1, file2) => {
  const formData = new FormData();
  formData.append('file1', file1);
  formData.append('file2', file2);
  const response = await API.post('/api/contract/compare', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export default {
  analyzeContract,
  compareContracts,
};
