/**
 * lib/api.ts
 * Centralized Axios instance for all API calls.
 * Automatically attaches JWT token from localStorage.
 */
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001/api/v1';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = document.cookie
      .split('; ')
      .find((row) => row.startsWith('sentinel_token='))
      ?.split('=')[1];
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Redirect to login on 401
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (typeof window !== 'undefined' && err.response?.status === 401) {
      document.cookie = 'sentinel_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

// ── API methods ───────────────────────────────────────────────────────────────
export const sentinelApi = {
  login: (email: string, password: string) =>
    api.post('/auth/login', new URLSearchParams({ username: email, password })),

  scoreCustomer: (customerId: string) =>
    api.post('/score', { customer_id: customerId }),

  getCustomers: (params?: { risk_tier?: string; limit?: number }) =>
    api.get('/customers', { params }),

  getCustomer: (customerId: string) =>
    api.get(`/customers/${customerId}`),

  getPortfolioMetrics: () =>
    api.get('/portfolio/metrics'),
};
