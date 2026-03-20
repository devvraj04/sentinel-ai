/**
 * lib/authStore.ts
 * Zustand store for authentication state.
 * SSR-safe: wraps localStorage calls with typeof window checks.
 */
import { create } from 'zustand';

interface AuthState {
  token: string | null;
  role: string | null;
  fullName: string | null;
  isAuthenticated: boolean;
  login: (token: string, role: string, fullName: string) => void;
  logout: () => void;
}

function safeGet(key: string): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(key);
}

export const useAuthStore = create<AuthState>((set) => ({
  token: safeGet('sentinel_token'),
  role: safeGet('sentinel_role'),
  fullName: safeGet('sentinel_name'),
  isAuthenticated: !!safeGet('sentinel_token'),

  login: (token, role, fullName) => {
    localStorage.setItem('sentinel_token', token);
    localStorage.setItem('sentinel_role', role);
    localStorage.setItem('sentinel_name', fullName);
    set({ token, role, fullName, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem('sentinel_token');
    localStorage.removeItem('sentinel_role');
    localStorage.removeItem('sentinel_name');
    set({ token: null, role: null, fullName: null, isAuthenticated: false });
  },
}));
