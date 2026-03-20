/**
 * src/store/authStore.ts
 * Zustand store for authentication state.
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
 
export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem('sentinel_token'),
  role: localStorage.getItem('sentinel_role'),
  fullName: localStorage.getItem('sentinel_name'),
  isAuthenticated: !!localStorage.getItem('sentinel_token'),
 
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
