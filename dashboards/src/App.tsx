import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './store/authStore';
import { CreditOfficerDashboard } from './components/CreditOfficer/CreditOfficerDashboard';
 
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30000, retry: 2 } },
});
 
const LoginPage: React.FC = () => {
  const login = useAuthStore(s => s.login);
  const [email, setEmail] = React.useState('admin@sentinel.bank');
  const [password, setPassword] = React.useState('sentinel_admin');
  const [error, setError] = React.useState('');
 
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const { sentinelApi } = await import('./utils/api');
      const res = await sentinelApi.login(email, password);
      login(res.data.access_token, res.data.role, res.data.full_name);
      window.location.href = '/dashboard';
    } catch {
      setError('Invalid credentials. Try admin@sentinel.bank / sentinel_admin');
    }
  };
 
  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl border border-slate-200 p-8 w-full max-w-sm shadow-sm">
        <div className="text-center mb-8">
          <div className="text-3xl font-bold text-navy-DEFAULT mb-1" style={{ color: '#1B2A4A' }}>SENTINEL</div>
          <p className="text-sm text-slate-500">Pre-Delinquency Intelligence Platform</p>
        </div>
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" required />
          </div>
          {error && <p className="text-red-600 text-xs">{error}</p>}
          <button type="submit" className="w-full bg-blue-600 text-white rounded-lg py-2 text-sm font-semibold hover:bg-blue-700 transition-colors">
            Sign In
          </button>
        </form>
      </div>
    </div>
  );
};
 
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const isAuthenticated = useAuthStore(s => s.isAuthenticated);
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
};
 
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/dashboard" element={<ProtectedRoute><CreditOfficerDashboard /></ProtectedRoute>} />
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
