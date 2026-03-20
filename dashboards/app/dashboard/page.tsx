'use client';

/**
 * Credit Officer Dashboard — main portfolio view
 * Ported from the original React CreditOfficerDashboard.tsx
 */
import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  AlertTriangle, TrendingUp, DollarSign,
  CreditCard, Search, ChevronLeft, ChevronRight,
} from 'lucide-react';
import { sentinelApi } from '@/lib/api';

const RISK_COLORS: Record<string, string> = {
  green:  '#16A34A',
  yellow: '#D97706',
  orange: '#EA580C',
  red:    '#DC2626',
};

const TIER_LABELS: Record<string, string> = {
  green:  'Safe',
  yellow: 'Watch',
  orange: 'At Risk',
  red:    'Critical',
};

const PAGE_SIZE = 50;

function KPICard({
  title, value, subtitle, icon, color,
}: {
  title: string;
  value: string | number;
  subtitle: string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-start gap-4">
      <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}15` }}>
        <div style={{ color }}>{icon}</div>
      </div>
      <div>
        <p className="text-sm text-slate-500">{title}</p>
        <p className="text-2xl font-bold text-slate-800">{value}</p>
        <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [filterTier, setFilterTier] = useState('all');
  const [page, setPage] = useState(0);

  const { data: metrics } = useQuery({
    queryKey: ['portfolio-metrics'],
    queryFn: () => sentinelApi.getPortfolioMetrics().then((r) => r.data),
    refetchInterval: 30000,
  });

  const { data: customers = [], isLoading: customersLoading } = useQuery({
    queryKey: ['customers', filterTier],
    queryFn: () =>
      sentinelApi
        .getCustomers({
          risk_tier: filterTier === 'all' ? undefined : filterTier,
          limit: 1500,
        })
        .then((r) => r.data),
    refetchInterval: 30000,
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const filtered: any[] = (customers as any[]).filter((c: any) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      c.customer_id?.toLowerCase().includes(q) ||
      c.full_name?.toLowerCase().includes(q)
    );
  });

  useEffect(() => {
    setPage(0);
  }, [search, filterTier]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageSlice = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const pieData = [
    { name: 'Safe',     value: metrics?.safe_count          || 0, fill: RISK_COLORS.green },
    { name: 'Watch',    value: metrics?.watch_count         || 0, fill: RISK_COLORS.yellow },
    { name: 'At Risk',  value: metrics?.at_risk_count       || 0, fill: RISK_COLORS.orange },
    { name: 'Critical', value: metrics?.critical_risk_count || 0, fill: RISK_COLORS.red },
  ].filter((d) => d.value > 0);

  const barData = [
    { tier: 'Safe',     score: 12, fill: RISK_COLORS.green },
    { tier: 'Watch',    score: 35, fill: RISK_COLORS.yellow },
    { tier: 'At Risk',  score: 58, fill: RISK_COLORS.orange },
    { tier: 'Critical', score: 82, fill: RISK_COLORS.red },
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-8 py-5">
        <h1 className="text-2xl font-bold text-slate-800">Credit Officer Dashboard</h1>
        <p className="text-sm text-slate-500 mt-0.5">Real-time pre-delinquency risk monitoring</p>
      </div>

      <div className="px-8 py-6 space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Critical Risk"
            value={metrics?.critical_risk_count ?? '-'}
            subtitle="Require immediate intervention"
            icon={<AlertTriangle size={20} />}
            color="#DC2626"
          />
          <KPICard
            title="At Risk (Early Warning)"
            value={metrics?.at_risk_count ?? '-'}
            subtitle="Need proactive support"
            icon={<TrendingUp size={20} />}
            color="#D97706"
          />
          <KPICard
            title="Total Portfolio Debt"
            value={metrics ? `₹${(metrics.total_portfolio_debt / 100000).toFixed(1)}L` : '-'}
            subtitle="Across all customers"
            icon={<DollarSign size={20} />}
            color="#2563EB"
          />
          <KPICard
            title="Avg Credit Utilization"
            value={metrics ? `${(metrics.avg_credit_utilization * 100).toFixed(1)}%` : '-'}
            subtitle="Portfolio average"
            icon={<CreditCard size={20} />}
            color="#7C3AED"
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h2 className="text-base font-semibold text-slate-700 mb-4">Portfolio Risk Distribution</h2>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={85}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h2 className="text-base font-semibold text-slate-700 mb-4">Average Pulse Score by Tier</h2>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData}>
                <XAxis dataKey="tier" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                  {barData.map((_, i) => (
                    <Cell key={i} fill={barData[i].fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Customer Table */}
        <div className="bg-white rounded-xl border border-slate-200">
          {/* Controls */}
          <div className="p-5 border-b border-slate-100 flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <h2 className="text-base font-semibold text-slate-700 flex-1">Find & Manage Customers</h2>
            <div className="relative">
              <Search size={16} className="absolute left-3 top-2.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search by ID or name..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9 pr-4 py-2 text-sm border border-slate-200 rounded-lg
                           focus:outline-none focus:ring-2 focus:ring-blue-500 w-56"
              />
            </div>
            <select
              value={filterTier}
              onChange={(e) => setFilterTier(e.target.value)}
              className="text-sm border border-slate-200 rounded-lg px-3 py-2
                         focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Tiers</option>
              <option value="red">Critical</option>
              <option value="orange">At Risk</option>
              <option value="yellow">Watch</option>
              <option value="green">Safe</option>
            </select>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100">
                  {['Customer ID', 'Name', 'Pulse Score', 'Risk Tier',
                    'Credit Util.', 'Income', 'Days Late', 'Top Signal', ''].map((h, i) => (
                    <th
                      key={i}
                      className="px-4 py-3 text-left text-xs font-semibold
                                 text-slate-500 uppercase tracking-wide"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {customersLoading ? (
                  <tr>
                    <td colSpan={9} className="px-4 py-10 text-center text-slate-400">
                      <div className="flex justify-center mb-2">
                        <div className="animate-spin h-6 w-6 rounded-full
                                        border-2 border-blue-500 border-t-transparent" />
                      </div>
                      Loading customers…
                    </td>
                  </tr>
                ) : pageSlice.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="px-4 py-10 text-center text-slate-400">
                      No customers found
                    </td>
                  </tr>
                ) : (
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  pageSlice.map((c: any, i: number) => {
                    const color = RISK_COLORS[c.risk_tier] || '#64748B';
                    return (
                      <tr
                        key={c.customer_id}
                        className={`border-b border-slate-50 hover:bg-blue-50 cursor-pointer
                                    transition-colors group
                                    ${i % 2 === 0 ? '' : 'bg-slate-50/40'}`}
                        onClick={() => router.push(`/dashboard/${c.customer_id}`)}
                      >
                        <td className="px-4 py-3 font-mono text-xs text-slate-500">
                          {c.customer_id}
                        </td>
                        <td className="px-4 py-3 font-medium text-slate-800">
                          {c.full_name}
                        </td>
                        <td className="px-4 py-3">
                          <span className="font-bold text-lg" style={{ color }}>
                            {c.pulse_score}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className="px-2 py-0.5 rounded-full text-xs font-semibold uppercase"
                            style={{ backgroundColor: `${color}20`, color }}
                          >
                            {TIER_LABELS[c.risk_tier] || c.risk_tier}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-600">
                          {(c.credit_utilization * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-slate-600">
                          ₹{c.monthly_income?.toLocaleString('en-IN')}
                        </td>
                        <td className="px-4 py-3">
                          {c.days_past_due > 0 ? (
                            <span className="text-red-600 font-semibold">{c.days_past_due}d</span>
                          ) : (
                            <span className="text-slate-300">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-xs text-slate-500 max-w-[140px] truncate">
                          {c.top_factor ? c.top_factor.replace(/_/g, ' ') : '—'}
                        </td>
                        <td className="px-4 py-3">
                          <button
                            className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700
                                       text-white rounded-lg transition font-medium
                                       opacity-0 group-hover:opacity-100"
                          >
                            View →
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="px-5 py-3 border-t border-slate-100 flex items-center justify-between">
            <p className="text-xs text-slate-400">
              {filtered.length.toLocaleString()} customers · showing{' '}
              {Math.min(page * PAGE_SIZE + 1, filtered.length)}–
              {Math.min((page + 1) * PAGE_SIZE, filtered.length)}
            </p>
            <div className="flex items-center gap-2">
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
                className="p-1.5 rounded-lg border border-slate-200 disabled:opacity-40
                           hover:bg-slate-100 transition"
              >
                <ChevronLeft size={16} className="text-slate-600" />
              </button>
              <span className="text-xs text-slate-500 min-w-[60px] text-center">
                {page + 1} / {Math.max(1, totalPages)}
              </span>
              <button
                disabled={page + 1 >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="p-1.5 rounded-lg border border-slate-200 disabled:opacity-40
                           hover:bg-slate-100 transition"
              >
                <ChevronRight size={16} className="text-slate-600" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
