'use client';

/**
 * Customer Detail Page — full customer risk profile
 * Ported from the original React CustomerDetail.tsx
 */
import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
} from 'recharts';
import { api } from '@/lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────
interface TopFactor {
  feature_name: string;
  contribution: number;
  human_readable: string;
  direction: 'increases_risk' | 'decreases_risk';
  raw_value: number;
}

interface CustomerProfile {
  customer_id: string;
  full_name: string;
  pulse_score: number;
  risk_tier: string;
  pd_probability: number;
  confidence: number;
  credit_utilization: number;
  outstanding_balance: number;
  monthly_income: number;
  credit_limit: number;
  days_past_due: number;
  employment_status: string;
  segment: string;
  geography: string;
  preferred_channel: string;
  top_factor?: string;
  intervention_flag?: boolean;
  intervention_type?: string;
  updated_at?: string;
}

interface ScoreImpactTxn {
  txn_timestamp: string;
  txn_type: string;
  amount: number;
  category: string;
  channel: string;
  status: string;
  score_impact: number | null;   // real delta between consecutive model scores; null if no history
  score_after: number | null;    // real model score at this timestamp; null if no history
  impact_reason: string;
  is_stress_signal: boolean;
}

interface Transaction {
  txn_id: string;
  txn_timestamp: string;
  txn_type: string;
  amount: number;
  category: string;
  channel: string;
  status: string;
  is_stress_signal: boolean;
}

interface PulseHistoryPoint {
  pulse_score: number;
  risk_tier: string;
  pd_probability: number;
  confidence: number;
  top_factors: string[];
  intervention_flag: boolean;
  intervention_type: string;
  model_version: string;
  scored_at: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const TIER_COLORS: Record<string, string> = {
  green: '#22c55e',
  yellow: '#eab308',
  orange: '#f97316',
  red: '#ef4444',
};

const TIER_LABELS: Record<string, string> = {
  green: 'Safe',
  yellow: 'Watch',
  orange: 'At Risk',
  red: 'Critical',
};

function fmt(amount: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(amount);
}

function fmtDate(iso: string): string {
  if (!iso) return '';
  return new Date(iso).toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function txnLabel(txn_type: string): string {
  const labels: Record<string, string> = {
    salary_credit: 'Salary Credit',
    upi_debit: 'UPI Payment',
    atm_withdrawal: 'ATM Withdrawal',
    auto_debit: 'EMI Auto-Debit',
    utility_payment: 'Utility Payment',
    savings_withdrawal: 'Savings Withdrawal',
    credit_card_payment: 'Credit Card',
    neft_rtgs: 'NEFT/RTGS',
    other: 'Other',
  };
  return labels[txn_type] || txn_type.replace(/_/g, ' ').toUpperCase();
}

// ── Pulse Gauge ───────────────────────────────────────────────────────────────
function PulseGauge({ score, tier }: { score: number; tier: string }) {
  const color = TIER_COLORS[tier] || '#6b7280';
  const angle = -135 + (score / 100) * 270;

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" width="200" height="120">
        {/* Background arc */}
        <path
          d="M 20 110 A 80 80 0 1 1 180 110"
          fill="none"
          stroke="#374151"
          strokeWidth="16"
          strokeLinecap="round"
        />
        {/* Green zone */}
        <path
          d="M 20 110 A 80 80 0 0 1 60 38"
          fill="none"
          stroke="#22c55e"
          strokeWidth="16"
          strokeLinecap="round"
        />
        {/* Yellow zone */}
        <path d="M 60 38 A 80 80 0 0 1 100 24" fill="none" stroke="#eab308" strokeWidth="16" />
        {/* Orange zone */}
        <path d="M 100 24 A 80 80 0 0 1 140 38" fill="none" stroke="#f97316" strokeWidth="16" />
        {/* Red zone */}
        <path
          d="M 140 38 A 80 80 0 0 1 180 110"
          fill="none"
          stroke="#ef4444"
          strokeWidth="16"
          strokeLinecap="round"
        />
        {/* Needle */}
        <line
          x1="100"
          y1="110"
          x2={100 + 65 * Math.cos((angle * Math.PI) / 180)}
          y2={110 + 65 * Math.sin((angle * Math.PI) / 180)}
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx="100" cy="110" r="6" fill={color} />
        {/* Score label */}
        <text x="100" y="105" textAnchor="middle" fill={color} fontSize="22" fontWeight="bold">
          {score}
        </text>
      </svg>
      <span className="text-sm font-semibold mt-1" style={{ color }}>
        {TIER_LABELS[tier] || tier}
      </span>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
type Tab = 'overview' | 'transactions' | 'score-impact' | 'history';

export default function CustomerDetailPage() {
  const params = useParams();
  const router = useRouter();
  const customerId = params.customerId as string;

  const [profile, setProfile] = useState<CustomerProfile | null>(null);
  const [scoreData, setScoreData] = useState<{ top_factors: TopFactor[] } | null>(null);
  const [txns, setTxns] = useState<Transaction[]>([]);
  const [txnTotal, setTxnTotal] = useState(0);
  const [impact, setImpact] = useState<ScoreImpactTxn[]>([]);
  const [history, setHistory] = useState<PulseHistoryPoint[]>([]);
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [loading, setLoading] = useState(true);
  const [txnPage, setTxnPage] = useState(0);
  const TXN_PAGE_SIZE = 20;

  // Load profile (DynamoDB current score) + latest SHAP from history on mount.
  // We do NOT call POST /score automatically — that would create a new history
  // entry every time the page is opened. Score is only updated when the user
  // explicitly clicks "Re-score" or when the real-time pipeline runs.
  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.get(`/customers/${customerId}`),
      api.get(`/customers/${customerId}/pulse-history`, { params: { limit: 1 } }),
    ])
      .then(([profRes, histRes]) => {
        setProfile(profRes.data);
        // Use the most recent history entry for SHAP top_factors if available
        const latest = histRes.data?.history?.[0];
        if (latest?.top_factors?.length) {
          setScoreData({ top_factors: latest.top_factors.map((f: string, i: number) => ({
            feature_name: f,
            contribution: 1 / (i + 1),
            human_readable: f.replace(/_/g, ' '),
            direction: 'increases_risk' as const,
            raw_value: 0,
          })) });
        }
      })
      .finally(() => setLoading(false));
  }, [customerId]);

  // Manual re-score: calls the model, updates both stores, refreshes the page data
  const [rescoring, setRescoring] = useState(false);
  const handleRescore = () => {
    setRescoring(true);
    api.post('/score', { customer_id: customerId, force_refresh: true })
      .then((res) => {
        setScoreData(res.data);
        // Refresh profile to pick up the new DynamoDB score
        return api.get(`/customers/${customerId}`);
      })
      .then((res) => setProfile(res.data))
      .finally(() => setRescoring(false));
  };

  // Load tab data
  useEffect(() => {
    if (activeTab === 'transactions') {
      api
        .get(`/customers/${customerId}/transactions`, {
          params: { limit: TXN_PAGE_SIZE, offset: txnPage * TXN_PAGE_SIZE },
        })
        .then((r) => {
          setTxns(r.data.transactions);
          setTxnTotal(r.data.total);
        });
    }
    if (activeTab === 'score-impact') {
      api
        .get(`/customers/${customerId}/score-impact`, { params: { limit: 50 } })
        .then((r) => setImpact(r.data.transactions));
    }
    if (activeTab === 'history') {
      api
        .get(`/customers/${customerId}/pulse-history`, { params: { limit: 200 } })
        .then((r) => setHistory(r.data.history));
    }
  }, [activeTab, customerId, txnPage]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="text-center text-red-400 p-8">
        Customer not found.
        <button onClick={() => router.push('/dashboard')} className="ml-4 text-blue-400 underline">
          Back
        </button>
      </div>
    );
  }

  const tierColor = TIER_COLORS[profile.risk_tier] || '#6b7280';

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={() => router.push('/dashboard')}
          className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition"
        >
          ← Back
        </button>
        <div>
          <h1 className="text-2xl font-bold">{profile.full_name}</h1>
          <p className="text-gray-400 text-sm">
            {profile.customer_id} · {profile.segment} · {profile.geography}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <span
            className="px-3 py-1 rounded-full text-sm font-bold"
            style={{
              background: `${tierColor}22`,
              color: tierColor,
              border: `1px solid ${tierColor}`,
            }}
          >
            {TIER_LABELS[profile.risk_tier] || profile.risk_tier}
          </span>
          {profile.intervention_flag && (
            <span className="px-3 py-1 bg-red-900/40 border border-red-500 text-red-400 rounded-full text-xs font-semibold">
              Intervention Recommended
            </span>
          )}
          <button
            onClick={handleRescore}
            disabled={rescoring}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition"
            title="Run the LightGBM model now and update the score in all databases"
          >
            {rescoring ? 'Scoring…' : 'Re-score'}
          </button>
        </div>
      </div>

      {/* ── Top stats ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'Monthly Income', value: fmt(profile.monthly_income) },
          { label: 'Outstanding Balance', value: fmt(profile.outstanding_balance) },
          { label: 'Credit Utilization', value: `${(profile.credit_utilization * 100).toFixed(1)}%` },
          { label: 'Days Past Due', value: profile.days_past_due, warn: profile.days_past_due > 0 },
        ].map((stat) => (
          <div key={stat.label} className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <p className="text-gray-400 text-xs mb-1">{stat.label}</p>
            <p className={`text-xl font-bold ${stat.warn ? 'text-red-400' : 'text-white'}`}>
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      {/* ── Gauge + Key info ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Gauge */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 flex flex-col items-center">
          <h3 className="text-sm text-gray-400 mb-2">Pulse Score</h3>
          {profile.pulse_score !== null && profile.pulse_score !== undefined ? (
            <>
              <PulseGauge score={profile.pulse_score} tier={profile.risk_tier} />
              <p className="text-xs text-gray-500 mt-2">
                PD: {(profile.pd_probability * 100).toFixed(1)}% · Confidence:{' '}
                {profile.confidence !== null ? `${(profile.confidence * 100).toFixed(0)}%` : 'n/a'}
              </p>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-32 gap-2">
              <p className="text-gray-500 text-sm text-center">Not yet scored</p>
              <p className="text-gray-600 text-xs text-center">Click Re-score to run the model</p>
            </div>
          )}
          {profile.intervention_type && (
            <p className="text-xs text-orange-400 mt-1 text-center">
              {profile.intervention_type.replace(/_/g, ' ')}
            </p>
          )}
        </div>

        {/* Customer info */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 className="text-sm text-gray-400 mb-3">Customer Profile</h3>
          <div className="space-y-2 text-sm">
            {[
              ['Employment', profile.employment_status],
              ['Segment', profile.segment],
              ['Credit Limit', fmt(profile.credit_limit)],
              ['Preferred Channel', profile.preferred_channel],
              ['Last Scored', profile.updated_at ? fmtDate(profile.updated_at) : '—'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-gray-400">{k}</span>
                <span className="text-white font-medium">{v}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Top risk driver */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 className="text-sm text-gray-400 mb-3">Top Risk Driver</h3>
          {scoreData?.top_factors?.length ? (
            <div className="space-y-2">
              {scoreData.top_factors.slice(0, 5).map((f, i) => (
                <div key={i}>
                  <div className="flex justify-between text-xs mb-1">
                    <span
                      className={
                        f.direction === 'increases_risk' ? 'text-red-400' : 'text-green-400'
                      }
                    >
                      {f.direction === 'increases_risk' ? '↑' : '↓'} {f.human_readable}
                    </span>
                    <span className="text-gray-400">{(f.contribution * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, f.contribution * 500)}%`,
                        background: f.direction === 'increases_risk' ? '#ef4444' : '#22c55e',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No SHAP data available</p>
          )}
        </div>
      </div>

      {/* ── Tabs ───────────────────────────────────────────────────────────── */}
      <div className="flex gap-2 mb-4 border-b border-gray-800 pb-2">
        {(['overview', 'transactions', 'score-impact', 'history'] as Tab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition ${
              activeTab === tab
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            {tab === 'overview'
              ? '📊 SHAP Overview'
              : tab === 'transactions'
                ? '📋 All Transactions'
                : tab === 'score-impact'
                  ? '⚡ Score Impact'
                  : '📈 Score History'}
          </button>
        ))}
      </div>

      {/* ── Tab: SHAP Overview ─────────────────────────────────────────────── */}
      {activeTab === 'overview' && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 className="text-base font-semibold mb-4">
            SHAP Factor Contributions — Why is this score {profile.pulse_score}?
          </h3>
          {scoreData?.top_factors?.length ? (
            <>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={scoreData.top_factors.map((f) => ({
                    name:
                      f.human_readable.length > 30
                        ? f.human_readable.slice(0, 28) + '…'
                        : f.human_readable,
                    value: f.contribution,
                    direction: f.direction,
                    raw: f.raw_value,
                  }))}
                  layout="vertical"
                  margin={{ left: 10, right: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    width={200}
                    tick={{ fill: '#d1d5db', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(val: any, _: any, props: any) => [
                      `${(val * 100).toFixed(2)}% contribution | raw: ${props.payload.raw}`,
                      props.payload.direction === 'increases_risk'
                        ? '↑ Increases Risk'
                        : '↓ Decreases Risk',
                    ]}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {scoreData.top_factors.map((f, i) => (
                      <Cell
                        key={i}
                        fill={f.direction === 'increases_risk' ? '#ef4444' : '#22c55e'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-800">
                      <th className="text-left py-2">Signal</th>
                      <th className="text-right py-2">Raw Value</th>
                      <th className="text-right py-2">Contribution</th>
                      <th className="text-right py-2">Direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scoreData.top_factors.map((f, i) => (
                      <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                        <td className="py-2 text-gray-200">{f.human_readable}</td>
                        <td className="py-2 text-right text-gray-300">{f.raw_value}</td>
                        <td className="py-2 text-right font-mono">
                          {(f.contribution * 100).toFixed(2)}%
                        </td>
                        <td
                          className={`py-2 text-right font-semibold ${
                            f.direction === 'increases_risk' ? 'text-red-400' : 'text-green-400'
                          }`}
                        >
                          {f.direction === 'increases_risk' ? '↑ Risk' : '↓ Risk'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="text-gray-500">Score this customer first to see SHAP explanations.</p>
          )}
        </div>
      )}

      {/* ── Tab: All Transactions ──────────────────────────────────────────── */}
      {activeTab === 'transactions' && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-base font-semibold">
              Transaction History ({txnTotal.toLocaleString()} total)
            </h3>
            <div className="flex gap-2">
              <button
                disabled={txnPage === 0}
                onClick={() => setTxnPage((p) => p - 1)}
                className="px-3 py-1 bg-gray-800 rounded text-sm disabled:opacity-40 hover:bg-gray-700"
              >
                ← Prev
              </button>
              <span className="px-3 py-1 text-sm text-gray-400">
                {txnPage * TXN_PAGE_SIZE + 1}–
                {Math.min((txnPage + 1) * TXN_PAGE_SIZE, txnTotal)}
              </span>
              <button
                disabled={(txnPage + 1) * TXN_PAGE_SIZE >= txnTotal}
                onClick={() => setTxnPage((p) => p + 1)}
                className="px-3 py-1 bg-gray-800 rounded text-sm disabled:opacity-40 hover:bg-gray-700"
              >
                Next →
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-800 text-left">
                  <th className="py-2 pr-4">Date & Time</th>
                  <th className="py-2 pr-4">Type</th>
                  <th className="py-2 pr-4">Category</th>
                  <th className="py-2 pr-4">Channel</th>
                  <th className="py-2 pr-4 text-right">Amount</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2">Signal</th>
                </tr>
              </thead>
              <tbody>
                {txns.map((t, i) => (
                  <tr
                    key={i}
                    className={`border-b border-gray-800/40 hover:bg-gray-800/30 ${
                      t.is_stress_signal ? 'bg-red-950/10' : ''
                    }`}
                  >
                    <td className="py-2 pr-4 text-gray-300 whitespace-nowrap">
                      {fmtDate(t.txn_timestamp)}
                    </td>
                    <td className="py-2 pr-4 text-white">{txnLabel(t.txn_type)}</td>
                    <td className="py-2 pr-4 text-gray-300 capitalize">{t.category}</td>
                    <td className="py-2 pr-4 text-gray-400 uppercase text-xs">{t.channel}</td>
                    <td className="py-2 pr-4 text-right font-mono font-medium">{fmt(t.amount)}</td>
                    <td className="py-2 pr-4">
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-semibold ${
                          t.status === 'failed'
                            ? 'bg-red-900/50 text-red-400'
                            : 'bg-green-900/30 text-green-400'
                        }`}
                      >
                        {t.status}
                      </span>
                    </td>
                    <td className="py-2">
                      {t.is_stress_signal && (
                        <span className="text-orange-400 text-xs font-semibold">⚡ Stress</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab: Score Impact ──────────────────────────────────────────────── */}
      {activeTab === 'score-impact' && (
        <div className="space-y-4">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-base font-semibold mb-4">
              How Transactions Changed the Pulse Score
            </h3>
            {impact.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={[...impact].reverse()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      dataKey="txn_timestamp"
                      tickFormatter={(v) =>
                        new Date(v).toLocaleDateString('en-IN', {
                          day: '2-digit',
                          month: 'short',
                        })
                      }
                      tick={{ fill: '#9ca3af', fontSize: 10 }}
                    />
                    <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
                      labelFormatter={(v) => fmtDate(v)}
                      formatter={(val) => [val, 'Pulse Score']}
                    />
                    <ReferenceLine
                      y={70}
                      stroke="#ef4444"
                      strokeDasharray="4 4"
                      label={{ value: 'Critical', fill: '#ef4444', fontSize: 10 }}
                    />
                    <ReferenceLine
                      y={45}
                      stroke="#f97316"
                      strokeDasharray="4 4"
                      label={{ value: 'At Risk', fill: '#f97316', fontSize: 10 }}
                    />
                    <ReferenceLine
                      y={25}
                      stroke="#eab308"
                      strokeDasharray="4 4"
                      label={{ value: 'Watch', fill: '#eab308', fontSize: 10 }}
                    />
                    <Line dataKey="score_after" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>

                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-400 border-b border-gray-800 text-left">
                        <th className="py-2 pr-4">Date</th>
                        <th className="py-2 pr-4">Transaction</th>
                        <th className="py-2 pr-4 text-right">Amount</th>
                        <th className="py-2 pr-4 text-center">Score Change</th>
                        <th className="py-2 pr-4 text-center">Score After</th>
                        <th className="py-2">Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {impact.map((t, i) => (
                        <tr
                          key={i}
                          className={`border-b border-gray-800/40 hover:bg-gray-800/30 ${
                            t.is_stress_signal ? 'bg-red-950/10' : ''
                          }`}
                        >
                          <td className="py-2 pr-4 text-gray-400 text-xs whitespace-nowrap">
                            {fmtDate(t.txn_timestamp)}
                          </td>
                          <td className="py-2 pr-4 text-white">{txnLabel(t.txn_type)}</td>
                          <td className="py-2 pr-4 text-right font-mono">{fmt(t.amount)}</td>
                          <td className="py-2 pr-4 text-center">
                            {t.score_impact !== null && t.score_impact !== undefined ? (
                              <span
                                className={`font-bold text-base ${
                                  t.score_impact > 0
                                    ? 'text-red-400'
                                    : t.score_impact < 0
                                      ? 'text-green-400'
                                      : 'text-gray-500'
                                }`}
                              >
                                {t.score_impact > 0 ? `+${t.score_impact}` : t.score_impact === 0 ? '—' : t.score_impact}
                              </span>
                            ) : (
                              <span className="text-gray-700 text-xs">—</span>
                            )}
                          </td>
                          <td className="py-2 pr-4 text-center">
                            {t.score_after !== null && t.score_after !== undefined ? (
                              <span
                                className="font-mono font-bold"
                                style={{ color: TIER_COLORS[profile.risk_tier] || '#fff' }}
                              >
                                {t.score_after}
                              </span>
                            ) : (
                              <span className="text-gray-700 text-xs">no history</span>
                            )}
                          </td>
                          <td className="py-2 text-gray-400 text-xs">{t.impact_reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <p className="text-gray-500">No transaction impact data available.</p>
            )}
          </div>
        </div>
      )}

      {/* ── Tab: Score History ─────────────────────────────────────────────── */}
      {activeTab === 'history' && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-semibold">Pulse Score Timeline</h3>
            <span className="text-xs text-gray-500">{history.length} records</span>
          </div>

          {history.length > 0 ? (
            <>
              {/* ── Chart ── */}
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={[...history].reverse()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="scored_at"
                    tickFormatter={(v) =>
                      new Date(v).toLocaleString('en-IN', {
                        day: '2-digit', month: 'short',
                        hour: '2-digit', minute: '2-digit',
                      })
                    }
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
                    labelFormatter={(v) => fmtDate(v)}
                    formatter={(val: number, name: string) => [
                      name === 'pulse_score' ? val : `${(val * 100).toFixed(2)}%`,
                      name === 'pulse_score' ? 'Pulse Score' : 'PD Probability',
                    ]}
                  />
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Critical', fill: '#ef4444', fontSize: 10 }} />
                  <ReferenceLine y={45} stroke="#f97316" strokeDasharray="3 3" label={{ value: 'At Risk', fill: '#f97316', fontSize: 10 }} />
                  <ReferenceLine y={25} stroke="#eab308" strokeDasharray="3 3" label={{ value: 'Watch', fill: '#eab308', fontSize: 10 }} />
                  <Line
                    dataKey="pulse_score"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={({ cx, cy, payload }: any) => (
                      <circle
                        key={payload.scored_at}
                        cx={cx} cy={cy} r={3}
                        fill={TIER_COLORS[payload.risk_tier] || '#3b82f6'}
                        stroke="none"
                      />
                    )}
                  />
                </LineChart>
              </ResponsiveContainer>

              {/* ── Full timeline table ── */}
              <div className="mt-5 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700 text-left text-xs uppercase tracking-wide">
                      <th className="py-2 pr-3">Timestamp</th>
                      <th className="py-2 pr-3 text-center">Score</th>
                      <th className="py-2 pr-3 text-center">Δ</th>
                      <th className="py-2 pr-3">Tier</th>
                      <th className="py-2 pr-3 text-right">PD %</th>
                      <th className="py-2 pr-3">Top Factor</th>
                      <th className="py-2 pr-3">Intervention</th>
                      <th className="py-2 text-gray-600">Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h, i) => {
                      const prev = history[i + 1];
                      const delta = prev ? h.pulse_score - prev.pulse_score : null;
                      const deltaColor =
                        delta === null ? 'text-gray-600'
                        : delta > 0 ? 'text-red-400'
                        : delta < 0 ? 'text-green-400'
                        : 'text-gray-500';
                      const deltaStr =
                        delta === null ? '—'
                        : delta > 0 ? `+${delta}`
                        : delta < 0 ? `${delta}`
                        : '0';
                      return (
                        <tr
                          key={h.scored_at + i}
                          className="border-b border-gray-800/40 hover:bg-gray-800/30"
                        >
                          <td className="py-2 pr-3 text-gray-400 text-xs whitespace-nowrap">
                            {fmtDate(h.scored_at)}
                          </td>
                          <td className="py-2 pr-3 text-center">
                            <span
                              className="font-bold text-sm"
                              style={{ color: TIER_COLORS[h.risk_tier] || '#fff' }}
                            >
                              {h.pulse_score}
                            </span>
                          </td>
                          <td className={`py-2 pr-3 text-center font-mono text-xs font-semibold ${deltaColor}`}>
                            {deltaStr}
                          </td>
                          <td className="py-2 pr-3">
                            <span
                              className="text-xs font-semibold px-2 py-0.5 rounded"
                              style={{
                                color: TIER_COLORS[h.risk_tier],
                                background: `${TIER_COLORS[h.risk_tier]}22`,
                              }}
                            >
                              {TIER_LABELS[h.risk_tier] || h.risk_tier}
                            </span>
                          </td>
                          <td className="py-2 pr-3 text-right font-mono text-gray-300 text-xs">
                            {(h.pd_probability * 100).toFixed(2)}%
                          </td>
                          <td className="py-2 pr-3 text-gray-400 text-xs max-w-[180px] truncate">
                            {h.top_factors[0] || '—'}
                          </td>
                          <td className="py-2 pr-3">
                            {h.intervention_flag ? (
                              <span className="text-xs px-2 py-0.5 rounded bg-orange-500/20 text-orange-400 whitespace-nowrap">
                                {h.intervention_type?.replace(/_/g, ' ') || 'yes'}
                              </span>
                            ) : (
                              <span className="text-xs text-gray-700">—</span>
                            )}
                          </td>
                          <td className="py-2 text-gray-700 text-xs font-mono">
                            {h.model_version || '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="text-gray-500 text-sm">
              No score history yet. Scores are recorded here every time the
              customer is scored — run the simulator or call POST /api/v1/score.
            </p>
          )}
        </div>
      )}
    </div>
  );
}