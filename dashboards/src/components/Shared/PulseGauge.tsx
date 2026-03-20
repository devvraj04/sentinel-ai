/**
 * Pulse Score Gauge component — displays 0-100 score with color bands.
 * Used on the individual customer profile page.
 */
import React from 'react';
 
interface PulseGaugeProps {
  score: number;
  size?: number;
}
 
const TIER_COLORS: Record<string, string> = {
  green:  '#16A34A',
  yellow: '#D97706',
  orange: '#EA580C',
  red:    '#DC2626',
};
 
function getTier(score: number): string {
  if (score >= 75) return 'red';
  if (score >= 50) return 'orange';
  if (score >= 30) return 'yellow';
  return 'green';
}
 
export const PulseGauge: React.FC<PulseGaugeProps> = ({ score, size = 120 }) => {
  const tier = getTier(score);
  const color = TIER_COLORS[tier];
  const pct = score / 100;
  const radius = (size / 2) - 10;
  const circumference = Math.PI * radius; // half circle
  const offset = circumference * (1 - pct);
  const cx = size / 2;
  const cy = size / 2;
 
  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 20} viewBox={`0 0 ${size} ${size / 2 + 20}`}>
        {/* Background arc */}
        <path
          d={`M 10 ${cy} A ${radius} ${radius} 0 0 1 ${size - 10} ${cy}`}
          fill="none"
          stroke="#E2E8F0"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Score arc */}
        <path
          d={`M 10 ${cy} A ${radius} ${radius} 0 0 1 ${size - 10} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${circumference}`}
          strokeDashoffset={`${circumference - (circumference * pct)}`}
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
        {/* Score label */}
        <text x={cx} y={cy - 5} textAnchor="middle" fontSize="28" fontWeight="700" fill={color}>
          {score}
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fontSize="11" fill="#64748B">
          Pulse Score
        </text>
      </svg>
      <span
        className="mt-1 px-3 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wide"
        style={{ backgroundColor: `${color}20`, color }}
      >
        {tier}
      </span>
    </div>
  );
};

