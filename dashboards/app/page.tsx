'use client';

/**
 * Sentinel AI — Landing Page
 * Glassmorphism design inspired by the Stitch project.
 */
import Link from 'next/link';
import {
  Shield, Activity, TrendingUp, Zap,
  BarChart3, Brain, Bell, ArrowRight, ChevronRight,
} from 'lucide-react';

const features = [
  {
    icon: <Brain className="w-7 h-7" />,
    title: 'ML-Powered Risk Scoring',
    desc: 'XGBoost models trained on behavioral signals deliver real-time Pulse Scores for every customer.',
  },
  {
    icon: <Activity className="w-7 h-7" />,
    title: 'Behavioral Analytics',
    desc: 'Track spending velocity, transaction patterns, and stress signals before delinquency surfaces.',
  },
  {
    icon: <Shield className="w-7 h-7" />,
    title: 'SHAP Explainability',
    desc: 'Understand exactly which features drive each risk score with transparent factor contributions.',
  },
  {
    icon: <Bell className="w-7 h-7" />,
    title: 'Proactive Interventions',
    desc: 'Automated alerts and intervention recommendations triggered before customers miss payments.',
  },
  {
    icon: <BarChart3 className="w-7 h-7" />,
    title: 'Portfolio Intelligence',
    desc: 'Real-time dashboards with tier-level risk distribution, exposure analytics, and trend monitoring.',
  },
  {
    icon: <Zap className="w-7 h-7" />,
    title: 'Real-Time Processing',
    desc: 'Kafka-powered streaming pipeline scores transactions within seconds of occurrence.',
  },
];

const steps = [
  {
    num: '01',
    title: 'Ingest & Stream',
    desc: 'Transaction data flows through a Kafka pipeline, enriched with behavioral features in real time.',
  },
  {
    num: '02',
    title: 'Score & Analyze',
    desc: 'XGBoost models compute Pulse Scores and SHAP explanations for every customer interaction.',
  },
  {
    num: '03',
    title: 'Alert & Intervene',
    desc: 'Risk tier changes trigger automated interventions — restructuring offers, payment reminders, or escalation.',
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen relative overflow-hidden text-white"
      style={{
        background: 'linear-gradient(135deg, #0B1120 0%, #0F172A 30%, #1E1B4B 60%, #0B1120 100%)',
      }}
    >
      {/* Ambient glow effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full opacity-20"
          style={{ background: 'radial-gradient(circle, #3B82F6 0%, transparent 70%)' }} />
        <div className="absolute top-1/3 -right-40 w-[500px] h-[500px] rounded-full opacity-15"
          style={{ background: 'radial-gradient(circle, #8B5CF6 0%, transparent 70%)' }} />
        <div className="absolute -bottom-40 left-1/3 w-80 h-80 rounded-full opacity-10"
          style={{ background: 'radial-gradient(circle, #06B6D4 0%, transparent 70%)' }} />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)' }}>
            <Shield className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">SENTINEL</span>
          <span className="text-xs text-blue-400/70 font-medium ml-1">AI</span>
        </div>
        <div className="flex items-center gap-6">
          <Link href="#features" className="text-sm text-slate-400 hover:text-white transition">
            Features
          </Link>
          <Link href="#how-it-works" className="text-sm text-slate-400 hover:text-white transition">
            How It Works
          </Link>
          <Link
            href="/login"
            className="px-5 py-2 text-sm font-semibold rounded-lg transition"
            style={{
              background: 'linear-gradient(135deg, #3B82F6, #6366F1)',
              boxShadow: '0 0 20px rgba(59,130,246,0.3)',
            }}
          >
            Sign In
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 pt-20 pb-24 text-center">
        <div className="inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full text-xs font-medium"
          style={{
            background: 'rgba(59,130,246,0.1)',
            border: '1px solid rgba(59,130,246,0.2)',
            color: '#60A5FA',
          }}>
          <Zap className="w-3.5 h-3.5" />
          Pre-Delinquency Intelligence Platform
        </div>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-tight mb-6">
          <span className="block">Predict Risk</span>
          <span className="block"
            style={{
              background: 'linear-gradient(135deg, #60A5FA, #A78BFA, #34D399)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
            Before It Happens
          </span>
        </h1>

        <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
          Real-time ML-powered behavioral scoring detects financial stress signals
          before they become delinquencies. Protect your portfolio with proactive,
          explainable intelligence.
        </p>

        <div className="flex items-center justify-center gap-4">
          <Link
            href="/login"
            className="inline-flex items-center gap-2 px-8 py-3.5 text-sm font-semibold rounded-xl transition-all hover:scale-105"
            style={{
              background: 'linear-gradient(135deg, #3B82F6, #6366F1)',
              boxShadow: '0 4px 30px rgba(59,130,246,0.4)',
            }}
          >
            Launch Dashboard
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="#features"
            className="inline-flex items-center gap-2 px-8 py-3.5 text-sm font-semibold rounded-xl transition-all hover:bg-white/10"
            style={{
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
            }}
          >
            Explore Features
            <ChevronRight className="w-4 h-4" />
          </Link>
        </div>

        {/* Stats bar */}
        <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto">
          {[
            { value: '< 2s', label: 'Scoring Latency' },
            { value: '94%', label: 'Prediction Accuracy' },
            { value: '40%', label: 'Fewer Defaults' },
            { value: '1000+', label: 'Customers Scored' },
          ].map((stat, i) => (
            <div key={i} className="rounded-xl px-5 py-4"
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                backdropFilter: 'blur(12px)',
              }}
            >
              <p className="text-2xl font-bold"
                style={{
                  background: 'linear-gradient(135deg, #60A5FA, #A78BFA)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                {stat.value}
              </p>
              <p className="text-xs text-slate-500 mt-1">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto px-8 py-24">
        <div className="text-center mb-16">
          <p className="text-sm font-semibold text-blue-400 mb-2 uppercase tracking-wider">Features</p>
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
            Intelligence at Every Layer
          </h2>
          <p className="text-slate-400 mt-4 max-w-xl mx-auto">
            From real-time data ingestion to explainable risk scoring —
            everything you need for proactive credit management.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f, i) => (
            <div key={i} className="group rounded-2xl p-6 transition-all duration-300 hover:scale-[1.02]"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                backdropFilter: 'blur(16px)',
              }}
            >
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15))',
                  color: '#60A5FA',
                }}>
                {f.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="relative z-10 max-w-5xl mx-auto px-8 py-24">
        <div className="text-center mb-16">
          <p className="text-sm font-semibold text-blue-400 mb-2 uppercase tracking-wider">How it works</p>
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
            From Data to Decision in Seconds
          </h2>
        </div>

        <div className="space-y-6">
          {steps.map((s, i) => (
            <div key={i} className="flex gap-6 items-start rounded-2xl p-6"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                backdropFilter: 'blur(16px)',
              }}
            >
              <div className="text-3xl font-bold shrink-0"
                style={{
                  background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                {s.num}
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-1">{s.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative z-10 max-w-4xl mx-auto px-8 py-20 text-center">
        <div className="rounded-2xl p-12"
          style={{
            background: 'linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.08))',
            border: '1px solid rgba(59,130,246,0.15)',
            backdropFilter: 'blur(16px)',
          }}
        >
          <h2 className="text-3xl font-bold mb-4">Ready to Protect Your Portfolio?</h2>
          <p className="text-slate-400 mb-8 max-w-lg mx-auto">
            Start monitoring customer risk in real time. Catch stress signals before they become losses.
          </p>
          <Link
            href="/login"
            className="inline-flex items-center gap-2 px-8 py-3.5 text-sm font-semibold rounded-xl transition-all hover:scale-105"
            style={{
              background: 'linear-gradient(135deg, #3B82F6, #6366F1)',
              boxShadow: '0 4px 30px rgba(59,130,246,0.4)',
            }}
          >
            Get Started
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 py-8 text-center text-xs text-slate-500">
        <p>© 2026 Sentinel AI. Pre-Delinquency Intelligence Platform.</p>
      </footer>
    </div>
  );
}
