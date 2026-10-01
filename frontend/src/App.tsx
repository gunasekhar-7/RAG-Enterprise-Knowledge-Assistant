import { useState, useEffect } from 'react';
import { Database, Server, Cpu, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

interface HealthStatus {
  status: string;
  environment: string;
  version: string;
}

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const checkBackendHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/health');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setHealth(data);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to FastAPI backend');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkBackendHealth();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-indigo-600/20 text-indigo-400 p-2 rounded-lg border border-indigo-500/30">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white tracking-tight">
              RAG Enterprise Knowledge Assistant
            </h1>
            <p className="text-xs text-slate-400">Local Development Environment</p>
          </div>
        </div>
        <button
          onClick={checkBackendHealth}
          disabled={loading}
          className="flex items-center space-x-2 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-md border border-slate-700 transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Status</span>
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-5xl w-full mx-auto p-6 md:p-10 space-y-8">
        {/* Status Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
          <h2 className="text-base font-semibold text-slate-200 mb-4 flex items-center space-x-2">
            <span>System Connectivity Status</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* FastAPI Card */}
            <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2.5">
                  <Server className="w-5 h-5 text-indigo-400" />
                  <span className="text-sm font-medium text-slate-300">FastAPI Backend</span>
                </div>
                {health ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-amber-400" />
                )}
              </div>
              <div className="mt-4">
                {loading ? (
                  <span className="text-xs text-slate-400 animate-pulse">Checking connectivity...</span>
                ) : health ? (
                  <div className="text-xs text-emerald-400">
                    Online (v{health.version} - {health.environment})
                  </div>
                ) : (
                  <div className="text-xs text-amber-400 font-mono break-all">{error}</div>
                )}
              </div>
            </div>

            {/* PostgreSQL & pgvector Card */}
            <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2.5">
                  <Database className="w-5 h-5 text-emerald-400" />
                  <span className="text-sm font-medium text-slate-300">PostgreSQL (pgvector)</span>
                </div>
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              </div>
              <div className="mt-4 text-xs text-slate-400">
                Port 5432 · Vector Extension Enabled
              </div>
            </div>

            {/* Redis Card */}
            <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2.5">
                  <Cpu className="w-5 h-5 text-rose-400" />
                  <span className="text-sm font-medium text-slate-300">Redis 7 & MinIO</span>
                </div>
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              </div>
              <div className="mt-4 text-xs text-slate-400">
                Port 6379 (Redis) · Port 9000 (MinIO S3)
              </div>
            </div>
          </div>
        </div>

        {/* Quick Instructions */}
        <div className="border border-slate-800/80 bg-slate-900/40 rounded-xl p-6 text-sm text-slate-400 space-y-3">
          <h3 className="font-semibold text-slate-200">🚀 Environment Initialized Successfully</h3>
          <p>
            Your developer workspace is ready. You can interact with the API docs at{' '}
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-indigo-400 hover:underline"
            >
              http://localhost:8000/docs
            </a>
            .
          </p>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 py-4 text-center text-xs text-slate-500">
        RAG Enterprise Knowledge Assistant · Dev Environment Initialized
      </footer>
    </div>
  );
}
