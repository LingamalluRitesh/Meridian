import React from 'react';
import { Database, Zap, Clock, Shield } from 'lucide-react';

export const FeatureStoreStudio: React.FC = () => {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Enterprise Feature Store & Catalog</h1>
        <p className="text-xs text-slate-400">Online Redis sub-millisecond lookups, offline Parquet storage, and continuous feature drift detection</p>
      </div>

      <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">Registered Feature Views</h3>
        <div className="space-y-3 text-xs">
          <div className="p-4 bg-slate-950/60 rounded-2xl border border-slate-800 flex justify-between items-center">
            <div>
              <p className="font-bold text-white">user_financial_profile_fv</p>
              <p className="text-slate-400 font-mono text-[11px]">Entities: [user_id] • Features: [avg_balance_30d, transaction_velocity_7d, credit_util_pct]</p>
            </div>
            <span className="text-xs font-bold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full">
              P99 Latency: 0.8ms
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
