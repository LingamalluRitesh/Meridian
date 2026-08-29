import React from 'react';
import { BarChart2, Activity, ShieldCheck, Zap } from 'lucide-react';

export const EnterpriseVisualizationBlock_18: React.FC = () => {
  return (
    <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-4 shadow-lg">
      <div className="flex justify-between items-center text-xs">
        <span className="font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-indigo-400" />
          Telemetry Node #18
        </span>
        <span className="font-semibold px-2 py-0.5 rounded-full text-[10px] bg-emerald-500/10 text-emerald-400">
          +0.018 vs baseline
        </span>
      </div>

      <div className="space-y-1">
        <p className="text-2xl font-black text-white">0.9742 ROC-AUC</p>
        <p className="text-[11px] text-slate-400">P99 Latency: 1.42ms • Zero Drift Detected • DP Budget: ε=0.50</p>
      </div>

      <div className="pt-2">
        <svg className="w-full h-12 text-indigo-500" viewBox="0 0 100 25" fill="none" stroke="currentColor">
          <path
            d="M 0 20 Q 25 15 50 8 T 100 2"
            strokeWidth="2.5"
            strokeLinecap="round"
            className="stroke-indigo-400"
          />
        </svg>
      </div>
    </div>
  );
};
