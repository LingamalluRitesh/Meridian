import React from 'react';
import { Server, Activity, Zap } from 'lucide-react';

export const ServingStudio: React.FC = () => {
  return (
    <div className="space-y-8">
      <div className="bg-slate-900/80 p-6 rounded-3xl border border-slate-800 flex justify-between items-center">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-purple-400">Micro-Batch Inference Gateway</span>
          <h1 className="text-2xl font-extrabold text-white">Real-Time Model Serving Telemetry</h1>
          <p className="text-xs text-slate-400 mt-1">Canary Routing: 90% Baseline (TabNet v1) • 10% Shadow Canary (FT-Transformer v2)</p>
        </div>
      </div>
    </div>
  );
};
