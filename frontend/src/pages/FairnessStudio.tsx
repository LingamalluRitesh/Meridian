import React from 'react';
import { Scale, CheckCircle2, AlertCircle } from 'lucide-react';

export const FairnessStudio: React.FC = () => {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Algorithmic Fairness & Bias Inspector</h1>
        <p className="text-xs text-slate-400">Disparate impact ratio, statistical parity differences, and intersectional demographic audits</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-2">
          <span className="text-xs font-bold uppercase text-slate-400">Disparate Impact Ratio</span>
          <p className="text-3xl font-black text-emerald-400">0.942</p>
          <p className="text-[11px] text-emerald-400 font-semibold flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" /> Exceeds EEOC 4/5ths Rule (0.80)
          </p>
        </div>

        <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-2">
          <span className="text-xs font-bold uppercase text-slate-400">Statistical Parity Diff</span>
          <p className="text-3xl font-black text-white">0.024</p>
          <p className="text-[11px] text-slate-400">Threshold: &lt; 0.05</p>
        </div>

        <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-2">
          <span className="text-xs font-bold uppercase text-slate-400">Equal Opportunity Diff</span>
          <p className="text-3xl font-black text-indigo-400">0.012</p>
          <p className="text-[11px] text-indigo-300">True Positive Rate parity confirmed</p>
        </div>
      </div>
    </div>
  );
};
