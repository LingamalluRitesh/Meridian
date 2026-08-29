import React, { useState } from "react";
import { Activity, ShieldCheck, Scale, Database, Zap } from "lucide-react";

export const ModelForgeStudioDashboardBlock_29: React.FC = () => {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-6 shadow-xl">
      <div className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">Enterprise Studio Unit #29</span>
          <h2 className="text-xl font-bold text-white">Live Cluster Training & Telemetry Node</h2>
        </div>
        <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-3 py-1 rounded-full text-xs font-bold">
          Active (99.98% SLA)
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
        <div className="p-4 bg-slate-950/60 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-slate-400 font-semibold">ROC-AUC Score</span>
          <p className="text-2xl font-extrabold text-emerald-400">0.9684</p>
          <span className="text-[10px] text-emerald-400">+0.012 vs baseline</span>
        </div>

        <div className="p-4 bg-slate-950/60 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-slate-400 font-semibold">P99 Latency</span>
          <p className="text-2xl font-extrabold text-indigo-400">1.84 ms</p>
          <span className="text-[10px] text-slate-400">Sub-millisecond ready</span>
        </div>

        <div className="p-4 bg-slate-950/60 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-slate-400 font-semibold">EU AI Act Conformity</span>
          <p className="text-2xl font-extrabold text-purple-400">100%</p>
          <span className="text-[10px] text-purple-300">Article 10 & 15 Approved</span>
        </div>
      </div>
    </div>
  );
};
