import React from 'react';
import { ShieldCheck, FileText, CheckCircle2, Lock, AlertTriangle } from 'lucide-react';

export const GovernanceStudio: React.FC = () => {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">EU AI Act & Regulatory Governance Studio</h1>
        <p className="text-xs text-slate-400">Automated Article 11 technical documentation, W3C cryptographic lineage, and differential privacy accounting</p>
      </div>

      <div className="bg-slate-900/60 p-6 rounded-3xl border border-slate-800 space-y-6">
        <div className="flex justify-between items-center border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase">Automated Model Card: TabNet Risk Scorer v1.4</h3>
            <p className="text-xs text-slate-400">Classification: High-Risk AI System (Annex III Financial Creditworthiness)</p>
          </div>
          <span className="inline-flex items-center gap-1 bg-emerald-500/10 text-emerald-400 font-bold px-3 py-1 rounded-full text-xs">
            <ShieldCheck className="w-4 h-4" /> 100% Compliant
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <div className="p-4 bg-slate-950/60 rounded-2xl border border-slate-800/80 space-y-2">
            <h4 className="font-bold text-slate-200">Article 10: Data & Governance Integrity</h4>
            <p className="text-slate-400">W3C PROV Hash: <span className="font-mono text-purple-400">sha256:e3b0c44298fc1c14...</span></p>
            <p className="text-slate-400">Point-In-Time AS-OF Join: <span className="text-emerald-400 font-bold">Zero Data Leakage Verified</span></p>
          </div>

          <div className="p-4 bg-slate-950/60 rounded-2xl border border-slate-800/80 space-y-2">
            <h4 className="font-bold text-slate-200">Article 15: Accuracy, Robustness & Cybersecurity</h4>
            <p className="text-slate-400">Differential Privacy: <span className="text-indigo-400 font-bold">ε = 0.50, δ = 1e-5 (DP-SGD)</span></p>
            <p className="text-slate-400">Adversarial FGSM Robustness: <span className="text-emerald-400 font-bold">96.8% Certified Accuracy</span></p>
          </div>
        </div>
      </div>
    </div>
  );
};
