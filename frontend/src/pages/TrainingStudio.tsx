import React, { useState } from 'react';
import { Play, Pause, RotateCcw, Cpu, Activity, Zap, CheckCircle2 } from 'lucide-react';

export const TrainingStudio: React.FC = () => {
  const [isTraining, setIsTraining] = useState(true);
  const [epoch, setEpoch] = useState(48);

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="bg-slate-900/80 p-6 rounded-3xl border border-slate-800 flex justify-between items-center">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">Deep Tabular AutoML Engine</span>
          <h1 className="text-2xl font-extrabold text-white">TabNet Credit Risk Classifier (Trial #14)</h1>
          <p className="text-xs text-slate-400 mt-1">Sparsemax Entropy Regularization • Ghost Batch Size: 128 • Epoch {epoch}/100</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setIsTraining(!isTraining)}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl text-xs flex items-center gap-1.5 shadow-lg shadow-indigo-500/20"
          >
            {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isTraining ? 'Pause Training' : 'Resume Training'}
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400 font-semibold">Validation ROC-AUC</span>
          <p className="text-2xl font-bold text-emerald-400">0.9642</p>
          <span className="text-[10px] text-emerald-400">+0.012 vs FT-Transformer baseline</span>
        </div>

        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400 font-semibold">Training Loss (Log-Loss)</span>
          <p className="text-2xl font-bold text-white">0.1420</p>
          <span className="text-[10px] text-indigo-300">Converging steadily</span>
        </div>

        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400 font-semibold">Sparsity Selection Ratio</span>
          <p className="text-2xl font-bold text-purple-400">84.2%</p>
          <span className="text-[10px] text-purple-300">18 / 114 active features</span>
        </div>

        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800 space-y-1">
          <span className="text-xs text-slate-400 font-semibold">GPU Cluster Throughput</span>
          <p className="text-2xl font-bold text-amber-400">42,500 samples/s</p>
          <span className="text-[10px] text-amber-300">4x NVIDIA A100 (98% Utilization)</span>
        </div>
      </div>
    </div>
  );
};
