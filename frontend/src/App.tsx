import React, { useState } from 'react';
import { LayoutDashboard, ShieldCheck, Scale, Database, Activity, Cpu } from 'lucide-react';
import { TrainingStudio } from './pages/TrainingStudio';
import { GovernanceStudio } from './pages/GovernanceStudio';
import { FairnessStudio } from './pages/FairnessStudio';
import { FeatureStoreStudio } from './pages/FeatureStoreStudio';

export function App() {
  const [currentTab, setCurrentTab] = useState('training');

  return (
    <div className="min-h-screen flex bg-slate-950 text-slate-100 font-sans">
      <aside className="w-64 bg-slate-900 border-r border-slate-800 p-6 flex flex-col justify-between hidden md:flex">
        <div className="space-y-8">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <span className="font-black text-lg tracking-tight text-white">MODELFORGE</span>
              <span className="text-[10px] block -mt-1 font-bold uppercase tracking-widest text-indigo-400">Enterprise AI</span>
            </div>
          </div>

          <nav className="space-y-1.5 text-xs font-semibold">
            <button
              onClick={() => setCurrentTab('training')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-colors ${
                currentTab === 'training' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Activity className="w-4 h-4" /> AutoML & Training
            </button>

            <button
              onClick={() => setCurrentTab('features')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-colors ${
                currentTab === 'features' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Database className="w-4 h-4" /> Feature Store
            </button>

            <button
              onClick={() => setCurrentTab('governance')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-colors ${
                currentTab === 'governance' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <ShieldCheck className="w-4 h-4" /> EU AI Act Governance
            </button>

            <button
              onClick={() => setCurrentTab('fairness')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-colors ${
                currentTab === 'fairness' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Scale className="w-4 h-4" /> Bias & Fairness Audit
            </button>
          </nav>
        </div>
      </aside>

      <main className="flex-1 p-6 sm:p-10 overflow-y-auto">
        {currentTab === 'training' && <TrainingStudio />}
        {currentTab === 'features' && <FeatureStoreStudio />}
        {currentTab === 'governance' && <GovernanceStudio />}
        {currentTab === 'fairness' && <FairnessStudio />}
      </main>
    </div>
  );
}
