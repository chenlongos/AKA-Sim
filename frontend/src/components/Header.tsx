import React from 'react';

export const Header: React.FC = () => {
    return (
        <header className="glass-panel border-b border-slate-800 px-6 py-4 flex justify-between items-center z-20">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg">🤖</div>
                <div>
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">LeRobot ACT Simulator</h1>
                    <p className="text-xs text-slate-400 mono">Action Chunking with Transformers - Educational Edition</p>
                </div>
            </div>
            <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    <span className="text-slate-300">Simulation Active</span>
                </div>
                <div className="mono text-xs text-slate-500">60 FPS</div>
            </div>
        </header>
    );
};
