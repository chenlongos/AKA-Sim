import React from 'react';

interface SidebarRightProps {
    cameraCanvasRef: React.RefObject<HTMLCanvasElement>;
    showAttention: boolean;
    actionChunks: number[];
    logs: { message: string, type: string, time: string }[];
    logContainerRef: React.RefObject<HTMLDivElement>;
    clearLogs: () => void;
}

export const SidebarRight: React.FC<SidebarRightProps> = ({
    cameraCanvasRef,
    showAttention,
    actionChunks,
    logs,
    logContainerRef,
    clearLogs
}) => {
    return (
        <aside className="w-96 glass-panel border-l border-slate-800 flex flex-col">
            <div className="h-48 camera-feed border-b border-slate-800 relative">
                <canvas ref={cameraCanvasRef} className="w-full h-full object-cover"></canvas>
                <div className="absolute top-2 left-2 text-xs mono text-green-400 bg-black/50 px-2 py-1 rounded">CAM_01 (Onboard)</div>
                <div className="absolute bottom-2 right-2 text-xs text-slate-500">30 FPS</div>
                
                <div className={`absolute inset-0 attention-heatmap pointer-events-none transition-opacity duration-300 ${showAttention ? 'opacity-100' : 'opacity-0'}`}></div>
            </div>

            <div className="h-32 border-b border-slate-800 p-3 bg-slate-900/30">
                <h4 className="text-xs font-semibold text-slate-400 mb-2 uppercase">Action Chunking (ACT)</h4>
                <div className="flex items-end gap-1 h-16">
                    {actionChunks.length > 0 ? actionChunks.map((act, idx) => {
                        const colors = ['bg-slate-700', 'bg-blue-500', 'bg-blue-400', 'bg-purple-500', 'bg-purple-400'];
                        return (
                            <div key={idx} className={`flex-1 ${colors[act]} rounded-t transition-all duration-300`} style={{ height: `${20 + Math.random() * 60}%`, opacity: 1 - (idx * 0.1) }}></div>
                        );
                    }) : (
                        <div className="flex-1 bg-slate-800 rounded-t text-center text-[10px] text-slate-600 pt-2">Waiting...</div>
                    )}
                </div>
                <div className="flex justify-between text-[10px] text-slate-600 mt-1 mono">
                    <span>t+0</span>
                    <span>t+4</span>
                    <span>t+8</span>
                </div>
            </div>

            <div className="flex-1 flex flex-col min-h-0">
                <div className="p-3 border-b border-slate-800 flex justify-between items-center">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase">System Logs</h4>
                    <button onClick={clearLogs} className="text-[10px] text-slate-600 hover:text-slate-400">Clear</button>
                </div>
                <div ref={logContainerRef} className="flex-1 overflow-y-auto p-3 space-y-1 text-xs mono">
                    {logs.map((log, i) => (
                        <div key={i} className={`log-entry log-${log.type}`}>
                            [{log.time}] {log.message}
                        </div>
                    ))}
                </div>
            </div>
        </aside>
    );
};
