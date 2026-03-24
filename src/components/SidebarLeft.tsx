import React from 'react';
import { CloudDataset, CloudModel, CloudTrainingStatus } from '../types';

interface SidebarLeftProps {
    trainingMode: 'frontend' | 'cloud';
    setTrainingMode: (mode: 'frontend' | 'cloud') => void;
    
    // Cloud Props
    cloudDatasets: CloudDataset[];
    selectedCloudDataset: string;
    setSelectedCloudDataset: (id: string) => void;
    fetchCloudDatasets: () => void;
    cloudModels: CloudModel[];
    selectedCloudModel: string;
    setSelectedCloudModel: (id: string) => void;
    fetchCloudModels: () => void;
    cloudTrainingStatus: CloudTrainingStatus | null;

    // Scene Props
    sceneType: string;
    setSceneType: (type: string) => void;
    sceneSize: string;
    setSceneSize: (size: string) => void;
    sceneComplexity: string;
    setSceneComplexity: (complexity: string) => void;

    // Robot Props
    robotConfig: string[];
    setRobotConfig: (config: string[]) => void;
    lightPos: { x: number, y: number, z: number };
    setLightPos: (pos: { x: number, y: number, z: number }) => void;
    resetRobot: () => void;

    // Control Props
    speed: number;
    setSpeed: (speed: number) => void;
    turnSpeed: number;
    setTurnSpeed: (speed: number) => void;
    simRef: React.MutableRefObject<any>; // Using any for simplicity here, but could be typed
    sendCommand: (cmd: string) => void;

    // Recording Props
    isRecording: boolean;
    toggleRecording: () => void;
    episodesCount: number;
    frameCount: number;
    actionCount: number;
    saveDataset: () => void;

    // Training Props
    startTraining: () => void;
    isTraining: boolean;
    trainingProgress: number;
    trainingStatus: string;

    // Inference Props
    trainedModel: { name: string } | null;
    startInference: () => void;
    isInferencing: boolean;
    showAttention: boolean;
    setShowAttention: (show: boolean) => void;

    // Ball Motion Props
    ballMotionMode: 'fixed' | 'random' | 'mixed';
    setBallMotionMode: (mode: 'fixed' | 'random' | 'mixed') => void;
    ballRadius: number;
    setBallRadius: (radius: number) => void;
    ballSpeed: number;
    setBallSpeed: (speed: number) => void;
    ballRandomIntensity: number;
    setBallRandomIntensity: (intensity: number) => void;
}

export const SidebarLeft: React.FC<SidebarLeftProps> = ({
    trainingMode, setTrainingMode,
    cloudDatasets, selectedCloudDataset, setSelectedCloudDataset, fetchCloudDatasets,
    cloudModels, selectedCloudModel, setSelectedCloudModel, fetchCloudModels, cloudTrainingStatus,
    sceneType, setSceneType, sceneSize, setSceneSize, sceneComplexity, setSceneComplexity,
    robotConfig, setRobotConfig, lightPos, setLightPos, resetRobot,
    speed, setSpeed, turnSpeed, setTurnSpeed, simRef, sendCommand,
    isRecording, toggleRecording, episodesCount, frameCount, actionCount, saveDataset,
    startTraining, isTraining, trainingProgress, trainingStatus,
    trainedModel, startInference, isInferencing, showAttention, setShowAttention,
    ballMotionMode, setBallMotionMode, ballRadius, setBallRadius, ballSpeed, setBallSpeed, ballRandomIntensity, setBallRandomIntensity
}) => {
    return (
        <aside className="w-80 glass-panel border-r border-slate-800 flex flex-col overflow-y-auto">
            <div className="p-4 space-y-6">
                <div className="bg-slate-900/50 p-1 rounded-lg flex text-xs font-medium border border-slate-800">
                    <button 
                        onClick={() => setTrainingMode('frontend')}
                        className={`flex-1 py-1.5 rounded-md transition-all ${trainingMode === 'frontend' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-slate-400 hover:text-slate-300'}`}
                    >
                        前端训练 (Frontend)
                    </button>
                    <button 
                        onClick={() => setTrainingMode('cloud')}
                        className={`flex-1 py-1.5 rounded-md transition-all ${trainingMode === 'cloud' ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20' : 'text-slate-400 hover:text-slate-300'}`}
                    >
                        云端训练 (Cloud)
                    </button>
                </div>

                {trainingMode === 'cloud' && (
                    <div className="space-y-3 p-3 bg-purple-900/10 border border-purple-500/20 rounded-lg">
                        <h3 className="text-xs font-semibold text-purple-400 uppercase tracking-wider flex items-center gap-2">
                            ☁️ 云端配置
                        </h3>
                        
                        <div className="space-y-2">
                            <label className="text-xs text-slate-400 block">选择数据集</label>
                            <div className="flex gap-2">
                                <select 
                                    value={selectedCloudDataset} 
                                    onChange={e => setSelectedCloudDataset(e.target.value)}
                                    className="flex-1 bg-slate-800 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-purple-500"
                                >
                                    <option value="">-- Select Dataset --</option>
                                    {cloudDatasets.map((ds, i) => (
                                        <option key={i} value={ds.path}>{ds.name} ({ds.size})</option>
                                    ))}
                                </select>
                                <button onClick={fetchCloudDatasets} className="p-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-400">
                                    ↻
                                </button>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-xs text-slate-400 block">选择模型</label>
                            <div className="flex gap-2">
                                <select 
                                    value={selectedCloudModel} 
                                    onChange={e => setSelectedCloudModel(e.target.value)}
                                    className="flex-1 bg-slate-800 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-purple-500"
                                >
                                    <option value="">-- Select Model --</option>
                                    {cloudModels.map((m, i) => (
                                        <option key={i} value={m.id}>{m.name} (v{m.version})</option>
                                    ))}
                                </select>
                                <button onClick={fetchCloudModels} className="p-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-400">
                                    ↻
                                </button>
                            </div>
                        </div>

                        {cloudTrainingStatus && (
                            <div className="text-[10px] mono bg-black/30 p-2 rounded border border-purple-500/10 text-purple-300">
                                Status: {cloudTrainingStatus.status}<br/>
                                Epoch: {cloudTrainingStatus.epoch}/{cloudTrainingStatus.total_epochs}<br/>
                                Loss: {cloudTrainingStatus.loss?.toFixed(4)}
                            </div>
                        )}
                    </div>
                )}

                <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">场景设置</h3>
                    <div className="space-y-2">
                        <select value={sceneType} onChange={e => setSceneType(e.target.value)} className="w-full bg-slate-800/50 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500 appearance-none">
                            <option value="basic">基础场景 (Basic)</option>
                            <option value="living_room">客厅场景 (Living Room)</option>
                            <option value="classroom">教室场景 (Classroom)</option>
                            <option value="tennis_court">网球场 (Tennis Court)</option>
                        </select>
                        <div className="flex gap-2">
                            <select value={sceneSize} onChange={e => setSceneSize(e.target.value)} className="flex-1 bg-slate-800/50 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500 appearance-none">
                                <option value="small">小尺寸</option>
                                <option value="medium">中尺寸</option>
                                <option value="large">大尺寸</option>
                            </select>
                            <select value={sceneComplexity} onChange={e => setSceneComplexity(e.target.value)} className="flex-1 bg-slate-800/50 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500 appearance-none">
                                <option value="low">低复杂度</option>
                                <option value="medium">中复杂度</option>
                                <option value="high">高复杂度</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">小车配置</h3>
                    <div className="space-y-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded border border-slate-800">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={robotConfig.includes('arm')}
                                onChange={(e) => {
                                    if (e.target.checked) {
                                        setRobotConfig([...robotConfig, 'arm']);
                                    } else {
                                        setRobotConfig(robotConfig.filter(c => c !== 'arm'));
                                    }
                                }}
                                className="accent-blue-500 rounded bg-slate-800 border-slate-700"
                            />
                            <span>机械臂 (Robotic Arm)</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={robotConfig.includes('gripper')}
                                onChange={(e) => {
                                    if (e.target.checked) {
                                        setRobotConfig([...robotConfig, 'gripper']);
                                    } else {
                                        setRobotConfig(robotConfig.filter(c => c !== 'gripper'));
                                    }
                                }}
                                className="accent-blue-500 rounded bg-slate-800 border-slate-700"
                            />
                            <span>夹爪 (Gripper)</span>
                        </label>
                    </div>
                </div>

                <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">光源设置</h3>
                    <div className="space-y-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded border border-slate-800">
                        <label className="flex items-center gap-2">
                            <span className="w-4">X:</span> <input type="range" min="-30" max="30" value={lightPos.x} onChange={e => setLightPos({...lightPos, x: Number(e.target.value)})} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                        </label>
                        <label className="flex items-center gap-2">
                            <span className="w-4">Y:</span> <input type="range" min="5" max="40" value={lightPos.y} onChange={e => setLightPos({...lightPos, y: Number(e.target.value)})} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                        </label>
                        <label className="flex items-center gap-2">
                            <span className="w-4">Z:</span> <input type="range" min="-30" max="30" value={lightPos.z} onChange={e => setLightPos({...lightPos, z: Number(e.target.value)})} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                        </label>
                    </div>
                </div>

                <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">🎾 小球运动控制</h3>
                    <div className="space-y-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded border border-slate-800">
                        <label className="block">
                            <span className="text-slate-300 mb-1 block">运动模式</span>
                            <select
                                value={ballMotionMode}
                                onChange={(e) => setBallMotionMode(e.target.value as 'fixed' | 'random' | 'mixed')}
                                className="w-full bg-slate-800 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500"
                            >
                                <option value="fixed">固定航迹 (Fixed)</option>
                                <option value="random">随机运动 (Random)</option>
                                <option value="mixed">混合运动 (Mixed)</option>
                            </select>
                        </label>
                        <label className="flex items-center gap-2">
                            <span className="w-12">半径:</span>
                            <input
                                type="range"
                                min="1"
                                max="10"
                                step="0.5"
                                value={ballRadius}
                                onChange={(e) => setBallRadius(Number(e.target.value))}
                                className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                            <span className="w-8 text-right">{ballRadius.toFixed(1)}</span>
                        </label>
                        <label className="flex items-center gap-2">
                            <span className="w-12">速度:</span>
                            <input
                                type="range"
                                min="0.1"
                                max="3"
                                step="0.1"
                                value={ballSpeed}
                                onChange={(e) => setBallSpeed(Number(e.target.value))}
                                className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                            <span className="w-8 text-right">{ballSpeed.toFixed(1)}</span>
                        </label>
                        {(ballMotionMode === 'random' || ballMotionMode === 'mixed') && (
                            <label className="flex items-center gap-2">
                                <span className="w-12">随机:</span>
                                <input
                                    type="range"
                                    min="0"
                                    max="2"
                                    step="0.1"
                                    value={ballRandomIntensity}
                                    onChange={(e) => setBallRandomIntensity(Number(e.target.value))}
                                    className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                />
                                <span className="w-8 text-right">{ballRandomIntensity.toFixed(1)}</span>
                            </label>
                        )}
                    </div>
                </div>

                <div className="space-y-3">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">小车控制</h3>
                        <button onClick={resetRobot} className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 px-2 py-1 rounded transition-all" title="复位小车位置">
                            ↺ 复位
                        </button>
                    </div>
                    <div className="space-y-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded border border-slate-800">
                        <label className="flex items-center gap-2">
                            <span className="w-8">速度:</span>
                            <input type="range" min="0.05" max="0.5" step="0.01" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                            <span className="w-8 text-right">{speed.toFixed(2)}</span>
                        </label>
                        <label className="flex items-center gap-2">
                            <span className="w-8">转向:</span>
                            <input type="range" min="0.01" max="0.2" step="0.01" value={turnSpeed} onChange={(e) => setTurnSpeed(Number(e.target.value))} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                            <span className="w-8 text-right">{turnSpeed.toFixed(2)}</span>
                        </label>
                    </div>
                    <div className="grid grid-cols-3 gap-2 p-3 bg-slate-900/50 rounded-lg border border-slate-800">
                        <div></div>
                        <button 
                            className="control-btn bg-slate-800 hover:bg-slate-700 text-white p-2 rounded border border-slate-600 flex flex-col items-center gap-1 active:bg-blue-600"
                            onMouseDown={() => simRef.current.keys['w'] = true}
                            onMouseUp={() => simRef.current.keys['w'] = false}
                            onMouseLeave={() => simRef.current.keys['w'] = false}
                            onTouchStart={(e) => { e.preventDefault(); simRef.current.keys['w'] = true; }}
                            onTouchEnd={(e) => { e.preventDefault(); simRef.current.keys['w'] = false; }}
                        >
                            <span className="text-lg leading-none">↑</span>
                            <span className="text-[10px]">W</span>
                        </button>
                        <div></div>
                        <button 
                            className="control-btn bg-slate-800 hover:bg-slate-700 text-white p-2 rounded border border-slate-600 flex flex-col items-center gap-1 active:bg-blue-600"
                            onMouseDown={() => simRef.current.keys['a'] = true}
                            onMouseUp={() => simRef.current.keys['a'] = false}
                            onMouseLeave={() => simRef.current.keys['a'] = false}
                            onTouchStart={(e) => { e.preventDefault(); simRef.current.keys['a'] = true; }}
                            onTouchEnd={(e) => { e.preventDefault(); simRef.current.keys['a'] = false; }}
                        >
                            <span className="text-lg leading-none">←</span>
                            <span className="text-[10px]">A</span>
                        </button>
                        <button 
                            className="control-btn bg-slate-800 hover:bg-slate-700 text-white p-2 rounded border border-slate-600 flex flex-col items-center gap-1 active:bg-blue-600"
                            onMouseDown={() => simRef.current.keys['s'] = true}
                            onMouseUp={() => simRef.current.keys['s'] = false}
                            onMouseLeave={() => simRef.current.keys['s'] = false}
                            onTouchStart={(e) => { e.preventDefault(); simRef.current.keys['s'] = true; }}
                            onTouchEnd={(e) => { e.preventDefault(); simRef.current.keys['s'] = false; }}
                        >
                            <span className="text-lg leading-none">↓</span>
                            <span className="text-[10px]">S</span>
                        </button>
                        <button 
                            className="control-btn bg-slate-800 hover:bg-slate-700 text-white p-2 rounded border border-slate-600 flex flex-col items-center gap-1 active:bg-blue-600"
                            onMouseDown={() => simRef.current.keys['d'] = true}
                            onMouseUp={() => simRef.current.keys['d'] = false}
                            onMouseLeave={() => simRef.current.keys['d'] = false}
                            onTouchStart={(e) => { e.preventDefault(); simRef.current.keys['d'] = true; }}
                            onTouchEnd={(e) => { e.preventDefault(); simRef.current.keys['d'] = false; }}
                        >
                            <span className="text-lg leading-none">→</span>
                            <span className="text-[10px]">D</span>
                        </button>
                    </div>
                    <p className="text-[10px] text-slate-500 text-center">使用键盘 WASD 或方向键控制</p>
                </div>

                <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full bg-red-500 ${isRecording ? 'recording-pulse opacity-100' : 'opacity-30'}`}></span>
                        数据采集
                    </h3>
                    <div className="flex gap-2">
                        <button onClick={toggleRecording} className={`flex-1 ${isRecording ? 'bg-red-600/40' : 'bg-red-600/20'} hover:bg-red-600/30 text-red-400 border border-red-500/30 py-2 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2`}>
                            <span className={`w-2 h-2 rounded-full bg-red-500 ${isRecording ? 'recording-pulse' : ''}`}></span>
                            {isRecording ? '停止采集' : '开始采集'}
                        </button>
                    </div>
                    <div className="text-xs text-slate-400 mono bg-slate-900/50 p-2 rounded border border-slate-800">
                        <div>Episodes: <span className="text-blue-400">{episodesCount}</span></div>
                        <div>Frames: <span className="text-blue-400">{frameCount}</span></div>
                        <div>Actions: <span className="text-blue-400">{actionCount}</span></div>
                    </div>
                    <button onClick={saveDataset} disabled={episodesCount === 0} className={`w-full ${trainingMode === 'cloud' ? 'bg-purple-600 hover:bg-purple-500 text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'} border border-slate-600 py-2 rounded-lg text-sm transition-all disabled:opacity-50`}>
                        {trainingMode === 'cloud' ? '上传数据集到云端 (Upload)' : '保存数据集 (JSON)'}
                    </button>
                </div>

                <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                        {trainingMode === 'cloud' ? 'ACT 云端训练' : 'ACT 模型训练'}
                    </h3>
                    <button 
                        onClick={startTraining} 
                        disabled={trainingMode === 'cloud' ? !selectedCloudDataset : (episodesCount === 0 || isTraining)} 
                        className={`w-full ${trainingMode === 'cloud' ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500' : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500'} text-white py-3 rounded-lg font-medium transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50`}
                    >
                        {trainingMode === 'cloud' ? '开始云端训练' : '开始训练模型'}
                    </button>
                    {isTraining && (
                        <div className="space-y-2">
                            <div className="flex justify-between text-xs text-slate-400">
                                <span>Training...</span>
                                <span>{Math.floor(trainingProgress)}%</span>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                                <div className="h-full training-bar" style={{ width: `${trainingProgress}%` }}></div>
                            </div>
                            <div className="text-xs text-slate-500 mono">{trainingStatus}</div>
                        </div>
                    )}
                </div>

                <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">模型推理</h3>
                    
                    {trainingMode === 'cloud' ? (
                        <div className="text-xs text-purple-300 bg-purple-900/20 p-2 rounded border border-purple-500/20 mb-2">
                            当前云端模型: {selectedCloudModel || '未选择'}
                        </div>
                    ) : (
                        <select className="w-full bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500" defaultValue="">
                            <option value="">选择训练好的模型...</option>
                            {trainedModel ? (
                                <option value="act_model_v1">ACT_Model_v1 (Ready)</option>
                            ) : (
                                <option value="act_model_v1" disabled>ACT_Model_v1 (请先训练)</option>
                            )}
                        </select>
                    )}

                    <button 
                        onClick={startInference} 
                        disabled={trainingMode === 'cloud' ? !selectedCloudModel : !trainedModel} 
                        className={`w-full ${isInferencing ? 'bg-red-600/20 text-red-400 border-red-500/30 hover:bg-red-600/30' : 'bg-green-600/20 text-green-400 border-green-500/30 hover:bg-green-600/30'} border py-2 rounded-lg font-medium transition-all disabled:opacity-50`}
                    >
                        {isInferencing ? '停止推理' : (trainingMode === 'cloud' ? '启动云端推理' : '启动自主推理')}
                    </button>
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                        <input type="checkbox" id="show-attention" checked={showAttention} onChange={(e) => setShowAttention(e.target.checked)} className="rounded bg-slate-800 border-slate-600" />
                        <label htmlFor="show-attention">显示注意力热力图</label>
                    </div>
                </div>
            </div>
        </aside>
    );
};
