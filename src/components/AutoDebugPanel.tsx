import React from 'react';
import { AutoState } from '../services/autoController';

interface AutoDebugPanelProps {
  autoState: AutoState;
  ballPosition: { x: number; y: number; z: number } | null;
  bucketPosition: { x: number; y: number; z: number } | null;
  robotPosition: { x: number; y: number; z: number };
  isAutoMode: boolean;
}

export function AutoDebugPanel({
  autoState,
  ballPosition,
  bucketPosition,
  robotPosition,
  isAutoMode
}: AutoDebugPanelProps) {

  if (!isAutoMode) {
    return null;
  }

  const getStateColor = (state: AutoState): string => {
    switch (state) {
      case AutoState.IDLE: return 'text-gray-400';
      case AutoState.TRACKING: return 'text-blue-400';
      case AutoState.APPROACHING: return 'text-yellow-400';
      case AutoState.GRABBING: return 'text-orange-400';
      case AutoState.CARRYING: return 'text-purple-400';
      case AutoState.PLACING: return 'text-pink-400';
      case AutoState.COMPLETE: return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  const getStateLabel = (state: AutoState): string => {
    const labels: Record<AutoState, string> = {
      [AutoState.IDLE]: '空闲',
      [AutoState.TRACKING]: '追踪小球',
      [AutoState.APPROACHING]: '接近小球',
      [AutoState.GRABBING]: '抓取小球',
      [AutoState.CARRYING]: '搬运到红桶',
      [AutoState.PLACING]: '放入红桶',
      [AutoState.COMPLETE]: '任务完成'
    };
    return labels[state] || state;
  };

  const calculateDistance = (
    from: { x: number; y: number; z: number },
    to: { x: number; y: number; z: number }
  ): number => {
    return Math.sqrt(
      Math.pow(to.x - from.x, 2) +
      Math.pow(to.y - from.y, 2) +
      Math.pow(to.z - from.z, 2)
    );
  };

  const distanceToBall = ballPosition ? calculateDistance(robotPosition, ballPosition) : 0;
  const distanceToBucket = bucketPosition ? calculateDistance(robotPosition, bucketPosition) : 0;

  return (
    <div className="auto-debug-panel fixed top-4 right-4 bg-slate-900/90 backdrop-blur-sm border border-slate-700 rounded-lg p-4 shadow-xl z-50 w-64">
      <h3 className="text-white font-bold mb-3 text-sm border-b border-slate-700 pb-2">
        🤖 自动控制状态
      </h3>

      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-slate-400 text-xs">当前状态:</span>
          <span className={`text-xs font-bold ${getStateColor(autoState)}`}>
            {getStateLabel(autoState)}
          </span>
        </div>

        {ballPosition && (
          <div className="flex justify-between items-center">
            <span className="text-slate-400 text-xs">小球位置:</span>
            <span className="text-green-400 text-xs font-mono">
              ({ballPosition.x.toFixed(1)}, {ballPosition.z.toFixed(1)})
            </span>
          </div>
        )}

        {bucketPosition && (
          <div className="flex justify-between items-center">
            <span className="text-slate-400 text-xs">红桶位置:</span>
            <span className="text-red-400 text-xs font-mono">
              ({bucketPosition.x.toFixed(1)}, {bucketPosition.z.toFixed(1)})
            </span>
          </div>
        )}

        <div className="flex justify-between items-center">
          <span className="text-slate-400 text-xs">机器人位置:</span>
          <span className="text-blue-400 text-xs font-mono">
            ({robotPosition.x.toFixed(1)}, {robotPosition.z.toFixed(1)})
          </span>
        </div>

        {ballPosition && (
          <div className="flex justify-between items-center">
            <span className="text-slate-400 text-xs">距离小球:</span>
            <span className={`text-xs font-mono ${distanceToBall < 1.0 ? 'text-green-400' : 'text-yellow-400'}`}>
              {distanceToBall.toFixed(2)}m
            </span>
          </div>
        )}

        {bucketPosition && (
          <div className="flex justify-between items-center">
            <span className="text-slate-400 text-xs">距离红桶:</span>
            <span className={`text-xs font-mono ${distanceToBucket < 1.0 ? 'text-green-400' : 'text-yellow-400'}`}>
              {distanceToBucket.toFixed(2)}m
            </span>
          </div>
        )}

        <div className="mt-3 pt-2 border-t border-slate-700">
          <div className="text-xs text-slate-500 mb-1">任务进度:</div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                autoState === AutoState.IDLE ? 'bg-gray-500' :
                autoState === AutoState.TRACKING ? 'bg-blue-500' :
                autoState === AutoState.APPROACHING ? 'bg-yellow-500' :
                autoState === AutoState.GRABBING ? 'bg-orange-500' :
                autoState === AutoState.CARRYING ? 'bg-purple-500' :
                autoState === AutoState.PLACING ? 'bg-pink-500' : 'bg-green-500'
              }`}
              style={{
                width: `${(Object.values(AutoState).indexOf(autoState) / Object.values(AutoState).length) * 100}%`
              }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
}
