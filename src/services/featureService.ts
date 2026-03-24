import * as THREE from 'three';
import { SimulationState } from '../types';

/**
 * 特征工程工厂 (Feature Factory)
 * 职责：纯粹的物理和几何特征计算逻辑
 */
export const FeatureFactory = {
    /**
     * 计算相对于摄像头的目标特征
     */
    calculateRelativeTarget: (robotPos: {x: number, z: number, rotation: number}, targetPos: {x: number, z: number}) => {
        // 摄像头向前偏移 0.85
        const camX = robotPos.x + Math.sin(robotPos.rotation) * 0.85;
        const camZ = robotPos.z + Math.cos(robotPos.rotation) * 0.85;

        const dx = targetPos.x - camX;
        const dz = targetPos.z - camZ;
        const distance = Math.sqrt(dx * dx + dz * dz);

        const angleToTarget = Math.atan2(dx, dz);
        let angleDiff = angleToTarget - robotPos.rotation;

        while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
        while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

        const FOV = Math.PI / 4; // 45度
        const isVisible = Math.abs(angleDiff) <= FOV;

        return { distance, angleDiff, isVisible, dx, dz, camX, camZ };
    },

    /**
     * 射线投射检测遮挡
     */
    checkOcclusion: (camPos: {x: number, z: number}, targetPos: {x: number, z: number}, walls: THREE.Mesh[], target: THREE.Mesh) => {
        const dx = targetPos.x - camPos.x;
        const dz = targetPos.z - camPos.z;
        const distance = Math.sqrt(dx * dx + dz * dz);

        if (distance <= 1.0) return false;

        const steps = Math.floor(distance / 0.5);
        const stepX = dx / steps;
        const stepZ = dz / steps;

        for (let i = 1; i < steps; i++) {
            const px = camPos.x + stepX * i;
            const pz = camPos.z + stepZ * i;

            for (const wall of walls) {
                if (wall === target) continue;
                const w = wall.userData.w;
                const d = wall.userData.d;
                const wx = wall.position.x;
                const wz = wall.position.z;

                if (px > wx - w/2 && px < wx + w/2 && pz > wz - d/2 && pz < wz + d/2) {
                    return true; // 被遮挡
                }
            }
        }
        return false;
    }
};

// ─── 特征提取函数类型 ───
type FeatureExtractor = (sim: any, rel: any, occ: boolean) => number;

/**
 * V1 特征映射表（14 维，兼容远程 origin/3d-front 旧版本）
 * 语义：[x, z, angle, speed, targetDist, isColliding, 0, 0, ...]
 */
const V1_FEATURE_MAP: FeatureExtractor[] = [
    /* 0  */ (sim) => sim.robotState.x,
    /* 1  */ (sim) => sim.robotState.z,
    /* 2  */ (sim) => sim.robotState.rotation,                // 原始角度值
    /* 3  */ (sim) => sim.robotState.velocity,
    /* 4  */ (_sim, rel) => rel.distance,                     // targetDist
    /* 5  */ (sim) => sim.isColliding ? 1.0 : 0.0,
    // 6-13 留空（补零）
];

/**
 * V2 特征映射表（16 维，当前版本）
 * 语义：[x, z, sin(θ), cos(θ), speed, targetX, targetZ, ballVel, ...]
 * 顺序极其重要：新的特征应该总是追加在末尾，以保持向后兼容。
 */
const V2_FEATURE_MAP: FeatureExtractor[] = [
    /* 0  */ (sim) => sim.robotState.x,
    /* 1  */ (sim) => sim.robotState.z,
    /* 2  */ (sim) => Math.sin(sim.robotState.rotation),
    /* 3  */ (sim) => Math.cos(sim.robotState.rotation),
    /* 4  */ (sim) => sim.robotState.velocity,
    /* 5  */ (sim) => sim.target.position.x,
    /* 6  */ (sim) => sim.target.position.z,
    /* 7  */ (sim) => sim.ballVelX || 0,
    /* 8  */ (sim) => sim.ballVelZ || 0,
    /* 9  */ (_sim, rel) => rel.distance,
    /* 10 */ (_sim, rel) => rel.angleDiff,
    /* 11 */ (sim) => sim.isColliding ? 1.0 : 0.0,
    /* 12 */ (_sim, rel) => rel.isVisible ? 1.0 : 0.0,
    /* 13 */ (_sim, _rel, occ) => occ ? 1.0 : 0.0,
    /* 14 */ (sim) => sim.ballAccelX || 0,
    /* 15 */ (sim) => sim.ballAccelZ || 0,
    // --- 可以在此追加 16, 17, 18 维 ---
];

/**
 * 版本路由表：根据维度选择对应的特征映射
 * - 14 维 → V1（旧版兼容）
 * - 16 维 → V2（当前版本）
 * - 其他维度 → 默认使用 V2 并裁切/填零
 */
const VERSION_MAP: Record<number, FeatureExtractor[]> = {
    14: V1_FEATURE_MAP,
    16: V2_FEATURE_MAP,
};

/**
 * 特征适配器 (Feature Adapter)
 * 职责：根据模型版本/维度要求，组装特征向量
 */
export const FeatureAdapter = {
    /** 已知的维度版本列表 */
    KNOWN_DIMS: [14, 16] as readonly number[],

    /**
     * 根据指定的维度组装状态向量
     * @param dim 目标维度 (14 → V1, 16 → V2, 其他 → 默认 V2 裁切/填零)
     * @param simData 仿真环境实时数据
     */
    getFeatures: (dim: number, simData: any) => {
        const { robotState, target, walls } = simData;

        // 1. 预计算核心几何特征（供 Map 使用）
        const rel = FeatureFactory.calculateRelativeTarget(robotState, target.position);
        const isBlocked = FeatureFactory.checkOcclusion({ x: rel.camX, z: rel.camZ }, target.position, walls, target);

        // 2. 根据维度选择对应的特征映射表
        const featureMap = VERSION_MAP[dim] || V2_FEATURE_MAP;

        if (dim > featureMap.length) {
            console.warn(`Requested dim ${dim} exceeds feature map (${featureMap.length}). Extra dims filled with zeros.`);
        }

        // 3. 动态切片组装
        const state = new Array(dim).fill(0);
        for (let i = 0; i < dim; i++) {
            if (featureMap[i]) {
                state[i] = featureMap[i](simData, rel, isBlocked);
            }
            // 超出映射表范围的维度保持为 0
        }

        return state;
    }
};
