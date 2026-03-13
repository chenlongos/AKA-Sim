import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';

export interface RobotState {
    x: number;
    z: number;
    rotation: number;
    velocity: number;
    angularVelocity: number;
}

export type RobotConfig = string[];
export type SceneType = 'basic' | 'living_room' | 'classroom' | 'tennis_court';
export type SceneSize = 'small' | 'medium' | 'large';
export type SceneComplexity = 'low' | 'medium' | 'high';

export interface LogEntry {
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
    timestamp: number;
}

export interface Frame {
    state: number[];
    image: number[][][];
    imageBase64?: string;
    action: number[];
}

export type Episode = Frame[];

export interface CloudModel {
    id: string;
    path: string;
    created_at: string;
    updated_at: string;
}

export interface CloudDataset {
    id: string;
    path: string;
    size_bytes: number;
    created_at: string;
    updated_at: string;
}

export interface CloudTrainingStatus {
    status: string;
    epoch: number;
    num_epochs: number;
    avg_loss: number | null;
    progress: number;
    error: string | null;
    message: string | null;
}

export interface SimulationState {
    robotState: RobotState;
    isRecording: boolean;
    isTraining: boolean;
    isInferencing: boolean;
    episodes: Episode[];
    currentEpisode: Episode[];
    target: THREE.Mesh | null;
    walls: THREE.Mesh[];
    environmentGroup: THREE.Group | null;
    dirLight: THREE.DirectionalLight | null;
    plane: THREE.Mesh | null;
    robot: THREE.Group | null;
    camera: THREE.PerspectiveCamera | null;
    scene: THREE.Scene | null;
    renderer: THREE.WebGLRenderer | null;
    onboardCamera: THREE.PerspectiveCamera | null;
    onboardRenderTarget: THREE.WebGLRenderTarget | null;
    keys: Record<string, boolean>;
    animationFrameId: number;
    inferenceTimeoutId: ReturnType<typeof setTimeout>;
    model: tf.LayersModel | null;
    recordingIntervalId: ReturnType<typeof setInterval> | null;
    lastX: number;
    lastZ: number;
    stuckCounter: number;
    actionBuffer: any[];
}
