import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import { cloudService } from './services/cloudService';
import { actService } from './services/actService';
import { CloudModel, CloudDataset, CloudTrainingStatus, SimulationState, RobotConfig, SceneType, SceneSize, SceneComplexity, LogEntry } from './types';
import { createScene, createCamera, createRenderer, createLights, createFloor } from './sim/scene';
import { createRobot } from './sim/robot';
import { updateEnvironment } from './sim/environment';
import { tr } from 'motion/react-client';

interface Obstacle {
    id: string;
    x: number;
    z: number;
    w: number;
    h: number;
    d: number;
    color: number;
    type: 'box' | 'cylinder';
    rotation: number;
}

interface ScenePreset {
    id: string;
    name: string;
    sceneType: SceneType;
    sceneSize: SceneSize;
    sceneComplexity: SceneComplexity;
    obstacles: Omit<Obstacle, 'id'>[];
    targetPos: { x: number; z: number };
    createdAt: string;
}

const SCENE_PRESETS_KEY = 'scenePresets';

let obstacleIdCounter = 0;
const nextObstacleId = () => `obs_${++obstacleIdCounter}_${Date.now()}`;

export default function App() {
    // State
    const [isRecording, setIsRecording] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [isInferencing, setIsInferencing] = useState(false);
    const [episodesCount, setEpisodesCount] = useState(0);
    const [frameCount, setFrameCount] = useState(0);
    const [actionCount, setActionCount] = useState(0);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [trainingStatus, setTrainingStatus] = useState('');
    const [trainedModels, setTrainedModels] = useState<{ name: string }[]>([
        { name: '跟随网球示例' },
        { name: '自动避障示例' }
    ]);
    const [selectedModel, setSelectedModel] = useState<string>('跟随网球示例');
    const [trainedModel, setTrainedModel] = useState<{ name: string } | null>(null);
    const [showAttention, setShowAttention] = useState(false);
    const [sceneType, setSceneType] = useState(() => localStorage.getItem('sceneType') || 'basic');
    const [sceneSize, setSceneSize] = useState(() => localStorage.getItem('sceneSize') || 'medium');
    const [sceneComplexity, setSceneComplexity] = useState(() => localStorage.getItem('sceneComplexity') || 'low');
    const [hasArm, setHasArm] = useState(() => localStorage.getItem('hasArm') === 'true');
    const [lightPos, setLightPos] = useState(() => {
        const saved = localStorage.getItem('lightPos');
        return saved ? JSON.parse(saved) : { x: 10, y: 20, z: 10 };
    });
    const [speed, setSpeed] = useState(() => Number(localStorage.getItem('speed')) || 0.1);
    const [turnSpeed, setTurnSpeed] = useState(() => Number(localStorage.getItem('turnSpeed')) || 0.05);
    const [logs, setLogs] = useState<{ message: string, type: string, time: string }[]>([
        { message: 'System initialized. Waiting for commands...', type: 'info', time: new Date().toLocaleTimeString() }
    ]);
    const [actionChunks, setActionChunks] = useState<number[]>([]);
    const [activeKeys, setActiveKeys] = useState<Record<string, boolean>>({});

    // Obstacle placement
    const [obstacles, setObstacles] = useState<Obstacle[]>(() => {
        const saved = localStorage.getItem('customObstacles');
        return saved ? JSON.parse(saved) : [];
    });
    const [placementMode, setPlacementMode] = useState<'none' | 'box' | 'cylinder'>('none');
    const [selectedObstacleId, setSelectedObstacleId] = useState<string | null>(null);
    const obstacleMeshes = useRef<Map<string, THREE.Mesh>>(new Map());
    const placementModeRef = useRef(placementMode);
    placementModeRef.current = placementMode;

    // Scene presets
    const [scenePresets, setScenePresets] = useState<ScenePreset[]>(() => {
        const saved = localStorage.getItem(SCENE_PRESETS_KEY);
        return saved ? JSON.parse(saved) : [];
    });
    const [savingPreset, setSavingPreset] = useState(false);

    useEffect(() => {
        localStorage.setItem(SCENE_PRESETS_KEY, JSON.stringify(scenePresets));
    }, [scenePresets]);

    // Persist settings
    useEffect(() => {
        localStorage.setItem('sceneType', sceneType);
        localStorage.setItem('sceneSize', sceneSize);
        localStorage.setItem('sceneComplexity', sceneComplexity);
        localStorage.setItem('hasArm', hasArm.toString());
        localStorage.setItem('lightPos', JSON.stringify(lightPos));
        localStorage.setItem('speed', speed.toString());
        localStorage.setItem('turnSpeed', turnSpeed.toString());
    }, [sceneType, sceneSize, sceneComplexity, hasArm, lightPos, speed, turnSpeed]);

    useEffect(() => {
        localStorage.setItem('customObstacles', JSON.stringify(obstacles));
    }, [obstacles]);

    // Cloud Training State
    const [trainingMode, setTrainingMode] = useState<'frontend' | 'cloud'>('frontend');
    const [cloudModels, setCloudModels] = useState<any[]>([]);
    const [cloudDatasets, setCloudDatasets] = useState<any[]>([]);
    const [selectedCloudModel, setSelectedCloudModel] = useState('');
    const [selectedCloudDataset, setSelectedCloudDataset] = useState('');
    const [cloudTrainingStatus, setCloudTrainingStatus] = useState<any>(null);

    // Refs for DOM elements updated frequently
    const posXRef = useRef<HTMLSpanElement>(null);
    const posZRef = useRef<HTMLSpanElement>(null);
    const rotYRef = useRef<HTMLSpanElement>(null);
    const velocityRef = useRef<HTMLSpanElement>(null);
    const canvasContainerRef = useRef<HTMLDivElement>(null);
    const cameraCanvasRef = useRef<HTMLCanvasElement>(null);
    const logContainerRef = useRef<HTMLDivElement>(null);
    const smallCanvasRef = useRef<HTMLCanvasElement | null>(null);

    // Simulation State
    const sim = useRef({
        robotState: { x: 0, z: 0, rotation: Math.PI, velocity: 0, angularVelocity: 0 },
        isRecording: false,
        isTraining: false,
        isInferencing: false,
        episodes: [] as any[],
        currentEpisode: [] as any[],
        target: null as THREE.Mesh | null,
        walls: [] as THREE.Mesh[],
        environmentGroup: null as THREE.Group | null,
        dirLight: null as THREE.DirectionalLight | null,
        plane: null as THREE.Mesh | null,
        robot: null as THREE.Group | null,
        armGroup: null as THREE.Group | null,
        arm: null as any,
        wheels: [] as THREE.Group[],
        camera: null as THREE.PerspectiveCamera | null,
        scene: null as THREE.Scene | null,
        renderer: null as THREE.WebGLRenderer | null,
        onboardCamera: null as THREE.PerspectiveCamera | null,
        onboardRenderTarget: null as THREE.WebGLRenderTarget | null,
        keys: {} as Record<string, boolean>,
        animationFrameId: 0,
        inferenceTimeoutId: 0 as unknown as ReturnType<typeof setTimeout>,
        model: null as tf.LayersModel | null,
        recordingIntervalId: null as unknown as ReturnType<typeof setInterval>,
        lastX: 0,
        lastZ: 0,
        stuckCounter: 0,
        actionBuffer: [] as any[],
        lastInferenceLogTime: 0,
        lastManualLogTime: 0
    });

    const isAtBottom = useRef(true);
    const isDragging = useRef(false);
    const raycaster = useRef(new THREE.Raycaster());
    const mouse = useRef(new THREE.Vector2());

    const handleLogScroll = () => {
        if (logContainerRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current;
            isAtBottom.current = Math.abs(scrollHeight - scrollTop - clientHeight) < 10;
        }
    };

    useEffect(() => {
        if (isAtBottom.current && logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const addLog = useCallback((message: string, type = 'info') => {
        setLogs(prev => {
            const newLogs = [...prev, { message, type, time: new Date().toLocaleTimeString() }];
            if (newLogs.length > 200) {
                return newLogs.slice(-200);
            }
            return newLogs;
        });
    }, []);

    const clearLogs = () => setLogs([]);

    // Obstacle management helpers
    const addObstacleToScene = useCallback((obs: Obstacle) => {
        const group = sim.current.environmentGroup;
        if (!group) return;
        let mesh: THREE.Mesh;
        if (obs.type === 'cylinder') {
            const geo = new THREE.CylinderGeometry(obs.w / 2, obs.w / 2, obs.h, 16);
            const mat = new THREE.MeshStandardMaterial({ color: obs.color, roughness: 0.7 });
            mesh = new THREE.Mesh(geo, mat);
        } else {
            const geo = new THREE.BoxGeometry(obs.w, obs.h, obs.d);
            const mat = new THREE.MeshStandardMaterial({ color: obs.color, roughness: 0.7 });
            mesh = new THREE.Mesh(geo, mat);
        }
        mesh.position.set(obs.x, obs.h / 2, obs.z);
        mesh.rotation.y = obs.rotation;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { w: obs.w, d: obs.d, obstacleId: obs.id };
        group.add(mesh);
        sim.current.walls.push(mesh);
        obstacleMeshes.current.set(obs.id, mesh);
    }, []);

    const removeObstacleFromScene = useCallback((id: string) => {
        const mesh = obstacleMeshes.current.get(id);
        if (!mesh) return;
        const group = sim.current.environmentGroup;
        if (group) group.remove(mesh);
        sim.current.walls = sim.current.walls.filter(w => w.userData.obstacleId !== id);
        obstacleMeshes.current.delete(id);
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
    }, []);

    const updateObstacleInScene = useCallback((id: string, updates: Partial<Obstacle>) => {
        const mesh = obstacleMeshes.current.get(id);
        if (!mesh) return;
        if (updates.x !== undefined) mesh.position.x = updates.x;
        if (updates.z !== undefined) mesh.position.z = updates.z;
        if (updates.h !== undefined) mesh.position.y = updates.h / 2;
        if (updates.rotation !== undefined) mesh.rotation.y = updates.rotation;
    }, []);

    const addObstacle = useCallback((x: number, z: number, type: 'box' | 'cylinder' = 'box') => {
        const w = type === 'cylinder' ? 0.8 + Math.random() * 0.6 : 0.8 + Math.random() * 1.5;
        const h = type === 'cylinder' ? 0.6 + Math.random() * 1.0 : 0.8 + Math.random() * 1.5;
        const d = type === 'cylinder' ? w : 0.8 + Math.random() * 1.5;
        const colors = [0x64748b, 0x475569, 0x78716c, 0x92400e, 0x1e3a5f, 0x7c2d12];
        const color = colors[Math.floor(Math.random() * colors.length)];
        const obs: Obstacle = { id: nextObstacleId(), x, z, w, h, d, color, type, rotation: 0 };
        setObstacles(prev => [...prev, obs]);
        addObstacleToScene(obs);
        addLog(`Placed ${type} obstacle at (${x.toFixed(1)}, ${z.toFixed(1)})`, 'info');
    }, [addObstacleToScene, addLog]);

    const removeObstacle = useCallback((id: string) => {
        removeObstacleFromScene(id);
        setObstacles(prev => prev.filter(o => o.id !== id));
        if (selectedObstacleId === id) setSelectedObstacleId(null);
    }, [removeObstacleFromScene, selectedObstacleId]);

    const clearAllObstacles = useCallback(() => {
        obstacles.forEach(o => removeObstacleFromScene(o.id));
        setObstacles([]);
        setSelectedObstacleId(null);
        addLog('All custom obstacles cleared', 'info');
    }, [obstacles, removeObstacleFromScene, addLog]);

    // Scene preset management
    const saveScenePreset = useCallback((name: string) => {
        const target = sim.current.target;
        const preset: ScenePreset = {
            id: `preset_${Date.now()}`,
            name,
            sceneType,
            sceneSize,
            sceneComplexity,
            obstacles: obstacles.map(({ id: _, ...rest }) => rest),
            targetPos: target ? { x: target.position.x, z: target.position.z } : { x: 0, z: -5 },
            createdAt: new Date().toLocaleString(),
        };
        setScenePresets(prev => [...prev, preset]);
        setSavingPreset(false);
        addLog(`Scene saved: "${name}"`, 'success');
    }, [sceneType, sceneSize, sceneComplexity, obstacles, addLog]);

    const loadScenePreset = useCallback((preset: ScenePreset) => {
        // Clear current obstacles from scene
        obstacles.forEach(o => removeObstacleFromScene(o.id));

        // Restore scene settings
        setSceneType(preset.sceneType);
        setSceneSize(preset.sceneSize);
        setSceneComplexity(preset.sceneComplexity);

        // Restore obstacles (re-assign IDs)
        const newObstacles: Obstacle[] = preset.obstacles.map(o => ({
            ...o,
            id: nextObstacleId(),
        }));
        setObstacles(newObstacles);

        // Restore target position
        if (sim.current.target) {
            sim.current.target.position.set(preset.targetPos.x, 0.25, preset.targetPos.z);
        }

        addLog(`Scene loaded: "${preset.name}" (${newObstacles.length} obstacles)`, 'success');
    }, [obstacles, removeObstacleFromScene, addLog]);

    const deleteScenePreset = useCallback((id: string) => {
        setScenePresets(prev => {
            const preset = prev.find(p => p.id === id);
            if (preset) addLog(`Scene deleted: "${preset.name}"`, 'info');
            return prev.filter(p => p.id !== id);
        });
    }, [addLog]);

    const exportScenePreset = useCallback((preset: ScenePreset) => {
        const json = JSON.stringify(preset, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `scene_${preset.name.replace(/\s+/g, '_')}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }, []);

    const importScenePreset = useCallback(() => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = () => {
            const file = input.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = () => {
                try {
                    const preset = JSON.parse(reader.result as string) as ScenePreset;
                    if (!preset.name || !preset.sceneType || !preset.obstacles) {
                        addLog('Invalid scene file format', 'error');
                        return;
                    }
                    setScenePresets(prev => [...prev, { ...preset, id: `preset_${Date.now()}` }]);
                    addLog(`Scene imported: "${preset.name}"`, 'success');
                } catch {
                    addLog('Failed to parse scene file', 'error');
                }
            };
            reader.readAsText(file);
        };
        input.click();
    }, [addLog]);

    useEffect(() => {
        if (!canvasContainerRef.current) return;

        const container = canvasContainerRef.current;

        // Scene
        const scene = createScene();
        sim.current.scene = scene;

        // Camera
        const camera = createCamera(container.clientWidth, container.clientHeight);
        sim.current.camera = camera;

        // Renderer
        const renderer = createRenderer(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);
        sim.current.renderer = renderer;

        // Lights
        const dirLight = createLights(scene, { x: 10, y: 20, z: 10 });
        sim.current.dirLight = dirLight;

        // Floor
        const plane = createFloor(scene);
        sim.current.plane = plane;

        // Environment Group
        const environmentGroup = new THREE.Group();
        scene.add(environmentGroup);
        sim.current.environmentGroup = environmentGroup;

        // Robot
        const { robot, onboardCamera, onboardRenderTarget, wheelMeshes, armGroup, lowerArm, elbow, wrist, gripper } = createRobot(scene);
        sim.current.robot = robot;
        sim.current.armGroup = armGroup;
        sim.current.arm = {
            lowerArm, elbow, wrist, gripper,
            state: 'idle',
            grabbedObject: null,
            targetRotations: { lowerArm: Math.PI/4, elbow: Math.PI/2.5, wrist: -Math.PI/6 }
        };
        sim.current.wheels = wheelMeshes;
        sim.current.onboardCamera = onboardCamera;
        sim.current.onboardRenderTarget = onboardRenderTarget;
        
        robot.rotation.y = Math.PI;

        scene.add(robot);

        // Dragging Logic
        let draggingTarget = 'none' as 'none' | 'target' | 'obstacle';
        let draggingObstacleId: string | null = null;

        const onMouseDown = (event: MouseEvent) => {
            if (!container || !sim.current.camera || !sim.current.target) return;
            
            const rect = container.getBoundingClientRect();
            mouse.current.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
            mouse.current.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;
            
            raycaster.current.setFromCamera(mouse.current, sim.current.camera);

            // Placement mode: click on ground to place obstacle
            if (placementModeRef.current !== 'none') {
                const planeIntersects = raycaster.current.intersectObject(sim.current.plane!);
                if (planeIntersects.length > 0) {
                    const point = planeIntersects[0].point;
                    addObstacle(point.x, point.z, placementModeRef.current);
                }
                return;
            }

            // Check obstacles first (they are on top visually)
            const obstacleMeshArr = Array.from(obstacleMeshes.current.values());
            if (obstacleMeshArr.length > 0) {
                const obsIntersects = raycaster.current.intersectObjects(obstacleMeshArr);
                if (obsIntersects.length > 0) {
                    draggingTarget = 'obstacle';
                    draggingObstacleId = obsIntersects[0].object.userData.obstacleId;
                    isDragging.current = true;
                    setSelectedObstacleId(draggingObstacleId);
                    if (renderer.domElement) renderer.domElement.style.cursor = 'grabbing';
                    return;
                }
            }

            // Check target ball
            const intersects = raycaster.current.intersectObject(sim.current.target);
            if (intersects.length > 0) {
                draggingTarget = 'target';
                isDragging.current = true;
                if (renderer.domElement) renderer.domElement.style.cursor = 'grabbing';
            }
        };

        const onMouseMove = (event: MouseEvent) => {
            if (!container || !sim.current.camera || !sim.current.target) return;

            const rect = container.getBoundingClientRect();
            mouse.current.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
            mouse.current.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;

            raycaster.current.setFromCamera(mouse.current, sim.current.camera);

            if (!isDragging.current) {
                // Hover cursor
                const obstacleMeshArr = Array.from(obstacleMeshes.current.values());
                const obsIntersects = obstacleMeshArr.length > 0 ? raycaster.current.intersectObjects(obstacleMeshArr) : [];
                const targetIntersects = raycaster.current.intersectObject(sim.current.target);
                if (renderer.domElement) {
                    if (obsIntersects.length > 0 || targetIntersects.length > 0) {
                        renderer.domElement.style.cursor = placementModeRef.current !== 'none' ? 'crosshair' : 'grab';
                    } else {
                        renderer.domElement.style.cursor = placementModeRef.current !== 'none' ? 'crosshair' : 'default';
                    }
                }
                return;
            }

            if (!sim.current.plane) return;
            
            const intersects = raycaster.current.intersectObject(sim.current.plane);
            
            if (intersects.length > 0) {
                const point = intersects[0].point;
                if (draggingTarget === 'target') {
                    sim.current.target.position.x = point.x;
                    sim.current.target.position.z = point.z;
                } else if (draggingTarget === 'obstacle' && draggingObstacleId) {
                    setObstacles(prev => prev.map(o => o.id === draggingObstacleId ? { ...o, x: point.x, z: point.z } : o));
                    updateObstacleInScene(draggingObstacleId, { x: point.x, z: point.z });
                }
            }
        };

        const onMouseUp = () => {
            isDragging.current = false;
            draggingTarget = 'none';
            draggingObstacleId = null;
            if (renderer.domElement) renderer.domElement.style.cursor = placementModeRef.current !== 'none' ? 'crosshair' : 'default';
        };

        const onTouchStart = (event: TouchEvent) => {
            if (event.touches.length > 0) {
                const touch = event.touches[0];
                onMouseDown({ clientX: touch.clientX, clientY: touch.clientY } as MouseEvent);
            }
        };

        const onTouchMove = (event: TouchEvent) => {
            if (event.touches.length > 0) {
                const touch = event.touches[0];
                onMouseMove({ clientX: touch.clientX, clientY: touch.clientY } as MouseEvent);
            }
        };

        const onTouchEnd = () => {
            onMouseUp();
        };

        renderer.domElement.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        renderer.domElement.addEventListener('touchstart', onTouchStart, { passive: false });
        window.addEventListener('touchmove', onTouchMove, { passive: false });
        window.addEventListener('touchend', onTouchEnd);

        // Resize handler
        const onWindowResize = () => {
            if (!container || !sim.current.camera || !sim.current.renderer) return;
            sim.current.camera.aspect = container.clientWidth / container.clientHeight;
            sim.current.camera.updateProjectionMatrix();
            sim.current.renderer.setSize(container.clientWidth, container.clientHeight);
        };
        window.addEventListener('resize', onWindowResize);

        return () => {
            window.removeEventListener('resize', onWindowResize);
            cancelAnimationFrame(sim.current.animationFrameId);
            if (container && sim.current.renderer) {
                container.removeChild(sim.current.renderer.domElement);
            }
        };
    }, [addLog]);

    useEffect(() => {
        if (sim.current.armGroup) {
            sim.current.armGroup.visible = hasArm;
        }
    }, [hasArm]);

    useEffect(() => {
        if (sim.current.dirLight) {
            sim.current.dirLight.position.set(lightPos.x, lightPos.y, lightPos.z);
        }
    }, [lightPos]);

    useEffect(() => {
        if (!sim.current.environmentGroup || !sim.current.plane) return;

        const group = sim.current.environmentGroup;
        const plane = sim.current.plane;

        while (group.children.length > 0) {
            group.remove(group.children[0]);
        }
        sim.current.walls = [];

        let sizeVal = 20;
        if (sceneSize === 'small') sizeVal = 10;
        if (sceneSize === 'large') sizeVal = 30;

        const halfSize = sizeVal / 2;

        const createTexture = (type: string) => {
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 512;
            const ctx = canvas.getContext('2d')!;

            if (type === 'wood') {
                ctx.fillStyle = '#8b5a2b';
                ctx.fillRect(0, 0, 512, 512);
                for (let i = 0; i < 200; i++) {
                    ctx.fillStyle = `rgba(60, 30, 10, ${Math.random() * 0.15})`;
                    ctx.fillRect(0, Math.random() * 512, 512, Math.random() * 10);
                }
            } else if (type === 'tile') {
                ctx.fillStyle = '#e2e8f0';
                ctx.fillRect(0, 0, 512, 512);
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 4;
                for (let i = 0; i <= 512; i += 64) {
                    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 512); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(512, i); ctx.stroke();
                }
            } else if (type === 'tennis') {
                ctx.fillStyle = '#1e5631';
                ctx.fillRect(0, 0, 512, 512);
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 4;
                ctx.strokeRect(64, 32, 384, 448);
                ctx.beginPath(); ctx.moveTo(100, 32); ctx.lineTo(100, 480); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(412, 32); ctx.lineTo(412, 480); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(100, 128); ctx.lineTo(412, 128); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(100, 384); ctx.lineTo(412, 384); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(256, 128); ctx.lineTo(256, 384); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(256, 32); ctx.lineTo(256, 40); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(256, 480); ctx.lineTo(256, 472); ctx.stroke();
                ctx.lineWidth = 6;
                ctx.beginPath(); ctx.moveTo(64, 256); ctx.lineTo(448, 256); ctx.stroke();
            } else if (type === 'wall') {
                ctx.fillStyle = '#f8fafc';
                ctx.fillRect(0, 0, 512, 512);
                for (let i = 0; i < 500; i++) {
                    ctx.fillStyle = `rgba(0,0,0,${Math.random() * 0.03})`;
                    ctx.beginPath();
                    ctx.arc(Math.random() * 512, Math.random() * 512, Math.random() * 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            const tex = new THREE.CanvasTexture(canvas);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            if (type !== 'tennis') tex.repeat.set(sizeVal / 5, sizeVal / 5);
            return tex;
        };

        const woodTex = createTexture('wood');
        const tileTex = createTexture('tile');
        const tennisTex = createTexture('tennis');
        const wallTex = createTexture('wall');

        const planeMat = plane.material as THREE.MeshStandardMaterial;
        if (sceneType === 'living_room') {
            planeMat.map = woodTex;
            planeMat.color.setHex(0xffffff);
        } else if (sceneType === 'classroom') {
            planeMat.map = tileTex;
            planeMat.color.setHex(0xffffff);
        } else if (sceneType === 'tennis_court') {
            planeMat.map = tennisTex;
            planeMat.color.setHex(0xffffff);
        } else {
            planeMat.map = null;
            planeMat.color.setHex(0x1e293b);
        }
        planeMat.needsUpdate = true;

        const wallMat = new THREE.MeshStandardMaterial({ map: wallTex, color: 0xd1d5db, roughness: 0.9 });

        const addWall = (x: number, z: number, w: number, h: number, d: number, color?: number, map?: THREE.Texture) => {
            const mat = color ? new THREE.MeshStandardMaterial({ color, map: map || null }) : wallMat;
            const geo = new THREE.BoxGeometry(w, h, d);
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(x, h / 2, z);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.userData = { w, d };
            group.add(mesh);
            sim.current.walls.push(mesh);
        };

        if (sceneType !== 'tennis_court') {
            addWall(0, -halfSize, sizeVal, 2, 0.5);
            addWall(0, halfSize, sizeVal, 2, 0.5);
            addWall(-halfSize, 0, 0.5, 2, sizeVal);
            addWall(halfSize, 0, 0.5, 2, sizeVal);
        } else {
            addWall(0, -30, 60, 2, 0.5);
            addWall(0, 30, 60, 2, 0.5);
            addWall(-30, 0, 0.5, 2, 60);
            addWall(30, 0, 0.5, 2, 60);
        }

        let numObstacles = 0;
        if (sceneComplexity === 'medium') numObstacles = 5;
        if (sceneComplexity === 'high') numObstacles = 12;

        if (sceneType === 'basic') {
            for (let i = 0; i < numObstacles; i++) {
                const w = 1 + Math.random() * 2;
                const d = 1 + Math.random() * 2;
                const x = (Math.random() - 0.5) * (sizeVal - 4);
                const z = (Math.random() - 0.5) * (sizeVal - 4);
                if (Math.abs(x) < 2 && Math.abs(z) < 2) continue;
                addWall(x, z, w, 1.5, d, 0x64748b);
            }
        } else if (sceneType === 'living_room') {
            addWall(0, -halfSize + 2, 4, 1, 1.5, 0x334155);
            addWall(0, -halfSize + 1.2, 4, 2, 0.5, 0x334155);
            addWall(0, halfSize - 1, 3, 0.8, 1, 0x8b5cf6);
            addWall(0, -halfSize + 4, 2, 0.5, 1.5, 0xffffff, woodTex);

            for (let i = 0; i < numObstacles - 3; i++) {
                const w = 0.8 + Math.random() * 1;
                const d = 0.8 + Math.random() * 1;
                const x = (Math.random() - 0.5) * (sizeVal - 4);
                const z = (Math.random() - 0.5) * (sizeVal - 4);
                if (Math.abs(x) < 3 && Math.abs(z) < 3) continue;
                addWall(x, z, w, 1 + Math.random(), d, 0x475569);
            }
        } else if (sceneType === 'classroom') {
            const rows = Math.min(4, Math.max(2, Math.floor(numObstacles / 2)));
            const cols = Math.min(4, Math.max(2, Math.floor(numObstacles / 2)));
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const x = -halfSize / 2 + 2 + c * 3;
                    const z = -halfSize / 2 + 2 + r * 3;
                    if (Math.abs(x) < 2 && Math.abs(z) < 2) continue;
                    addWall(x, z, 1.5, 0.8, 1, 0xffffff, woodTex);
                }
            }
            addWall(0, halfSize - 2, 3, 1, 1.5, 0xffffff, woodTex);
        } else if (sceneType === 'tennis_court') {
            const netMat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.5, wireframe: true });
            const netGeo = new THREE.BoxGeometry(60, 1, 0.5);
            const net = new THREE.Mesh(netGeo, netMat);
            net.position.set(0, 0.5, 0);
            group.add(net);
        }

        const ballGeo = new THREE.SphereGeometry(0.25, 16, 16);
        const ballMat = new THREE.MeshStandardMaterial({ color: 0xccff00, roughness: 0.8 });
        const ball = new THREE.Mesh(ballGeo, ballMat);
        ball.position.set(0, 0.25, -5);
        ball.castShadow = true;
        ball.userData = { w: 0.5, d: 0.5 };
        group.add(ball);
        sim.current.walls.push(ball);
        sim.current.target = ball; // Use ball as the target since red cube is removed

        addLog(`Scene updated: ${sceneType}, Size: ${sceneSize}, Complexity: ${sceneComplexity}`, 'info');

        // Restore custom obstacles after scene regeneration
        obstacles.forEach(obs => addObstacleToScene(obs));
    }, [sceneType, sceneSize, sceneComplexity, addLog, obstacles, addObstacleToScene]);

    const [enableCollisionProtection, setEnableCollisionProtection] = useState(() => {
        const saved = localStorage.getItem('enableCollisionProtection');
        return saved !== null ? saved === 'true' : true;
    });

    useEffect(() => {
        localStorage.setItem('enableCollisionProtection', enableCollisionProtection.toString());
    }, [enableCollisionProtection]);

    const captureImage = useCallback(() => {
        if (!cameraCanvasRef.current) return null;

        if (!smallCanvasRef.current) {
            const c = document.createElement('canvas');
            c.width = 64;
            c.height = 64;
            smallCanvasRef.current = c;
        }

        const smallCtx = smallCanvasRef.current.getContext('2d', { willReadFrequently: true });
        if (smallCtx) {
            smallCtx.drawImage(cameraCanvasRef.current, 0, 0, 64, 64);
        }

        return tf.tidy(() => {
            const img = tf.browser.fromPixels(smallCanvasRef.current!);
            const normalized = img.div(255.0);
            return normalized.arraySync() as number[][][];
        });
    }, []);

    const recordFrame = useCallback(() => {
        if (!sim.current.target || !sim.current.isRecording) return;

        // Smart Recording: Check for collision
        if (enableCollisionProtection && sim.current.isColliding) {
            addLog('Collision detected! Stopping recording and discarding last 1s...', 'warning');

            // Discard last 1 second (approx 10 frames at 10Hz recording)
            const framesToDiscard = 10;
            if (sim.current.currentEpisode.length > framesToDiscard) {
                sim.current.currentEpisode.splice(-framesToDiscard, framesToDiscard);
            } else {
                sim.current.currentEpisode = [];
            }

            toggleRecording(); // Stop recording
            return;
        }

        const image = captureImage();
        if (!image) return;

        // Use small canvas for Base64 - much faster
        const imageBase64 = smallCanvasRef.current?.toDataURL('image/jpeg', 0.7).split(',')[1];

        // Match reference state structure:
        // state: [x, y, angle, speed, ballDist, isColliding, ...zeros] (14 dims)

        const x = sim.current.robotState.x;
        const z = sim.current.robotState.z; // treating z as y in 2D
        const angle = sim.current.robotState.rotation;
        const speed = sim.current.robotState.velocity;

        const targetDist = Math.hypot(sim.current.target.position.x - x, sim.current.target.position.z - z);

        const state = new Array(14).fill(0);
        state[0] = x;
        state[1] = z;
        state[2] = angle;
        state[3] = speed;
        state[4] = targetDist;
        state[5] = sim.current.isColliding ? 1.0 : 0.0; // Explicitly tell model we are colliding

        const envState = new Array(10).fill(0);
        envState[0] = x;
        envState[1] = z;
        envState[2] = angle;
        envState[3] = speed;
        envState[4] = sim.current.isColliding ? 1 : 0;
        envState[5] = targetDist;
        // Relative angle to ball (normalized to [-PI, PI])
        const relAngleToBall = Math.atan2(
            sim.current.target.position.x - x,
            sim.current.target.position.z - z
        ) - angle;
        envState[6] = Math.sin(relAngleToBall); // sin component for angle diff
        envState[7] = Math.cos(relAngleToBall); // cos component for angle diff
        // Ball position relative to robot
        envState[8] = sim.current.target.position.x - x;
        envState[9] = sim.current.target.position.z - z;

        // Action: [velocity, angularVelocity] (continuous, 2-dim)
        // Supports simultaneous forward + turning, matching real robot control.
        const keys = sim.current.keys;
        let v = 0, w = 0;
        if (keys['w'] || keys['arrowup']) v += sim.current.speed;
        if (keys['s'] || keys['arrowdown']) v -= sim.current.speed;
        if (keys['a'] || keys['arrowleft']) w -= sim.current.turnSpeed;
        if (keys['d'] || keys['arrowright']) w += sim.current.turnSpeed;
        const action = [v, w];

        const frame = {
            state: state,
            envState: envState,
            image: image,
            imageBase64: imageBase64,
            action: action
        };

        sim.current.currentEpisode.push(frame);

        // Throttle UI updates to avoid lag
        if (sim.current.currentEpisode.length % 5 === 0) {
            setFrameCount(sim.current.currentEpisode.length);
            setActionCount(sim.current.currentEpisode.length);
        }
    }, [captureImage]);

    const sendCommand = useCallback((cmd: string) => {
        if (sim.current.isTraining) return;

        switch (cmd) {
            case 'forward': sim.current.robotState.velocity = speed; break;
            case 'backward': sim.current.robotState.velocity = -speed; break;
            case 'left': sim.current.robotState.angularVelocity = turnSpeed; break;
            case 'right': sim.current.robotState.angularVelocity = -turnSpeed; break;
        }

        addLog(`Command: ${cmd.toUpperCase()}`, 'info');
    }, [addLog, speed, turnSpeed]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            const key = e.key.toLowerCase();
            sim.current.keys[key] = true;
            if (['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
                e.preventDefault();
            }
            // Obstacle interaction shortcuts (only when not in placement mode)
            if (placementMode === 'none' && selectedObstacleId) {
                if (key === 'r') {
                    setObstacles(prev => prev.map(o => {
                        if (o.id !== selectedObstacleId) return o;
                        const newRot = o.rotation + Math.PI / 12;
                        updateObstacleInScene(o.id, { rotation: newRot });
                        return { ...o, rotation: newRot };
                    }));
                }
                if (key === 'delete' || key === 'backspace') {
                    e.preventDefault();
                    removeObstacle(selectedObstacleId);
                }
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            sim.current.keys[e.key.toLowerCase()] = false;
        };

        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.removeEventListener('keyup', handleKeyUp);
        };
    }, [placementMode, selectedObstacleId, removeObstacle, updateObstacleInScene]);

    const updateRobotMovement = useCallback(() => {
        if (sim.current.isInferencing || sim.current.isTraining) return;

        const keys = sim.current.keys;
        let v = 0;
        let w = 0;

        if (keys['w'] || keys['arrowup']) v += speed;
        if (keys['s'] || keys['arrowdown']) v -= speed;
        if (keys['a'] || keys['arrowleft']) w += turnSpeed;
        if (keys['d'] || keys['arrowright']) w -= turnSpeed;

        if (v !== 0 || w !== 0) {
            if (!sim.current.lastManualLogTime || Date.now() - sim.current.lastManualLogTime > 200) {
                addLog(`Manual Movement: V=${v.toFixed(2)}, W=${w.toFixed(2)}`, 'info');
                sim.current.lastManualLogTime = Date.now();
            }
        }

        // Update active keys for UI highlighting
        const newActiveKeys: Record<string, boolean> = {};
        if (v > 0) newActiveKeys['w'] = true;
        if (v < 0) newActiveKeys['s'] = true;
        if (w > 0) newActiveKeys['a'] = true;
        if (w < 0) newActiveKeys['d'] = true;
        if (keys['q']) newActiveKeys['q'] = true;
        if (keys['e']) newActiveKeys['e'] = true;
        
        setActiveKeys(prev => {
            const keysChanged = Object.keys(newActiveKeys).length !== Object.keys(prev).length || 
                               Object.keys(newActiveKeys).some(k => newActiveKeys[k] !== prev[k]);
            return keysChanged ? newActiveKeys : prev;
        });

        sim.current.robotState.velocity = v;
        sim.current.robotState.angularVelocity = w;
    }, [speed, turnSpeed, addLog]);

    const updatePhysics = useCallback(() => {
        const state = sim.current.robotState;
        const robot = sim.current.robot;
        const camera = sim.current.camera;

        const nextX = state.x + Math.sin(state.rotation) * state.velocity;
        const nextZ = state.z + Math.cos(state.rotation) * state.velocity;

        let collided = false;
        const robotRadius = 0.9; // Increased to cover camera offset (1.2) and body
        const checkAABB = (objX: number, objZ: number, w: number, d: number) => {
            return (nextX > objX - w / 2 - robotRadius &&
                nextX < objX + w / 2 + robotRadius &&
                nextZ > objZ - d / 2 - robotRadius &&
                nextZ < objZ + d / 2 + robotRadius);
        };

        sim.current.walls.forEach(wall => {
            if (checkAABB(wall.position.x, wall.position.z, wall.userData.w, wall.userData.d)) {
                collided = true;
            }
        });

        if (sim.current.target && checkAABB(sim.current.target.position.x, sim.current.target.position.z, sim.current.target.userData.w, sim.current.target.userData.d)) {
            collided = true;
        }

        sim.current.isColliding = collided; // Expose collision state

        if (!collided) {
            state.x = nextX;
            state.z = nextZ;
        } else {
            state.velocity = 0;
        }

        state.rotation += state.angularVelocity;

        state.velocity *= 0.9;
        state.angularVelocity *= 0.9;

        if (robot) {
            robot.position.x = state.x;
            robot.position.z = state.z;
            robot.rotation.y = state.rotation;
            
            if (sim.current.wheels && sim.current.wheels.length === 2) {
                const wheelRadius = 0.3;
                const rotationAmount = state.velocity / wheelRadius;
                // Also add differential rotation for turning
                const turnRotation = state.angularVelocity * 0.5 / wheelRadius;
                
                sim.current.wheels[0].rotateY(-rotationAmount - turnRotation); // Right wheel
                sim.current.wheels[1].rotateY(-rotationAmount + turnRotation); // Left wheel
            }
        }

        if (camera) {
            camera.position.x = state.x;
            camera.position.z = state.z + 12;
            camera.lookAt(state.x, 0, state.z);
        }

        if (posXRef.current) posXRef.current.textContent = state.x.toFixed(2);
        if (posZRef.current) posZRef.current.textContent = state.z.toFixed(2);
        if (rotYRef.current) rotYRef.current.textContent = (state.rotation * 180 / Math.PI).toFixed(0) + '°';
        if (velocityRef.current) velocityRef.current.textContent = (Math.abs(state.velocity) * 10).toFixed(1);
    }, []);

    const updateOnboardCamera = useCallback(() => {
        const { onboardCamera, onboardRenderTarget, renderer, scene, robotState } = sim.current;
        const canvas = cameraCanvasRef.current;
        if (!onboardCamera || !onboardRenderTarget || !renderer || !scene || !canvas) return;

        const dirX = Math.sin(robotState.rotation);
        const dirZ = Math.cos(robotState.rotation);

        onboardCamera.position.set(
            robotState.x + dirX * 0.85,
            0.7,
            robotState.z + dirZ * 0.85
        );

        onboardCamera.lookAt(
            robotState.x + dirX * 10.0,
            0.7,
            robotState.z + dirZ * 10.0
        );

        renderer.setRenderTarget(onboardRenderTarget);
        renderer.clear();
        renderer.render(scene, onboardCamera);
        renderer.setRenderTarget(null);

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;
        canvas.width = 320;
        canvas.height = 240;

        const pixels = new Uint8Array(320 * 240 * 4);
        renderer.readRenderTargetPixels(onboardRenderTarget, 0, 0, 320, 240, pixels);

        const imageData = ctx.createImageData(320, 240);

        for (let y = 0; y < 240; y++) {
            for (let x = 0; x < 320; x++) {
                const srcIdx = ((239 - y) * 320 + x) * 4;
                const dstIdx = (y * 320 + x) * 4;

                imageData.data[dstIdx] = pixels[srcIdx];
                imageData.data[dstIdx + 1] = pixels[srcIdx + 1];
                imageData.data[dstIdx + 2] = pixels[srcIdx + 2];
                imageData.data[dstIdx + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }, []);

    const updateArm = useCallback(() => {
        if (!sim.current.arm || !hasArm) return;
        const arm = sim.current.arm;

        // Handle inputs
        if (sim.current.keys['q']) {
            sim.current.keys['q'] = false;
            if (arm.state === 'idle') {
                // Find object
                const robotPos = sim.current.robot!.position.clone();
                const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(sim.current.robot!.quaternion);
                const grabPos = robotPos.clone().add(forward.multiplyScalar(1.5));

                let closestObj = null;
                let minDistance = 1.5;

                for (const obj of sim.current.walls) {
                    if (obj.geometry.type === 'SphereGeometry' || obj.userData.isGrabbable) {
                        const dist = obj.position.distanceTo(grabPos);
                        if (dist < minDistance) {
                            minDistance = dist;
                            closestObj = obj;
                        }
                    }
                }

                if (closestObj) {
                    arm.grabbedObject = closestObj;
                    arm.state = 'picking_down';
                    arm.targetRotations = { lowerArm: Math.PI/2.2, elbow: Math.PI/4, wrist: Math.PI/4 };
                    addLog('Picking up object...', 'info');
                } else {
                    addLog('No object in range to pick up', 'warn');
                }
            }
        }

        if (sim.current.keys['e']) {
            sim.current.keys['e'] = false;
            if (arm.state === 'holding') {
                arm.state = 'dropping_down';
                arm.targetRotations = { lowerArm: Math.PI/2.2, elbow: Math.PI/4, wrist: Math.PI/4 };
                addLog('Dropping object...', 'info');
            }
        }

        // Animate joints
        arm.lowerArm.rotation.x += (arm.targetRotations.lowerArm - arm.lowerArm.rotation.x) * 0.1;
        arm.elbow.rotation.x += (arm.targetRotations.elbow - arm.elbow.rotation.x) * 0.1;
        arm.wrist.rotation.x += (arm.targetRotations.wrist - arm.wrist.rotation.x) * 0.1;

        // State machine transitions
        const isAtTarget = 
            Math.abs(arm.lowerArm.rotation.x - arm.targetRotations.lowerArm) < 0.05 &&
            Math.abs(arm.elbow.rotation.x - arm.targetRotations.elbow) < 0.05 &&
            Math.abs(arm.wrist.rotation.x - arm.targetRotations.wrist) < 0.05;

        if (isAtTarget) {
            if (arm.state === 'picking_down') {
                if (arm.grabbedObject) {
                    arm.gripper.add(arm.grabbedObject);
                    arm.grabbedObject.position.set(0, 0.35, 0);
                }
                arm.state = 'picking_up';
                arm.targetRotations = { lowerArm: -Math.PI/6, elbow: Math.PI/1.5, wrist: -Math.PI/4 };
            } else if (arm.state === 'picking_up') {
                arm.state = 'holding';
                addLog('Object picked up', 'success');
            } else if (arm.state === 'dropping_down') {
                if (arm.grabbedObject) {
                    const worldPos = new THREE.Vector3();
                    arm.grabbedObject.getWorldPosition(worldPos);
                    sim.current.environmentGroup!.add(arm.grabbedObject);
                    arm.grabbedObject.position.copy(worldPos);
                    // Drop to ground level
                    arm.grabbedObject.position.y = 0.25;
                    arm.grabbedObject = null;
                }
                arm.state = 'dropping_up';
                arm.targetRotations = { lowerArm: Math.PI/4, elbow: Math.PI/2.5, wrist: -Math.PI/6 };
            } else if (arm.state === 'dropping_up') {
                arm.state = 'idle';
                addLog('Object dropped', 'success');
            }
        }
    }, [hasArm, addLog]);

    const animate = useCallback(() => {
        sim.current.animationFrameId = requestAnimationFrame(animate);

        updateRobotMovement();
        updatePhysics();
        updateOnboardCamera();
        updateArm();

        if (sim.current.renderer && sim.current.scene && sim.current.camera) {
            sim.current.renderer.render(sim.current.scene, sim.current.camera);
        }
    }, [updateRobotMovement, updatePhysics, updateOnboardCamera, updateArm]);

    useEffect(() => {
        sim.current.animationFrameId = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(sim.current.animationFrameId);
    }, [animate]);

    const toggleRecording = () => {
        if (!sim.current.isRecording) {
            sim.current.isRecording = true;
            sim.current.currentEpisode = [];
            setIsRecording(true);
            addLog('Started recording episode...', 'success');

            sim.current.recordingIntervalId = setInterval(recordFrame, 100);
        } else {
            sim.current.isRecording = false;
            setIsRecording(false);
            if (sim.current.recordingIntervalId) clearInterval(sim.current.recordingIntervalId);

            if (sim.current.currentEpisode.length > 0) {
                sim.current.episodes.push([...sim.current.currentEpisode]);
                setEpisodesCount(sim.current.episodes.length);
                addLog(`Episode saved. Total episodes: ${sim.current.episodes.length}`, 'success');
            }
        }
    };

    const resetRobot = () => {
        sim.current.robotState.x = 0;
        sim.current.robotState.z = 0;
        sim.current.robotState.rotation = Math.PI;
        sim.current.robotState.velocity = 0;
        sim.current.robotState.angularVelocity = 0;
        setActiveKeys({});
        setActionChunks([]);

        if (sim.current.robot) {
            sim.current.robot.position.set(0, 0, 0);
            sim.current.robot.rotation.y = Math.PI;
        }

        if (sim.current.target) {
            sim.current.target.position.set(0, 0.25, -5);
        }

        addLog('Robot and target reset', 'warning');
    };

    const saveDataset = () => {
        if (trainingMode === 'cloud') {
            saveCloudDataset();
            return;
        }

        if (sim.current.episodes.length === 0) return;

        const dataset = {
            metadata: {
                robot_type: "diff_drive",
                action_space: "discrete_5",
                fps: 10,
                created: new Date().toISOString()
            },
            episodes: sim.current.episodes
        };

        const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `lerobot_act_dataset_${Date.now()}.json`;
        a.click();

        addLog('Dataset downloaded successfully', 'success');
    };

    const clearDataset = () => {
        sim.current.episodes = [];
        setEpisodesCount(0);
        setFrameCount(0);
        setActionCount(0);
        addLog('Dataset cleared.', 'info');
    };

    const handleImportDataset = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const content = e.target?.result as string;
                const dataset = JSON.parse(content);

                if (dataset.episodes && Array.isArray(dataset.episodes)) {
                    sim.current.episodes = dataset.episodes;
                    setEpisodesCount(dataset.episodes.length);

                    // Calculate total frames and actions
                    let totalFrames = 0;
                    dataset.episodes.forEach((ep: any[]) => totalFrames += ep.length);
                    setFrameCount(totalFrames);
                    setActionCount(totalFrames); // Assuming 1 action per frame

                    addLog(`Dataset imported: ${dataset.episodes.length} episodes`, 'success');
                } else {
                    addLog('Invalid dataset format', 'error');
                }
            } catch (err) {
                console.error(err);
                addLog('Failed to parse dataset file', 'error');
            }
        };
        reader.readAsText(file);
    };

    const loadTrainedModels = useCallback(async () => {
        try {
            const models = await tf.io.listModels();
            const loadedModels = Object.keys(models)
                .filter(key => key.startsWith('indexeddb://'))
                .map(key => ({ name: key.replace('indexeddb://', '') }));
            
            setTrainedModels([
                { name: '跟随网球示例' },
                { name: '自动避障示例' },
                ...loadedModels
            ]);
        } catch (err) {
            console.error('Failed to load models from IndexedDB', err);
        }
    }, []);

    const handleExportModels = async () => {
        const modelsToExport = trainedModels.filter(m => !m.name.endsWith('示例'));
        if (modelsToExport.length === 0) {
            addLog('No user-trained models to export.', 'warning');
            return;
        }
        addLog('Preparing models for export...', 'info');
        try {
            const JSZip = (await import('jszip')).default;
            const zip = new JSZip();
            
            for (const modelInfo of modelsToExport) {
                const modelName = modelInfo.name;
                const model = await tf.loadLayersModel('indexeddb://' + modelName);
                
                await model.save(tf.io.withSaveHandler(async (artifacts) => {
                    const folder = zip.folder(modelName);
                    if (folder) {
                        folder.file(`${modelName}.json`, JSON.stringify({
                            modelTopology: artifacts.modelTopology,
                            format: artifacts.format,
                            generatedBy: artifacts.generatedBy,
                            convertedBy: artifacts.convertedBy,
                            weightsManifest: [{
                                paths: [`${modelName}.weights.bin`],
                                weights: artifacts.weightSpecs
                            }]
                        }));
                        if (artifacts.weightData) {
                            let data: ArrayBuffer;
                            if (Array.isArray(artifacts.weightData)) {
                                const totalLength = artifacts.weightData.reduce((acc, val) => acc + val.byteLength, 0);
                                const tmp = new Uint8Array(totalLength);
                                let offset = 0;
                                for (const buf of artifacts.weightData) {
                                    tmp.set(new Uint8Array(buf), offset);
                                    offset += buf.byteLength;
                                }
                                data = tmp.buffer;
                            } else {
                                data = artifacts.weightData;
                            }
                            folder.file(`${modelName}.weights.bin`, data);
                        }
                    }
                    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
                }));
            }
            
            const blob = await zip.generateAsync({ type: 'blob' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `lerobot_models_${Date.now()}.zip`;
            a.click();
            URL.revokeObjectURL(url);
            addLog('Models exported successfully.', 'success');
        } catch (err) {
            console.error('Export failed:', err);
            addLog('Failed to export models.', 'error');
        }
    };

    const handleImportModelsFolder = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        addLog('Importing models...', 'info');
        const modelFiles: Record<string, { json?: File, bin?: File }> = {};

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const parts = file.webkitRelativePath.split('/');
            if (parts.length < 2) continue;

            const fileName = file.name;
            const modelName = parts[parts.length - 2];

            if (!modelFiles[modelName]) modelFiles[modelName] = {};

            if (fileName.endsWith('.json')) {
                modelFiles[modelName].json = file;
            } else if (fileName.endsWith('.bin')) {
                modelFiles[modelName].bin = file;
            }
        }

        let importCount = 0;
        for (const [modelName, pair] of Object.entries(modelFiles)) {
            if (pair.json && pair.bin) {
                try {
                    const model = await tf.loadLayersModel(tf.io.browserFiles([pair.json, pair.bin]));
                    await model.save('indexeddb://' + modelName);
                    importCount++;
                } catch (err) {
                    console.error(`Failed to import model ${modelName}:`, err);
                    addLog(`Failed to import ${modelName}`, 'error');
                }
            }
        }

        if (importCount > 0) {
            addLog(`Successfully imported ${importCount} models.`, 'success');
            loadTrainedModels();
        } else {
            addLog('No valid models found in the selected folder.', 'warning');
        }
        
        e.target.value = '';
    };

    const finishTraining = useCallback(async () => {
        sim.current.isTraining = false;
        setIsTraining(false);
        const newModelName = `ACT_Model_v${trainedModels.length + 1}`;
        
        if (sim.current.model) {
            try {
                await sim.current.model.save(`indexeddb://${newModelName}`);
            } catch (err) {
                console.error('Failed to save model to IndexedDB', err);
            }
        }

        const newModel = { name: newModelName };
        setTrainedModels(prev => [...prev, newModel]);
        setTrainedModel(newModel);
        setSelectedModel(newModelName);

        addLog('Training complete! Model ready for inference.', 'success');
        addLog(`Model: ${newModelName} with CVAE prior, Chunk size: 8`, 'info');
    }, [addLog, trainedModels.length]);

    // Cloud Functions
    const fetchCloudModels = useCallback(async () => {
        const models = await cloudService.fetchModels();
        setCloudModels(models);
        if (models.length > 0 && !selectedCloudModel) {
            setSelectedCloudModel(models[0].id);
        }
    }, [selectedCloudModel]);

    const fetchCloudDatasets = useCallback(async () => {
        const datasets = await cloudService.fetchDatasets();
        setCloudDatasets(datasets);
        if (datasets.length > 0 && !selectedCloudDataset) {
            setSelectedCloudDataset(datasets[0].path);
        }
    }, [selectedCloudDataset]);

    useEffect(() => {
        if (trainingMode === 'cloud') {
            fetchCloudModels();
            fetchCloudDatasets();
        }
    }, [trainingMode, fetchCloudModels, fetchCloudDatasets]);

    const CHUNK_SIZE = 10;
    const ACTION_DIM = 2;

    const packDataset = (episodes: any[]) => {
        const states: number[][] = [];
        const env_states: number[][] = [];
        const actions: number[][][] = [];
        const action_is_pad: number[][] = [];
        const images: string[][][] = [];
        let hasImages = false;

        episodes.forEach(ep => {
            for (let i = 0; i < ep.length; i += CHUNK_SIZE) {
                const chunkSteps = ep.slice(i, i + CHUNK_SIZE);
                const firstStep = chunkSteps[0];

                const stateChunk = firstStep.state || new Array(14).fill(0);
                const envChunk = firstStep.envState || new Array(10).fill(0);

                const actionChunk: number[][] = [];
                const padChunk: number[] = [];
                const imageChunk: string[][] = [];

                chunkSteps.forEach((step: any) => {
                    actionChunk.push(step.action);
                    padChunk.push(0);
                    const image = step.imageBase64 || "";
                    if (image) hasImages = true;
                    imageChunk.push([image]);
                });

                for (let pad = chunkSteps.length; pad < CHUNK_SIZE; pad += 1) {
                    actionChunk.push(new Array(ACTION_DIM).fill(0));
                    padChunk.push(1);
                    imageChunk.push([""]);
                }

                states.push(stateChunk);
                env_states.push(envChunk);
                actions.push(actionChunk);
                action_is_pad.push(padChunk);
                images.push(imageChunk);
            }
        });

        if (hasImages) {
            return { states, env_states, actions, action_is_pad, images };
        }
        return { states, env_states, actions, action_is_pad };
    };

    const saveCloudDataset = async () => {
        if (sim.current.episodes.length === 0) return;
        addLog('Uploading dataset to cloud...', 'info');

        const dataset = packDataset(sim.current.episodes);
        const success = await cloudService.saveDataset(dataset);

        if (success) {
            addLog('Dataset uploaded successfully!', 'success');
            fetchCloudDatasets();
        } else {
            addLog('Failed to upload dataset.', 'error');
        }
    };

    const startCloudTraining = async () => {
        if (!selectedCloudDataset) {
            addLog('Please select a dataset for cloud training.', 'error');
            return;
        }

        setIsTraining(true);
        setTrainingStatus('Initiating Cloud Training...');

        const success = await cloudService.startTraining(selectedCloudDataset);

        if (success) {
            addLog('Cloud training started.', 'success');
            // Poll for status
            const interval = setInterval(async () => {
                const status = await cloudService.getTrainingStatus();
                setCloudTrainingStatus(status);

                if (status) {
                    if (status.num_epochs > 0) {
                        setTrainingProgress((status.epoch / status.num_epochs) * 100);
                        setTrainingStatus(`Cloud Training: Epoch ${status.epoch}/${status.num_epochs} - Loss: ${status.avg_loss?.toFixed(4) ?? 'N/A'}`);
                    }

                    if (status.status === 'completed' || status.status === 'failed') {
                        clearInterval(interval);
                        setIsTraining(false);
                        if (status.status === 'completed') {
                            addLog('Cloud training completed!', 'success');
                            fetchCloudModels(); // Refresh models list
                        } else {
                            addLog(`Cloud training failed: ${status.error || 'Unknown error'}`, 'error');
                        }
                    }
                }
            }, 2000);
        } else {
            setIsTraining(false);
            addLog('Failed to start cloud training.', 'error');
        }
    };

    const runCloudInference = useCallback(async () => {
        if (!sim.current.isInferencing) return;

        const x = sim.current.robotState.x;
        const z = sim.current.robotState.z;
        const angle = sim.current.robotState.rotation;
        const speed = sim.current.robotState.velocity;
        const targetDist = sim.current.target ? Math.hypot(sim.current.target.position.x - x, sim.current.target.position.z - z) : 0;

        const state = new Array(14).fill(0);
        state[0] = x;
        state[1] = z;
        state[2] = angle;
        state[3] = speed;
        state[4] = targetDist;
        state[5] = sim.current.isColliding ? 1.0 : 0.0;

        const envState = new Array(10).fill(0);
        envState[0] = x;
        envState[1] = z;
        envState[2] = angle;
        envState[3] = speed;
        envState[4] = sim.current.isColliding ? 1 : 0;
        envState[5] = targetDist;
        if (sim.current.target) {
            const relAngleToBall = Math.atan2(
                sim.current.target.position.x - x,
                sim.current.target.position.z - z
            ) - angle;
            envState[6] = Math.sin(relAngleToBall);
            envState[7] = Math.cos(relAngleToBall);
            envState[8] = sim.current.target.position.x - x;
            envState[9] = sim.current.target.position.z - z;
        }

        const action = await cloudService.runInferenceStep(state, envState);

        if (!action) {
            return;
        }

        // Handle continuous action vector [velocity, angularVelocity]
        let v = 0;
        let w = 0;
        if (Array.isArray(action)) {
            v = action[0] || 0;
            w = action[1] || 0;
        } else if (typeof action === 'string') {
            // Legacy discrete command fallback
            const moveSpeed = 0.1;
            const turnSpeedVal = 0.05;
            switch (action) {
                case 'up': v = moveSpeed; break;
                case 'down': v = -moveSpeed; break;
                case 'left': w = turnSpeedVal; break;
                case 'right': w = -turnSpeedVal; break;
                case 'stop': v = 0; w = 0; break;
            }
        }

        // Clamp
        v = Math.max(-0.15, Math.min(0.15, v));
        w = Math.max(-0.08, Math.min(0.08, w));

        sim.current.robotState.velocity = v;
        sim.current.robotState.angularVelocity = w;

        sim.current.inferenceTimeoutId = setTimeout(runCloudInference, 200);
    }, [captureImage]);

    const startCloudInference = async () => {
        if (!selectedCloudModel) {
            addLog('Please select a cloud model.', 'error');
            return;
        }

        const success = await cloudService.startInference(selectedCloudModel);

        if (success) {
            sim.current.isInferencing = true;
            setIsInferencing(true);
            addLog('Cloud inference started.', 'success');
            runCloudInference();
        } else {
            addLog('Failed to start cloud inference.', 'error');
        }
    };

    const stopInference = useCallback(async () => {
        sim.current.isInferencing = false;
        setIsInferencing(false);
        clearTimeout(sim.current.inferenceTimeoutId);
        setActiveKeys({});
        setActionChunks([]);

        if (trainingMode === 'cloud') {
            const success = await cloudService.stopInference();
            if (success) {
                addLog('Cloud inference stopped.', 'warning');
            }
        } else {
            addLog('Inference stopped.', 'warning');
        }
    }, [trainingMode, addLog]);

    const startTraining = async () => {
        if (trainingMode === 'cloud') {
            startCloudTraining();
            return;
        }

        if (sim.current.episodes.length === 0) return;

        sim.current.isTraining = true;
        setIsTraining(true);
        setTrainingProgress(0);
        setTrainingStatus('Preparing data with Action Chunking...');

        addLog(`Initializing ACT training (Chunk Size: ${actService.CHUNK_SIZE})...`, 'info');

        // Prepare Data with Action Chunking
        const data = actService.prepareTrainingData(sim.current.episodes);

        if (data.imageInputs.length === 0) {
            addLog('No visual data found in episodes!', 'error');
            setIsTraining(false);
            sim.current.isTraining = false;
            return;
        }

        // Define ACT Model
        const model = actService.createModel();
        sim.current.model = model;

        // Train
        await actService.trainModel(model, data, (epoch, logs) => {
            const progress = ((epoch + 1) / 50) * 100;
            setTrainingProgress(progress);
            setTrainingStatus(`Epoch ${epoch + 1}/50 - Loss: ${logs?.loss.toFixed(4)}`);
        });

        await finishTraining();
    };

    const runInference = useCallback(() => {
        if (!sim.current.isInferencing || !sim.current.target) return;

        const { robotState, target, model } = sim.current;

        if (selectedModel === '跟随网球示例' || selectedModel === '自动避障示例') {
            let targetSpeed = 0;
            let targetTurn = 0;

            if (selectedModel === '跟随网球示例') {
                const dirX = Math.sin(robotState.rotation);
                const dirZ = Math.cos(robotState.rotation);
                
                // Calculate from camera position (0.85 units forward) to match visual FOV
                const camX = robotState.x + dirX * 0.85;
                const camZ = robotState.z + dirZ * 0.85;
                
                const dx = target.position.x - camX;
                const dz = target.position.z - camZ;
                const distance = Math.sqrt(dx * dx + dz * dz);
                
                const targetAngle = Math.atan2(dx, dz);
                let angleDiff = targetAngle - robotState.rotation;
                
                while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
                while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

                // Camera horizontal FOV is ~96 degrees (48 left/right). Use 45 degrees for strict visibility.
                const FOV = Math.PI / 4; 
                let isVisible = Math.abs(angleDiff) <= FOV;

                // Check for obstacles blocking the view
                if (isVisible && distance > 1.0) {
                    const steps = Math.floor(distance / 0.5);
                    const stepX = dx / steps;
                    const stepZ = dz / steps;
                    let px = camX;
                    let pz = camZ;
                    
                    for (let i = 1; i < steps; i++) {
                        px += stepX;
                        pz += stepZ;
                        for (const wall of sim.current.walls) {
                            if (wall === target) continue;
                            const w = wall.userData.w;
                            const depth = wall.userData.d;
                            const wx = wall.position.x;
                            const wz = wall.position.z;
                            if (px > wx - w/2 && px < wx + w/2 && pz > wz - depth/2 && pz < wz + depth/2) {
                                isVisible = false;
                                break;
                            }
                        }
                        if (!isVisible) break;
                    }
                }

                if (distance <= 1.5 && isVisible) {
                    // Reached the ball and it's visible: stop completely, no jittering
                    targetSpeed = 0;
                    targetTurn = 0;
                } else if (!isVisible) {
                    // Ball out of FOV or blocked: rotate in place to search
                    targetSpeed = 0;
                    targetTurn = turnSpeed;
                } else {
                    // Ball in FOV: track and approach
                    // Use a smaller multiplier for smoother turning
                    targetTurn = angleDiff * 1.0;
                    
                    // Cap the turn speed to prevent violent swings
                    targetTurn = Math.max(-turnSpeed * 1.5, Math.min(turnSpeed * 1.5, targetTurn));
                    
                    // Smooth speed approach
                    targetSpeed = Math.min(speed, (distance - 1.5) * 0.5);
                    
                    // If the angle is too large, slow down to turn
                    if (Math.abs(angleDiff) > Math.PI / 6) {
                        targetSpeed *= 0.5;
                    }
                }
            } else if (selectedModel === '自动避障示例') {
                const checkDistance = (angleOffset: number) => {
                    const checkAngle = robotState.rotation + angleOffset;
                    const dirX = Math.sin(checkAngle);
                    const dirZ = Math.cos(checkAngle);
                    
                    let minDist = 5; // max lookahead
                    const robotRadius = 1.0; // slightly larger for safety

                    for (const wall of sim.current.walls) {
                        for (let d = 0.5; d < minDist; d += 0.5) {
                            const px = robotState.x + dirX * d;
                            const pz = robotState.z + dirZ * d;
                            
                            const w = wall.userData.w;
                            const depth = wall.userData.d;
                            const wx = wall.position.x;
                            const wz = wall.position.z;
                            
                            if (px > wx - w/2 - robotRadius && px < wx + w/2 + robotRadius &&
                                pz > wz - depth/2 - robotRadius && pz < wz + depth/2 + robotRadius) {
                                minDist = d;
                                break;
                            }
                        }
                    }
                    return minDist;
                };

                const distFront = checkDistance(0);
                const distLeft = checkDistance(Math.PI / 4);
                const distRight = checkDistance(-Math.PI / 4);

                targetSpeed = speed * 0.8;
                targetTurn = 0;

                if (distFront < 3) {
                    targetSpeed = 0; 
                    if (distLeft > distRight) {
                        targetTurn = turnSpeed;
                    } else {
                        targetTurn = -turnSpeed;
                    }
                    if (distLeft < 2 && distRight < 2) {
                        targetSpeed = -speed * 0.5;
                        targetTurn = turnSpeed;
                    }
                } else {
                    if (distLeft < 3) targetTurn = -turnSpeed * 0.5;
                    if (distRight < 3) targetTurn = turnSpeed * 0.5;
                }
            }

            targetSpeed = Math.max(-speed, Math.min(speed, targetSpeed));
            targetTurn = Math.max(-turnSpeed, Math.min(turnSpeed, targetTurn));

            robotState.velocity = robotState.velocity * 0.5 + targetSpeed * 0.5;
            robotState.angularVelocity = robotState.angularVelocity * 0.5 + targetTurn * 0.5;

            const newActiveKeys: Record<string, boolean> = {};
            if (robotState.velocity > 0.02) newActiveKeys['w'] = true;
            if (robotState.velocity < -0.02) newActiveKeys['s'] = true;
            if (robotState.angularVelocity > 0.01) newActiveKeys['a'] = true;
            if (robotState.angularVelocity < -0.01) newActiveKeys['d'] = true;
            
            setActiveKeys(prev => {
                const keysChanged = Object.keys(newActiveKeys).length !== Object.keys(prev).length || 
                                   Object.keys(newActiveKeys).some(k => newActiveKeys[k] !== prev[k]);
                return keysChanged ? newActiveKeys : prev;
            });

            sim.current.inferenceTimeoutId = setTimeout(runInference, 100);
            return;
        }

        if (!model) return;

        // Stuck detection logic
        if (typeof sim.current.lastX === 'undefined') {
            sim.current.lastX = robotState.x;
            sim.current.lastZ = robotState.z;
            sim.current.stuckCounter = 0;
        }

        const distMoved = Math.sqrt(
            Math.pow(robotState.x - sim.current.lastX, 2) +
            Math.pow(robotState.z - sim.current.lastZ, 2)
        );
        sim.current.lastX = robotState.x;
        sim.current.lastZ = robotState.z;

        if (Math.abs(robotState.velocity) > 0.05 && distMoved < 0.005) {
            sim.current.stuckCounter++;
        } else {
            sim.current.stuckCounter = Math.max(0, sim.current.stuckCounter - 1);
        }

        if (sim.current.stuckCounter > 20) {
            addLog('Stuck detected! Recovering...', 'warning');
            robotState.velocity = -0.2;
            robotState.angularVelocity = (Math.random() - 0.5) * 2.0; // Increased rotation range
            sim.current.stuckCounter = 0;
            // Increase recovery time to 1.5s to allow robot to back away fully
            sim.current.inferenceTimeoutId = setTimeout(runInference, 1500);
            return;
        }

        const image = captureImage();
        if (!image) return;

        const dx = target.position.x - robotState.x;
        const dz = target.position.z - robotState.z;
        const targetDist = Math.sqrt(dx * dx + dz * dz);

        const state = new Array(14).fill(0);
        state[0] = robotState.x;
        state[1] = robotState.z;
        state[2] = robotState.rotation;
        state[3] = robotState.velocity;
        state[4] = targetDist;

        const prediction = actService.predict(model, image, state);

        // Temporal Ensembling (simplified)
        if (!sim.current.actionBuffer) sim.current.actionBuffer = [];

        // Parse prediction into chunk
        const newChunk = [];
        for (let i = 0; i < actService.CHUNK_SIZE; i++) {
            newChunk.push({
                v: prediction[i * 2],
                w: prediction[i * 2 + 1],
                weight: Math.exp(-0.5 * i) // Exponential weighting
            });
        }
        setActionChunks(newChunk.map(a => a.v));
        sim.current.actionBuffer.push(newChunk);

        // Keep buffer size limited
        if (sim.current.actionBuffer.length > actService.CHUNK_SIZE) {
            sim.current.actionBuffer.shift();
        }

        // Aggregate actions
        let sumV = 0, sumW = 0, totalWeight = 0;

        sim.current.actionBuffer.forEach((chunk: any[], index: number) => {
            const offset = sim.current.actionBuffer.length - 1 - index;
            if (offset < chunk.length) {
                const action = chunk[offset];
                sumV += action.v * action.weight;
                sumW += action.w * action.weight;
                totalWeight += action.weight;
            }
        });

        if (totalWeight > 0) {
            // Apply slight smoothing to velocity to prevent sudden jumps
            const targetV = Math.max(-speed, Math.min(speed, sumV / totalWeight));
            const targetW = Math.max(-turnSpeed, Math.min(turnSpeed, sumW / totalWeight));

            // Simple low-pass filter (alpha = 0.5)
            robotState.velocity = robotState.velocity * 0.5 + targetV * 0.5;
            robotState.angularVelocity = robotState.angularVelocity * 0.5 + targetW * 0.5;

            // Log movement periodically (every 200ms)
            if (!sim.current.lastInferenceLogTime || Date.now() - sim.current.lastInferenceLogTime > 200) {
                addLog(`Inference Movement: V=${robotState.velocity.toFixed(2)}, W=${robotState.angularVelocity.toFixed(2)}`, 'info');
                sim.current.lastInferenceLogTime = Date.now();
            }
        } else {
            robotState.velocity = Math.max(-speed, Math.min(speed, prediction[0]));
            robotState.angularVelocity = Math.max(-turnSpeed, Math.min(turnSpeed, prediction[1]));
        }

        // Update active keys for UI highlighting during inference
        const newActiveKeys: Record<string, boolean> = {};
        if (robotState.velocity > 0.02) newActiveKeys['w'] = true;
        if (robotState.velocity < -0.02) newActiveKeys['s'] = true;
        if (robotState.angularVelocity > 0.01) newActiveKeys['a'] = true;
        if (robotState.angularVelocity < -0.01) newActiveKeys['d'] = true;
        
        setActiveKeys(prev => {
            const keysChanged = Object.keys(newActiveKeys).length !== Object.keys(prev).length || 
                               Object.keys(newActiveKeys).some(k => newActiveKeys[k] !== prev[k]);
            return keysChanged ? newActiveKeys : prev;
        });

        // Check goal
        if (targetDist < 1.0) {
            addLog('Target reached!', 'success');
            target.position.set(
                (Math.random() - 0.5) * 12,
                0.4,
                (Math.random() - 0.5) * 12
            );
            sim.current.actionBuffer = [];
        }

        // Match training frequency (10Hz = 100ms)
        sim.current.inferenceTimeoutId = setTimeout(runInference, 100);
    }, [addLog, captureImage, speed, turnSpeed, selectedModel]);

    const startInference = async () => {
        if (isInferencing) {
            stopInference();
            return;
        }

        if (trainingMode === 'cloud') {
            startCloudInference();
        } else {
            if (!selectedModel) {
                addLog('No frontend model selected!', 'error');
                return;
            }

            try {
                if (selectedModel === '跟随网球示例' || selectedModel === '自动避障示例') {
                    addLog(`Loading ${selectedModel}...`, 'info');
                    sim.current.isInferencing = true;
                    setIsInferencing(true);
                    addLog(`Starting Frontend ACT inference with ${selectedModel}...`, 'success');
                    runInference();
                } else {
                    addLog(`Loading model ${selectedModel}...`, 'info');
                    sim.current.model = await tf.loadLayersModel(`indexeddb://${selectedModel}`);
                    sim.current.isInferencing = true;
                    setIsInferencing(true);
                    addLog(`Starting Frontend ACT inference with ${selectedModel}...`, 'success');
                    runInference();
                }
            } catch (err) {
                console.error(err);
                addLog(`Failed to load model ${selectedModel}`, 'error');
            }
        }
    };

    const [showManual, setShowManual] = useState(false);
    const [manualPage, setManualPage] = useState(1);

    return (
        <div className="h-screen flex flex-col grid-bg text-[#e0e0e0] font-sans overflow-x-hidden bg-[#0a0a0f]">
            {showManual && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={() => setShowManual(false)}>
                    <div className="bg-slate-900 border border-slate-700 p-8 rounded-lg max-w-4xl w-full max-h-[85vh] overflow-y-auto relative" onClick={e => e.stopPropagation()}>
                        <button className="absolute top-4 right-4 text-slate-400 hover:text-white text-xl" onClick={() => setShowManual(false)}>×</button>
                        
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-2xl font-bold text-blue-400">
                                {manualPage === 1 ? '训练手册 - 操作指南' : '训练手册 - 技术文档'}
                            </h2>
                            <div className="flex gap-2">
                                <button 
                                    className={`px-3 py-1 rounded text-xs ${manualPage === 1 ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'}`}
                                    onClick={() => setManualPage(1)}
                                >
                                    操作指南
                                </button>
                                <button 
                                    className={`px-3 py-1 rounded text-xs ${manualPage === 2 ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'}`}
                                    onClick={() => setManualPage(2)}
                                >
                                    技术架构
                                </button>
                            </div>
                        </div>

                        <div className="text-sm text-slate-300 space-y-6">
                            {manualPage === 1 ? (
                                <>
                                    <section className="space-y-3">
                                        <h3 className="font-bold text-slate-100 text-lg border-b border-slate-800 pb-2">1. 快速开始</h3>
                                        <p>本模拟器允许您通过“动作分块（ACT）”技术训练小车自动导航。基本流程如下：</p>
                                        <ul className="list-disc pl-5 space-y-2">
                                            <li><strong>手动控制</strong>：使用键盘 <kbd className="bg-slate-800 px-1 rounded">WASD</kbd> 或方向键控制小车移动。</li>
                                            <li><strong>数据采集</strong>：点击“开始采集”，在手动控制的同时记录小车的行为。</li>
                                            <li><strong>模型训练</strong>：采集足够数据后（建议至少 10 个 Episode），点击“开始训练”。</li>
                                            <li><strong>自主推理</strong>：训练完成后，在下拉列表中选择模型，点击“启动自主推理”。</li>
                                        </ul>
                                    </section>
                                    <section className="space-y-3">
                                        <h3 className="font-bold text-slate-100 text-lg border-b border-slate-800 pb-2">2. 训练要求</h3>
                                        <ul className="list-disc pl-5 space-y-2">
                                            <li><strong>多样性</strong>：从不同位置开始，以不同角度接近目标。</li>
                                            <li><strong>平滑性</strong>：手动操作时尽量保持平滑，避免剧烈的无意义转向。</li>
                                            <li><strong>数据量</strong>：建议每个场景采集 15-20 个成功的 Episode。</li>
                                        </ul>
                                    </section>
                                    <section className="space-y-3">
                                        <h3 className="font-bold text-slate-100 text-lg border-b border-slate-800 pb-2">3. 训练技巧</h3>
                                        <ul className="list-disc pl-5 space-y-2">
                                            <li><strong>碰撞保护</strong>：开启碰撞保护可以自动过滤掉撞墙的无效数据。</li>
                                            <li><strong>速度设置</strong>：训练时的速度设置会影响模型的学习。建议先用低速采集，模型稳定后再尝试高速。</li>
                                            <li><strong>目标距离</strong>：小车距离目标越近，采集到的“冲刺”数据越精准。</li>
                                        </ul>
                                    </section>
                                </>
                            ) : (
                                <>
                                    <section className="space-y-3">
                                        <h3 className="font-bold text-slate-100 text-lg border-b border-slate-800 pb-2">1. 系统架构</h3>
                                        <p>本系统基于 React 和 Three.js 构建，采用前端本地训练与推理架构。核心组件包括：</p>
                                        <ul className="list-disc pl-5 space-y-2">
                                            <li><strong>Three.js 场景渲染</strong>：负责小车、环境、光源和机载摄像头的 3D 渲染。</li>
                                            <li><strong>TensorFlow.js 模型引擎</strong>：在浏览器端执行 ACT (Action Chunking with Transformers) 模型的训练与推理。</li>
                                            <li><strong>状态管理</strong>：使用 React Hooks 和 `useRef` 管理仿真状态（机器人位置、速度、动作缓冲区）。</li>
                                        </ul>
                                    </section>
                                    <section className="space-y-3">
                                        <h3 className="font-bold text-slate-100 text-lg border-b border-slate-800 pb-2">2. ACT 原理与实现</h3>
                                        <p>ACT (Action Chunking with Transformers) 旨在解决机器人动作预测中的平滑性与多模态问题。</p>
                                        <ul className="list-disc pl-5 space-y-2">
                                            <li><strong>动作分块 (Action Chunking)</strong>：模型一次预测未来多个时间步的动作序列，而非单一动作，从而显著降低动作抖动。</li>
                                            <li><strong>Transformer 架构</strong>：利用自注意力机制捕捉动作序列的时序依赖。</li>
                                            <li><strong>CVAE (条件变分自编码器)</strong>：在训练时建模动作的多模态分布，推理时通过采样实现动作的多样性。</li>
                                        </ul>
                                    </section>
                                    <section className="space-y-3">
                                        <h3 className="font-bold text-slate-100 text-lg border-b border-slate-800 pb-2">3. 训练与推理细节</h3>
                                        <p>整个生命周期分为以下阶段：</p>
                                        <ul className="list-disc pl-5 space-y-2">
                                            <li><strong>数据预处理</strong>：将图像缩放至 224x224，并对状态向量进行归一化。</li>
                                            <li><strong>损失函数</strong>：结合 L1 动作损失和 KL 散度（用于 CVAE 潜空间正则化）。</li>
                                            <li><strong>时间集成 (Temporal Ensembling)</strong>：在推理时，对重叠的动作块进行加权平均，进一步提升平滑度。</li>
                                        </ul>
                                    </section>
                                </>
                            )}
                        </div>
                        
                        <div className="mt-8 flex justify-between items-center border-t border-slate-800 pt-6">
                            <span className="text-xs text-slate-500">页码: {manualPage} / 2</span>
                            <div className="flex gap-4">
                                {manualPage === 1 ? (
                                    <button className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded transition-colors" onClick={() => setManualPage(2)}>下一页: 技术架构</button>
                                ) : (
                                    <button className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-6 py-2 rounded transition-colors" onClick={() => setManualPage(1)}>上一页: 操作指南</button>
                                )}
                                <button className="bg-slate-700 hover:bg-slate-600 text-white px-6 py-2 rounded transition-colors" onClick={() => setShowManual(false)}>完成</button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
            <header className="glass-panel border-b border-slate-800 px-6 py-4 flex justify-between items-center z-20">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg">🤖</div>
                    <div>
                        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">AKA ACT Simulator</h1>
                        <p className="text-xs text-slate-400 mono">Action Chunking with Transformers - Educational Edition</p>
                    </div>
                </div>
                <div className="flex items-center gap-4 text-sm">
                    <button onClick={() => setShowManual(true)} className="text-blue-400 hover:text-blue-300">训练手册</button>
                    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700">
                        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                        <span className="text-slate-300">Simulation Active</span>
                    </div>
                    <div className="mono text-xs text-slate-500">60 FPS</div>
                </div>
            </header>

            <div className="flex-1 flex overflow-hidden">
                <aside className="w-80 glass-panel border-r border-slate-800 flex flex-col overflow-y-auto">
                    <div className="p-4 space-y-6">
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">1. 配置环境</h3>
                            <div className="space-y-3">
                                <button
                                    className="w-full flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider focus:outline-none"
                                    onClick={(e) => {
                                        const content = e.currentTarget.nextElementSibling;
                                        const chevron = e.currentTarget.querySelector('.chevron');
                                        if (content && chevron) {
                                            content.classList.toggle('hidden');
                                            chevron.textContent = content.classList.contains('hidden') ? '▼' : '▲';
                                        }
                                    }}
                                >
                                    <span>场景设置</span>
                                    <span className="chevron text-[10px]">▼</span>
                                </button>
                                <div className="space-y-2 hidden">
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

                            {/* Obstacle Placement Panel */}
                            <div className="space-y-3">
                                <button
                                    className="w-full flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider focus:outline-none"
                                    onClick={(e) => {
                                        const content = e.currentTarget.nextElementSibling;
                                        const chevron = e.currentTarget.querySelector('.chevron');
                                        if (content && chevron) {
                                            content.classList.toggle('hidden');
                                            chevron.textContent = content.classList.contains('hidden') ? '▼' : '▲';
                                        }
                                    }}
                                >
                                    <span>放置障碍物</span>
                                    <div className="flex items-center gap-1">
                                        {obstacles.length > 0 && <span className="text-[10px] text-slate-500">{obstacles.length}</span>}
                                        <span className="chevron text-[10px]">▼</span>
                                    </div>
                                </button>
                                <div className="space-y-2 hidden">
                                    {placementMode !== 'none' && (
                                        <div className="flex items-center gap-2 px-2 py-1.5 rounded bg-amber-900/30 border border-amber-700/50 text-xs text-amber-300">
                                            <span className="animate-pulse">+</span>
                                            <span>点击地面放置{placementMode === 'box' ? '方块' : '圆柱'}</span>
                                            <button onClick={() => setPlacementMode('none')} className="ml-auto text-amber-400 hover:text-amber-200">取消</button>
                                        </div>
                                    )}
                                    <div className="flex gap-1.5">
                                        <button
                                            onClick={() => setPlacementMode(placementMode === 'box' ? 'none' : 'box')}
                                            className={`flex-1 text-xs py-1.5 rounded border transition-all ${placementMode === 'box' ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_8px_rgba(37,99,235,0.4)]' : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-600'}`}
                                        >
                                            + 方块
                                        </button>
                                        <button
                                            onClick={() => setPlacementMode(placementMode === 'cylinder' ? 'none' : 'cylinder')}
                                            className={`flex-1 text-xs py-1.5 rounded border transition-all ${placementMode === 'cylinder' ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_8px_rgba(37,99,235,0.4)]' : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-600'}`}
                                        >
                                            + 圆柱
                                        </button>
                                    </div>
                                    {selectedObstacleId && (
                                        <div className="flex gap-1.5 px-2 py-1.5 rounded bg-slate-800/50 border border-blue-800/50 text-xs">
                                            <span className="text-blue-300">已选中</span>
                                            <button onClick={() => {
                                                setObstacles(prev => prev.map(o => {
                                                    if (o.id !== selectedObstacleId) return o;
                                                    const newRot = o.rotation + Math.PI / 12;
                                                    updateObstacleInScene(o.id, { rotation: newRot });
                                                    return { ...o, rotation: newRot };
                                                }));
                                            }} className="text-slate-300 hover:text-white bg-slate-700 px-1.5 rounded">旋转 R</button>
                                            <button onClick={() => removeObstacle(selectedObstacleId)} className="text-red-400 hover:text-red-300 bg-slate-700 px-1.5 rounded">删除</button>
                                            <button onClick={() => setSelectedObstacleId(null)} className="text-slate-400 hover:text-white bg-slate-700 px-1.5 rounded ml-auto">取消</button>
                                        </div>
                                    )}
                                    {obstacles.length > 0 && (
                                        <div className="max-h-32 overflow-y-auto space-y-1 text-xs text-slate-400">
                                            {obstacles.map(obs => (
                                                <div
                                                    key={obs.id}
                                                    onClick={() => setSelectedObstacleId(obs.id)}
                                                    className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer transition-all ${selectedObstacleId === obs.id ? 'bg-blue-900/40 border border-blue-700/50' : 'hover:bg-slate-800 border border-transparent'}`}
                                                >
                                                    <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#' + obs.color.toString(16).padStart(6, '0') }}></span>
                                                    <span className="flex-1 truncate">{obs.type === 'box' ? '方块' : '圆柱'}</span>
                                                    <span className="text-slate-500">({obs.x.toFixed(1)}, {obs.z.toFixed(1)})</span>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); removeObstacle(obs.id); }}
                                                        className="text-slate-500 hover:text-red-400 ml-1"
                                                    >x</button>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {obstacles.length > 0 && (
                                        <button onClick={clearAllObstacles} className="w-full text-xs py-1 rounded border border-slate-700 text-slate-400 hover:text-red-400 hover:border-red-700/50 transition-all">
                                            清除全部 ({obstacles.length})
                                        </button>
                                    )}
                                    <div className="text-[10px] text-slate-600 leading-tight">
                                        提示: 拖拽移动 | R 旋转 | Del 删除
                                    </div>
                                </div>
                            </div>

                            {/* Scene Presets Panel */}
                            <div className="space-y-3">
                                <button
                                    className="w-full flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider focus:outline-none"
                                    onClick={(e) => {
                                        const content = e.currentTarget.nextElementSibling;
                                        const chevron = e.currentTarget.querySelector('.chevron');
                                        if (content && chevron) {
                                            content.classList.toggle('hidden');
                                            chevron.textContent = content.classList.contains('hidden') ? '▼' : '▲';
                                        }
                                    }}
                                >
                                    <span>场景存档</span>
                                    <div className="flex items-center gap-1">
                                        {scenePresets.length > 0 && <span className="text-[10px] text-slate-500">{scenePresets.length}</span>}
                                        <span className="chevron text-[10px]">▼</span>
                                    </div>
                                </button>
                                <div className="space-y-2 hidden">
                                    {/* Save current scene */}
                                    {savingPreset ? (
                                        <div className="flex gap-1.5">
                                            <input
                                                id="preset-name-input"
                                                type="text"
                                                placeholder="输入场景名称..."
                                                className="flex-1 bg-slate-800/50 border border-slate-600 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500 placeholder:text-slate-600"
                                                autoFocus
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter') {
                                                        const val = (e.target as HTMLInputElement).value.trim();
                                                        if (val) saveScenePreset(val);
                                                    }
                                                    if (e.key === 'Escape') setSavingPreset(false);
                                                }}
                                            />
                                            <button onClick={() => { const el = document.getElementById('preset-name-input') as HTMLInputElement | null; if (el && el.value.trim()) saveScenePreset(el.value.trim()); }} className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-2 py-1.5 rounded transition-all">保存</button>
                                            <button onClick={() => setSavingPreset(false)} className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 px-2 py-1.5 rounded transition-all">取消</button>
                                        </div>
                                    ) : (
                                        <div className="flex gap-1.5">
                                            <button onClick={() => setSavingPreset(true)} className="flex-1 text-xs py-1.5 rounded border border-green-700/50 text-green-400 hover:bg-green-900/30 transition-all">
                                                + 保存当前场景
                                            </button>
                                            <button onClick={importScenePreset} className="text-xs py-1.5 px-2 rounded border border-slate-600 text-slate-400 hover:bg-slate-800 transition-all" title="导入场景文件">
                                                导入
                                            </button>
                                        </div>
                                    )}

                                    {/* Preset list */}
                                    {scenePresets.length > 0 && (
                                        <div className="max-h-48 overflow-y-auto space-y-1 text-xs">
                                            {scenePresets.map(preset => (
                                                <div
                                                    key={preset.id}
                                                    className="flex items-center gap-2 px-2 py-1.5 rounded bg-slate-800/30 border border-slate-800 hover:border-slate-600 transition-all group"
                                                >
                                                    <div className="flex-1 min-w-0">
                                                        <div className="text-slate-300 truncate">{preset.name}</div>
                                                        <div className="text-[10px] text-slate-600">
                                                            {preset.sceneType} · {preset.obstacles.length}个障碍物 · {preset.createdAt}
                                                        </div>
                                                    </div>
                                                    <button
                                                        onClick={() => loadScenePreset(preset)}
                                                        className="text-blue-400 hover:text-blue-300 opacity-60 hover:opacity-100 transition-all"
                                                        title="加载场景"
                                                    >加载</button>
                                                    <button
                                                        onClick={() => exportScenePreset(preset)}
                                                        className="text-slate-500 hover:text-slate-300 opacity-60 hover:opacity-100 transition-all"
                                                        title="导出为JSON"
                                                    >导出</button>
                                                    <button
                                                        onClick={() => deleteScenePreset(preset.id)}
                                                        className="text-slate-600 hover:text-red-400 opacity-40 hover:opacity-100 transition-all"
                                                        title="删除"
                                                    >x</button>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {scenePresets.length === 0 && (
                                        <div className="text-[10px] text-slate-600 text-center py-2">
                                            暂无存档，配置好场景后点击保存
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="space-y-3">
                                <button
                                    className="w-full flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider focus:outline-none"
                                    onClick={(e) => {
                                        const content = e.currentTarget.nextElementSibling;
                                        const chevron = e.currentTarget.querySelector('.chevron');
                                        if (content && chevron) {
                                            content.classList.toggle('hidden');
                                            chevron.textContent = content.classList.contains('hidden') ? '▼' : '▲';
                                        }
                                    }}
                                >
                                    <span>光源设置</span>
                                    <span className="chevron text-[10px]">▼</span>
                                </button>
                                <div className="space-y-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded border border-slate-800 hidden">
                                    <label className="flex items-center gap-2">
                                        <span className="w-4">X:</span> <input type="range" min="-30" max="30" value={lightPos.x} onChange={e => setLightPos({ ...lightPos, x: Number(e.target.value) })} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                    </label>
                                    <label className="flex items-center gap-2">
                                        <span className="w-4">Y:</span> <input type="range" min="5" max="40" value={lightPos.y} onChange={e => setLightPos({ ...lightPos, y: Number(e.target.value) })} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                    </label>
                                    <label className="flex items-center gap-2">
                                        <span className="w-4">Z:</span> <input type="range" min="-30" max="30" value={lightPos.z} onChange={e => setLightPos({ ...lightPos, z: Number(e.target.value) })} className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                    </label>
                                </div>
                            </div>
                            <div className="space-y-3">
                                <button
                                    className="w-full flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider focus:outline-none"
                                    onClick={(e) => {
                                        const content = e.currentTarget.nextElementSibling;
                                        const chevron = e.currentTarget.querySelector('.chevron');
                                        if (content && chevron) {
                                            content.classList.toggle('hidden');
                                            chevron.textContent = content.classList.contains('hidden') ? '▼' : '▲';
                                        }
                                    }}
                                >
                                    <span>小车配置</span>
                                    <span className="chevron text-[10px]">▼</span>
                                </button>
                                <div className="space-y-2 text-xs text-slate-400 bg-slate-900/50 p-2 rounded border border-slate-800 hidden">
                                    <label className="flex items-center gap-2 cursor-pointer">
                                        <input 
                                            type="checkbox" 
                                            checked={hasArm} 
                                            onChange={e => setHasArm(e.target.checked)} 
                                            className="w-4 h-4 rounded bg-slate-800 border-slate-700 accent-blue-500"
                                        />
                                        <span>安装机械臂</span>
                                    </label>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-3">
                            <div
                                className="flex justify-between items-center cursor-pointer"
                                onClick={(e) => {
                                    const content = e.currentTarget.nextElementSibling;
                                    const chevron = e.currentTarget.querySelector('.chevron');
                                    if (content && chevron) {
                                        content.classList.toggle('hidden');
                                        chevron.textContent = content.classList.contains('hidden') ? '▼' : '▲';
                                    }
                                }}
                            >
                                <span className="text-sm font-semibold text-slate-300 uppercase tracking-wider">2. 操控本体</span>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={(e) => { e.stopPropagation(); resetRobot(); }}
                                        className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 px-2 py-1 rounded transition-all"
                                        title="复位小车位置"
                                    >
                                        ↺ 复位
                                    </button>
                                    <span className="chevron text-[10px] text-slate-400">▲</span>
                                </div>
                            </div>
                            <div className="space-y-3">
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
                                    {hasArm ? (
                                        <button
                                            className={`control-btn p-2 rounded border flex flex-col items-center gap-1 transition-all ${activeKeys['q'] ? 'bg-purple-600 border-purple-400 text-white shadow-[0_0_10px_rgba(147,51,234,0.5)]' : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-600'}`}
                                            onMouseDown={() => sim.current.keys['q'] = true}
                                            onMouseUp={() => sim.current.keys['q'] = false}
                                            onMouseLeave={() => sim.current.keys['q'] = false}
                                            onTouchStart={(e) => { e.preventDefault(); sim.current.keys['q'] = true; }}
                                            onTouchEnd={(e) => { e.preventDefault(); sim.current.keys['q'] = false; }}
                                        >
                                            <span className="text-xs font-bold">夹起</span>
                                            <span className="text-[10px]">Q</span>
                                        </button>
                                    ) : <div></div>}
                                    <button
                                        className={`control-btn p-2 rounded border flex flex-col items-center gap-1 transition-all ${activeKeys['w'] ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_10px_rgba(37,99,235,0.5)]' : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-600'}`}
                                        onMouseDown={() => sim.current.keys['w'] = true}
                                        onMouseUp={() => sim.current.keys['w'] = false}
                                        onMouseLeave={() => sim.current.keys['w'] = false}
                                        onTouchStart={(e) => { e.preventDefault(); sim.current.keys['w'] = true; }}
                                        onTouchEnd={(e) => { e.preventDefault(); sim.current.keys['w'] = false; }}
                                    >
                                        <span className="text-lg leading-none">↑</span>
                                        <span className="text-[10px]">W</span>
                                    </button>
                                    {hasArm ? (
                                        <button
                                            className={`control-btn p-2 rounded border flex flex-col items-center gap-1 transition-all ${activeKeys['e'] ? 'bg-purple-600 border-purple-400 text-white shadow-[0_0_10px_rgba(147,51,234,0.5)]' : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-600'}`}
                                            onMouseDown={() => sim.current.keys['e'] = true}
                                            onMouseUp={() => sim.current.keys['e'] = false}
                                            onMouseLeave={() => sim.current.keys['e'] = false}
                                            onTouchStart={(e) => { e.preventDefault(); sim.current.keys['e'] = true; }}
                                            onTouchEnd={(e) => { e.preventDefault(); sim.current.keys['e'] = false; }}
                                        >
                                            <span className="text-xs font-bold">放下</span>
                                            <span className="text-[10px]">E</span>
                                        </button>
                                    ) : <div></div>}
                                    <button
                                        className={`control-btn p-2 rounded border flex flex-col items-center gap-1 transition-all ${activeKeys['a'] ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_10px_rgba(37,99,235,0.5)]' : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-600'}`}
                                        onMouseDown={() => sim.current.keys['a'] = true}
                                        onMouseUp={() => sim.current.keys['a'] = false}
                                        onMouseLeave={() => sim.current.keys['a'] = false}
                                        onTouchStart={(e) => { e.preventDefault(); sim.current.keys['a'] = true; }}
                                        onTouchEnd={(e) => { e.preventDefault(); sim.current.keys['a'] = false; }}
                                    >
                                        <span className="text-lg leading-none">←</span>
                                        <span className="text-[10px]">A</span>
                                    </button>
                                    <button
                                        className={`control-btn p-2 rounded border flex flex-col items-center gap-1 transition-all ${activeKeys['s'] ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_10px_rgba(37,99,235,0.5)]' : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-600'}`}
                                        onMouseDown={() => sim.current.keys['s'] = true}
                                        onMouseUp={() => sim.current.keys['s'] = false}
                                        onMouseLeave={() => sim.current.keys['s'] = false}
                                        onTouchStart={(e) => { e.preventDefault(); sim.current.keys['s'] = true; }}
                                        onTouchEnd={(e) => { e.preventDefault(); sim.current.keys['s'] = false; }}
                                    >
                                        <span className="text-lg leading-none">↓</span>
                                        <span className="text-[10px]">S</span>
                                    </button>
                                    <button
                                        className={`control-btn p-2 rounded border flex flex-col items-center gap-1 transition-all ${activeKeys['d'] ? 'bg-blue-600 border-blue-400 text-white shadow-[0_0_10px_rgba(37,99,235,0.5)]' : 'bg-slate-800 hover:bg-slate-700 text-white border-slate-600'}`}
                                        onMouseDown={() => sim.current.keys['d'] = true}
                                        onMouseUp={() => sim.current.keys['d'] = false}
                                        onMouseLeave={() => sim.current.keys['d'] = false}
                                        onTouchStart={(e) => { e.preventDefault(); sim.current.keys['d'] = true; }}
                                        onTouchEnd={(e) => { e.preventDefault(); sim.current.keys['d'] = false; }}
                                    >
                                        <span className="text-lg leading-none">→</span>
                                        <span className="text-[10px]">D</span>
                                    </button>
                                </div>
                                <p className="text-[10px] text-slate-500 text-center mt-2">使用键盘 WASD 或方向键控制</p>
                            </div>
                        </div>
                        <div className="bg-slate-900/50 p-1 rounded-lg flex text-xs font-medium border border-slate-800">
                            <button
                                onClick={() => setTrainingMode('frontend')}
                                className={`flex-1 py-1.5 rounded-md transition-all ${trainingMode === 'frontend' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-slate-400 hover:text-slate-300'}`}
                            >
                                浏览器训练
                            </button>
                            <button
                                onClick={() => setTrainingMode('cloud')}
                                className={`flex-1 py-1.5 rounded-md transition-all ${trainingMode === 'cloud' ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20' : 'text-slate-400 hover:text-slate-300'}`}
                            >
                                服务器训练
                            </button>
                        </div>
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                                {/* <span className={`w-2 h-2 rounded-full bg-red-500 ${isRecording ? 'recording-pulse opacity-100' : 'opacity-30'}`}></span> */}
                                3. 采集数据
                            </h3>
                            <div className="flex gap-2">
                                <button onClick={toggleRecording} className={`flex-1 ${isRecording ? 'bg-red-600/40' : 'bg-red-600/20'} hover:bg-red-600/30 text-red-400 border border-red-500/30 py-2 px-4 rounded-lg font-medium transition-all flex items-center justify-center gap-2`}>
                                    <span className={`w-2 h-2 rounded-full bg-red-500 ${isRecording ? 'recording-pulse' : ''}`}></span>
                                    {isRecording ? '停止采集' : '开始采集'}
                                </button>
                            </div>
                            <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
                                <input
                                    type="checkbox"
                                    checked={enableCollisionProtection}
                                    onChange={e => setEnableCollisionProtection(e.target.checked)}
                                    className="w-4 h-4 rounded bg-slate-800 border-slate-700 accent-blue-500"
                                />
                                <span>开启碰撞保护 (撞墙自动停止)</span>
                            </label>
                            <div className="text-xs text-slate-400 mono bg-slate-900/50 p-2 rounded border border-slate-800">
                                <div>Episodes: <span className="text-blue-400">{episodesCount}</span></div>
                                <div>Frames: <span className="text-blue-400">{frameCount}</span></div>
                                <div>Actions: <span className="text-blue-400">{actionCount}</span></div>
                            </div>
                            <div className="flex gap-2">
                                <button onClick={saveDataset} disabled={episodesCount === 0} className={`flex-1 ${trainingMode === 'cloud' ? 'bg-purple-600 hover:bg-purple-500 text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'} border border-slate-600 py-2 rounded-lg text-sm transition-all disabled:opacity-50`}>
                                    {trainingMode === 'cloud' ? '上传' : '保存'}
                                </button>
                                <label className="flex-1 cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 py-2 rounded-lg text-sm transition-all flex items-center justify-center">
                                    导入
                                    <input type="file" accept=".json" onChange={handleImportDataset} className="hidden" />
                                </label>
                                <button onClick={clearDataset} disabled={episodesCount === 0} className="flex-1 bg-red-900/30 hover:bg-red-800/50 text-red-400 border border-red-800/50 py-2 rounded-lg text-sm transition-all disabled:opacity-50">
                                    清空
                                </button>
                            </div>
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
                                                // id 去掉前14位，只显示后10位
                                                <option key={i} value={ds.path}>{ds.id.slice(-20)} ({(ds.size_bytes / 1024).toFixed(1)} KB)</option>
                                            ))}
                                        </select>
                                        <button onClick={fetchCloudDatasets} className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs">
                                            ↻
                                        </button>
                                    </div>
                                </div>



                                {cloudTrainingStatus && (
                                    <div className="text-[10px] mono bg-black/30 p-2 rounded border border-purple-500/10 text-purple-300">
                                        Status: {cloudTrainingStatus.status}<br />
                                        Epoch: {cloudTrainingStatus.epoch}/{cloudTrainingStatus.num_epochs}<br />
                                        Loss: {cloudTrainingStatus.avg_loss?.toFixed(4) ?? 'N/A'}
                                    </div>
                                )}
                            </div>
                        )}
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                                4. 训练模型
                            </h3>
                            <button
                                onClick={startTraining}
                                disabled={trainingMode === 'cloud' ? !selectedCloudDataset : (episodesCount === 0 || isTraining)}
                                className={`w-full ${trainingMode === 'cloud' ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500' : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500'} text-white py-3 rounded-lg font-medium transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50`}
                            >
                                {trainingMode === 'cloud' ? '开始云端训练' : '开始训练模型'}
                            </button>
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs text-slate-400">
                                    <span>{isTraining ? '训练中' : '等待训练'}</span>
                                    <span>{Math.floor(trainingProgress)}%</span>
                                </div>
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                                    <div className="h-full training-bar" style={{ width: `${trainingProgress}%` }}></div>
                                </div>
                                <div className="text-xs text-slate-500 mono">{trainingStatus}</div>
                            </div>
                        </div>

                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">5. 执行推理</h3>

                            {trainingMode === 'cloud' ? (
                                <div className="text-xs text-purple-300 bg-purple-900/20 p-2 rounded border border-purple-500/20 mb-2">
                                    <div className="font-bold mb-2">选择模型</div>
                                    <div className="flex gap-2">
                                        <select
                                            value={selectedCloudModel}
                                            onChange={e => setSelectedCloudModel(e.target.value)}
                                            className="flex-1 bg-slate-800 border border-slate-700 text-slate-300 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-purple-500"
                                        >
                                            <option value="">-- Select Model --</option>
                                            {cloudModels.map((m, i) => (
                                                <option key={i} value={m.id}>{m.id}</option>
                                            ))}
                                        </select>
                                        <button onClick={fetchCloudModels} className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs">
                                            ↻
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <select 
                                        className="w-full bg-slate-900 border border-slate-700 text-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500" 
                                        value={selectedModel}
                                        onChange={e => setSelectedModel(e.target.value)}
                                    >
                                        <option value="">选择训练好的模型...</option>
                                        {trainedModels.map((model, i) => (
                                            <option key={i} value={model.name}>{model.name} (Ready)</option>
                                        ))}
                                    </select>
                                    <div className="flex gap-2">
                                        <button onClick={handleExportModels} disabled={trainedModels.length === 0} className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 py-1.5 rounded text-xs transition-all disabled:opacity-50">
                                            导出模型
                                        </button>
                                        <label className="flex-1 cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 py-1.5 rounded text-xs transition-all flex items-center justify-center">
                                            导入文件夹
                                            <input type="file" webkitdirectory="" directory="" multiple onChange={handleImportModelsFolder} className="hidden" />
                                        </label>
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={startInference}
                                disabled={trainingMode === 'cloud' ? !selectedCloudModel : !selectedModel}
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

                <main className="flex-1 relative bg-slate-950">
                    <div ref={canvasContainerRef} className="absolute inset-0"></div>

                    <div className="absolute top-4 left-4 glass-panel rounded-lg p-3 text-xs mono space-y-1 pointer-events-none">
                        <div className="text-slate-400">Position: <span ref={posXRef} className="text-blue-400">0.00</span>, <span ref={posZRef} className="text-blue-400">0.00</span></div>
                        <div className="text-slate-400">Rotation: <span ref={rotYRef} className="text-purple-400">0°</span></div>
                        <div className="text-slate-400">Velocity: <span ref={velocityRef} className="text-green-400">0.0</span> m/s</div>
                    </div>

                    <div className={`absolute top-4 right-4 glass-panel px-4 py-2 rounded-full text-sm font-medium border ${isTraining ? 'text-blue-400 border-blue-500/30' : isInferencing ? 'text-purple-400 border-purple-500/30 recording-pulse' : trainedModel && !isInferencing ? 'text-green-400 border-green-500/30' : 'text-slate-300 border-slate-700'}`}>
                        {isTraining ? '训练中...' : isInferencing ? 'ACT 自主推理中' : trainedModel && !isInferencing ? '训练完成' : '手动控制模式'}
                    </div>
                </main>

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
                            {actionChunks.length > 0 ? actionChunks.map((v, idx) => {
                                const height = Math.max(10, Math.min(100, Math.abs(v) * 100 / speed));
                                const color = v > 0 ? 'bg-blue-500' : 'bg-purple-500';
                                return (
                                    <div key={idx} className={`flex-1 ${color} rounded-t transition-all duration-300`} style={{ height: `${height}%`, opacity: 1 - (idx * 0.05) }}></div>
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
                        <div 
                            ref={logContainerRef} 
                            onScroll={handleLogScroll}
                            className="flex-1 overflow-y-auto p-3 space-y-1 text-[10px] mono"
                        >
                            {logs.map((log, i) => (
                                <div key={i} className={`log-entry log-${log.type}`}>
                                    [{log.time}] {log.message}
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>
            </div>
        </div>
    );
}