import * as THREE from 'three';
import { PathFinder } from './pathfinding';
import { ArmController } from './armController';
import { detectObjects, DetectedObject } from './vision';

export type ACTTask = 'findAndPickBall' | 'findBucketAndDrop' | 'idle';

export interface ACTExecutorState {
    task: ACTTask;
    subTask: string;
    path: THREE.Vector3[];
    currentWaypoint: THREE.Vector3 | null;
    isComplete: boolean;
    isExecuting: boolean;
}

export class ACTExecutor {
    private state: ACTExecutorState;
    private pathFinder: PathFinder;
    private armController: ArmController | null;
    private scene: THREE.Scene;
    private robotGroup: THREE.Group | null;
    private robotState: { x: number; z: number; rotation: number; velocity: number; angularVelocity: number };
    private addLog: (msg: string, type?: string) => void;
    private walls: THREE.Mesh[];
    
    // Configuration
    private readonly WAYPOINT_THRESHOLD = 0.3;  // 更小的阈值
    private readonly GRAB_DISTANCE = 1.0;        // 减小接近距离，让机器人更接近小球
    private readonly DROP_DISTANCE = 1.0;
    private readonly SPEED = 0.08;               // 降低速度
    private readonly TURN_SPEED = 0.04;          // 降低转向速度

    constructor(
        scene: THREE.Scene,
        walls: THREE.Mesh[],
        addLog: (msg: string, type?: string) => void,
        armController?: ArmController
    ) {
        this.scene = scene;
        this.walls = walls;
        this.addLog = addLog;
        this.armController = armController || null;
        this.pathFinder = new PathFinder();
        
        this.robotState = {
            x: 0,
            z: 0,
            rotation: Math.PI,
            velocity: 0,
            angularVelocity: 0
        };
        
        this.state = {
            task: 'idle',
            subTask: '',
            path: [],
            currentWaypoint: null,
            isComplete: false,
            isExecuting: false
        };
    }

    setRobotRef(robot: THREE.Group, state: any): void {
        this.robotGroup = robot;
        this.robotState = state;
    }

    setArmController(armController: ArmController): void {
        this.armController = armController;
    }

    updateObstacles(): void {
        this.pathFinder.updateObstacles(this.walls);
    }

    updateWalls(walls: THREE.Mesh[]): void {
        this.walls = walls;
    }

    private findBallInScene(): THREE.Mesh | null {
        let ballMesh: THREE.Mesh | null = null;
        
        this.scene.traverse((object) => {
            if (object instanceof THREE.Mesh && object.geometry.type === 'SphereGeometry') {
                const material = object.material as THREE.MeshStandardMaterial;
                const color = material.color;
                
                // Check if it's a green ball
                if (color.g > 0.8 && color.r > 0.6 && color.b < 0.2) {
                    ballMesh = object;
                }
            }
        });
        
        return ballMesh;
    }

    startTask(task: ACTTask): void {
        if (this.state.isExecuting) {
            this.addLog('Already executing a task', 'warning');
            return;
        }

        this.state = {
            task,
            subTask: 'searching',
            path: [],
            currentWaypoint: null,
            isComplete: false,
            isExecuting: true
        };

        this.updateObstacles();
        this.addLog(`Starting task: ${task}`, 'info');
    }

    stop(): void {
        this.state.isExecuting = false;
        this.state.isComplete = false;
        this.robotState.velocity = 0;
        this.robotState.angularVelocity = 0;
        this.addLog('Task stopped', 'warning');
    }

    reset(): void {
        this.state = {
            task: 'idle',
            subTask: '',
            path: [],
            currentWaypoint: null,
            isComplete: false,
            isExecuting: false
        };
        this.robotState.velocity = 0;
        this.robotState.angularVelocity = 0;
        
        if (this.armController) {
            this.armController.reset();
        }
    }

    update(): void {
        if (!this.state.isExecuting || this.state.isComplete) {
            return;
        }

        const robotPos = new THREE.Vector3(this.robotState.x, 0, this.robotState.z);

        // Detect objects
        const vision = detectObjects(this.scene, robotPos, this.robotState.rotation);

        // Debug log every second
        if (Date.now() % 1000 < 50) {
            this.addLog(`Vision: ball=${vision.ball ? vision.ball.distance.toFixed(2) : 'null'}, bucket=${vision.redBucket ? vision.redBucket.distance.toFixed(2) : 'null'}, vel=${this.robotState.velocity.toFixed(2)}, subTask=${this.state.subTask}`, 'info');
            this.addLog(`RobotPos: (${this.robotState.x.toFixed(2)}, ${this.robotState.z.toFixed(2)}), rot=${(this.robotState.rotation * 180 / Math.PI).toFixed(0)}°`, 'info');
        }

        switch (this.state.task) {
            case 'findAndPickBall':
                this.executeFindAndPickBall(vision, robotPos);
                break;

            case 'findBucketAndDrop':
                this.executeFindBucketAndDrop(vision, robotPos);
                break;
        }

        // Update arm controller if exists
        if (this.armController) {
            this.armController.update();
            this.armController.advancePhase(this.addLog);
        }
    }

    private executeFindAndPickBall(vision: any, robotPos: THREE.Vector3): void {
        if (!this.armController) {
            this.addLog('No arm controller available', 'error');
            this.state.isComplete = true;
            this.state.isExecuting = false;
            return;
        }

        switch (this.state.subTask) {
            case 'searching':
                if (vision.ball) {
                    this.state.subTask = 'approaching';
                    this.addLog(`✓ Ball detected at distance ${vision.ball.distance.toFixed(2)}m`, 'success');
                } else {
                    // Search pattern: rotate in place
                    this.robotState.velocity = 0;
                    this.robotState.angularVelocity = this.TURN_SPEED;
                    this.addLog(`🔄 Searching... (rotating)`);
                }
                break;

            case 'approaching':
                if (vision.ball) {
                    // Use vision distance directly - relaxed condition for easier pickup
                    if (vision.ball.distance < this.GRAB_DISTANCE + 0.5) {
                        // Close enough, start picking (distance threshold: 1.7m)
                        this.state.subTask = 'picking';
                        
                        // Find the ball mesh
                        const ballMesh = this.findBallInScene();
                        if (ballMesh) {
                            // Start pick sequence (exactly like pressing Q)
                            this.armController!.startPickBall(ballMesh);
                            this.addLog(`🎯 In position (dist=${vision.ball.distance.toFixed(2)}m), picking up ball...`, 'info');
                        }
                    } else {
                        // Navigate towards ball
                        const targetPos = vision.ball.position.clone();
                        const approachPos = targetPos.clone().sub(robotPos).normalize().multiplyScalar(this.GRAB_DISTANCE).add(robotPos);
                        this.navigateTo(approachPos, robotPos, () => {
                            this.state.subTask = 'picking';
                            
                            // Find the ball mesh
                            const ballMesh = this.findBallInScene();
                            if (ballMesh) {
                                // Start pick sequence (exactly like pressing Q)
                                this.armController!.startPickBall(ballMesh);
                                this.addLog('🎯 Reached target position, picking up ball...', 'info');
                            }
                        });
                    }
                } else {
                    this.state.subTask = 'searching';
                }
                break;

            case 'picking':
                this.robotState.velocity = 0;
                this.robotState.angularVelocity = 0;
                
                if (this.armController.isAtTarget() && this.armController.getState().phase >= 2) {
                    this.state.subTask = 'completed';
                    this.state.isComplete = true;
                    this.state.isExecuting = false;
                    this.addLog('✅ Task completed: Ball picked up', 'success');
                }
                break;

            case 'completed':
                this.robotState.velocity = 0;
                this.robotState.angularVelocity = 0;
                break;
        }
    }

    private executeFindBucketAndDrop(vision: any, robotPos: THREE.Vector3): void {
        if (!this.armController) {
            this.addLog('No arm controller available', 'error');
            this.state.isComplete = true;
            this.state.isExecuting = false;
            return;
        }

        if (!this.armController.hasObject()) {
            this.addLog('No object to drop, please pick up ball first', 'warning');
            this.state.isComplete = true;
            this.state.isExecuting = false;
            return;
        }

        switch (this.state.subTask) {
            case 'searching':
                if (vision.redBucket) {
                    this.state.subTask = 'approaching';
                    this.addLog(`Red bucket detected at distance ${vision.redBucket.distance.toFixed(2)}m`, 'info');
                } else {
                    // Search pattern
                    this.robotState.velocity = 0;
                    this.robotState.angularVelocity = this.TURN_SPEED;
                }
                break;

            case 'approaching':
                if (vision.redBucket) {
                    // Use vision distance directly: when distance to bucket < 2.0, start dropping
                    if (vision.redBucket.distance < 2.2) {
                        this.state.subTask = 'dropping';
                        this.armController!.startDropInBucket();
                        this.addLog('In position, dropping ball...', 'info');
                    } else {
                        // Move towards bucket using simple navigation
                        const targetPos = vision.redBucket.position.clone();
                        const robotPos = new THREE.Vector3(this.robotState.x, 0, this.robotState.z);
                        this.navigateTo(targetPos, robotPos, () => {
                            // This callback may not be reached due to collision, 
                            // but we check vision.redBucket.distance < 2.0 above
                            this.state.subTask = 'dropping';
                            this.armController!.startDropInBucket();
                            this.addLog('In position, dropping ball...', 'info');
                        });
                    }
                } else {
                    this.state.subTask = 'searching';
                }
                break;

            case 'dropping':
                this.robotState.velocity = 0;
                this.robotState.angularVelocity = 0;
                
                if (this.armController.isAtTarget() && this.armController.getState().phase >= 2) {
                    this.state.subTask = 'completed';
                    this.state.isComplete = true;
                    this.state.isExecuting = false;
                    this.addLog('Task completed: Ball dropped in bucket', 'success');
                }
                break;

            case 'completed':
                this.robotState.velocity = 0;
                this.robotState.angularVelocity = 0;
                break;
        }
    }

    private navigateTo(
        targetPos: THREE.Vector3,
        robotPos: THREE.Vector3,
        onArrive: () => void
    ): void {
        const distance = robotPos.distanceTo(targetPos);
        
        if (distance < this.WAYPOINT_THRESHOLD) {
            this.robotState.velocity = 0;
            this.robotState.angularVelocity = 0;
            onArrive();
            return;
        }

        // Calculate direction to target
        const direction = targetPos.clone().sub(robotPos).normalize();
        
        // Calculate desired angle to face the target
        const desiredAngle = Math.atan2(direction.x, direction.z);
        
        // Get current rotation (normalized to [0, 2PI])
        let currentRotation = this.robotState.rotation % (2 * Math.PI);
        if (currentRotation < 0) currentRotation += 2 * Math.PI;
        
        // Calculate angle difference (shortest path)
        let angleDiff = desiredAngle - currentRotation;
        
        // Normalize to [-PI, PI]
        while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
        while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
        
        // Relaxed angle threshold: allow movement with larger angle error
        const ANGLE_THRESHOLD = 0.4;  // 放宽到 0.4 弧度（约 23 度）
        
        // Turn towards target
        if (Math.abs(angleDiff) > ANGLE_THRESHOLD) {
            // Large angle difference: stop and turn
            this.robotState.velocity = 0;
            this.robotState.angularVelocity = angleDiff > 0 ? this.TURN_SPEED : -this.TURN_SPEED;
            if (Date.now() % 500 < 50) {
                this.addLog(`🧭 Turning: angleDiff=${angleDiff.toFixed(2)}rad (${(angleDiff * 180 / Math.PI).toFixed(0)}°)`);
            }
        } else {
            // Small angle difference: move forward with minor correction
            this.robotState.velocity = this.SPEED;
            // Apply proportional correction while moving
            this.robotState.angularVelocity = angleDiff * 0.1;  // 温和的角度修正
            if (Date.now() % 500 < 50) {
                this.addLog(`⬆️ Moving: dist=${distance.toFixed(2)}, angleDiff=${angleDiff.toFixed(2)}rad`);
            }
        }
    }

    getState(): ACTExecutorState {
        return { ...this.state };
    }

    // For data collection: get current action
    getCurrentAction(): number[] {
        // Return [velocity, angularVelocity] normalized
        return [
            this.robotState.velocity / this.SPEED,
            this.robotState.angularVelocity / this.TURN_SPEED
        ];
    }
}
