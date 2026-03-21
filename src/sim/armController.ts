import * as THREE from 'three';

export type ArmTask = 'idle' | 'picking_down' | 'picking_up' | 'holding' | 'dropping_down' | 'dropping_up' | 'pickBall' | 'dropInBucket' | 'pickUp' | 'dropDown';

export interface ArmControllerState {
    task: ArmTask;
    phase: number;
    targetRotations: {
        lowerArm: number;
        elbow: number;
        wrist: number;
    };
    grabbedObject: THREE.Mesh | null;
}

export class ArmController {
    private state: ArmControllerState;
    private armGroup: THREE.Group;
    private lowerArm: THREE.Group;
    private elbow: THREE.Group;
    private wrist: THREE.Group;
    private gripper: THREE.Group;
    private worldGroup: THREE.Group;
    private speed: number;

    constructor(
        armGroup: THREE.Group,
        lowerArm: THREE.Group,
        elbow: THREE.Group,
        wrist: THREE.Group,
        gripper: THREE.Group,
        worldGroup: THREE.Group,
        speed: number = 0.1
    ) {
        this.armGroup = armGroup;
        this.lowerArm = lowerArm;
        this.elbow = elbow;
        this.wrist = wrist;
        this.gripper = gripper;
        this.worldGroup = worldGroup;
        this.speed = speed;
        
        this.state = {
            task: 'idle',
            phase: 0,
            targetRotations: {
                lowerArm: Math.PI / 4,
                elbow: Math.PI / 2.5,
                wrist: -Math.PI / 6
            },
            grabbedObject: null
        };
    }

    update(): void {
        // Smoothly interpolate to target rotations (same as manual control: 0.1)
        this.lowerArm.rotation.x += (this.state.targetRotations.lowerArm - this.lowerArm.rotation.x) * 0.1;
        this.elbow.rotation.x += (this.state.targetRotations.elbow - this.elbow.rotation.x) * 0.1;
        this.wrist.rotation.x += (this.state.targetRotations.wrist - this.wrist.rotation.x) * 0.1;

        // Note: Do NOT modify grabbedObject position here!
        // - During picking_down: object stays at its world position until gripper reaches it
        // - After added to gripper: Three.js automatically handles relative positioning
        // Setting position.set(0, 0.35, 0) here would cause the ball to jump to world origin
    }

    isAtTarget(): boolean {
        return (
            Math.abs(this.lowerArm.rotation.x - this.state.targetRotations.lowerArm) < 0.05 &&
            Math.abs(this.elbow.rotation.x - this.state.targetRotations.elbow) < 0.05 &&
            Math.abs(this.wrist.rotation.x - this.state.targetRotations.wrist) < 0.05
        );
    }

    startPickBall(ballMesh: THREE.Mesh): void {
        if (this.state.task !== 'idle') return;
        
        // Exactly like pressing Q: set grabbedObject FIRST (but don't add to gripper yet)
        this.state.grabbedObject = ballMesh;
        this.state.task = 'picking_down';
        this.state.phase = 0;
        
        // Target rotations exactly like manual Q press
        this.state.targetRotations = {
            lowerArm: Math.PI / 2.2,
            elbow: Math.PI / 4,
            wrist: Math.PI / 4
        };
    }

    startDropInBucket(): void {
        if (this.state.task !== 'holding' && this.state.task !== 'pickUp') return;
        
        this.state.task = 'dropInBucket';
        this.state.phase = 0;
        this.state.targetRotations = {
            lowerArm: Math.PI / 2.2,
            elbow: Math.PI / 4,
            wrist: Math.PI / 4
        };
    }

    grabObject(object: THREE.Mesh): void {
        this.state.grabbedObject = object;
        this.gripper.add(object);
        object.position.set(0, 0.35, 0);
    }

    hasObject(): boolean {
        return this.state.grabbedObject !== null;
    }

    releaseObject(): void {
        if (!this.state.grabbedObject) return;

        const worldPos = new THREE.Vector3();
        this.state.grabbedObject.getWorldPosition(worldPos);
        
        this.worldGroup.add(this.state.grabbedObject);
        this.state.grabbedObject.position.copy(worldPos);
        this.state.grabbedObject.position.y = 0.25;
        this.state.grabbedObject = null;
    }

    advancePhase(addLog: (msg: string, type?: string) => void): void {
        if (!this.isAtTarget()) return;

        // Debug log
        console.log('[ArmController] advancePhase called, task:', this.state.task, 'phase:', this.state.phase);

        // Exactly like manual Q/E press state machine
        if (this.state.task === 'picking_down') {
            // Reached down position, add object to gripper (exactly like line 902-903 in App.tsx)
            if (this.state.grabbedObject) {
                this.gripper.add(this.state.grabbedObject);
                this.state.grabbedObject.position.set(0, 0.35, 0);
            }
            
            // Lift up
            this.state.task = 'picking_up';
            this.state.phase++;
            this.state.targetRotations = {
                lowerArm: -Math.PI / 6,
                elbow: Math.PI / 1.5,
                wrist: -Math.PI / 4
            };
        } else if (this.state.task === 'picking_up') {
            // Reached up position, task complete
            this.state.task = 'holding';
            this.state.phase++;
            addLog('Ball picked up', 'success');
        } else if (this.state.task === 'dropInBucket') {
            console.log('[ArmController] dropInBucket: releasing object');
            // Reached first down position (like pressing E: dropping_down)
            // Release object (exactly like manual E press)
            if (this.state.grabbedObject) {
                const worldPos = new THREE.Vector3();
                this.state.grabbedObject.getWorldPosition(worldPos);
                
                this.worldGroup.add(this.state.grabbedObject);
                this.state.grabbedObject.position.copy(worldPos);
                // Drop to ground level
                this.state.grabbedObject.position.y = 0.25;
                this.state.grabbedObject = null;
                addLog('Object released', 'success');
                console.log('[ArmController] Object released successfully');
            } else {
                console.log('[ArmController] No object to release!');
            }
            
            // Lift up (like dropping_up)
            this.state.task = 'dropping_up';
            this.state.phase++;
            this.state.targetRotations = {
                lowerArm: Math.PI / 4,
                elbow: Math.PI / 2.5,
                wrist: -Math.PI / 6
            };
        } else if (this.state.task === 'dropping_down') {
            // Reached down position for dropping, release object (exactly like manual E press)
            if (this.state.grabbedObject) {
                const worldPos = new THREE.Vector3();
                this.state.grabbedObject.getWorldPosition(worldPos);
                
                this.worldGroup.add(this.state.grabbedObject);
                this.state.grabbedObject.position.copy(worldPos);
                // Drop to ground level
                this.state.grabbedObject.position.y = 0.25;
                this.state.grabbedObject = null;
            }
            
            // Lift up
            this.state.task = 'dropping_up';
            this.state.phase++;
            this.state.targetRotations = {
                lowerArm: Math.PI / 4,
                elbow: Math.PI / 2.5,
                wrist: -Math.PI / 6
            };
        } else if (this.state.task === 'dropping_up') {
            // Reached up position, task complete
            this.state.task = 'idle';
            this.state.phase++;
            addLog('Ball dropped', 'success');
        }
    }

    reset(): void {
        this.state = {
            task: 'idle',
            phase: 0,
            targetRotations: {
                lowerArm: Math.PI / 4,
                elbow: Math.PI / 2.5,
                wrist: -Math.PI / 6
            },
            grabbedObject: null
        };
    }

    getState(): ArmControllerState {
        return { ...this.state };
    }
}
