import * as THREE from 'three';

export enum AutoState {
  IDLE = 'idle',
  TRACKING = 'tracking',
  APPROACHING = 'approaching',
  GRABBING = 'grabbing',
  CARRYING = 'carrying',
  PLACING = 'placing',
  COMPLETE = 'complete'
}

export interface AutoControllerConfig {
  approachSpeed: number;
  rotationSpeed: number;
  grabDistance: number;
  placeDistance: number;
}

export class AutoController {
  private state: AutoState = AutoState.IDLE;
  private config: AutoControllerConfig;
  private targetPosition: THREE.Vector3 | null = null;
  private bucketPosition: THREE.Vector3 | null = null;
  private ballPosition: THREE.Vector3 | null = null;
  private gripperOpen: boolean = true;
  private grabbedObject: THREE.Mesh | null = null;

  constructor(config: Partial<AutoControllerConfig> = {}) {
    this.config = {
      approachSpeed: 0.02,
      rotationSpeed: 0.05,
      grabDistance: 0.5,
      placeDistance: 0.3,
      ...config
    };
  }

  getCurrentState(): AutoState {
    return this.state;
  }

  getGripperPosition(robot: THREE.Group): THREE.Vector3 {
    const gripper = robot.getObjectByName('gripper');
    if (gripper) {
      return new THREE.Vector3().setFromMatrixPosition(gripper.matrixWorld);
    }
    return new THREE.Vector3(0, 1.5, 0.5);
  }

  calculateAngleToTarget(from: THREE.Vector3, to: THREE.Vector3): number {
    return Math.atan2(to.z - from.z, to.x - from.x);
  }

  adjustRobotOrientation(robot: THREE.Group, targetAngle: number): void {
    const currentRotation = robot.rotation.y;
    let angleDiff = targetAngle - currentRotation;

    while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
    while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

    const kP = 2.0;
    const newRotation = currentRotation + angleDiff * kP * this.config.rotationSpeed;
    robot.rotation.y = newRotation;
  }

  moveRobotForward(robot: THREE.Group, speed: number, deltaTime: number): void {
    const direction = new THREE.Vector3(0, 0, -1).applyQuaternion(robot.quaternion);
    const movement = direction.multiplyScalar(speed * deltaTime);
    robot.position.add(movement);
  }

  moveRobotToPosition(robot: THREE.Group, targetPos: THREE.Vector3, deltaTime: number): void {
    const currentPosition = new THREE.Vector3();
    robot.getWorldPosition(currentPosition);

    const direction = new THREE.Vector3()
      .subVectors(targetPos, currentPosition)
      .normalize();

    const speed = this.config.approachSpeed;
    const movement = direction.multiplyScalar(speed * deltaTime);

    const distance = currentPosition.distanceTo(targetPos);
    if (distance > 0.1) {
      robot.position.add(movement);
    }
  }

  setGripperOpen(robot: THREE.Group, isOpen: boolean): void {
    this.gripperOpen = isOpen;
    const gripper = robot.getObjectByName('gripper');
    if (gripper) {
      gripper.traverse((child) => {
        if (child instanceof THREE.Mesh && child.name.includes('finger')) {
          const fingerWidth = isOpen ? 0.3 : 0.1;
          child.scale.set(isOpen ? 1 : 0.7, 1, 1);
          if (child.name.includes('left')) {
            child.position.x = isOpen ? -0.3 : -0.15;
          } else if (child.name.includes('right')) {
            child.position.x = isOpen ? 0.3 : 0.15;
          }
        }
      });
    }
  }

  grabObject(robot: THREE.Group, ball: THREE.Mesh): void {
    if (!this.gripperOpen) return;

    this.grabbedObject = ball;
    const gripperPos = this.getGripperPosition(robot);
    ball.position.copy(gripperPos);
    ball.position.y += 0.2;

    ball.userData.isGrabbed = true;
  }

  releaseObject(robot: THREE.Group): void {
    if (this.grabbedObject) {
      this.grabbedObject.userData.isGrabbed = false;
      this.grabbedObject = null;
    }
  }

  update(
    robot: THREE.Group,
    ball: THREE.Mesh | null,
    bucket: THREE.Mesh | null,
    deltaTime: number
  ): void {
    if (!robot) return;

    this.ballPosition = ball ? ball.position.clone() : null;
    this.bucketPosition = bucket ? bucket.position.clone() : null;

    const gripperPos = this.getGripperPosition(robot);
    const robotPos = robot.position.clone();

    // Update grabbed object position
    if (this.grabbedObject) {
      this.grabbedObject.position.copy(gripperPos);
      this.grabbedObject.position.y += 0.3;
    }

    switch (this.state) {
      case AutoState.IDLE:
        if (ball && this.ballPosition) {
          this.state = AutoState.TRACKING;
          console.log('AutoController: Starting to track ball');
        }
        break;

      case AutoState.TRACKING:
        if (!ball || !this.ballPosition) {
          this.state = AutoState.IDLE;
          break;
        }

        const angleToBall = this.calculateAngleToTarget(gripperPos, this.ballPosition);
        this.adjustRobotOrientation(robot, angleToBall);

        if (Math.abs(robot.rotation.y - angleToBall) < 0.1) {
          this.state = AutoState.APPROACHING;
          console.log('AutoController: Approaching ball');
        }
        break;

      case AutoState.APPROACHING:
        if (!ball || !this.ballPosition) {
          this.state = AutoState.IDLE;
          break;
        }

        this.moveRobotToPosition(robot, this.ballPosition, deltaTime);

        const distanceToBall = gripperPos.distanceTo(this.ballPosition);
        if (distanceToBall < this.config.grabDistance) {
          this.state = AutoState.GRABBING;
          console.log('AutoController: Grabbing ball');
        }
        break;

      case AutoState.GRABBING:
        if (!ball) {
          this.state = AutoState.IDLE;
          break;
        }

        this.setGripperOpen(robot, false);
        this.grabObject(robot, ball);

        setTimeout(() => {
          this.state = AutoState.CARRYING;
          console.log('AutoController: Carrying ball to bucket');
        }, 500);
        break;

      case AutoState.CARRYING:
        if (!bucket || !this.bucketPosition) {
          console.warn('AutoController: No bucket found, staying in carrying state');
          break;
        }

        if (this.grabbedObject) {
          this.grabbedObject.position.copy(this.getGripperPosition(robot));
          this.grabbedObject.position.y += 0.2;
        }

        const bucketTop = new THREE.Vector3(
          this.bucketPosition.x,
          this.bucketPosition.y + 1.0,
          this.bucketPosition.z
        );

        const angleToBucket = this.calculateAngleToTarget(gripperPos, bucketTop);
        this.adjustRobotOrientation(robot, angleToBucket);

        const distanceToBucket = gripperPos.distanceTo(bucketTop);
        if (distanceToBucket < this.config.placeDistance && Math.abs(robot.rotation.y - angleToBucket) < 0.2) {
          this.state = AutoState.PLACING;
          console.log('AutoController: Placing ball in bucket');
        } else {
          this.moveRobotToPosition(robot, bucketTop, deltaTime);
        }
        break;

      case AutoState.PLACING:
        if (this.grabbedObject && this.bucketPosition) {
          this.grabbedObject.position.set(
            this.bucketPosition.x,
            this.bucketPosition.y + 0.5,
            this.bucketPosition.z
          );
        }

        this.setGripperOpen(robot, true);
        this.releaseObject(robot);

        setTimeout(() => {
          this.state = AutoState.COMPLETE;
          console.log('AutoController: Task complete!');
        }, 1000);
        break;

      case AutoState.COMPLETE:
        setTimeout(() => {
          this.state = AutoState.IDLE;
          console.log('AutoController: Ready for next task');
        }, 3000);
        break;
    }
  }

  reset(): void {
    this.state = AutoState.IDLE;
    this.gripperOpen = true;
    this.grabbedObject = null;
    this.targetPosition = null;
  }
}
