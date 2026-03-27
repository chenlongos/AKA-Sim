import * as THREE from 'three';
import { AutoController, AutoState } from './autoController';
import { createBucket, createBall } from '../sim/objects';

export class AutoDemoService {
  private autoController: AutoController;
  private bucket: THREE.Mesh | null = null;
  private ball: THREE.Mesh | null = null;
  private scene: THREE.Scene | null = null;
  private isRunning: boolean = false;
  private updateCallback: (() => void) | null = null;

  constructor() {
    this.autoController = new AutoController({
      approachSpeed: 0.05,
      rotationSpeed: 0.08,
      grabDistance: 1.0,
      placeDistance: 0.8
    });
  }

  initializeDemo(scene: THREE.Scene) {
    this.scene = scene;

    const bucketPosition = new THREE.Vector3(5, 0, 5);
    this.bucket = createBucket(0xff0000, bucketPosition);
    this.bucket.name = 'bucket';
    scene.add(this.bucket);

    const ballPosition = new THREE.Vector3(-5, 0.25, -3);
    this.ball = createBall(0xccff00, ballPosition);
    this.ball.name = 'ball';
    scene.add(this.ball);

    return { bucket: this.bucket, ball: this.ball };
  }

  startAutoMode(robot: THREE.Group, deltaTime: number) {
    if (!this.ball || !this.bucket) {
      console.warn('AutoDemoService: Ball or bucket not initialized');
      return;
    }

    this.autoController.update(robot, this.ball, this.bucket, deltaTime);
  }

  getCurrentState(): AutoState {
    return this.autoController.getCurrentState();
  }

  reset() {
    this.autoController.reset();

    if (this.scene && this.bucket && this.ball) {
      this.scene.remove(this.bucket);
      this.scene.remove(this.ball);
    }

    this.bucket = null;
    this.ball = null;
    this.isRunning = false;
  }

  updateBallPosition(position: THREE.Vector3) {
    if (this.ball) {
      this.ball.position.copy(position);
    }
  }

  updateBucketPosition(position: THREE.Vector3) {
    if (this.bucket) {
      this.bucket.position.copy(position);
    }
  }

  getBallPosition(): THREE.Vector3 | null {
    return this.ball ? this.ball.position.clone() : null;
  }

  getBucketPosition(): THREE.Vector3 | null {
    return this.bucket ? this.bucket.position.clone() : null;
  }

  cleanup() {
    this.reset();
    this.scene = null;
  }
}
