import * as THREE from 'three';

export function createBucket(color: number = 0xff0000, position: THREE.Vector3 = new THREE.Vector3(5, 0, 5)): THREE.Mesh {
  const bucketGroup = new THREE.Group();

  const bucketMat = new THREE.MeshStandardMaterial({
    color: color,
    roughness: 0.3,
    metalness: 0.6
  });

  const bucketGeo = new THREE.CylinderGeometry(0.6, 0.4, 1.2, 32, 1, true);
  const bucketMesh = new THREE.Mesh(bucketGeo, bucketMat);
  bucketMesh.position.y = 0.6;
  bucketMesh.castShadow = true;
  bucketMesh.receiveShadow = true;
  bucketGroup.add(bucketMesh);

  const rimGeo = new THREE.TorusGeometry(0.6, 0.05, 16, 32);
  const rimMat = new THREE.MeshStandardMaterial({
    color: 0xcc0000,
    roughness: 0.3,
    metalness: 0.6
  });
  const rim = new THREE.Mesh(rimGeo, rimMat);
  rim.rotation.x = Math.PI / 2;
  rim.position.y = 1.2;
  rim.castShadow = true;
  bucketGroup.add(rim);

  const handleMat = new THREE.MeshStandardMaterial({
    color: 0x888888,
    roughness: 0.5,
    metalness: 0.8
  });

  const handleGeo = new THREE.TorusGeometry(0.4, 0.03, 8, 16, Math.PI);
  const handle1 = new THREE.Mesh(handleGeo, handleMat);
  handle1.rotation.z = Math.PI / 2;
  handle1.position.set(0.6, 1.5, 0);
  handle1.castShadow = true;
  bucketGroup.add(handle1);

  const handle2 = new THREE.Mesh(handleGeo, handleMat);
  handle2.rotation.z = Math.PI / 2;
  handle2.position.set(-0.6, 1.5, 0);
  handle2.castShadow = true;
  bucketGroup.add(handle2);

  const combinedBucket = new THREE.Mesh(
    new THREE.CylinderGeometry(1, 1, 1, 32),
    new THREE.MeshStandardMaterial({ visible: false })
  );
  combinedBucket.add(bucketGroup);
  combinedBucket.position.copy(position); // Set position on the combined bucket mesh

  return combinedBucket;
}

export function createBall(color: number = 0xccff00, position: THREE.Vector3 = new THREE.Vector3(-3, 0.25, 3)): THREE.Mesh {
  const ballGeo = new THREE.SphereGeometry(0.25, 32, 32);
  const ballMat = new THREE.MeshStandardMaterial({
    color: color,
    roughness: 0.8,
    metalness: 0.1
  });
  const ball = new THREE.Mesh(ballGeo, ballMat);
  ball.position.copy(position);
  ball.castShadow = true;
  ball.receiveShadow = true;
  ball.userData = { isBall: true, isGrabbed: false };

  return ball;
}
