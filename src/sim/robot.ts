import * as THREE from 'three';

export function createRobot(scene: THREE.Scene) {
    const robot = new THREE.Group();
    
    // Body
    const bodyGeo = new THREE.BoxGeometry(1.2, 0.4, 1.8);
    const bodyMat = new THREE.MeshStandardMaterial({ 
        color: 0x3b82f6,
        roughness: 0.5,
        metalness: 0.8
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.4;
    body.castShadow = true;
    robot.add(body);
    
    // Wheels
    const wheelGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 32);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x1e293b });
    const spokeGeo = new THREE.BoxGeometry(0.5, 0.22, 0.05);
    const spokeMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8 });
    
    const wheelPositions = [
        { x: -0.7, z: 0 }, { x: 0.7, z: 0 }
    ];
    
    const wheelMeshes: THREE.Group[] = [];

    wheelPositions.forEach(pos => {
        const wheelGroup = new THREE.Group();
        wheelGroup.position.set(pos.x, 0.3, pos.z);
        wheelGroup.rotation.z = Math.PI / 2;

        const wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.castShadow = true;
        wheelGroup.add(wheel);
        
        const spoke1 = new THREE.Mesh(spokeGeo, spokeMat);
        const spoke2 = new THREE.Mesh(spokeGeo, spokeMat);
        spoke2.rotation.y = Math.PI / 2;
        
        wheelGroup.add(spoke1);
        wheelGroup.add(spoke2);

        robot.add(wheelGroup);
        wheelMeshes.push(wheelGroup);
    });
    
    // Head/Camera Mount
    const headGeo = new THREE.BoxGeometry(0.4, 0.4, 0.4);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x64748b });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.set(0, 0.8, 0.6);
    head.castShadow = true;
    robot.add(head);
    
    // Camera Lens
    const lensGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.1, 16);
    const lensMat = new THREE.MeshStandardMaterial({ color: 0x000000 });
    const lens = new THREE.Mesh(lensGeo, lensMat);
    lens.rotation.x = Math.PI / 2;
    lens.position.set(0, 0.8, 0.85);
    robot.add(lens);
    
    // Onboard Camera
    const onboardCamera = new THREE.PerspectiveCamera(80, 320/240, 0.1, 50);
    onboardCamera.position.set(0, 0.7, 0.6);
    
    const onboardRenderTarget = new THREE.WebGLRenderTarget(320, 240);
    
    // Robotic Arm
    const armGroup = new THREE.Group();
    armGroup.position.set(0, 0.6, 0.2); // Base of the arm on the robot body (moved forward)

    const orangeMat = new THREE.MeshStandardMaterial({ color: 0xff5722, roughness: 0.6 });
    const blackMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.8 });

    // Base pillar (Black)
    const basePillar = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.5, 0.4), blackMat);
    basePillar.position.y = 0.25;
    basePillar.castShadow = true;
    armGroup.add(basePillar);

    // Shoulder (Orange)
    const shoulder = new THREE.Group();
    shoulder.position.y = 0.5;
    armGroup.add(shoulder);

    const shoulderBracket = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.3), orangeMat);
    shoulderBracket.position.y = 0.15;
    shoulderBracket.castShadow = true;
    shoulder.add(shoulderBracket);

    // Lower Arm
    const lowerArm = new THREE.Group();
    lowerArm.position.y = 0.3;
    lowerArm.rotation.x = Math.PI / 4; // Lean forward
    shoulder.add(lowerArm);

    const lowerArmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.8, 0.2), orangeMat);
    lowerArmMesh.position.y = 0.4;
    lowerArmMesh.castShadow = true;
    lowerArm.add(lowerArmMesh);

    // Elbow
    const elbow = new THREE.Group();
    elbow.position.y = 0.8;
    elbow.rotation.x = Math.PI / 2.5; // Bend down a bit
    lowerArm.add(elbow);

    const elbowBracket = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.25, 0.25), orangeMat);
    elbowBracket.castShadow = true;
    elbow.add(elbowBracket);

    const elbowServo = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.2, 0.35), blackMat);
    elbowServo.position.x = 0.05;
    elbow.add(elbowServo);

    // Upper Arm
    const upperArmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.4, 0.2), orangeMat);
    upperArmMesh.position.y = 0.2;
    upperArmMesh.castShadow = true;
    elbow.add(upperArmMesh);

    // Wrist
    const wrist = new THREE.Group();
    wrist.position.y = 0.4;
    wrist.rotation.x = -Math.PI / 6;
    elbow.add(wrist);

    const wristBracket = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.2, 0.25), orangeMat);
    wristBracket.castShadow = true;
    wrist.add(wristBracket);

    const wristServo = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.25, 0.2), blackMat);
    wristServo.position.y = 0.1;
    wrist.add(wristServo);

    // Gripper
    const gripper = new THREE.Group();
    gripper.position.y = 0.1;
    wrist.add(gripper);

    const gripperBase = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.1, 0.4), orangeMat);
    gripperBase.position.y = 0.05;
    gripperBase.castShadow = true;
    gripper.add(gripperBase);

    // Fingers
    const leftFinger = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.6, 0.4), orangeMat);
    leftFinger.position.set(-0.3, 0.35, 0);
    leftFinger.castShadow = true;
    gripper.add(leftFinger);

    const rightFinger = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.6, 0.4), orangeMat);
    rightFinger.position.set(0.3, 0.35, 0);
    rightFinger.castShadow = true;
    gripper.add(rightFinger);

    robot.add(armGroup);
    
    return { robot, onboardCamera, onboardRenderTarget, wheelMeshes, armGroup, lowerArm, elbow, wrist, gripper };
}