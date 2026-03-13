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
    
    const wheels = [
        { x: -0.7, z: 0 }, { x: 0.7, z: 0 }
    ];
    
    wheels.forEach(pos => {
        const wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(pos.x, 0.3, pos.z);
        wheel.castShadow = true;
        robot.add(wheel);
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
    
    return { robot, onboardCamera, onboardRenderTarget };
}
