import * as THREE from 'three';

export function updateEnvironment(
    scene: THREE.Scene, 
    sceneType: string, 
    sceneSize: string, 
    sceneComplexity: string,
    currentWalls: THREE.Mesh[],
    currentEnvironmentGroup: THREE.Group | null,
    currentTarget: THREE.Mesh | null
) {
    // Clean up
    currentWalls.forEach(wall => scene.remove(wall));
    if (currentEnvironmentGroup) scene.remove(currentEnvironmentGroup);
    if (currentTarget) scene.remove(currentTarget);
    
    const walls: THREE.Mesh[] = [];
    const group = new THREE.Group();
    scene.add(group);
    
    let sizeVal = 20;
    if (sceneSize === 'small') sizeVal = 10;
    if (sceneSize === 'large') sizeVal = 30;
    
    let numObstacles = 5;
    if (sceneComplexity === 'low') numObstacles = 2;
    if (sceneComplexity === 'high') numObstacles = 10;
    
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x475569 });
    const woodTex = new THREE.MeshStandardMaterial({ color: 0x8b5a2b });
    
    const addWall = (x: number, z: number, w: number, h: number, d: number, color: number, mat?: THREE.Material) => {
        const geo = new THREE.BoxGeometry(w, h, d);
        const m = mat || new THREE.MeshStandardMaterial({ color });
        const mesh = new THREE.Mesh(geo, m);
        mesh.position.set(x, h/2, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { w, d };
        group.add(mesh);
        walls.push(mesh);
    };
    
    // Boundary Walls
    const halfSize = sizeVal / 2;
    addWall(0, -halfSize, sizeVal, 2, 1, 0x1e293b);
    addWall(0, halfSize, sizeVal, 2, 1, 0x1e293b);
    addWall(-halfSize, 0, 1, 2, sizeVal, 0x1e293b);
    addWall(halfSize, 0, 1, 2, sizeVal, 0x1e293b);
    
    if (sceneType === 'living_room') {
        // Sofa
        addWall(-halfSize + 3, -halfSize + 3, 4, 1, 2, 0x5c5c5c);
        // Table
        addWall(0, 0, 3, 0.8, 2, 0x8b5a2b, woodTex);
        // TV Stand
        addWall(halfSize - 2, 0, 1, 1.2, 4, 0x1e1e1e);
        
        for(let i=0; i<numObstacles; i++) {
            const w = 0.5 + Math.random();
            const d = 0.5 + Math.random();
            const x = (Math.random() - 0.5) * (sizeVal - 4);
            const z = (Math.random() - 0.5) * (sizeVal - 4);
            if (Math.abs(x) < 3 && Math.abs(z) < 3) continue;
            addWall(x, z, w, 1 + Math.random(), d, 0x475569);
        }
    } else if (sceneType === 'classroom') {
        const rows = Math.min(4, Math.max(2, Math.floor(numObstacles / 2)));
        const cols = Math.min(4, Math.max(2, Math.floor(numObstacles / 2)));
        for(let r=0; r<rows; r++) {
            for(let c=0; c<cols; c++) {
                const x = -halfSize/2 + 2 + c * 3;
                const z = -halfSize/2 + 2 + r * 3;
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
    ball.position.set(-halfSize/2 + 1.5, 0.25, halfSize/2 - 1.5);
    ball.castShadow = true;
    ball.userData = { w: 0.5, d: 0.5 };
    group.add(ball);
    walls.push(ball);

    // Add bucket for ball placement
    const bucketGeo = new THREE.CylinderGeometry(1.0, 1.2, 1.6, 32);
    const bucketMat = new THREE.MeshStandardMaterial({ color: 0xff4444, roughness: 0.8 });
    const bucket = new THREE.Mesh(bucketGeo, bucketMat);
    bucket.position.set(-halfSize/2 + 3.5, 0.8, halfSize/2 - 1.5);
    bucket.castShadow = true;
    bucket.receiveShadow = true;
    bucket.userData = { w: 1.2, d: 1.2, type: 'bucket' };
    group.add(bucket);
    walls.push(bucket);

    return { walls, group, target: null };
}
