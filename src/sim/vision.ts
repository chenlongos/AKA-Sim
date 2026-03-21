import * as THREE from 'three';

export interface DetectedObject {
    type: 'ball' | 'redBucket';
    position: THREE.Vector3;
    distance: number;
    angle: number;
    confidence: number;
}

export interface VisionResult {
    ball: DetectedObject | null;
    redBucket: DetectedObject | null;
}

export function detectObjects(
    scene: THREE.Scene,
    robotPosition: THREE.Vector3,
    robotRotation: number
): VisionResult {
    const result: VisionResult = {
        ball: null,
        redBucket: null
    };

    // Detect ball (green sphere)
    scene.traverse((object) => {
        if (object instanceof THREE.Mesh && object.geometry.type === 'SphereGeometry') {
            const material = object.material as THREE.MeshStandardMaterial;
            const color = material.color;
            
            // Check if it's a green ball (approximately #ccff00)
            // R: 0.8, G: 1.0, B: 0.0
            if (color.g > 0.8 && color.r > 0.6 && color.b < 0.2) {
                const worldPos = new THREE.Vector3();
                object.getWorldPosition(worldPos);
                
                const relativePos = worldPos.clone().sub(robotPosition);
                const distance = relativePos.length();
                
                result.ball = {
                    type: 'ball',
                    position: worldPos,
                    distance,
                    angle: 0,
                    confidence: 1.0
                };
            }
        }
        
        // Detect red bucket (red cylinder)
        if (object instanceof THREE.Mesh && object.geometry.type === 'CylinderGeometry') {
            const material = object.material as THREE.MeshStandardMaterial;
            const color = material.color;
            
            // Check if it's a red bucket (approximately #ff4444)
            // R: 1.0, G: 0.267, B: 0.267
            if (color.r > 0.8 && color.g < 0.6 && color.b < 0.6) {
                const worldPos = new THREE.Vector3();
                object.getWorldPosition(worldPos);
                
                const relativePos = worldPos.clone().sub(robotPosition);
                const distance = relativePos.length();
                
                result.redBucket = {
                    type: 'redBucket',
                    position: worldPos,
                    distance,
                    angle: 0,
                    confidence: 1.0
                };
            }
        }
    });

    return result;
}

export function getClosestObject(
    objects: DetectedObject[],
    maxDistance: number = 10
): DetectedObject | null {
    if (objects.length === 0) return null;
    
    const validObjects = objects.filter(obj => obj.distance <= maxDistance);
    if (validObjects.length === 0) return null;
    
    return validObjects.reduce((closest, current) => 
        current.distance < closest.distance ? current : closest
    );
}
