import * as THREE from 'three';

export interface PathNode {
    x: number;
    z: number;
    gCost: number;
    hCost: number;
    fCost: number;
    parent: PathNode | null;
    walkable: boolean;
}

export interface PathfindingConfig {
    gridSize: number;
    cellSize: number;
    maxIterations: number;
}

const defaultConfig: PathfindingConfig = {
    gridSize: 40,
    cellSize: 0.5,
    maxIterations: 1000
};

export class PathFinder {
    private grid: PathNode[][];
    private obstacles: THREE.Vector3[] = [];
    private config: PathfindingConfig;

    constructor(config: Partial<PathfindingConfig> = {}) {
        this.config = { ...defaultConfig, ...config };
        this.grid = this.createGrid();
    }

    private createGrid(): PathNode[][] {
        const grid: PathNode[][] = [];
        const halfSize = (this.config.gridSize * this.config.cellSize) / 2;

        for (let x = 0; x < this.config.gridSize; x++) {
            grid[x] = [];
            for (let z = 0; z < this.config.gridSize; z++) {
                grid[x][z] = {
                    x: x * this.config.cellSize - halfSize,
                    z: z * this.config.cellSize - halfSize,
                    gCost: 0,
                    hCost: 0,
                    fCost: 0,
                    parent: null,
                    walkable: true
                };
            }
        }

        return grid;
    }

    updateObstacles(walls: THREE.Mesh[], robotRadius: number = 0.5): void {
        // Reset grid
        for (let x = 0; x < this.config.gridSize; x++) {
            for (let z = 0; z < this.config.gridSize; z++) {
                this.grid[x][z].walkable = true;
            }
        }

        // Mark obstacles
        walls.forEach(wall => {
            const w = wall.userData.w || 1;
            const d = wall.userData.d || 1;
            const halfW = w / 2 + robotRadius;
            const halfD = d / 2 + robotRadius;

            for (let x = 0; x < this.config.gridSize; x++) {
                for (let z = 0; z < this.config.gridSize; z++) {
                    const node = this.grid[x][z];
                    if (node.x > wall.position.x - halfW &&
                        node.x < wall.position.x + halfW &&
                        node.z > wall.position.z - halfD &&
                        node.z < wall.position.z + halfD) {
                        node.walkable = false;
                    }
                }
            }
        });
    }

    findPath(start: THREE.Vector3, end: THREE.Vector3): THREE.Vector3[] {
        const startNode = this.getNodeFromWorldPosition(start);
        const endNode = this.getNodeFromWorldPosition(end);

        if (!startNode || !endNode || !startNode.walkable || !endNode.walkable) {
            return [];
        }

        const openSet: PathNode[] = [];
        const closedSet = new Set<PathNode>();

        openSet.push(startNode);

        let iterations = 0;

        while (openSet.length > 0 && iterations < this.config.maxIterations) {
            iterations++;

            // Get node with lowest fCost
            let currentNode = openSet.reduce((min, node) => 
                node.fCost < min.fCost ? node : min
            );

            // Remove from open set
            const index = openSet.indexOf(currentNode);
            openSet.splice(index, 1);
            closedSet.add(currentNode);

            // Reached destination
            if (currentNode === endNode) {
                return this.retracePath(currentNode);
            }

            // Check neighbors
            const neighbors = this.getNeighbors(currentNode);

            for (const neighbor of neighbors) {
                if (!neighbor.walkable || closedSet.has(neighbor)) {
                    continue;
                }

                const newGCost = currentNode.gCost + this.getDistance(currentNode, neighbor);

                if (newGCost < neighbor.gCost || !openSet.includes(neighbor)) {
                    neighbor.gCost = newGCost;
                    neighbor.hCost = this.getDistance(neighbor, endNode);
                    neighbor.fCost = neighbor.gCost + neighbor.hCost;
                    neighbor.parent = currentNode;

                    if (!openSet.includes(neighbor)) {
                        openSet.push(neighbor);
                    }
                }
            }
        }

        // No path found
        return [];
    }

    private getNodeFromWorldPosition(position: THREE.Vector3): PathNode | null {
        const halfSize = (this.config.gridSize * this.config.cellSize) / 2;
        
        const x = Math.floor((position.x + halfSize) / this.config.cellSize);
        const z = Math.floor((position.z + halfSize) / this.config.cellSize);

        if (x >= 0 && x < this.config.gridSize && z >= 0 && z < this.config.gridSize) {
            return this.grid[x][z];
        }

        return null;
    }

    private getNeighbors(node: PathNode): PathNode[] {
        const neighbors: PathNode[] = [];
        const halfSize = this.config.gridSize;
        
        const indices = [
            [1, 0], [-1, 0], [0, 1], [0, -1], // Cardinal directions
            [1, 1], [1, -1], [-1, 1], [-1, -1] // Diagonals
        ];

        const nodeX = Math.floor((node.x + halfSize * this.config.cellSize) / this.config.cellSize);
        const nodeZ = Math.floor((node.z + halfSize * this.config.cellSize) / this.config.cellSize);

        for (const [dx, dz] of indices) {
            const newX = nodeX + dx;
            const newZ = nodeZ + dz;

            if (newX >= 0 && newX < this.config.gridSize && newZ >= 0 && newZ < this.config.gridSize) {
                neighbors.push(this.grid[newX][newZ]);
            }
        }

        return neighbors;
    }

    private getDistance(a: PathNode, b: PathNode): number {
        const distX = Math.abs(a.x - b.x);
        const distZ = Math.abs(a.z - b.z);

        if (distX > distZ) {
            return 14 * distZ + 10 * (distX - distZ); // Diagonal distance
        }
        return 14 * distX + 10 * (distZ - distX);
    }

    private retracePath(endNode: PathNode): THREE.Vector3[] {
        const path: THREE.Vector3[] = [];
        let currentNode: PathNode | null = endNode;

        while (currentNode) {
            path.push(new THREE.Vector3(currentNode.x, 0, currentNode.z));
            currentNode = currentNode.parent;
        }

        path.reverse();
        return path;
    }

    // Simple direct path for open spaces
    findDirectPath(start: THREE.Vector3, end: THREE.Vector3): THREE.Vector3[] {
        return [start, end];
    }

    // Get next waypoint from current position
    getNextWaypoint(path: THREE.Vector3[], currentPosition: THREE.Vector3, threshold: number = 0.5): THREE.Vector3 | null {
        if (path.length <= 1) return null;

        // Find the first waypoint that's not yet reached
        for (let i = 1; i < path.length; i++) {
            const waypoint = path[i];
            const distance = currentPosition.distanceTo(waypoint);
            
            if (distance > threshold) {
                return waypoint;
            }
        }

        return null;
    }
}
