class GameLogic {
    constructor(grid) {
        this.grid = grid;
    }

    countNeighbors(x, y, z) {
        let count = 0;
        for (let dz = -1; dz <= 1; dz++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (dx === 0 && dy === 0 && dz === 0) continue;

                    const nx = (x + dx + this.grid.width) % this.grid.width;
                    const ny = (y + dy + this.grid.height) % this.grid.height;
                    const nz = (z + dz + this.grid.depth) % this.grid.depth;

                    if (this.grid.isAlive(nx, ny, nz)) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    update() {
        for (let z = 0; z < this.grid.depth; z++) {
            for (let y = 0; y < this.grid.height; y++) {
                for (let x = 0; x < this.grid.width; x++) {
                    const neighbors = this.countNeighbors(x, y, z);
                    const currentAge = this.grid.getCellAge(x, y, z);
                    const idx = this.grid.getIndex(x, y, z);

                    if (currentAge > 0) { // Cell is alive
                        // Survival rule: 5-8 neighbors (coral-like growth)
                        if (neighbors >= 5 && neighbors <= 8) {
                            // Age normally
                            this.grid.nextState[idx] = currentAge + 1;
                        } else {
                            // Die if outside survival range
                            this.grid.nextState[idx] = Math.max(0, currentAge - 5);
                        }
                    } else { // Cell is dead
                        // Birth rules: 6-7 or 9 or 12 neighbors (branching points)
                        if ((neighbors >= 6 && neighbors <= 7) || 
                            neighbors === 9 || 
                            neighbors === 12) {
                            // Birth with 30% chance to prevent overcrowding
                            if (Math.random() < 0.3) {
                                this.grid.nextState[idx] = 1;
                            } else {
                                this.grid.nextState[idx] = 0;
                            }
                        } else {
                            this.grid.nextState[idx] = 0;
                        }
                    }
                }
            }
        }

        this.grid.swapBuffers();
    }
} 