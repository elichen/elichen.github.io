class Grid {
    constructor(width, height, depth) {
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.cells = new Uint8Array(width * height * depth);
        this.nextState = new Uint8Array(width * height * depth);
    }

    getIndex(x, y, z) {
        return x + this.width * (y + this.height * z);
    }

    getCellAge(x, y, z) {
        return this.cells[this.getIndex(x, y, z)];
    }

    setCell(x, y, z, value) {
        this.cells[this.getIndex(x, y, z)] = value;
    }

    isAlive(x, y, z) {
        return this.getCellAge(x, y, z) > 0;
    }

    getAliveCellCount() {
        return this.cells.reduce((count, cell) => count + (cell > 0 ? 1 : 0), 0);
    }

    randomize(probability) {
        for (let z = 0; z < this.depth; z++) {
            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    if (Math.random() < probability) {
                        this.setCell(x, y, z, 1);
                    }
                }
            }
        }
    }

    reset() {
        this.cells.fill(0);
        this.nextState.fill(0);
    }

    swapBuffers() {
        [this.cells, this.nextState] = [this.nextState, this.cells];
    }
} 