class Tile {
    constructor(type, edges) {
        this.type = type;
        this.edges = edges; // [top, right, bottom, left]
    }
}

class Cell {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.collapsed = false;
        this.options = [];
    }
    
    get entropy() {
        return this.options.length;
    }
}

class WaveFunctionCollapse {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.grid = [];
        this.tiles = this.initializeTiles();
        this.totalCells = width * height;
        this.collapsedCount = 0;
        
        this.initializeGrid();
    }
    
    initializeTiles() {
        // Define tile types with their edge constraints
        // 0 = empty, 1 = wall
        return [
            new Tile('empty', [0, 0, 0, 0]),      // Empty space
            new Tile('wall-h', [0, 1, 0, 1]),    // Horizontal wall
            new Tile('wall-v', [1, 0, 1, 0]),    // Vertical wall
            new Tile('corner-tl', [0, 1, 1, 0]), // Top-left corner
            new Tile('corner-tr', [0, 0, 1, 1]), // Top-right corner
            new Tile('corner-bl', [1, 1, 0, 0]), // Bottom-left corner
            new Tile('corner-br', [1, 0, 0, 1]), // Bottom-right corner
            new Tile('junction-t', [0, 1, 1, 1]), // T junction (top)
            new Tile('junction-r', [1, 0, 1, 1]), // T junction (right)
            new Tile('junction-b', [1, 1, 0, 1]), // T junction (bottom)
            new Tile('junction-l', [1, 1, 1, 0]), // T junction (left)
            new Tile('cross', [1, 1, 1, 1]),     // Cross junction
        ];
    }
    
    initializeGrid() {
        this.grid = [];
        this.collapsedCount = 0;
        
        for (let y = 0; y < this.height; y++) {
            const row = [];
            for (let x = 0; x < this.width; x++) {
                const cell = new Cell(x, y);
                cell.options = [...Array(this.tiles.length).keys()];
                row.push(cell);
            }
            this.grid.push(row);
        }
    }
    
    getCell(x, y) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
            return null;
        }
        return this.grid[y][x];
    }
    
    getLowestEntropyCell() {
        let minEntropy = Infinity;
        let candidates = [];
        
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const cell = this.grid[y][x];
                if (!cell.collapsed && cell.entropy > 0) {
                    if (cell.entropy < minEntropy) {
                        minEntropy = cell.entropy;
                        candidates = [cell];
                    } else if (cell.entropy === minEntropy) {
                        candidates.push(cell);
                    }
                }
            }
        }
        
        if (candidates.length === 0) {
            return null;
        }
        
        // Return random cell from candidates
        return candidates[Math.floor(Math.random() * candidates.length)];
    }
    
    collapseCell(cell) {
        if (cell.collapsed || cell.options.length === 0) {
            return false;
        }
        
        // Choose random option from available
        const choice = cell.options[Math.floor(Math.random() * cell.options.length)];
        cell.options = [choice];
        cell.collapsed = true;
        this.collapsedCount++;
        
        return true;
    }
    
    propagate(startCell) {
        const stack = [startCell];
        
        while (stack.length > 0) {
            const current = stack.pop();
            const currentTileIndex = current.options[0];
            const currentTile = this.tiles[currentTileIndex];
            
            // Check all neighbors
            const neighbors = [
                { x: current.x, y: current.y - 1, edge: 0, oppositeEdge: 2 }, // Top
                { x: current.x + 1, y: current.y, edge: 1, oppositeEdge: 3 }, // Right
                { x: current.x, y: current.y + 1, edge: 2, oppositeEdge: 0 }, // Bottom
                { x: current.x - 1, y: current.y, edge: 3, oppositeEdge: 1 }, // Left
            ];
            
            for (const neighbor of neighbors) {
                const neighborCell = this.getCell(neighbor.x, neighbor.y);
                if (!neighborCell || neighborCell.collapsed) {
                    continue;
                }
                
                const validOptions = [];
                const requiredEdge = currentTile.edges[neighbor.edge];
                
                for (const option of neighborCell.options) {
                    const tile = this.tiles[option];
                    if (tile.edges[neighbor.oppositeEdge] === requiredEdge) {
                        validOptions.push(option);
                    }
                }
                
                if (validOptions.length < neighborCell.options.length) {
                    neighborCell.options = validOptions;
                    if (validOptions.length > 0) {
                        stack.push(neighborCell);
                    }
                }
            }
        }
    }
    
    step() {
        const cell = this.getLowestEntropyCell();
        if (!cell) {
            return false; // No more cells to collapse
        }
        
        if (this.collapseCell(cell)) {
            this.propagate(cell);
            return true;
        }
        
        return false;
    }
    
    generate() {
        while (this.step()) {
            // Continue until all cells are collapsed
        }
    }
    
    reset() {
        this.initializeGrid();
    }
    
    manualCollapse(x, y, tileIndex = null) {
        const cell = this.getCell(x, y);
        if (!cell || cell.collapsed) {
            return false;
        }
        
        if (tileIndex !== null && cell.options.includes(tileIndex)) {
            cell.options = [tileIndex];
        } else if (cell.options.length > 0) {
            // Use random valid option
            const choice = cell.options[Math.floor(Math.random() * cell.options.length)];
            cell.options = [choice];
        } else {
            return false;
        }
        
        cell.collapsed = true;
        this.collapsedCount++;
        this.propagate(cell);
        return true;
    }
    
    getTotalEntropy() {
        let totalEntropy = 0;
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const cell = this.grid[y][x];
                if (!cell.collapsed) {
                    totalEntropy += cell.entropy;
                }
            }
        }
        return totalEntropy;
    }
}