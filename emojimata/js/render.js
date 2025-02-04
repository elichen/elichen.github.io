async function init() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const ca = new CAModel();
    
    // Set canvas to visible immediately
    canvas.style.display = 'block';
    
    // Load the model
    await ca.loadModel();
    
    const [_, h, w, ch] = ca.state.shape;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w * ca.scale}px`;
    canvas.style.height = `${h * ca.scale}px`;
    
    // Clear the entire canvas by applying damage everywhere
    for (let x = 0; x < w; x += 8) {
        for (let y = 0; y < h; y += 8) {
            ca.damage(x, y, 8);
        }
    }
    
    // Plant seeds immediately after clearing
    const quadrants = ['topLeft', 'topRight', 'bottomLeft', 'bottomRight'];
    quadrants.forEach(quadrant => {
        const pos = getRandomQuadrantPosition(quadrant);
        ca.plantSeed(pos.x, pos.y);
    });
    
    // Start rendering immediately
    render();
    
    // Function to get random position within a quadrant
    function getRandomQuadrantPosition(quadrant) {
        const tileSize = ca.tileSize;
        const padding = 20; // Keep seeds away from edges
        
        // Define quadrant boundaries
        const quadrants = {
            topLeft: { 
                x: [padding, tileSize - padding],
                y: [padding, tileSize - padding]
            },
            topRight: {
                x: [tileSize + padding, 2 * tileSize - padding],
                y: [padding, tileSize - padding]
            },
            bottomLeft: {
                x: [padding, tileSize - padding],
                y: [tileSize + padding, 2 * tileSize - padding]
            },
            bottomRight: {
                x: [tileSize + padding, 2 * tileSize - padding],
                y: [tileSize + padding, 2 * tileSize - padding]
            }
        };
        
        const bounds = quadrants[quadrant];
        const x = Math.floor(Math.random() * (bounds.x[1] - bounds.x[0])) + bounds.x[0];
        const y = Math.floor(Math.random() * (bounds.y[1] - bounds.y[0])) + bounds.y[0];
        return { x, y };
    }
    
    canvas.onmousedown = e => {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / ca.scale);
        const y = Math.floor((e.clientY - rect.top) / ca.scale);
        if (e.buttons == 1) {
            if (e.shiftKey) {
                ca.plantSeed(x, y);
            } else {
                ca.damage(x, y, 8);
            }
        }
    };
    
    canvas.onmousemove = e => {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / ca.scale);
        const y = Math.floor((e.clientY - rect.top) / ca.scale);
        if (e.buttons == 1 && !e.shiftKey) {
            ca.damage(x, y, 8);
        }
    };

    function render() {
        ca.step();

        const imageData = tf.tidy(() => {
            const rgba = ca.state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
            const img = rgba.mul(255);
            const rgbaBytes = new Uint8ClampedArray(img.dataSync());
            return new ImageData(rgbaBytes, w, h);
        });
        
        ctx.putImageData(imageData, 0, 0);
        requestAnimationFrame(render);
    }
}

window.onload = init; 