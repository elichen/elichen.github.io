async function init() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const ca = new CAModel();
    
    // Load the model
    await ca.loadModel();
    
    const [_, h, w, ch] = ca.state.shape;
    canvas.width = w;  // This will now be 2x the original tile width
    canvas.height = h;
    canvas.style.width = `${w * ca.scale}px`;
    canvas.style.height = `${h * ca.scale}px`;
    
    // Clear the entire canvas by applying damage everywhere
    for (let x = 0; x < w; x += 8) {
        for (let y = 0; y < h; y += 8) {
            ca.damage(x, y, 8);
        }
    }
    
    // Remove the automatic seed planting
    // const centerX1 = Math.floor(ca.tileSize / 2);  // Center of first tile
    // const centerX2 = Math.floor(ca.tileSize * 1.5);  // Center of second tile
    // const centerY = Math.floor(h / 2);
    // ca.plantSeed(centerX1, centerY);
    // ca.plantSeed(centerX2, centerY);
    
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
            // For white: RGB should be 1.0 (255), alpha should be 1.0
            const rgba = ca.state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
            // Just multiply by 255 to get white (1.0 -> 255)
            const img = rgba.mul(255);
            const rgbaBytes = new Uint8ClampedArray(img.dataSync());
            return new ImageData(rgbaBytes, w, h);
        });
        
        ctx.putImageData(imageData, 0, 0);
        requestAnimationFrame(render);
    }

    render();

    // Make the canvas visible after initialization
    canvas.classList.add('visible');
}

window.onload = init; 