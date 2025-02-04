async function init() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const ca = new CAModel();
    const seedSlider = document.getElementById('seedCount');
    const seedValue = document.getElementById('seedValue');
    let isRunning = false;
    let animationFrame = null;
    
    // Function to handle resize
    function handleResize() {
        // Recalculate tiles
        ca.calculateTiles();
        
        // Resize canvas to match the state dimensions
        const stateWidth = ca.tileSize * ca.numTilesX;
        const stateHeight = ca.tileSize * ca.numTilesY;
        
        canvas.width = stateWidth;
        canvas.height = stateHeight;
        
        // Scale up for display
        canvas.style.width = `${stateWidth * ca.scale}px`;
        canvas.style.height = `${stateHeight * ca.scale}px`;
        
        // Reinitialize state with new dimensions
        ca.initState();
        
        // Plant new seeds based on slider value
        const totalSeeds = parseInt(seedSlider.value);
        for (let i = 0; i < totalSeeds; i++) {
            const x = Math.floor(Math.random() * stateWidth);
            const y = Math.floor(Math.random() * stateHeight);
            ca.plantSeed(x, y);
        }
    }
    
    // Debounce helper
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Handle slider changes
    seedSlider.addEventListener('input', (e) => {
        seedValue.textContent = e.target.value;
    });
    
    seedSlider.addEventListener('change', () => {
        if (isRunning) {
            cancelAnimationFrame(animationFrame);
        }
        handleResize();
        render();
    });
    
    // Add resize listener
    window.addEventListener('resize', debounce(handleResize, 250));
    
    // Set canvas to visible immediately
    canvas.style.display = 'block';
    
    // Initial setup
    await ca.loadModel();
    handleResize();
    
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
        isRunning = true;
        ca.step();

        const imageData = tf.tidy(() => {
            const [_, h, w, ch] = ca.state.shape;
            const rgba = ca.state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
            const img = rgba.mul(255);
            const rgbaBytes = new Uint8ClampedArray(img.dataSync());
            return new ImageData(rgbaBytes, w, h);
        });
        
        ctx.putImageData(imageData, 0, 0);
        animationFrame = requestAnimationFrame(render);
    }
    
    render();
}

window.onload = init; 