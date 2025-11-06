// Environment simulation

let environment;
let animationId;
let currentAction = 1; // 0=left, 1=none, 2=right

function initializeApp() {
    environment = new StickBalancingEnv();

    // Start with a random state
    environment.reset();

    // Setup keyboard controls
    setupKeyboardControls();

    // Start animation loop
    animate();
}

function setupKeyboardControls() {
    const keysPressed = new Set();

    document.addEventListener('keydown', (e) => {
        if (e.repeat) return; // Ignore key repeat

        switch(e.key) {
            case 'ArrowLeft':
            case 'a':
                keysPressed.add('left');
                break;
            case 'ArrowRight':
            case 'd':
                keysPressed.add('right');
                break;
            case ' ':
            case 'r':
                environment.reset();
                break;
        }

        // Update action based on currently pressed keys
        if (keysPressed.has('left') && !keysPressed.has('right')) {
            currentAction = 0;
        } else if (keysPressed.has('right') && !keysPressed.has('left')) {
            currentAction = 2;
        } else {
            currentAction = 1; // Both or neither pressed
        }
    });

    document.addEventListener('keyup', (e) => {
        switch(e.key) {
            case 'ArrowLeft':
            case 'a':
                keysPressed.delete('left');
                break;
            case 'ArrowRight':
            case 'd':
                keysPressed.delete('right');
                break;
        }

        // Update action based on currently pressed keys
        if (keysPressed.has('left') && !keysPressed.has('right')) {
            currentAction = 0;
        } else if (keysPressed.has('right') && !keysPressed.has('left')) {
            currentAction = 2;
        } else {
            currentAction = 1; // Both or neither pressed
        }
    });
}

function animate() {
    // Step the environment with current action
    const [state, reward, done] = environment.step(currentAction);

    // Draw the current state
    drawEnvironment();

    // Continue animation
    animationId = requestAnimationFrame(animate);
}

// Expose environment globally
window.environment = environment;

// Initialize when window loads
window.onload = initializeApp;