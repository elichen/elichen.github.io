// Environment simulation

let environment;
let animationId;
let currentAction = 0.0; // Continuous value between -1 and 1 (left to right)
let aiController;
let aiActive = false;

async function initializeApp() {
    environment = new StickBalancingEnv();

    // Start with a random state
    environment.reset();

    // Initialize AI controller
    aiController = new AIController();

    // Setup controls
    setupKeyboardControls();
    setupAIControls();

    // Try to load AI model
    loadAIModel();

    // Start animation loop
    animate();
}

async function loadAIModel() {
    try {
        const success = await aiController.loadModel();
        if (success) {
            // Enable AI by default once loaded
            aiActive = true;
            aiController.setEnabled(true);
            console.log('AI model loaded and enabled');
        } else {
            console.warn('AI model not found');
        }
    } catch (error) {
        console.error('Failed to load AI model:', error);
    }
}

function setupAIControls() {
    const toggleButton = document.getElementById('aiToggle');

    // Button starts as "Stop AI" since AI is enabled by default
    toggleButton.classList.add('active');

    toggleButton.addEventListener('click', () => {
        if (!aiController.isLoaded) {
            console.warn('AI model not loaded yet');
            return;
        }

        aiActive = !aiActive;
        aiController.setEnabled(aiActive);

        if (aiActive) {
            toggleButton.textContent = 'Stop AI';
            toggleButton.classList.add('active');
        } else {
            toggleButton.textContent = 'Start AI';
            toggleButton.classList.remove('active');
        }
    });
}

function setupKeyboardControls() {
    // Make keysPressed globally accessible for override logic
    window.keysPressed = new Set();

    document.addEventListener('keydown', (e) => {
        if (e.repeat) return; // Ignore key repeat

        switch(e.key) {
            case 'ArrowLeft':
            case 'a':
            case 'A':
                window.keysPressed.add('left');
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                window.keysPressed.add('right');
                break;
            case ' ':
            case 'r':
            case 'R':
                environment.reset();
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        switch(e.key) {
            case 'ArrowLeft':
            case 'a':
            case 'A':
                window.keysPressed.delete('left');
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                window.keysPressed.delete('right');
                break;
        }
    });
}

function getKeyboardAction() {
    // Check if any movement keys are pressed
    if (window.keysPressed.has('left') && !window.keysPressed.has('right')) {
        return -1.0; // Full left
    } else if (window.keysPressed.has('right') && !window.keysPressed.has('left')) {
        return 1.0; // Full right
    } else {
        return null; // No keyboard input
    }
}

async function animate() {
    // Get current state for AI
    const state = environment.getState();

    // Priority: Keyboard input always overrides AI
    const keyboardAction = getKeyboardAction();

    if (keyboardAction !== null) {
        // User is pressing keys - use keyboard input
        currentAction = keyboardAction;
    } else if (aiActive && aiController.isActive()) {
        // No keyboard input and AI is active - use AI
        currentAction = await aiController.getAction(state);
    } else {
        // No keyboard input and AI not active - stop
        currentAction = 0.0;
    }

    // Step the environment with current action
    const [newState, reward, done] = environment.step(currentAction);

    // Draw the current state
    drawEnvironment();

    // Continue animation
    animationId = requestAnimationFrame(animate);
}

// Expose environment globally
window.environment = environment;

// Initialize when window loads
window.onload = initializeApp;