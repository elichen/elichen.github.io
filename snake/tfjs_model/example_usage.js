// Load and use the Snake PPO model in TensorFlow.js

// Load the model
async function loadSnakeModel() {
    const model = await tf.loadLayersModel('./tfjs_model/model.json');
    return model;
}

// Predict action from game state
function predictAction(model, state) {
    // State should be a 24-element array with the features:
    // - Direction one-hot (4)
    // - Food direction (2)
    // - Danger detection (4)
    // - Distance to walls (4)
    // - Snake length (1)
    // - Grid fill ratio (1)
    // - Food distance (1)
    // - Body pattern (3)
    // - Connectivity features (4)

    const input = tf.tensor2d([state]);
    const prediction = model.predict(input);
    const action = prediction.argMax(-1).dataSync()[0];

    // Clean up tensors
    input.dispose();
    prediction.dispose();

    return action;  // 0: Up, 1: Right, 2: Down, 3: Left
}

// Example usage
async function playSnake() {
    const model = await loadSnakeModel();

    // Game loop
    while (!gameOver) {
        const state = getGameState();  // Your function to get current state
        const action = predictAction(model, state);
        executeAction(action);  // Your function to execute the action
    }
}

// Feature extraction example
function getGameState(snake, food, gridSize) {
    const state = [];

    // 1. Direction one-hot (4 features)
    const directionMap = {
        'up': [1, 0, 0, 0],
        'right': [0, 1, 0, 0],
        'down': [0, 0, 1, 0],
        'left': [0, 0, 0, 1]
    };
    state.push(...directionMap[snake.direction]);

    // 2. Food direction (2 features)
    const head = snake.body[0];
    state.push((food.x - head.x) / gridSize);
    state.push((food.y - head.y) / gridSize);

    // 3. Danger detection (4 features)
    const dangers = [
        isDanger(head.x, head.y - 1),  // Up
        isDanger(head.x + 1, head.y),  // Right
        isDanger(head.x, head.y + 1),  // Down
        isDanger(head.x - 1, head.y)   // Left
    ];
    state.push(...dangers.map(d => d ? 1 : 0));

    // 4. Distance to walls (4 features)
    state.push(head.y / gridSize);  // Top
        state.push((gridSize - 1 - head.x) / gridSize);  // Right
        state.push((gridSize - 1 - head.y) / gridSize);  // Bottom
        state.push(head.x / gridSize);  // Left

    // 5. Snake length (1 feature)
    state.push(snake.body.length / (gridSize * gridSize));

    // 6. Grid fill ratio (1 feature)
    state.push((snake.body.length - 1) / (gridSize * gridSize));

    // 7. Food distance (1 feature)
    const foodDist = Math.abs(food.x - head.x) + Math.abs(food.y - head.y);
    state.push(foodDist / (2 * gridSize));

    // 8. Body pattern (3 features) - simplified
    state.push(0, 0, 0);  // Placeholder for body patterns

    // 9. Connectivity features (4 features) - simplified
    state.push(0.5, 0.5, 0.5, 0.5);  // Placeholder for connectivity

    return state;
}
