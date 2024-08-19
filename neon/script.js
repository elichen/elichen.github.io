const NUM_NEURONS = 40;
const ACTIVATION_THRESHOLD = 0.5;
const DECAY_RATE = 0.95;
const NEURON_RADIUS = 6; // pixels
const MIN_INTERVAL = 500; // Minimum time between activations (ms)
const MAX_INTERVAL = 3000; // Maximum time between activations (ms)

let neurons = [];
let boardWidth, boardHeight;
let isAutoMode = false;
let autoTimeout;

function initializeNetwork() {
    const gameBoard = document.getElementById('game-board');
    gameBoard.innerHTML = '<svg id="connections"></svg>';
    const svg = document.getElementById('connections');
    neurons = [];

    boardWidth = gameBoard.clientWidth;
    boardHeight = gameBoard.clientHeight;

    svg.setAttribute('width', boardWidth);
    svg.setAttribute('height', boardHeight);

    for (let i = 0; i < NUM_NEURONS; i++) {
        const neuron = {
            id: i,
            x: Math.random() * (boardWidth - 2 * NEURON_RADIUS) + NEURON_RADIUS,
            y: Math.random() * (boardHeight - 2 * NEURON_RADIUS) + NEURON_RADIUS,
            activation: 0,
            connections: []
        };
        neurons.push(neuron);
    }

    // Generate connections
    neurons.forEach(neuron => {
        const numConnections = Math.floor(Math.random() * 3) + 1;
        for (let i = 0; i < numConnections; i++) {
            let targetId;
            do {
                targetId = Math.floor(Math.random() * NUM_NEURONS);
            } while (targetId === neuron.id || neuron.connections.includes(targetId));
            neuron.connections.push(targetId);
        }
    });

    // Draw connections
    neurons.forEach(neuron => {
        neuron.connections.forEach(targetId => {
            const target = neurons[targetId];
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', neuron.x);
            line.setAttribute('y1', neuron.y);
            line.setAttribute('x2', target.x);
            line.setAttribute('y2', target.y);
            line.setAttribute('stroke', 'rgba(52, 211, 153, 0.2)');
            line.setAttribute('stroke-width', '2');
            svg.appendChild(line);
        });
    });

    // Create neuron elements
    neurons.forEach(neuron => {
        const neuronElement = document.createElement('div');
        neuronElement.className = 'neuron';
        neuronElement.style.left = `${neuron.x}px`;
        neuronElement.style.top = `${neuron.y}px`;
        neuronElement.addEventListener('click', () => {
            if (!isAutoMode) activateNeuron(neuron.id);
        });
        gameBoard.appendChild(neuronElement);
    });

    updateDisplay();
}

function activateNeuron(id) {
    const neuron = neurons[id];
    neuron.activation = 1;
    propagateActivation(id, 1);
    updateDisplay();
}

function propagateActivation(id, depth) {
    if (depth > 3) return;
    const neuron = neurons[id];
    neuron.connections.forEach(targetId => {
        const target = neurons[targetId];
        const newActivation = Math.min(1, target.activation + (1 / (depth + 1)));
        if (newActivation > target.activation) {
            target.activation = newActivation;
            propagateActivation(targetId, depth + 1);
        }
    });
}

function updateDisplay() {
    neurons.forEach((neuron, index) => {
        const neuronElement = document.querySelectorAll('.neuron')[index];
        neuronElement.style.transform = `translate(-50%, -50%) scale(${1 + neuron.activation * 0.5})`;
        neuronElement.style.backgroundColor = `rgba(52, 211, 153, ${0.3 + neuron.activation * 0.7})`;
        neuronElement.style.boxShadow = `0 0 ${10 * neuron.activation}px ${5 * neuron.activation}px rgba(52, 211, 153, ${neuron.activation})`;
    });
}

function decayActivations() {
    neurons.forEach(neuron => {
        neuron.activation *= DECAY_RATE;
        if (neuron.activation < 0.01) neuron.activation = 0;
    });
    updateDisplay();
}

function toggleMode() {
    isAutoMode = !isAutoMode;
    const toggleButton = document.getElementById('toggle-mode');
    toggleButton.textContent = isAutoMode ? 'Switch to Manual' : 'Switch to Automatic';
    
    if (isAutoMode) {
        startAutoMode();
    } else {
        stopAutoMode();
    }
}

function startAutoMode() {
    function activateRandomNeuron() {
        const randomNeuronId = Math.floor(Math.random() * neurons.length);
        activateNeuron(randomNeuronId);
        const randomInterval = Math.random() * (MAX_INTERVAL - MIN_INTERVAL) + MIN_INTERVAL;
        autoTimeout = setTimeout(activateRandomNeuron, randomInterval);
    }
    activateRandomNeuron();
}

function stopAutoMode() {
    clearTimeout(autoTimeout);
}

// Initialize network
window.addEventListener('load', () => {
    initializeNetwork();
    const toggleButton = document.getElementById('toggle-mode');
    toggleButton.addEventListener('click', toggleMode);
});

window.addEventListener('resize', initializeNetwork);

// Set up decay interval
setInterval(decayActivations, 100);

// Reset network button
document.getElementById('reset-network').addEventListener('click', () => {
    stopAutoMode();
    isAutoMode = false;
    document.getElementById('toggle-mode').textContent = 'Switch to Automatic';
    initializeNetwork();
});