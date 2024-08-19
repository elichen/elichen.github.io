const NUM_NEURONS = {
    cloud: 40,
    feedforward: { input: 10, hidden: [15, 15, 15], output: 10 },
    transformer: { embedding: 10, attention: 6, ffn: 8, output: 10 }
};
const ACTIVATION_THRESHOLD = 0.5;
const DECAY_RATE = 0.95;
const NEURON_RADIUS = 6;
const MIN_INTERVAL = 100;
const MAX_INTERVAL = 500;

let neurons = [];
let boardWidth, boardHeight;
let isAutoMode = false;
let autoTimeout;
let currentNetworkType = 'cloud';

function initializeNetwork() {
    const gameBoard = document.getElementById('game-board');
    gameBoard.innerHTML = '<svg id="connections"></svg>';
    const svg = document.getElementById('connections');
    neurons = [];

    boardWidth = gameBoard.clientWidth;
    boardHeight = gameBoard.clientHeight;

    svg.setAttribute('width', boardWidth);
    svg.setAttribute('height', boardHeight);

    switch(currentNetworkType) {
        case 'cloud':
            initializeCloudNetwork();
            break;
        case 'feedforward':
            initializeFeedforwardNetwork();
            break;
        case 'transformer':
            initializeTransformerNetwork();
            break;
    }

    drawConnections();
    createNeuronElements();
    updateDisplay();
}

function initializeCloudNetwork() {
    for (let i = 0; i < NUM_NEURONS.cloud; i++) {
        neurons.push({
            id: i,
            x: Math.random() * (boardWidth - 2 * NEURON_RADIUS) + NEURON_RADIUS,
            y: Math.random() * (boardHeight - 2 * NEURON_RADIUS) + NEURON_RADIUS,
            activation: 0,
            connections: []
        });
    }

    neurons.forEach(neuron => {
        const numConnections = Math.floor(Math.random() * 3) + 1;
        for (let i = 0; i < numConnections; i++) {
            let targetId;
            do {
                targetId = Math.floor(Math.random() * NUM_NEURONS.cloud);
            } while (targetId === neuron.id || neuron.connections.includes(targetId));
            neuron.connections.push(targetId);
        }
    });
}

function initializeFeedforwardNetwork() {
    let layerSizes = [NUM_NEURONS.feedforward.input, ...NUM_NEURONS.feedforward.hidden, NUM_NEURONS.feedforward.output];
    let neuronId = 0;
    let xStep = boardWidth / (layerSizes.length + 1);

    layerSizes.forEach((size, layerIndex) => {
        let yStep = boardHeight / (size + 1);
        for (let i = 0; i < size; i++) {
            neurons.push({
                id: neuronId++,
                x: xStep * (layerIndex + 1),
                y: yStep * (i + 1),
                activation: 0,
                connections: [],
                layer: layerIndex
            });
        }
    });

    // Connect layers from left to right
    for (let i = 0; i < neurons.length; i++) {
        if (neurons[i].layer < layerSizes.length - 1) { // Not an output neuron
            let nextLayerStart = layerSizes.slice(0, neurons[i].layer + 1).reduce((a, b) => a + b, 0);
            let nextLayerEnd = nextLayerStart + layerSizes[neurons[i].layer + 1];
            for (let j = nextLayerStart; j < nextLayerEnd; j++) {
                neurons[i].connections.push(j);
            }
        }
    }
}

function initializeTransformerNetwork() {
    // Simplified transformer decoder architecture
    let layers = [
        { name: 'embedding', size: NUM_NEURONS.transformer.embedding },
        { name: 'attention1', size: NUM_NEURONS.transformer.attention },
        { name: 'ffn1', size: NUM_NEURONS.transformer.ffn },
        { name: 'attention2', size: NUM_NEURONS.transformer.attention },
        { name: 'ffn2', size: NUM_NEURONS.transformer.ffn },
        { name: 'output', size: NUM_NEURONS.transformer.output }
    ];

    let neuronId = 0;
    let xStep = boardWidth / (layers.length + 1);

    layers.forEach((layer, layerIndex) => {
        let yStep = boardHeight / (layer.size + 1);
        for (let i = 0; i < layer.size; i++) {
            neurons.push({
                id: neuronId++,
                x: xStep * (layerIndex + 1),
                y: yStep * (i + 1),
                activation: 0,
                connections: [],
                layerType: layer.name,
                layer: layerIndex
            });
        }
    });

    // Connect layers from left to right
    for (let i = 0; i < layers.length - 1; i++) {
        let currentLayerStart = layers.slice(0, i).reduce((a, b) => a + b.size, 0);
        let currentLayerEnd = currentLayerStart + layers[i].size;
        let nextLayerStart = currentLayerEnd;
        let nextLayerEnd = nextLayerStart + layers[i + 1].size;

        for (let j = currentLayerStart; j < currentLayerEnd; j++) {
            for (let k = nextLayerStart; k < nextLayerEnd; k++) {
                neurons[j].connections.push(k);
            }
        }
    }

    // Add self-attention connections
    layers.forEach((layer, layerIndex) => {
        if (layer.name.includes('attention')) {
            let layerStart = layers.slice(0, layerIndex).reduce((a, b) => a + b.size, 0);
            let layerEnd = layerStart + layer.size;
            for (let i = layerStart; i < layerEnd; i++) {
                for (let j = layerStart; j < layerEnd; j++) {
                    if (i !== j) neurons[i].connections.push(j);
                }
            }
        }
    });
}

function drawConnections() {
    const svg = document.getElementById('connections');
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
}

function createNeuronElements() {
    const gameBoard = document.getElementById('game-board');
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
}

function activateNeuron(id) {
    const neuron = neurons[id];
    neuron.activation = 1;
    updateDisplay();
    setTimeout(() => propagateActivation(id, 1), 50); // Short delay before propagation starts
}

function propagateActivation(id, depth) {
    if (depth > 5) return; // Increased max depth for longer propagation chains
    const neuron = neurons[id];
    neuron.connections.forEach(targetId => {
        const target = neurons[targetId];
        const newActivation = Math.min(1, target.activation + (1 / (depth + 1)));
        if (newActivation > target.activation) {
            target.activation = newActivation;
            setTimeout(() => {
                updateDisplay();
                propagateActivation(targetId, depth + 1);
            }, 50); // Shorter delay for faster propagation
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
        let randomNeuronId;
        switch(currentNetworkType) {
            case 'cloud':
                randomNeuronId = Math.floor(Math.random() * neurons.length);
                break;
            case 'feedforward':
                // Only activate input layer neurons
                randomNeuronId = Math.floor(Math.random() * NUM_NEURONS.feedforward.input);
                break;
            case 'transformer':
                // Only activate embedding layer neurons
                randomNeuronId = Math.floor(Math.random() * NUM_NEURONS.transformer.embedding);
                break;
        }
        activateNeuron(randomNeuronId);
        const randomInterval = Math.random() * (MAX_INTERVAL - MIN_INTERVAL) + MIN_INTERVAL;
        autoTimeout = setTimeout(activateRandomNeuron, randomInterval);
    }
    activateRandomNeuron();
}

function stopAutoMode() {
    clearTimeout(autoTimeout);
}

function changeNetworkType(type) {
    currentNetworkType = type;
    stopAutoMode();
    isAutoMode = false;
    document.getElementById('toggle-mode').textContent = 'Switch to Automatic';
    initializeNetwork();
    updateInstructions();
}

function updateInstructions() {
    document.getElementById('cloud-instructions').style.display = currentNetworkType === 'cloud' ? 'block' : 'none';
    document.getElementById('feedforward-instructions').style.display = currentNetworkType === 'feedforward' ? 'block' : 'none';
    document.getElementById('transformer-instructions').style.display = currentNetworkType === 'transformer' ? 'block' : 'none';
}

// Event Listeners
window.addEventListener('load', () => {
    initializeNetwork();
    document.getElementById('toggle-mode').addEventListener('click', toggleMode);
    document.getElementById('reset-network').addEventListener('click', () => {
        stopAutoMode();
        isAutoMode = false;
        document.getElementById('toggle-mode').textContent = 'Switch to Automatic';
        initializeNetwork();
    });
    document.getElementById('network-type').addEventListener('change', (e) => {
        changeNetworkType(e.target.value);
    });
});

window.addEventListener('resize', initializeNetwork);

// Set up decay interval
setInterval(decayActivations, 100);