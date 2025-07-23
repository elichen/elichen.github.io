// Heap's Algorithm Implementation and Interactive Visualization

// Global state
let animationState = {
    isPlaying: false,
    currentStep: 0,
    permutations: [],
    swapHistory: [],
    speed: 500,
    timeoutId: null,
    currentArray: [],
    originalArray: ['A', 'B', 'C', 'D']
};

// DOM Elements
const playBtn = document.getElementById('play-btn');
const pauseBtn = document.getElementById('pause-btn');
const stepBtn = document.getElementById('step-btn');
const resetBtn = document.getElementById('reset-btn');
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');
const currentPermDiv = document.getElementById('current-perm');
const swapIndicator = document.getElementById('swap-indicator');
const stepCount = document.getElementById('step-count');
const currentK = document.getElementById('current-k');
const swapType = document.getElementById('swap-type');
const historyList = document.getElementById('history-list');
const permutationDemo = document.querySelector('.permutation-list');

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    initializeDemo();
    initializeVisualization();
    setupEventListeners();
    setupTabs();
    drawRecursiveTree();
    drawPerformanceChart();
});

// Initialize the permutation demo
function initializeDemo() {
    const demoArray = ['A', 'B', 'C'];
    const perms = heapPermutation(demoArray);
    
    perms.forEach((perm, index) => {
        const permDiv = document.createElement('div');
        permDiv.className = 'perm-item';
        permDiv.textContent = perm.join('');
        permutationDemo.appendChild(permDiv);
        
        // Animate appearance
        setTimeout(() => {
            permDiv.classList.add('highlight');
            setTimeout(() => permDiv.classList.remove('highlight'), 500);
        }, index * 100);
    });
}

// Heap's Algorithm - Recursive Implementation
function heapPermutation(array) {
    const result = [];
    const swaps = [];
    
    function generate(k, arr) {
        if (k === 1) {
            result.push([...arr]);
            return;
        }
        
        generate(k - 1, arr);
        
        for (let i = 0; i < k - 1; i++) {
            let swapIndices;
            if (k % 2 === 0) {
                swapIndices = [i, k - 1];
                [arr[i], arr[k - 1]] = [arr[k - 1], arr[i]];
            } else {
                swapIndices = [0, k - 1];
                [arr[0], arr[k - 1]] = [arr[k - 1], arr[0]];
            }
            swaps.push({
                indices: swapIndices,
                k: k,
                type: k % 2 === 0 ? 'even' : 'odd',
                result: [...arr]
            });
            generate(k - 1, arr);
        }
    }
    
    generate(array.length, array);
    animationState.swapHistory = swaps;
    return result;
}

// Initialize visualization
function initializeVisualization() {
    animationState.currentArray = [...animationState.originalArray];
    animationState.permutations = heapPermutation([...animationState.originalArray]);
    animationState.currentStep = 0;
    
    updatePermutationDisplay();
    updateStateDisplay();
    clearHistory();
    
    // Add initial permutation to history
    addToHistory(animationState.currentArray);
}

// Update permutation display
function updatePermutationDisplay() {
    currentPermDiv.innerHTML = '';
    
    animationState.currentArray.forEach((element, index) => {
        const elemDiv = document.createElement('div');
        elemDiv.className = 'perm-element';
        elemDiv.textContent = element;
        elemDiv.setAttribute('data-index', index);
        currentPermDiv.appendChild(elemDiv);
    });
}

// Animate swap
function animateSwap(indices) {
    const elements = currentPermDiv.querySelectorAll('.perm-element');
    
    indices.forEach(index => {
        elements[index].classList.add('swapping');
    });
    
    swapIndicator.textContent = `Swapping positions ${indices[0]} and ${indices[1]}`;
    
    setTimeout(() => {
        indices.forEach(index => {
            elements[index].classList.remove('swapping');
        });
    }, animationState.speed * 0.8);
}

// Update state display
function updateStateDisplay() {
    stepCount.textContent = animationState.currentStep;
    
    if (animationState.currentStep > 0 && animationState.currentStep <= animationState.swapHistory.length) {
        const swap = animationState.swapHistory[animationState.currentStep - 1];
        currentK.textContent = swap.k;
        swapType.textContent = swap.type === 'even' ? 'Even (sequential)' : 'Odd (first element)';
    } else {
        currentK.textContent = '-';
        swapType.textContent = '-';
        swapIndicator.textContent = '';
    }
}

// Add permutation to history
function addToHistory(perm) {
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item new';
    historyItem.textContent = perm.join('');
    historyList.appendChild(historyItem);
    
    setTimeout(() => {
        historyItem.classList.remove('new');
    }, 500);
    
    // Auto-scroll to bottom
    historyList.scrollTop = historyList.scrollHeight;
}

// Clear history
function clearHistory() {
    historyList.innerHTML = '';
}

// Step through algorithm
function step() {
    if (animationState.currentStep >= animationState.swapHistory.length) {
        pause();
        return;
    }
    
    const swap = animationState.swapHistory[animationState.currentStep];
    
    // Animate the swap
    animateSwap(swap.indices);
    
    // Perform the swap
    setTimeout(() => {
        animationState.currentArray = [...swap.result];
        updatePermutationDisplay();
        addToHistory(animationState.currentArray);
        animationState.currentStep++;
        updateStateDisplay();
        
        if (animationState.isPlaying && animationState.currentStep < animationState.swapHistory.length) {
            animationState.timeoutId = setTimeout(step, animationState.speed);
        } else if (animationState.currentStep >= animationState.swapHistory.length) {
            pause();
        }
    }, animationState.speed * 0.8);
}

// Play animation
function play() {
    animationState.isPlaying = true;
    playBtn.style.display = 'none';
    pauseBtn.style.display = 'inline-block';
    
    if (animationState.currentStep >= animationState.swapHistory.length) {
        reset();
    }
    
    step();
}

// Pause animation
function pause() {
    animationState.isPlaying = false;
    playBtn.style.display = 'inline-block';
    pauseBtn.style.display = 'none';
    
    if (animationState.timeoutId) {
        clearTimeout(animationState.timeoutId);
    }
}

// Reset visualization
function reset() {
    pause();
    animationState.currentStep = 0;
    animationState.currentArray = [...animationState.originalArray];
    updatePermutationDisplay();
    updateStateDisplay();
    clearHistory();
    addToHistory(animationState.currentArray);
}

// Setup event listeners
function setupEventListeners() {
    playBtn.addEventListener('click', play);
    pauseBtn.addEventListener('click', pause);
    stepBtn.addEventListener('click', () => {
        if (!animationState.isPlaying) {
            step();
        }
    });
    resetBtn.addEventListener('click', reset);
    
    speedSlider.addEventListener('input', (e) => {
        animationState.speed = parseInt(e.target.value);
        speedValue.textContent = animationState.speed;
    });
}

// Setup tabs
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Update active states
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            button.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

// Draw recursive tree visualization
function drawRecursiveTree() {
    const treeContainer = document.getElementById('tree-vis');
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '800');
    svg.setAttribute('height', '400');
    svg.style.width = '100%';
    svg.style.height = 'auto';
    
    // Simple tree structure for ['A', 'B', 'C']
    const nodes = [
        {id: 'root', label: 'generate(3, [A,B,C])', x: 400, y: 50, level: 0},
        {id: 'n1', label: 'generate(2, [A,B,C])', x: 200, y: 150, level: 1},
        {id: 'n2', label: 'generate(2, [B,A,C])', x: 400, y: 150, level: 1},
        {id: 'n3', label: 'generate(2, [C,B,A])', x: 600, y: 150, level: 1},
        {id: 'l1', label: '[A,B,C]', x: 100, y: 250, level: 2, leaf: true},
        {id: 'l2', label: '[B,A,C]', x: 200, y: 250, level: 2, leaf: true},
        {id: 'l3', label: '[B,A,C]', x: 300, y: 250, level: 2, leaf: true},
        {id: 'l4', label: '[A,B,C]', x: 400, y: 250, level: 2, leaf: true},
        {id: 'l5', label: '[C,B,A]', x: 500, y: 250, level: 2, leaf: true},
        {id: 'l6', label: '[B,C,A]', x: 600, y: 250, level: 2, leaf: true}
    ];
    
    const edges = [
        {from: 'root', to: 'n1'},
        {from: 'root', to: 'n2'},
        {from: 'root', to: 'n3'},
        {from: 'n1', to: 'l1'},
        {from: 'n1', to: 'l2'},
        {from: 'n2', to: 'l3'},
        {from: 'n2', to: 'l4'},
        {from: 'n3', to: 'l5'},
        {from: 'n3', to: 'l6'}
    ];
    
    // Draw edges
    edges.forEach(edge => {
        const fromNode = nodes.find(n => n.id === edge.from);
        const toNode = nodes.find(n => n.id === edge.to);
        
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', fromNode.x);
        line.setAttribute('y1', fromNode.y);
        line.setAttribute('x2', toNode.x);
        line.setAttribute('y2', toNode.y);
        line.setAttribute('stroke', '#e5e7eb');
        line.setAttribute('stroke-width', '2');
        svg.appendChild(line);
    });
    
    // Draw nodes
    nodes.forEach(node => {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', node.x);
        circle.setAttribute('cy', node.y);
        circle.setAttribute('r', node.leaf ? '30' : '40');
        circle.setAttribute('fill', node.leaf ? '#0066cc' : '#f8f9fa');
        circle.setAttribute('stroke', node.leaf ? '#0066cc' : '#e5e7eb');
        circle.setAttribute('stroke-width', '2');
        g.appendChild(circle);
        
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', node.x);
        text.setAttribute('y', node.y + 5);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-family', 'sans-serif');
        text.setAttribute('font-size', node.leaf ? '16' : '12');
        text.setAttribute('fill', node.leaf ? 'white' : '#333');
        text.textContent = node.label;
        g.appendChild(text);
        
        svg.appendChild(g);
    });
    
    treeContainer.appendChild(svg);
}

// Draw performance comparison chart
function drawPerformanceChart() {
    const canvas = document.getElementById('performance-chart');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 800;
    canvas.height = 400;
    
    // Data for comparison
    const data = {
        labels: ['n=3', 'n=4', 'n=5', 'n=6', 'n=7', 'n=8'],
        heaps: [5, 23, 119, 719, 5039, 40319],
        lexicographic: [9, 48, 300, 2160, 17640, 161280],
        backtracking: [15, 96, 750, 6480, 58800, 564480]
    };
    
    // Chart settings
    const padding = 60;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    const barWidth = chartWidth / (data.labels.length * 3 + data.labels.length + 1);
    const maxValue = Math.max(...data.backtracking);
    
    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw axes
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    
    // Draw bars
    const colors = {
        heaps: '#0066cc',
        lexicographic: '#ff6b6b',
        backtracking: '#4ecdc4'
    };
    
    data.labels.forEach((label, i) => {
        const x = padding + (i * 3 + 1) * barWidth + i * barWidth;
        
        // Heap's algorithm bar
        const heapHeight = (data.heaps[i] / maxValue) * chartHeight;
        ctx.fillStyle = colors.heaps;
        ctx.fillRect(x, canvas.height - padding - heapHeight, barWidth, heapHeight);
        
        // Lexicographic bar
        const lexHeight = (data.lexicographic[i] / maxValue) * chartHeight;
        ctx.fillStyle = colors.lexicographic;
        ctx.fillRect(x + barWidth, canvas.height - padding - lexHeight, barWidth, lexHeight);
        
        // Backtracking bar
        const backHeight = (data.backtracking[i] / maxValue) * chartHeight;
        ctx.fillStyle = colors.backtracking;
        ctx.fillRect(x + 2 * barWidth, canvas.height - padding - backHeight, barWidth, backHeight);
        
        // Label
        ctx.fillStyle = '#333';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, x + 1.5 * barWidth, canvas.height - padding + 20);
    });
    
    // Legend
    const legendY = padding - 30;
    const legendItems = [
        {label: "Heap's Algorithm", color: colors.heaps},
        {label: 'Lexicographic', color: colors.lexicographic},
        {label: 'Backtracking', color: colors.backtracking}
    ];
    
    legendItems.forEach((item, i) => {
        const x = padding + i * 150;
        ctx.fillStyle = item.color;
        ctx.fillRect(x, legendY, 20, 20);
        ctx.fillStyle = '#333';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(item.label, x + 25, legendY + 15);
    });
    
    // Y-axis label
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#333';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Number of Operations', 0, 0);
    ctx.restore();
}