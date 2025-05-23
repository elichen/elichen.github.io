class NeuralBridgeGame {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.score = 0;
        this.level = 1;
        this.bridges = 0;
        this.accuracy = 100;
        this.isPlaying = false;
        
        this.nodes = [];
        this.connections = [];
        this.selectedNode = null;
        this.mousePos = { x: 0, y: 0 };
        this.drawingConnection = false;
        
        this.levels = this.generateLevels();
        this.currentLevelIndex = 0;
        
        this.setupEventListeners();
        this.init();
    }
    
    generateLevels() {
        return [
            {
                name: "Basic Connection",
                llmNodes: [
                    { id: 'llm1', x: 200, y: 200, label: 'Generate Text', description: 'LLM capability to generate human-like text' }
                ],
                kgNodes: [
                    { id: 'kg1', x: 800, y: 200, label: 'Factual Data', description: 'Verified facts from knowledge base' }
                ],
                requiredConnections: [['llm1', 'kg1']],
                hint: "Connect text generation with factual data for accurate responses"
            },
            {
                name: "Multiple Facts",
                llmNodes: [
                    { id: 'llm1', x: 200, y: 150, label: 'Question Understanding', description: 'Parse and understand user queries' },
                    { id: 'llm2', x: 200, y: 350, label: 'Answer Generation', description: 'Generate natural language answers' }
                ],
                kgNodes: [
                    { id: 'kg1', x: 800, y: 100, label: 'Historical Facts', description: 'Database of historical events' },
                    { id: 'kg2', x: 800, y: 250, label: 'Scientific Data', description: 'Verified scientific information' },
                    { id: 'kg3', x: 800, y: 400, label: 'Geographic Info', description: 'Location and mapping data' }
                ],
                requiredConnections: [['llm1', 'kg2'], ['llm2', 'kg2']],
                hint: "Scientific questions need scientific data!"
            },
            {
                name: "Complex Reasoning",
                llmNodes: [
                    { id: 'llm1', x: 150, y: 150, label: 'Context Analysis', description: 'Understand conversation context' },
                    { id: 'llm2', x: 150, y: 300, label: 'Reasoning Engine', description: 'Apply logical reasoning' },
                    { id: 'llm3', x: 150, y: 450, label: 'Response Synthesis', description: 'Create coherent responses' }
                ],
                kgNodes: [
                    { id: 'kg1', x: 850, y: 100, label: 'Entity Relations', description: 'How entities relate to each other' },
                    { id: 'kg2', x: 850, y: 250, label: 'Temporal Data', description: 'Time-based information' },
                    { id: 'kg3', x: 850, y: 400, label: 'Causal Links', description: 'Cause and effect relationships' },
                    { id: 'kg4', x: 850, y: 550, label: 'Statistical Facts', description: 'Numerical and statistical data' }
                ],
                requiredConnections: [['llm1', 'kg1'], ['llm2', 'kg3'], ['llm3', 'kg1']],
                hint: "Context needs entities, reasoning needs causality!"
            },
            {
                name: "Knowledge Integration",
                llmNodes: [
                    { id: 'llm1', x: 100, y: 100, label: 'Multi-hop QA', description: 'Answer questions requiring multiple steps' },
                    { id: 'llm2', x: 100, y: 250, label: 'Fact Checking', description: 'Verify claim accuracy' },
                    { id: 'llm3', x: 100, y: 400, label: 'Explanation Gen', description: 'Generate detailed explanations' },
                    { id: 'llm4', x: 300, y: 175, label: 'Inference Engine', description: 'Draw logical conclusions' },
                    { id: 'llm5', x: 300, y: 325, label: 'Summary Creation', description: 'Create concise summaries' }
                ],
                kgNodes: [
                    { id: 'kg1', x: 700, y: 100, label: 'Primary Sources', description: 'Original source documents' },
                    { id: 'kg2', x: 700, y: 250, label: 'Cross-References', description: 'Connected information' },
                    { id: 'kg3', x: 700, y: 400, label: 'Verification Data', description: 'Data for fact-checking' },
                    { id: 'kg4', x: 900, y: 175, label: 'Domain Ontology', description: 'Domain-specific relationships' },
                    { id: 'kg5', x: 900, y: 325, label: 'Meta-Knowledge', description: 'Knowledge about knowledge' }
                ],
                requiredConnections: [
                    ['llm1', 'kg2'], ['llm2', 'kg3'], ['llm3', 'kg4'], 
                    ['llm4', 'kg5'], ['llm5', 'kg1']
                ],
                hint: "Match each LLM capability with its ideal knowledge source!"
            }
        ];
    }
    
    init() {
        this.setupCanvas();
        this.loadLevel(0);
        this.animate();
    }
    
    setupCanvas() {
        this.canvas.width = 1000;
        this.canvas.height = 600;
    }
    
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        
        document.getElementById('startBtn').addEventListener('click', () => this.startGame());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetLevel());
        document.getElementById('tutorialBtn').addEventListener('click', () => this.showTutorial());
        
        document.querySelector('.close').addEventListener('click', () => this.closeTutorial());
    }
    
    loadLevel(levelIndex) {
        const level = this.levels[levelIndex];
        this.nodes = [];
        this.connections = [];
        this.selectedNode = null;
        
        level.llmNodes.forEach(node => {
            this.nodes.push({
                ...node,
                type: 'llm',
                radius: 40,
                connected: false
            });
        });
        
        level.kgNodes.forEach(node => {
            this.nodes.push({
                ...node,
                type: 'kg',
                radius: 40,
                connected: false
            });
        });
        
        document.getElementById('challenge-text').textContent = level.hint;
        this.updateStats();
    }
    
    handleMouseDown(e) {
        if (!this.isPlaying) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const clickedNode = this.getNodeAt(x, y);
        
        if (clickedNode) {
            if (!this.selectedNode) {
                this.selectedNode = clickedNode;
                this.drawingConnection = true;
                document.getElementById('node-info').innerHTML = `
                    <strong>${clickedNode.label}</strong><br>
                    ${clickedNode.description}<br>
                    Type: ${clickedNode.type.toUpperCase()}
                `;
            } else if (clickedNode !== this.selectedNode && clickedNode.type !== this.selectedNode.type) {
                this.createConnection(this.selectedNode, clickedNode);
                this.selectedNode = null;
                this.drawingConnection = false;
            } else {
                this.selectedNode = clickedNode;
            }
        } else {
            this.selectedNode = null;
            this.drawingConnection = false;
            document.getElementById('node-info').textContent = 'Click on a node to see details';
        }
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.mousePos.x = e.clientX - rect.left;
        this.mousePos.y = e.clientY - rect.top;
    }
    
    handleMouseUp(e) {
        if (this.drawingConnection && this.selectedNode) {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const targetNode = this.getNodeAt(x, y);
            
            if (targetNode && targetNode !== this.selectedNode && targetNode.type !== this.selectedNode.type) {
                this.createConnection(this.selectedNode, targetNode);
            }
        }
        
        this.drawingConnection = false;
    }
    
    getNodeAt(x, y) {
        return this.nodes.find(node => {
            const dx = x - node.x;
            const dy = y - node.y;
            return Math.sqrt(dx * dx + dy * dy) < node.radius;
        });
    }
    
    createConnection(node1, node2) {
        const connectionExists = this.connections.some(conn =>
            (conn.from === node1 && conn.to === node2) ||
            (conn.from === node2 && conn.to === node1)
        );
        
        if (!connectionExists) {
            const level = this.levels[this.currentLevelIndex];
            const isCorrect = level.requiredConnections.some(req =>
                (req[0] === node1.id && req[1] === node2.id) ||
                (req[0] === node2.id && req[1] === node1.id)
            );
            
            this.connections.push({
                from: node1,
                to: node2,
                correct: isCorrect
            });
            
            node1.connected = true;
            node2.connected = true;
            
            if (isCorrect) {
                this.score += 100;
                this.bridges++;
                this.playSuccessSound();
            } else {
                this.accuracy = Math.max(0, this.accuracy - 10);
                this.playErrorSound();
            }
            
            this.updateStats();
            this.checkLevelComplete();
        }
    }
    
    checkLevelComplete() {
        const level = this.levels[this.currentLevelIndex];
        const correctConnections = this.connections.filter(conn => conn.correct).length;
        
        if (correctConnections === level.requiredConnections.length) {
            setTimeout(() => {
                if (this.accuracy === 100) {
                    this.score *= 2;
                    alert('Perfect! Score doubled!');
                }
                
                if (this.currentLevelIndex < this.levels.length - 1) {
                    this.currentLevelIndex++;
                    this.level++;
                    this.loadLevel(this.currentLevelIndex);
                    alert(`Level ${this.level - 1} Complete! Moving to Level ${this.level}`);
                } else {
                    alert(`Congratulations! You've completed all levels!\nFinal Score: ${this.score}`);
                    this.isPlaying = false;
                }
            }, 500);
        }
    }
    
    playSuccessSound() {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 523.25; // C5
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.2);
    }
    
    playErrorSound() {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 200;
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.3);
    }
    
    updateStats() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('level').textContent = this.level;
        document.getElementById('bridges').textContent = this.bridges;
        document.getElementById('accuracy').textContent = this.accuracy + '%';
    }
    
    startGame() {
        this.isPlaying = true;
        this.score = 0;
        this.level = 1;
        this.bridges = 0;
        this.accuracy = 100;
        this.currentLevelIndex = 0;
        this.loadLevel(0);
        document.getElementById('startBtn').textContent = 'Restart';
    }
    
    resetLevel() {
        this.connections = [];
        this.selectedNode = null;
        this.nodes.forEach(node => node.connected = false);
        this.accuracy = 100;
        this.updateStats();
    }
    
    showTutorial() {
        document.getElementById('tutorial-modal').style.display = 'block';
    }
    
    closeTutorial() {
        document.getElementById('tutorial-modal').style.display = 'none';
    }
    
    animate() {
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid background
        this.drawGrid();
        
        // Draw connections
        this.connections.forEach(conn => {
            this.drawConnection(conn.from, conn.to, conn.correct);
        });
        
        // Draw temporary connection while dragging
        if (this.drawingConnection && this.selectedNode) {
            this.ctx.strokeStyle = '#ffaa00';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(this.selectedNode.x, this.selectedNode.y);
            this.ctx.lineTo(this.mousePos.x, this.mousePos.y);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
        
        // Draw nodes
        this.nodes.forEach(node => {
            this.drawNode(node);
        });
    }
    
    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.lineWidth = 1;
        
        for (let x = 0; x < this.canvas.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }
    
    drawNode(node) {
        const isSelected = node === this.selectedNode;
        const time = Date.now() * 0.001;
        
        // Node glow effect
        if (isSelected || node.connected) {
            const gradient = this.ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius * 2
            );
            gradient.addColorStop(0, node.type === 'llm' ? 'rgba(74, 158, 255, 0.3)' : 'rgba(0, 255, 136, 0.3)');
            gradient.addColorStop(1, 'transparent');
            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(node.x - node.radius * 2, node.y - node.radius * 2, node.radius * 4, node.radius * 4);
        }
        
        // Node circle
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius + (isSelected ? Math.sin(time * 3) * 5 : 0), 0, Math.PI * 2);
        this.ctx.fillStyle = node.type === 'llm' ? '#4a9eff' : '#00ff88';
        this.ctx.fill();
        
        // Node border
        this.ctx.strokeStyle = isSelected ? '#ffffff' : (node.connected ? '#ffaa00' : 'rgba(255, 255, 255, 0.3)');
        this.ctx.lineWidth = isSelected ? 4 : 2;
        this.ctx.stroke();
        
        // Node label
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(node.label, node.x, node.y);
        
        // Node type indicator
        this.ctx.font = '10px Arial';
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        this.ctx.fillText(node.type.toUpperCase(), node.x, node.y + node.radius + 15);
    }
    
    drawConnection(from, to, isCorrect) {
        const gradient = this.ctx.createLinearGradient(from.x, from.y, to.x, to.y);
        gradient.addColorStop(0, from.type === 'llm' ? '#4a9eff' : '#00ff88');
        gradient.addColorStop(1, to.type === 'llm' ? '#4a9eff' : '#00ff88');
        
        this.ctx.strokeStyle = isCorrect ? gradient : '#ff4444';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(from.x, from.y);
        this.ctx.lineTo(to.x, to.y);
        this.ctx.stroke();
        
        // Connection pulse effect
        if (isCorrect) {
            const time = Date.now() * 0.001;
            const pulsePos = (Math.sin(time * 2) + 1) / 2;
            const pulseX = from.x + (to.x - from.x) * pulsePos;
            const pulseY = from.y + (to.y - from.y) * pulsePos;
            
            this.ctx.beginPath();
            this.ctx.arc(pulseX, pulseY, 5, 0, Math.PI * 2);
            this.ctx.fillStyle = '#ffaa00';
            this.ctx.fill();
        }
    }
}

// Initialize game when page loads
window.addEventListener('DOMContentLoaded', () => {
    const game = new NeuralBridgeGame();
});