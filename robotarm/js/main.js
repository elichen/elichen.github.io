class SimulationApp {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isHumanMode = true;
        this.isTraining = false;
        
        // Initialize modules
        this.physics = new Physics();
        this.robotArm = new RobotArm();
        this.environment = new Environment(this.canvas.width, this.canvas.height);
        this.humanControl = new HumanControl(this.robotArm, this.environment);
        this.replayBuffer = new ReplayBuffer(5000);
        this.rlAgent = new RLAgent(this.replayBuffer);
        
        this.setupEventListeners();
        this.lastTimestamp = 0;
        this.animate = this.animate.bind(this);
    }

    setupEventListeners() {
        document.getElementById('mode-switch').addEventListener('click', () => {
            this.isHumanMode = !this.isHumanMode;
            document.getElementById('current-mode').textContent = 
                this.isHumanMode ? 'Human Control' : 'RL Agent';
        });

        document.getElementById('start-training').addEventListener('click', () => {
            this.isTraining = !this.isTraining;
            document.getElementById('start-training').textContent = 
                this.isTraining ? 'Stop Training' : 'Start Training';
        });

        document.getElementById('claw-control').addEventListener('click', () => {
            if (this.isHumanMode) {
                this.robotArm.toggleClaw();
            }
        });

        document.getElementById('reset').addEventListener('click', () => {
            this.reset();
        });

        this.canvas.addEventListener('click', (event) => {
            if (this.isHumanMode) {
                this.humanControl.handleClick(event);
            }
        });
    }

    reset() {
        this.environment.reset();
        this.robotArm.reset();
        this.updateStats();
    }

    updateStats() {
        document.getElementById('epsilon-value').textContent = 
            this.rlAgent.epsilon.toFixed(3);
        document.getElementById('episode-count').textContent = 
            this.rlAgent.episodeCount;
        document.getElementById('total-reward').textContent = 
            this.rlAgent.totalReward.toFixed(2);
    }

    animate(timestamp) {
        const deltaTime = timestamp - this.lastTimestamp;
        this.lastTimestamp = timestamp;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update simulation
        if (!this.isHumanMode && this.isTraining) {
            this.rlAgent.update(this.robotArm, this.environment);
        }

        // Update physics
        this.physics.update(this.robotArm, this.environment, deltaTime);

        // Render
        this.environment.render(this.ctx);
        this.robotArm.render(this.ctx);

        // Update stats
        this.updateStats();

        requestAnimationFrame(this.animate);
    }

    start() {
        this.reset();
        requestAnimationFrame(this.animate);
    }
}

// Start the application when the window loads
window.onload = () => {
    const app = new SimulationApp();
    app.start();
}; 