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
        this.replayBuffer = new ReplayBuffer(50000);
        this.humanControl = new HumanControl(this.robotArm, this.environment, this.replayBuffer);
        this.rlAgent = new RLAgent(this.replayBuffer);
        
        this.setupEventListeners();
        this.lastTimestamp = 0;
        this.animate = this.animate.bind(this);
    }

    setupEventListeners() {
        const modeButton = document.getElementById('mode-switch');
        
        modeButton.addEventListener('click', () => {
            this.isHumanMode = !this.isHumanMode;
            
            // Update button text to show what mode you'll switch to
            modeButton.textContent = this.isHumanMode ? 
                'Switch to AI Mode' : 'Switch to Human Mode';
            
            // Automatically enable training when switching to RL mode
            if (!this.isHumanMode) {
                this.isTraining = true;
                document.getElementById('start-training').textContent = 'Stop Training';
            }
            
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
                this.robotArm.isClawClosed = !this.robotArm.isClawClosed;
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

        // Update robot arm position
        this.robotArm.update();

        // Update simulation based on mode
        if (this.isHumanMode) {
            const { reward, done } = this.humanControl.update();
            // Update display with human mode rewards
            document.getElementById('total-reward').textContent = 
                this.humanControl.totalReward.toFixed(2);
        } else {
            // In RL mode, always run the agent
            // Pass shouldTrain=false to disable training but still take actions
            this.rlAgent.update(this.robotArm, this.environment, this.isTraining);
            
            // Log Q-values for debugging
            if (!this.isTraining) {
                const state = this.environment.getState(this.robotArm);
                const stateTensor = tf.tensor2d([state]);
                this.rlAgent.model.predict(stateTensor).array().then(predictions => {
                    console.log('Q-values:', predictions[0]);
                });
                stateTensor.dispose();
            }
            
            this.updateStats();
        }

        // Update physics
        this.physics.update(this.robotArm, this.environment, deltaTime);

        // Render
        this.environment.render(this.ctx);
        this.robotArm.render(this.ctx);

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