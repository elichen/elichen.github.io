class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas')
        this.canvas.width = 800
        this.canvas.height = 500
        this.mountainCar = new MountainCar(this.canvas)
        this.env = new MountainCarEnv(this.mountainCar)

        // Initialize training components
        this.isManualMode = false  // Default to training mode

        // Setup is async for SB3 agent loading
        this.initAsync()
    }

    async initAsync() {
        await this.setupAgent()
        this.setupControls()
        this.setupTrainingControls()

        // Reset environment for training mode start
        if (!this.isManualMode) {
            this.env.reset()
        }

        this.showSuccess = false
        this.successTimer = 0
        this.episodeRewards = new CircularBuffer(10)

        // Stats tracking (no visual display)
        this.episodeCount = 0;
        this.stepCount = 0;
        this.maxStepsPerEpisode = 10000; // Prevent infinite episodes
        this.currentEpisodeSteps = 0;
        this.currentEpisodeReturn = 0;

        this.gameLoop()
    }

    async setupAgent() {
        // Always use pre-trained SB3 model
        this.agent = new SB3Agent(this.env);
        await this.agent.loadModel('models/sb3_weights.json');
        console.log('Using pre-trained SB3 DQN agent');
    }

    setupTrainingControls() {
        const controls = document.createElement('div')
        controls.className = 'controls'

        const toggleButton = document.createElement('button')
        toggleButton.textContent = 'Switch to Manual Mode'  // Start in AI mode
        toggleButton.onclick = () => {
            this.isManualMode = !this.isManualMode
            toggleButton.textContent = this.isManualMode ?
                'Switch to AI Mode' :
                'Switch to Manual Mode'

            // Update mode text
            const modeText = document.getElementById('modeText')
            if (modeText) {
                modeText.textContent = this.isManualMode ?
                    'Use left and right arrow keys to control the car' :
                    'Watch a pre-trained AI agent solve this classic control problem!'
            }

            // Reset environment when switching modes
            if (this.isManualMode) {
                this.mountainCar = new MountainCar(this.canvas)
                this.env = new MountainCarEnv(this.mountainCar)
            } else {
                this.env.reset()
            }
        }

        controls.appendChild(toggleButton)
        document.querySelector('.container').appendChild(controls)
    }

    setupControls() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft':
                    this.currentAction = 'left'
                    break
                case 'ArrowRight':
                    this.currentAction = 'right'
                    break
            }
        })

        document.addEventListener('keyup', () => {
            this.currentAction = 'none'
        })
    }

    async gameLoop() {
        if (this.isManualMode) {
            this.manualStep()
        } else {
            await this.trainingStep()
        }

        if (this.showSuccess) {
            const ctx = this.canvas.getContext('2d')
            ctx.font = '40px "Press Start 2P"'
            ctx.fillStyle = '#39ff14'
            ctx.textAlign = 'center'
            ctx.fillText('SUCCESS!', this.canvas.width/2, this.canvas.height/2)

            this.successTimer--
            if (this.successTimer <= 0) {
                this.showSuccess = false
                if (this.isManualMode) {
                    // Reset the environment in manual mode
                    this.mountainCar = new MountainCar(this.canvas)
                    this.env = new MountainCarEnv(this.mountainCar)
                } else {
                    this.env.reset()
                }
            }
        }

        requestAnimationFrame(() => this.gameLoop())
    }

    manualStep() {
        if (this.showSuccess) {
            // Don't step while showing success message
            this.mountainCar.render()
            return
        }

        const success = this.mountainCar.step(this.currentAction)
        this.mountainCar.render()

        if (success) {
            this.showSuccess = true
            this.successTimer = 100
        }
    }

    async trainingStep() {
        if (this.currentEpisodeSteps >= this.maxStepsPerEpisode) {
            // Force episode end if max steps reached
            this.episodeCount++;
            this.logEpisodeStats(this.currentEpisodeReturn, this.currentEpisodeSteps, true);
            this.env.reset();
            this.currentEpisodeSteps = 0;
            this.currentEpisodeReturn = 0;
            return;
        }

        const state = this.env.getState();

        // Get action from pre-trained SB3 agent
        const action = this.agent.act(state, false); // inference mode

        const result = this.env.step(action);

        this.currentEpisodeSteps++;
        this.stepCount++;
        this.currentEpisodeReturn += result.reward;

        if (result.done) {
            this.episodeCount++;
            this.logEpisodeStats(this.currentEpisodeReturn, this.currentEpisodeSteps, false);
            this.env.reset();
            this.currentEpisodeSteps = 0;
            this.currentEpisodeReturn = 0;
        }

        this.env.render();
    }

    logEpisodeStats(episodeReturn, steps, timeout) {
        this.episodeRewards.push(episodeReturn);
        const avgReward = this.episodeRewards.average();

        const status = timeout ? "TIMEOUT" : (episodeReturn >= 100 ? "SUCCESS" : "FAILED");

        // Log to console periodically
        if (this.episodeCount % 10 === 0) {
            console.log(`Episode ${this.episodeCount}:`, {
                totalSteps: this.stepCount,
                episodeSteps: steps,
                status,
                return: episodeReturn.toFixed(1),
                avgReturn: avgReward.toFixed(1),
                maxHeight: Math.sin(3 * this.env.mountainCar.position).toFixed(3)
            });
        }
    }

    dispose() {
        if (this.agent) {
            this.agent.dispose()
        }
    }
} 