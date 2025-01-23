class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas')
        this.canvas.width = 800
        this.canvas.height = 500
        this.mountainCar = new MountainCar(this.canvas)
        this.env = new MountainCarEnv(this.mountainCar)
        
        // Initialize training components
        this.isManualMode = true
        this.setupAgent()
        this.setupControls()
        this.setupTrainingControls()
        
        this.showSuccess = false
        this.successTimer = 0
        this.episodeRewards = new CircularBuffer(10)
        this.stats = document.createElement('div')
        this.stats.className = 'stats'
        document.querySelector('.container').appendChild(this.stats)
        
        // Add these properties to the constructor
        this.episodeCount = 0;
        this.stepCount = 0;
        this.maxStepsPerEpisode = 10000; // Prevent infinite episodes
        this.currentEpisodeSteps = 0;
        this.currentEpisodeReturn = 0;
        
        this.gameLoop()
    }

    setupAgent() {
        // Wrap environment with normalization
        let env = this.env;
        env = new NormalizeObservation(env);
        env = new ScaleReward(env, 0.99); // Add reward scaling

        const config = {
            env: env, // Use wrapped environment
            numActions: 3,
            epsilonStart: 1.0,
            epsilonTarget: 0.01,
            totalSteps: 100000,
        }
        this.agent = new StreamQ(config)
    }

    setupTrainingControls() {
        const controls = document.createElement('div')
        controls.className = 'controls'
        
        const toggleButton = document.createElement('button')
        toggleButton.textContent = 'Switch to Training Mode'
        toggleButton.onclick = () => {
            this.isManualMode = !this.isManualMode
            toggleButton.textContent = this.isManualMode ? 
                'Switch to Training Mode' : 
                'Switch to Manual Mode'
            
            if (!this.isManualMode) {
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
                this.env.reset()
            }
        }

        requestAnimationFrame(() => this.gameLoop())
    }

    manualStep() {
        const success = this.mountainCar.step(this.currentAction)
        this.mountainCar.render()

        if (success && !this.showSuccess) {
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
        const { action } = await this.agent.sampleAction(state);
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

        await this.agent.update(
            state, 
            action,
            result.reward,
            result.state,
            result.done,
            false
        );

        this.env.render();
    }

    logEpisodeStats(episodeReturn, steps, timeout) {
        this.episodeRewards.push(episodeReturn);
        const avgReward = this.episodeRewards.average();
        
        const status = timeout ? "TIMEOUT" : (episodeReturn >= 100 ? "SUCCESS" : "FAILED");
        
        this.stats.innerHTML = `
            Episode: ${this.episodeCount}<br>
            Total Steps: ${this.stepCount}<br>
            Episode Steps: ${steps}<br>
            Status: ${status}<br>
            Last Return: ${episodeReturn.toFixed(1)}<br>
            Avg Return (${this.episodeRewards.size}): ${avgReward.toFixed(1)}<br>
            Epsilon: ${this.agent.epsilon.toFixed(3)}<br>
            Max Height: ${Math.sin(3 * this.env.mountainCar.position).toFixed(3)}
        `;

        // Log to console periodically
        if (this.episodeCount % 10 === 0) {
            console.log(`Episode ${this.episodeCount}:`, {
                totalSteps: this.stepCount,
                episodeSteps: steps,
                status,
                return: episodeReturn.toFixed(1),
                avgReturn: avgReward.toFixed(1),
                epsilon: this.agent.epsilon.toFixed(3),
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