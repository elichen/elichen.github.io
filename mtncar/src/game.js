class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas')
        this.canvas.width = 800
        this.canvas.height = 500
        this.mountainCar = new MountainCar(this.canvas)
        this.env = new MountainCarEnv(this.mountainCar)
        
        // Initialize training components
        this.isManualMode = false  // Default to training mode
        this.setupAgent()
        this.setupControls()
        this.setupTrainingControls()

        // Reset environment for training mode start
        if (!this.isManualMode) {
            this.env.reset()
        }

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
        // Check URL parameter for agent type
        const urlParams = new URLSearchParams(window.location.search);
        const agentType = urlParams.get('agent') || 'dqn'; // Default to DQN for better sample efficiency

        // Wrap environment with normalization
        let env = this.env;
        env = new NormalizeObservation(env);
        env = new ScaleReward(env, 0.99); // Add reward scaling

        if (agentType === 'dqn') {
            // Get DQN configuration from URL parameters
            const explorationMode = urlParams.get('exploration') || 'epsilon';
            const usePrioritizedReplay = urlParams.get('priority') !== 'false';
            const useDoubleDQN = urlParams.get('double') !== 'false';

            // Use new sample-efficient DQN agent
            const config = {
                // Network architecture
                hiddenSizes: [64, 64],
                activation: 'leakyReLU',
                layerNorm: true,

                // Q-learning parameters
                gamma: 0.99,
                learningRate: 0.0003,

                // Experience replay
                bufferSize: 50000,
                batchSize: 32,
                minBufferSize: 1000,
                usePrioritizedReplay: usePrioritizedReplay,

                // Target network
                targetUpdateFreq: 100,
                tau: 1.0,

                // Exploration
                explorationMode: explorationMode,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 30000, // Faster decay for better efficiency

                // UCB parameters
                ucbC: 2.0,

                // Boltzmann parameters
                boltzmannTemp: 1.0,
                boltzmannTempEnd: 0.1,

                // N-step returns
                nSteps: 3,

                // Double DQN
                useDoubleDQN: useDoubleDQN,

                // Training
                updateFreq: 4,
                gradientsPerStep: 1,
            };
            this.agent = new DQNAgent(env, config);
            console.log(`Using DQN agent with ${explorationMode} exploration, ` +
                       `priority replay: ${usePrioritizedReplay}, ` +
                       `double DQN: ${useDoubleDQN}`);
        } else {
            // Use original StreamQ agent
            const config = {
                env: env,
                numActions: 3,
                epsilonStart: 1.0,
                epsilonTarget: 0.01,
                totalSteps: 100000,
            };
            this.agent = new StreamQ(config);
            console.log('Using StreamQ agent');
        }
    }

    setupTrainingControls() {
        const controls = document.createElement('div')
        controls.className = 'controls'
        
        const toggleButton = document.createElement('button')
        toggleButton.textContent = 'Switch to Manual Mode'  // Start in training mode
        toggleButton.onclick = () => {
            this.isManualMode = !this.isManualMode
            toggleButton.textContent = this.isManualMode ?
                'Switch to Training Mode' :
                'Switch to Manual Mode'

            // Update mode text
            const modeText = document.getElementById('modeText')
            if (modeText) {
                modeText.textContent = this.isManualMode ?
                    'Use left and right arrow keys to control the car' :
                    'AI Training Mode Active - Watch it Learn!'
            }

            // Reset environment when switching modes
            this.env.reset()
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

        // Handle different agent interfaces
        let action;
        if (this.agent instanceof DQNAgent) {
            action = this.agent.act(state, true); // training = true
        } else {
            const result = await this.agent.sampleAction(state);
            action = result.action;
        }

        const result = this.env.step(action);

        this.currentEpisodeSteps++;
        this.stepCount++;
        this.currentEpisodeReturn += result.reward;

        // Store transition and train for DQN agent
        if (this.agent instanceof DQNAgent) {
            this.agent.remember(state, action, result.reward, result.state, result.done);
            this.agent.train(); // Will only train when buffer has enough samples
        } else {
            // Original StreamQ update
            await this.agent.update(
                state,
                action,
                result.reward,
                result.state,
                result.done,
                false
            );
        }

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

        // Get agent-specific stats
        let agentStats = '';
        if (this.agent instanceof DQNAgent) {
            const stats = this.agent.getStats();
            agentStats = `
                Agent: DQN (Sample-Efficient)<br>
                Buffer Size: ${stats.bufferSize}<br>
                Epsilon: ${stats.epsilon.toFixed(3)}<br>
                Updates: ${stats.updateCount}<br>
                Avg Loss: ${stats.avgLoss.toFixed(6)}
            `;
        } else {
            agentStats = `
                Agent: StreamQ<br>
                Epsilon: ${this.agent.epsilon.toFixed(3)}
            `;
        }

        this.stats.innerHTML = `
            Episode: ${this.episodeCount}<br>
            Total Steps: ${this.stepCount}<br>
            Episode Steps: ${steps}<br>
            Status: ${status}<br>
            Last Return: ${episodeReturn.toFixed(1)}<br>
            Avg Return (${this.episodeRewards.size}): ${avgReward.toFixed(1)}<br>
            ${agentStats}<br>
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