class RLAgent {
    constructor(replayBuffer) {
        this.replayBuffer = replayBuffer;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.gamma = 0.99;
        this.batchSize = 32;
        this.episodeCount = 0;
        this.totalReward = 0;
        this.angleStep = 0.1;
        this.lastState = null;
        this.lastAction = null;
        this.isTraining = false;
        
        this.initializeModel();
    }

    async initializeModel() {
        // Create Q-Network using TensorFlow.js
        this.model = tf.sequential();
        
        this.model.add(tf.layers.dense({
            units: 24,
            activation: 'relu',
            inputShape: [5]  // State size
        }));
        
        this.model.add(tf.layers.dense({
            units: 24,
            activation: 'relu'
        }));
        
        this.model.add(tf.layers.dense({
            units: 6  // Number of possible actions
        }));

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
    }

    async update(robotArm, environment) {
        let action;

        // Only take new actions if the arm isn't moving
        if (!robotArm.isMoving && !this.isTraining) {
            const state = environment.getState(robotArm);
            action = await this.selectAction(state);
            this.executeAction(action, robotArm);
        }

        // Update arm position
        robotArm.update();
        
        // Calculate rewards etc. only after movement is complete
        if (!robotArm.isMoving && !this.isTraining) {
            const { reward, done } = environment.calculateReward(robotArm);
            const nextState = environment.getState(robotArm);
            
            this.totalReward += reward;

            // Store experience and train
            if (this.lastState && this.lastAction !== null) {
                this.replayBuffer.store({
                    state: this.lastState,
                    action: this.lastAction,
                    reward: reward,
                    nextState: nextState,
                    done: done
                });
            }

            if (action !== undefined) {
                this.lastState = nextState;
                this.lastAction = action;
            }

            if (this.replayBuffer.size >= this.batchSize) {
                await this.train();
            }

            if (done) {
                this.episodeCount++;
                this.totalReward = 0;
                environment.reset();
                robotArm.reset();
                this.lastState = null;
                this.lastAction = null;
            }

            // Update epsilon
            this.epsilon = Math.max(
                this.epsilonMin,
                this.epsilon * this.epsilonDecay
            );
        }
    }

    async selectAction(state) {
        // Epsilon-greedy action selection
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * 6);
        }

        const stateTensor = tf.tensor2d([state]);
        const predictions = await this.model.predict(stateTensor).array();
        stateTensor.dispose();
        
        return predictions[0].indexOf(Math.max(...predictions[0]));
    }

    executeAction(action, robotArm) {
        // Don't execute new actions if the arm is still moving
        if (robotArm.isMoving) return;

        switch(action) {
            case 0: 
                robotArm.setTargetAngles(
                    robotArm.angle1 + this.angleStep, 
                    robotArm.angle2
                ); 
                break;
            case 1: 
                robotArm.setTargetAngles(
                    robotArm.angle1 - this.angleStep, 
                    robotArm.angle2
                ); 
                break;
            case 2: 
                robotArm.setTargetAngles(
                    robotArm.angle1, 
                    robotArm.angle2 + this.angleStep
                ); 
                break;
            case 3: 
                robotArm.setTargetAngles(
                    robotArm.angle1, 
                    robotArm.angle2 - this.angleStep
                ); 
                break;
            case 4: robotArm.targetClawClosed = false; break;
            case 5: robotArm.targetClawClosed = true; break;
        }
    }

    async train() {
        // Don't start training if already training
        if (this.isTraining) {
            console.log('Training already in progress, skipping...');
            return;
        }

        const batch = this.replayBuffer.sample(this.batchSize);
        if (!batch) return;

        try {
            this.isTraining = true;

            const states = batch.map(exp => exp.state);
            const nextStates = batch.map(exp => exp.nextState);

            const currentQs = await this.model.predict(tf.tensor2d(states)).array();
            const nextQs = await this.model.predict(tf.tensor2d(nextStates)).array();

            const x = [];
            const y = [];

            for (let i = 0; i < batch.length; i++) {
                const { state, action, reward, done } = batch[i];
                const currentQ = [...currentQs[i]];
                
                currentQ[action] = reward + (done ? 0 : this.gamma * Math.max(...nextQs[i]));
                
                x.push(state);
                y.push(currentQ);
            }

            await this.model.fit(tf.tensor2d(x), tf.tensor2d(y), {
                epochs: 1,
                verbose: 0
            });
        } catch (error) {
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
        }
    }
} 