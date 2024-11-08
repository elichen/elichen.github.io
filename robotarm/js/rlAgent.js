class RLAgent {
    constructor(replayBuffer) {
        this.replayBuffer = replayBuffer;
        this.epsilon = 1.0;
        this.epsilonMin = 0.1;
        this.epsilonDecay = 0.999;
        this.gamma = 0.99;
        this.batchSize = 32;
        this.episodeCount = 0;
        this.totalReward = 0;
        this.angleStep = 0.1;
        this.lastState = null;
        this.lastAction = null;
        this.isTraining = false;
        this.frameCount = 0;
        this.frameSkip = 4;
        this.targetUpdateFreq = 1000;  // Update target network every 1000 frames
        
        this.initializeModel();
    }

    async initializeModel() {
        // Create main Q-Network
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [9]
        }));
        this.model.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        this.model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        this.model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        this.model.add(tf.layers.dense({
            units: 6
        }));

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        // Create target network with same architecture
        this.targetModel = tf.sequential();
        this.targetModel.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [9]
        }));
        this.targetModel.add(tf.layers.dense({
            units: 128,
            activation: 'relu'
        }));
        this.targetModel.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        this.targetModel.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        this.targetModel.add(tf.layers.dense({
            units: 6
        }));

        // Initialize target network with main network's weights
        await this.updateTargetNetwork();
    }

    async updateTargetNetwork() {
        const weights = this.model.getWeights();
        const targetWeights = this.targetModel.getWeights();
        
        for (let i = 0; i < weights.length; i++) {
            const w = weights[i];
            targetWeights[i].assign(w);
        }
    }

    async update(robotArm, environment, shouldTrain = false) {
        this.frameCount++;

        const state = environment.getState(robotArm);
        const action = await this.selectAction(state, shouldTrain);
        
        // Store the previous state-action pair's outcome if we have one and we're training
        if (this.lastState && this.lastAction !== null && shouldTrain) {
            let reward;
            let done;

            // Check if the last action was invalid
            if (this.lastInvalidAction) {
                reward = -10;  // Penalty for invalid action
                done = false;
                this.lastInvalidAction = false;
            } else {
                ({ reward, done } = environment.calculateReward(robotArm));
            }
            
            // Store the experience from the last action
            this.replayBuffer.store({
                state: this.lastState,
                action: this.lastAction,
                reward: reward,
                nextState: state,
                done: done
            });
            
            this.totalReward += reward;

            if (done) {
                this.episodeCount++;
                this.totalReward = 0;
                console.log(`Episode ${this.episodeCount} - Replay Buffer Stats:`);
                console.log(`  AI Experiences: ${this.replayBuffer.aiExperienceCount}`);
                console.log(`  Human Experiences: ${this.replayBuffer.humanExperienceCount}`);
                console.log(`  Total: ${this.replayBuffer.size}`);
                
                environment.reset();
                robotArm.reset();
                this.lastState = null;
                this.lastAction = null;
                return;
            }
        }
        
        // Only store state and action if we're training
        if (shouldTrain) {
            this.lastState = state;
            this.lastAction = action;
        }
        
        // Execute action and track if it was invalid
        const success = this.executeAction(action, robotArm);
        if (!success && shouldTrain) {
            this.lastInvalidAction = true;
        }

        // Train if we have enough experiences and it's a training frame
        if (shouldTrain && 
            this.replayBuffer.size >= this.batchSize && 
            !this.isTraining && 
            this.frameCount % this.frameSkip === 0) {
            await this.train();
        }

        // Update epsilon only during training
        if (shouldTrain) {
            this.epsilon = Math.max(
                this.epsilonMin,
                this.epsilon * this.epsilonDecay
            );
        }
    }

    async selectAction(state, shouldTrain = false) {
        if (shouldTrain) {
            // Regular epsilon-greedy exploration
            if (Math.random() < this.epsilon) {
                return Math.floor(Math.random() * 6);
            }
            
            // Occasionally force a large configuration change
            if (Math.random() < 0.05) {  // 5% chance
                // Choose between elbow-up and elbow-down configurations
                const preferUp = Math.random() < 0.5;
                return preferUp ? 2 : 3;  // Force large angle2 change
            }
        }

        // Otherwise use model prediction
        const stateTensor = tf.tensor2d([state]);
        const predictions = await this.model.predict(stateTensor).array();
        stateTensor.dispose();
        
        return predictions[0].indexOf(Math.max(...predictions[0]));
    }

    executeAction(action, robotArm) {
        console.log(`Executing action ${action}`);
        let newAngle1 = robotArm.angle1;
        let newAngle2 = robotArm.angle2;

        switch(action) {
            case 0: 
                newAngle1 += this.angleStep;
                break;
            case 1: 
                newAngle1 -= this.angleStep;
                break;
            case 2: 
                newAngle2 += this.angleStep;
                break;
            case 3: 
                newAngle2 -= this.angleStep;
                break;
            case 4: 
                robotArm.isClawClosed = false; 
                console.log("Opening claw");
                return true;
            case 5: 
                robotArm.isClawClosed = true; 
                console.log("Closing claw");
                return true;
        }

        // Only apply the new angles if they're valid
        const success = robotArm.setTargetAngles(newAngle1, newAngle2);
        if (!success) {
            console.log(`Model predicted invalid action ${action} at angles (${robotArm.angle1.toFixed(2)}, ${robotArm.angle2.toFixed(2)})`);
        } else {
            console.log(`Successfully moved to angles (${newAngle1.toFixed(2)}, ${newAngle2.toFixed(2)})`);
        }
        return success;
    }

    async train() {
        if (this.isTraining) return;

        const batch = this.replayBuffer.sample(this.batchSize);
        if (!batch) return;

        try {
            this.isTraining = true;

            const states = batch.map(exp => exp.state);
            const nextStates = batch.map(exp => exp.nextState);

            const currentQs = await this.model.predict(tf.tensor2d(states)).array();
            // Use target network for next state Q-values
            const nextQs = await this.targetModel.predict(tf.tensor2d(nextStates)).array();

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

            // Update target network periodically
            if (this.frameCount % this.targetUpdateFreq === 0) {
                await this.updateTargetNetwork();
                console.log("Target network updated");
            }
        } catch (error) {
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
        }
    }
} 