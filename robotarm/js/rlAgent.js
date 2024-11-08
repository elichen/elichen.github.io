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
        const state = environment.getState(robotArm);
        const action = await this.selectAction(state);
        
        // Execute action
        this.executeAction(action, robotArm);
        
        // Get reward and next state
        const { reward, done } = environment.calculateReward(robotArm);
        const nextState = environment.getState(robotArm);
        
        // Store experience
        this.replayBuffer.store({
            state,
            action,
            reward,
            nextState,
            done
        });
        
        this.totalReward += reward;

        // Train the network
        if (this.replayBuffer.size >= this.batchSize) {
            await this.train();
        }

        // Update epsilon
        this.epsilon = Math.max(
            this.epsilonMin,
            this.epsilon * this.epsilonDecay
        );

        if (done) {
            this.episodeCount++;
            this.totalReward = 0;
            environment.reset();
            robotArm.reset();
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
        switch(action) {
            case 0: robotArm.moveJoint(1, 1); break;  // Increase angle1
            case 1: robotArm.moveJoint(1, -1); break; // Decrease angle1
            case 2: robotArm.moveJoint(2, 1); break;  // Increase angle2
            case 3: robotArm.moveJoint(2, -1); break; // Decrease angle2
            case 4: robotArm.isClawClosed = false; break; // Open claw
            case 5: robotArm.isClawClosed = true; break;  // Close claw
        }
    }

    async train() {
        const batch = this.replayBuffer.sample(this.batchSize);
        if (!batch) return;

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
    }
} 