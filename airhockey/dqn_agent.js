class DQNAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;  // 6 (puck x,y,dx,dy + both paddles x,y)
        this.actionSize = actionSize; // 9 (N,NE,E,SE,S,SW,W,NW,STAY)
        
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.gamma = 0.95;
        this.learningRate = 0.001;
        
        this.mainNetwork = this.createNetwork();
        this.targetNetwork = this.createNetwork();
        this.updateTargetNetwork();
        
        this.replayBuffer = [];
        this.replayBufferSize = 10000;
        this.batchSize = 32;
        this.updateFrequency = 4;
        this.targetUpdateFrequency = 100;
        this.frameCount = 0;
    }

    createNetwork() {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            inputShape: [this.stateSize]
        }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: this.actionSize,
            activation: 'linear'
        }));
        
        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    updateTargetNetwork() {
        const weights = this.mainNetwork.getWeights();
        this.targetNetwork.setWeights(weights);
    }

    getState(puck, playerPaddle, aiPaddle, isTopPlayer = false) {
        // For top player: normalize everything as if playing from bottom
        // For bottom player: keep original perspective
        let state;
        if (isTopPlayer) {
            state = [
                // Puck position relative to agent's paddle
                (puck.x - aiPaddle.x) / canvas.width,
                (canvas.height - puck.y - (canvas.height - aiPaddle.y)) / canvas.height,
                // Puck velocity (flipped for top player)
                puck.dx / maxSpeed,
                -puck.dy / maxSpeed,
                // Opponent paddle position relative to agent's paddle
                (playerPaddle.x - aiPaddle.x) / canvas.width,
                (canvas.height - playerPaddle.y - (canvas.height - aiPaddle.y)) / canvas.height
            ];
        } else {
            state = [
                // Puck position relative to agent's paddle
                (puck.x - playerPaddle.x) / canvas.width,
                (puck.y - playerPaddle.y) / canvas.height,
                // Puck velocity
                puck.dx / maxSpeed,
                puck.dy / maxSpeed,
                // Opponent paddle position relative to agent's paddle
                (aiPaddle.x - playerPaddle.x) / canvas.width,
                (aiPaddle.y - playerPaddle.y) / canvas.height
            ];
        }
        return state;
    }

    act(state, training = false) {
        if (training && Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        }

        return tf.tidy(() => {
            const stateTensor = tf.tensor2d([state], [1, this.stateSize]);
            const prediction = this.mainNetwork.predict(stateTensor);
            return prediction.argMax(1).dataSync()[0];
        });
    }

    remember(state, action, reward, nextState, done) {
        if (this.replayBuffer.length >= this.replayBufferSize) {
            this.replayBuffer.shift();
        }
        this.replayBuffer.push([state, action, reward, nextState, done]);
    }

    async train() {
        if (this.replayBuffer.length < this.batchSize) return;

        // Sample random batch
        const batch = this.sampleBatch();
        
        const states = batch.map(exp => exp[0]);
        const actions = batch.map(exp => exp[1]);
        const rewards = batch.map(exp => exp[2]);
        const nextStates = batch.map(exp => exp[3]);
        const dones = batch.map(exp => exp[4]);

        // Compute target Q values
        const nextStatesTensor = tf.tensor2d(nextStates);
        const nextQValues = this.targetNetwork.predict(nextStatesTensor);
        const maxNextQ = nextQValues.max(1);
        const targetQValues = maxNextQ.mul(tf.scalar(this.gamma)).add(tf.tensor1d(rewards));

        // Train main network
        const statesTensor = tf.tensor2d(states);
        const currentQ = this.mainNetwork.predict(statesTensor);
        const currentQArray = currentQ.arraySync();
        const targetQArray = currentQArray.map((qValues, i) => {
            // Create a copy of current Q values
            const newQValues = [...qValues];
            // Update only the Q value for the action that was taken
            newQValues[actions[i]] = dones[i] ? rewards[i] : targetQValues.dataSync()[i];
            return newQValues;
        });

        await this.mainNetwork.fit(statesTensor, tf.tensor2d(targetQArray), {
            epochs: 1,
            verbose: 0
        });

        // Cleanup tensors
        nextStatesTensor.dispose();
        nextQValues.dispose();
        maxNextQ.dispose();
        targetQValues.dispose();
        statesTensor.dispose();
        currentQ.dispose();

        // Decay epsilon
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    sampleBatch() {
        const batch = [];
        for (let i = 0; i < this.batchSize; i++) {
            const idx = Math.floor(Math.random() * this.replayBuffer.length);
            batch.push(this.replayBuffer[idx]);
        }
        return batch;
    }
} 