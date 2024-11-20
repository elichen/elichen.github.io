class DQNAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;  // 11 (puck x,y,dx,dy + opponent paddle x,y + distance + angle + behind + relative Y positions)
        this.actionSize = actionSize; // 5 (STAY,UP,RIGHT,DOWN,LEFT)
        
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.999;
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
        this.isTraining = false;
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

    getState(puck, playerPaddle, aiPaddle, isTopPlayer = false, canvasWidth, canvasHeight) {
        const paddleX = isTopPlayer ? aiPaddle.x : playerPaddle.x;
        const paddleY = isTopPlayer ? aiPaddle.y : playerPaddle.y;
        
        // Calculate distance and angle to puck
        let dx = puck.x - paddleX;
        let dy = puck.y - paddleY;
        if (isTopPlayer) {
            dx = -dx;
            dy = -dy;
        }
        const distance = Math.sqrt(dx * dx + dy * dy) / canvasHeight;
        const angle = Math.atan2(dy, dx) / Math.PI;
        
        // Calculate if puck is behind paddle
        const isPuckBehind = isTopPlayer ? 
            (puck.y < paddleY) :
            (puck.y > paddleY);

        // Calculate relative Y positions
        const puckToOwnGoal = isTopPlayer ?
            puck.y / (canvasHeight/2) :  // Distance from top goal for top player
            (canvasHeight - puck.y) / (canvasHeight/2);  // Distance from bottom goal for bottom player
            
        const paddleToOwnGoal = isTopPlayer ?
            paddleY / (canvasHeight/2) :  // Distance from top goal for top player
            (canvasHeight - paddleY) / (canvasHeight/2);  // Distance from bottom goal for bottom player

        let state;
        if (isTopPlayer) {
            state = [
                (canvasWidth - puck.x) / canvasWidth,
                (canvasHeight - puck.y) / canvasHeight,
                -puck.dx / maxSpeed,
                -puck.dy / maxSpeed,
                (canvasWidth - playerPaddle.x) / canvasWidth,
                (canvasHeight - playerPaddle.y) / canvasHeight,
                distance,
                angle,
                isPuckBehind ? 1 : 0,
                puckToOwnGoal,
                paddleToOwnGoal
            ];
        } else {
            state = [
                puck.x / canvasWidth,
                puck.y / canvasHeight,
                puck.dx / maxSpeed,
                puck.dy / maxSpeed,
                aiPaddle.x / canvasWidth,
                aiPaddle.y / canvasHeight,
                distance,
                angle,
                isPuckBehind ? 1 : 0,
                puckToOwnGoal,
                paddleToOwnGoal
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
        if (this.isTraining || this.replayBuffer.length < this.batchSize) return;

        try {
            this.isTraining = true;

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
                const newQValues = [...qValues];
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
        } finally {
            this.isTraining = false;
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