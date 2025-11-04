class TrainedAgent {
    constructor() {
        this.model = null;
        this.modelInfo = null;
        this.isLoaded = false;
        this.angleStep = 0.1;
        this.lastClawToggleStep = 0;
        this.stepCount = 0;
        this.minClawToggleInterval = 10; // Minimum steps between claw toggles
    }

    async loadModel(modelPath = './models/tfjs/model.json') {
        try {
            console.log('Loading trained model from:', modelPath);

            // Load model info first
            const infoResponse = await fetch('./models/tfjs/model_info.json');
            this.modelInfo = await infoResponse.json();

            // Load the TensorFlow.js model
            this.model = await tf.loadLayersModel(modelPath);

            console.log('Model loaded successfully');
            console.log('Model info:', this.modelInfo);

            this.isLoaded = true;
            return true;
        } catch (error) {
            console.error('Failed to load model:', error);
            this.isLoaded = false;
            return false;
        }
    }

    async selectAction(state) {
        if (!this.isLoaded || !this.model) {
            console.warn('Model not loaded, returning random action');
            return Math.floor(Math.random() * 5);
        }

        this.stepCount++;

        // Convert state to tensor
        const stateTensor = tf.tensor2d([state]);

        try {
            // Get Q-values from model
            const qValues = await this.model.predict(stateTensor).array();
            stateTensor.dispose();

            // Get valid actions from state (last 4 values indicate validity of movement actions)
            const validActions = [
                state[7],  // valid_action_0
                state[8],  // valid_action_1
                state[9],  // valid_action_2
                state[10], // valid_action_3
                1.0        // claw toggle is always valid
            ];

            // Prevent excessive claw toggling
            const stepsSinceLastToggle = this.stepCount - this.lastClawToggleStep;
            if (stepsSinceLastToggle < this.minClawToggleInterval) {
                validActions[4] = 0; // Temporarily disable claw toggle
            }

            // Mask invalid actions with -Infinity
            const maskedQValues = qValues[0].map((q, i) =>
                validActions[i] > 0 ? q : -Infinity
            );

            // Select action with highest Q-value
            const action = maskedQValues.indexOf(Math.max(...maskedQValues));

            // Track claw toggles
            if (action === 4) {
                this.lastClawToggleStep = this.stepCount;
            }

            // Debug logging (remove after testing)
            if (Math.random() < 0.01) {  // Log 1% of actions
                console.log('State:', {
                    angle1: state[0],
                    angle2: state[1],
                    blockHeld: state[6],
                    clawClosed: state[4],
                    distance: state[5]
                });
                console.log('Q-values:', qValues[0]);
                console.log('Valid actions:', validActions);
                console.log('Selected action:', action);
            }

            return action;
        } catch (error) {
            console.error('Error during action selection:', error);
            stateTensor.dispose();
            return Math.floor(Math.random() * 5);
        }
    }

    executeAction(action, robotArm) {
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
                robotArm.isClawClosed = !robotArm.isClawClosed;
                return true;
        }

        // Only apply the new angles if they're valid
        const success = robotArm.setTargetAngles(newAngle1, newAngle2);
        return success;
    }

    async update(robotArm, environment) {
        // Get current state
        const state = environment.getState(robotArm);

        // Select action using trained model
        const action = await this.selectAction(state);

        // Execute action
        this.executeAction(action, robotArm);

        // Check if task is complete
        const { reward, done } = environment.calculateReward(robotArm);

        if (done) {
            console.log('Task completed with reward:', reward);
            // Don't reset here - let the demo.html handle resets
            // This prevents double resets
        }
    }

    // Get a human-readable description of the action
    getActionDescription(action) {
        const actions = [
            'Increase angle1',
            'Decrease angle1',
            'Increase angle2',
            'Decrease angle2',
            'Toggle claw'
        ];
        return actions[action] || 'Unknown action';
    }
}