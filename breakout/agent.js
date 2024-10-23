class DQNAgent {
    constructor(inputSize, numActions) {
        this.inputSize = inputSize;
        this.numActions = numActions;
        this.model = new DQNModel(inputSize, numActions);
    }

    async loadModel(modelUrl) {
        await this.model.loadModel(modelUrl);
    }

    act(state) {
        const prediction = this.model.predict(state);
        const actionTensor = prediction.argMax(1);
        const action = actionTensor.dataSync()[0];

        prediction.dispose();
        actionTensor.dispose();

        return action;
    }
}
