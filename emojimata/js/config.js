const ModelConfig = {
    MODEL_TYPES: {
        'naive': { usePool: 0, damageN: 0 },
        'persistent': { usePool: 1, damageN: 0 },
        'regenerating': { usePool: 1, damageN: 3 }
    },

    getModelPath() {
        return `${path}/08000.weights.h5.json`;
    }
}; 