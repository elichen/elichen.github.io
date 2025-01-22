const ModelConfig = {
    EMOJI_MODELS: {
        '😀': '1F600',
        '💥': '1F4A5',
        '👁': '1F441',
        '🦎': '1F98E',
        '🐠': '1F420',
        '🦋': '1F98B',
        '🐞': '1F41E',
        '🕸': '1F578',
        '🥨': '1F968',
        '🎄': '1F384'
    },

    MODEL_TYPES: {
        'naive': { usePool: 0, damageN: 0 },
        'persistent': { usePool: 1, damageN: 0 },
        'regenerating': { usePool: 1, damageN: 3 }
    },

    getModelPath(emoji = '🦋', modelType = 'regenerating', fireRate = 0.5, run = 0) {
        let path = '';
        const config = this.MODEL_TYPES[modelType];
        
        if (fireRate === 0.5) {
            path += `use_sample_pool_${config.usePool}_damage_n_${config.damageN}_`;
        } else if (fireRate === 1.0) {
            path += 'fire_rate_1.0_';
        }
        
        const emojiCode = this.EMOJI_MODELS[emoji] || 
                         (typeof emoji === 'string' ? 
                             emoji.codePointAt(0).toString(16).toUpperCase() : 
                             this.EMOJI_MODELS['🦋']);
        
        path += `target_emoji_${emojiCode}_run_index_${run}`;
        return `${path}/08000.weights.h5.json`;
    }
}; 