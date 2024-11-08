class ReplayBuffer {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.buffer = [];              // AI experiences buffer
        this.humanBuffer = [];         // Permanent human experiences buffer
        this.position = 0;
    }

    store(experience, isHuman = false) {
        if (isHuman) {
            // Human experiences are always kept
            this.humanBuffer.push(experience);
        } else {
            // AI experiences use circular buffer
            if (this.buffer.length < this.maxSize) {
                this.buffer.push(experience);
            } else {
                this.buffer[this.position] = experience;
                this.position = (this.position + 1) % this.maxSize;
            }
        }
    }

    sample(batchSize) {
        const totalExperiences = this.buffer.length + this.humanBuffer.length;
        if (totalExperiences < batchSize) {
            return null;
        }

        const batch = [];
        const indices = new Set();
        
        // Determine how many samples to take from each buffer
        const humanSamples = Math.min(
            Math.floor(batchSize * 0.2),  // 20% of batch from human experiences
            this.humanBuffer.length        // But no more than available
        );
        const aiSamples = batchSize - humanSamples;

        // Sample from human buffer
        while (indices.size < humanSamples) {
            indices.add({
                isHuman: true,
                index: Math.floor(Math.random() * this.humanBuffer.length)
            });
        }

        // Sample from AI buffer
        while (indices.size < batchSize) {
            indices.add({
                isHuman: false,
                index: Math.floor(Math.random() * this.buffer.length)
            });
        }

        // Build batch from selected indices
        for (const {isHuman, index} of indices) {
            batch.push(isHuman ? this.humanBuffer[index] : this.buffer[index]);
        }

        return batch;
    }

    clear() {
        this.buffer = [];
        // Don't clear humanBuffer
        this.position = 0;
    }

    get size() {
        return this.buffer.length + this.humanBuffer.length;
    }

    get humanExperienceCount() {
        return this.humanBuffer.length;
    }

    get aiExperienceCount() {
        return this.buffer.length;
    }
} 