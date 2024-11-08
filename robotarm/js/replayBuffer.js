class ReplayBuffer {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.buffer = [];
        this.position = 0;
    }

    store(experience) {
        if (this.buffer.length < this.maxSize) {
            this.buffer.push(experience);
        } else {
            this.buffer[this.position] = experience;
            this.position = (this.position + 1) % this.maxSize;
        }
    }

    sample(batchSize) {
        if (this.buffer.length < batchSize) {
            return null;
        }

        const batch = [];
        const indices = new Set();

        while (indices.size < batchSize) {
            indices.add(Math.floor(Math.random() * this.buffer.length));
        }

        for (const index of indices) {
            batch.push(this.buffer[index]);
        }

        return batch;
    }

    clear() {
        this.buffer = [];
        this.position = 0;
    }

    get size() {
        return this.buffer.length;
    }
} 