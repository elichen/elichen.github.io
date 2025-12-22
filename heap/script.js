// Heap's Algorithm Interactive Demos

// Generate all swap steps using Heap's algorithm
function generateHeapSteps(arr) {
    const steps = [];
    const a = [...arr];

    function generate(k) {
        if (k === 1) {
            steps.push({ type: 'output', state: [...a] });
            return;
        }

        for (let i = 0; i < k; i++) {
            generate(k - 1);

            if (i < k - 1) {
                const swapIdx = k % 2 === 0 ? i : 0;
                steps.push({
                    type: 'swap',
                    indices: [swapIdx, k - 1],
                    k: k,
                    before: [...a]
                });
                [a[swapIdx], a[k - 1]] = [a[k - 1], a[swapIdx]];
                steps.push({ type: 'output', state: [...a] });
            }
        }
    }

    generate(arr.length);
    return steps;
}

// Demo state
class Demo {
    constructor(id, elements) {
        this.id = id;
        this.elements = elements;
        this.steps = generateHeapSteps([...elements]);
        this.stepIndex = 0;
        this.playing = false;
        this.speed = 400;
        this.timeoutId = null;

        this.stateEl = document.querySelector(`#demo-${id} .current-state`);
        this.infoEl = document.getElementById(`swap-info-${id}`);
        this.trailEl = document.getElementById(`trail-${id}`);
        this.permCount = document.getElementById(`perm-count-${id}`);
        this.swapCount = document.getElementById(`swap-count-${id}`);
        this.levelEl = document.getElementById(`level-${id}`);

        this.permutations = 1;
        this.swaps = 0;

        this.setupControls();
    }

    setupControls() {
        const playBtn = document.getElementById(`play-${this.id}`);
        const stepBtn = document.getElementById(`step-${this.id}`);
        const resetBtn = document.getElementById(`reset-${this.id}`);
        const speedInput = document.getElementById(`speed-${this.id}`);

        if (playBtn) playBtn.addEventListener('click', () => this.togglePlay());
        if (stepBtn) stepBtn.addEventListener('click', () => this.step());
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
        if (speedInput) {
            speedInput.addEventListener('input', (e) => {
                this.speed = 1100 - parseInt(e.target.value); // Invert for intuitive control
            });
        }
    }

    render() {
        // Find current state
        let currentState = [...this.elements];
        for (let i = 0; i < this.stepIndex; i++) {
            if (this.steps[i].type === 'output') {
                currentState = [...this.steps[i].state];
            }
        }

        // Render elements
        this.stateEl.innerHTML = currentState.map((el, idx) =>
            `<div class="element" data-idx="${idx}">${el}</div>`
        ).join('');
    }

    highlightSwap(indices) {
        const elements = this.stateEl.querySelectorAll('.element');
        indices.forEach(idx => {
            if (elements[idx]) elements[idx].classList.add('swapping');
        });
    }

    clearHighlight() {
        this.stateEl.querySelectorAll('.element').forEach(el => {
            el.classList.remove('swapping');
        });
    }

    addToTrail(perm) {
        // Remove active from previous
        this.trailEl.querySelectorAll('.trail-item').forEach(el => {
            el.classList.remove('active');
        });

        const item = document.createElement('div');
        item.className = 'trail-item active';
        item.textContent = perm.join('');
        this.trailEl.appendChild(item);
    }

    updateStats() {
        if (this.permCount) this.permCount.textContent = this.permutations;
        if (this.swapCount) this.swapCount.textContent = this.swaps;
    }

    updateLevel(k) {
        if (this.levelEl) {
            this.levelEl.innerHTML = `<span class="level-badge">k=${k} (${k % 2 === 0 ? 'even' : 'odd'})</span>`;
        }
    }

    step() {
        if (this.stepIndex >= this.steps.length) {
            this.playing = false;
            this.updatePlayButton();
            return false;
        }

        const step = this.steps[this.stepIndex];

        if (step.type === 'swap') {
            this.highlightSwap(step.indices);
            const swapType = step.k % 2 === 0 ? 'even (rotate)' : 'odd (pivot)';
            this.infoEl.textContent = `Swap positions ${step.indices[0]} ↔ ${step.indices[1]} [k=${step.k}, ${swapType}]`;
            this.updateLevel(step.k);
            this.swaps++;
            this.updateStats();
        } else if (step.type === 'output') {
            this.clearHighlight();
            this.render();

            if (this.stepIndex > 0) {
                this.addToTrail(step.state);
                this.permutations++;
                this.updateStats();
            }
        }

        this.stepIndex++;

        return true;
    }

    togglePlay() {
        if (this.playing) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        this.playing = true;
        this.updatePlayButton();
        this.runStep();
    }

    pause() {
        this.playing = false;
        this.updatePlayButton();
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
    }

    runStep() {
        if (!this.playing) return;

        const hasMore = this.step();
        if (hasMore && this.playing) {
            this.timeoutId = setTimeout(() => this.runStep(), this.speed);
        }
    }

    updatePlayButton() {
        const playBtn = document.getElementById(`play-${this.id}`);
        if (playBtn) {
            playBtn.textContent = this.playing ? '⏸ Pause' : '▶ Play';
        }
    }

    reset() {
        this.pause();
        this.stepIndex = 0;
        this.permutations = 1;
        this.swaps = 0;

        this.render();
        this.clearHighlight();

        this.trailEl.innerHTML = `<div class="trail-item active">${this.elements.join('')}</div>`;
        this.infoEl.textContent = 'Click Play or Step';

        this.updateStats();
        if (this.levelEl) {
            this.levelEl.innerHTML = `<span class="level-badge">k=${this.elements.length}</span>`;
        }
    }
}

// Initialize demos when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Demo for 3 elements
    const demo3 = new Demo('3', ['A', 'B', 'C']);
    demo3.render();

    // Demo for 4 elements
    const demo4 = new Demo('4', ['A', 'B', 'C', 'D']);
    demo4.render();
});
