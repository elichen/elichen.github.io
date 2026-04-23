// Heap's Algorithm Interactive Demos

// Generate all swap steps using Heap's algorithm
function generateHeapSteps(arr) {
    const steps = [];
    const a = [...arr];

    function generate(k) {
        if (k === 1) {
            steps.push({ type: 'output', state: [...a], k });
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
                // Don't push output here - next generate(k-1) will output at base case
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
        this.currentState = [...elements];
        this.playing = false;
        this.speed = 400;
        this.totalPermutations = this.steps.filter((step) => step.type === 'output').length;
        this.totalSwaps = this.steps.filter((step) => step.type === 'swap').length;
        this.timeoutId = null;
        this.animationFrameId = null;
        this.tiles = new Map();
        this.phase = 'current';
        this.motionQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
        this.reducedMotion = this.motionQuery ? this.motionQuery.matches : false;

        this.stateEl = document.querySelector(`#demo-${id} .current-state`);
        this.infoEl = document.getElementById(`swap-info-${id}`);
        this.traceEl = document.getElementById(`trace-${id}`);
        this.trailEl = document.getElementById(`trail-${id}`);
        this.permCount = document.getElementById(`perm-count-${id}`);
        this.swapCount = document.getElementById(`swap-count-${id}`);
        this.progressEl = document.getElementById(`progress-${id}`);
        this.progressText = document.getElementById(`progress-text-${id}`);
        this.levelEl = document.getElementById(`level-${id}`);
        this.blockEl = document.getElementById(`block-${id}`);
        this.playBtn = document.getElementById(`play-${this.id}`);
        this.stepBtn = document.getElementById(`step-${this.id}`);
        this.resetBtn = document.getElementById(`reset-${this.id}`);

        this.permutations = 1;
        this.swaps = 0;

        this.stateEl.innerHTML = '';

        this.setupMotionPreferences();
        this.setupControls();
    }

    setupMotionPreferences() {
        if (!this.motionQuery) {
            this.applyMotionSettings(this.speed);
            return;
        }

        const updateMotionPreference = (event) => {
            this.reducedMotion = event.matches;
            this.applyMotionSettings(this.speed);
            this.syncTiles();
        };

        if (this.motionQuery.addEventListener) {
            this.motionQuery.addEventListener('change', updateMotionPreference);
        } else {
            this.motionQuery.addListener(updateMotionPreference);
        }
    }

    setupControls() {
        const speedInput = document.getElementById(`speed-${this.id}`);

        if (this.playBtn) this.playBtn.addEventListener('click', () => this.togglePlay());
        if (this.stepBtn) this.stepBtn.addEventListener('click', () => this.step());
        if (this.resetBtn) this.resetBtn.addEventListener('click', () => this.reset());
        if (speedInput) {
            this.updateSpeed(speedInput.value);
            speedInput.addEventListener('input', (e) => {
                this.updateSpeed(e.target.value);
            });
        } else {
            this.applyMotionSettings(this.speed);
        }
    }

    updateSpeed(controlValue) {
        const sliderValue = parseInt(controlValue, 10);
        if (Number.isNaN(sliderValue)) return;

        this.applyMotionSettings(1100 - sliderValue);
    }

    applyMotionSettings(speed) {
        this.speed = speed;

        const motionDuration = this.reducedMotion
            ? 0
            : Math.max(80, Math.min(320, speed - 60));
        const trailDuration = this.reducedMotion
            ? 0
            : Math.max(100, Math.min(220, motionDuration - 30));

        this.stateEl.style.setProperty('--motion-ms', `${motionDuration}ms`);
        if (this.trailEl) {
            this.trailEl.style.setProperty('--trail-ms', `${trailDuration}ms`);
        }
    }

    getTile(value) {
        if (!this.tiles.has(value)) {
            const tile = document.createElement('div');
            tile.className = 'element';
            tile.textContent = value;
            tile.dataset.value = value;
            tile.style.setProperty('--tx', '0px');
            tile.style.setProperty('--ty', '0px');
            this.tiles.set(value, tile);
        }

        return this.tiles.get(value);
    }

    render() {
        this.phase = 'current';
        this.stepIndex = this.steps.length > 0 ? 1 : 0;
        this.syncTiles();
        this.resetTrail();
        this.updateLevel(this.elements.length);
        this.updateBlockState(this.currentState, 'current');
        this.setOutputInfo(this.currentState);
        this.updateStats();
        this.updateProgress();
        this.updateNextAction();
        this.updateControls();
    }

    syncTiles() {
        this.cancelPendingAnimation();

        this.currentState.forEach((value) => {
            const tile = this.getTile(value);
            tile.classList.remove('swapping');
            tile.style.transition = '';
            tile.style.setProperty('--tx', '0px');
            tile.style.setProperty('--ty', '0px');
            this.stateEl.appendChild(tile);
        });

        this.applyTileRoles();
    }

    captureTilePositions() {
        const positions = new Map();

        this.currentState.forEach((value) => {
            positions.set(value, this.getTile(value).getBoundingClientRect());
        });

        return positions;
    }

    cancelPendingAnimation() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    highlightValues(values) {
        this.tiles.forEach((tile, value) => {
            tile.classList.toggle('swapping', values.has(value));
        });
    }

    clearHighlight() {
        this.highlightValues(new Set());
    }

    applyTileRoles() {
        const fixedValue = this.currentState[this.currentState.length - 1];

        this.tiles.forEach((tile, value) => {
            const isFixedTail = value === fixedValue;
            tile.classList.toggle('fixed-tail', isFixedTail);
            tile.classList.toggle('preparing-tail', isFixedTail && this.phase === 'preparing');
        });
    }

    getBlockMeta(state) {
        return {
            slot: state.length - 1,
            value: state[state.length - 1]
        };
    }

    updateBlockState(state, mode) {
        if (!this.blockEl) return;

        const { slot, value } = this.getBlockMeta(state);
        const blockLabel = mode === 'preparing' ? 'Preparing next block' : 'Current block';
        this.blockEl.innerHTML = `
            <span class="block-badge ${mode}">
                ${blockLabel}: slot ${slot} fixed = ${value}
            </span>
        `;
    }

    setOutputInfo(state) {
        if (!this.infoEl) return;

        const { slot, value } = this.getBlockMeta(state);
        this.infoEl.textContent = `Output ${state.join('')} [slot ${slot} fixed = ${value}]`;
    }

    createTrailGroup(state) {
        const { slot, value } = this.getBlockMeta(state);
        const blockNumber = this.trailEl.querySelectorAll('.trail-group').length + 1;

        const group = document.createElement('section');
        group.className = 'trail-group current';
        group.dataset.slot = String(slot);
        group.dataset.fixed = value;

        const label = document.createElement('div');
        label.className = 'trail-group-label';
        label.innerHTML = `
            <span class="trail-group-title">Block ${blockNumber}</span>
            <span class="trail-group-fixed">slot ${slot} fixed = ${value}</span>
        `;

        const items = document.createElement('div');
        items.className = 'trail-group-items';

        group.append(label, items);
        return group;
    }

    getOrCreateTrailGroup(state) {
        const { slot, value } = this.getBlockMeta(state);
        let group = this.trailEl.lastElementChild;

        if (
            !group ||
            !group.classList.contains('trail-group') ||
            group.dataset.slot !== String(slot) ||
            group.dataset.fixed !== value
        ) {
            group = this.createTrailGroup(state);
            this.trailEl.appendChild(group);
        }

        this.trailEl.querySelectorAll('.trail-group').forEach((el) => {
            el.classList.toggle('current', el === group);
        });

        return group;
    }

    resetTrail() {
        if (!this.trailEl) return;

        this.trailEl.innerHTML = '';

        const group = this.getOrCreateTrailGroup(this.currentState);
        const item = document.createElement('div');
        item.className = 'trail-item active';
        item.textContent = this.currentState.join('');
        group.querySelector('.trail-group-items').appendChild(item);
    }

    animateStateChange(movingValues) {
        this.cancelPendingAnimation();

        if (this.reducedMotion) {
            this.currentState.forEach((value) => {
                const tile = this.getTile(value);
                tile.style.transition = '';
                tile.style.setProperty('--tx', '0px');
                tile.style.setProperty('--ty', '0px');
                this.stateEl.appendChild(tile);
            });
            this.applyTileRoles();
            this.highlightValues(movingValues);
            return;
        }

        const before = this.captureTilePositions();

        this.currentState.forEach((value) => {
            this.stateEl.appendChild(this.getTile(value));
        });

        const orderedTiles = this.currentState.map((value) => this.getTile(value));
        orderedTiles.forEach((tile) => {
            const first = before.get(tile.dataset.value);
            const last = tile.getBoundingClientRect();
            const deltaX = first ? first.left - last.left : 0;
            const deltaY = first ? first.top - last.top : 0;

            tile.style.transition = 'none';
            tile.style.setProperty('--tx', `${deltaX}px`);
            tile.style.setProperty('--ty', `${deltaY}px`);
        });

        this.applyTileRoles();
        this.highlightValues(movingValues);
        this.stateEl.getBoundingClientRect();

        this.animationFrameId = requestAnimationFrame(() => {
            this.animationFrameId = null;

            orderedTiles.forEach((tile) => {
                tile.style.transition = '';
                tile.style.setProperty('--tx', '0px');
                tile.style.setProperty('--ty', '0px');
            });
        });
    }

    addToTrail(perm) {
        // Remove active from previous
        this.trailEl.querySelectorAll('.trail-item').forEach(el => {
            el.classList.remove('active');
        });

        const group = this.getOrCreateTrailGroup(perm);
        const item = document.createElement('div');
        item.className = 'trail-item active entering';
        item.textContent = perm.join('');
        group.querySelector('.trail-group-items').appendChild(item);

        if (this.reducedMotion) {
            item.classList.remove('entering');
            return;
        }

        item.getBoundingClientRect();
        requestAnimationFrame(() => {
            item.classList.remove('entering');
        });
    }

    updateStats() {
        if (this.permCount) this.permCount.textContent = this.permutations;
        if (this.swapCount) this.swapCount.textContent = this.swaps;
    }

    updateProgress() {
        const outputRatio = this.totalPermutations
            ? this.permutations / this.totalPermutations
            : 0;

        if (this.progressEl) {
            this.progressEl.style.width = `${clamp(outputRatio * 100, 0, 100)}%`;
        }

        if (this.progressText) {
            this.progressText.textContent = `${this.permutations}/${this.totalPermutations} outputs · ${this.swaps}/${this.totalSwaps} swaps`;
        }
    }

    updateLevel(k) {
        if (this.levelEl) {
            this.levelEl.innerHTML = `<span class="level-badge">k=${k} (${k % 2 === 0 ? 'even' : 'odd'})</span>`;
        }
    }

    describeStep(step) {
        if (!step) {
            return {
                kind: 'done',
                title: 'All permutations emitted',
                detail: `${this.totalPermutations} outputs reached with ${this.totalSwaps} swaps. Reset to replay the schedule.`
            };
        }

        if (step.type === 'output') {
            return {
                kind: 'output',
                title: `Emit ${step.state.join('')}`,
                detail: 'The recursion reached k=1, so the current arrangement becomes the next output.'
            };
        }

        const [i, j] = step.indices;
        const parity = step.k % 2 === 0 ? 'even' : 'odd';
        const rule = parity === 'even'
            ? `k=${step.k} is even, so swap loop index ${i} with the last slot.`
            : `k=${step.k} is odd, so swap position 0 with the last slot.`;

        return {
            kind: 'swap',
            title: `Swap positions ${i} and ${j}`,
            detail: rule
        };
    }

    updateNextAction() {
        if (!this.traceEl) return;

        const summary = this.describeStep(this.steps[this.stepIndex]);
        this.traceEl.dataset.kind = summary.kind;
        this.traceEl.innerHTML = `
            <span class="trace-kicker">Next action</span>
            <strong>${summary.title}</strong>
            <span>${summary.detail}</span>
        `;
    }

    updateControls() {
        const complete = this.stepIndex >= this.steps.length;
        this.updatePlayButton();

        if (this.stepBtn) this.stepBtn.disabled = complete || this.playing;
        if (this.resetBtn) this.resetBtn.disabled = this.stepIndex <= 1 && this.permutations === 1 && this.swaps === 0;
    }

    step() {
        if (this.stepIndex >= this.steps.length) {
            this.playing = false;
            this.updateNextAction();
            this.updateControls();
            return false;
        }

        const step = this.steps[this.stepIndex];

        if (step.type === 'swap') {
            // Perform the swap on our state
            const [i, j] = step.indices;
            const movingValues = new Set([this.currentState[i], this.currentState[j]]);
            const isTopLevelSwap = step.k === this.elements.length;
            [this.currentState[i], this.currentState[j]] = [this.currentState[j], this.currentState[i]];
            this.phase = isTopLevelSwap ? 'preparing' : 'current';
            this.animateStateChange(movingValues);
            this.updateBlockState(this.currentState, isTopLevelSwap ? 'preparing' : 'current');

            const swapRule = step.k % 2 === 0
                ? `even: swap i (${step.indices[0]}) with last`
                : 'odd: swap 0 with last';
            this.infoEl.textContent = isTopLevelSwap
                ? `Preparing next block: swap positions ${step.indices[0]} ↔ ${step.indices[1]} [k=${step.k}, ${swapRule}]`
                : `Swap inside current block: positions ${step.indices[0]} ↔ ${step.indices[1]} [k=${step.k}, ${swapRule}]`;
            this.updateLevel(step.k);
            this.swaps++;
            this.updateStats();
        } else if (step.type === 'output') {
            this.clearHighlight();
            this.phase = 'current';
            this.currentState = [...step.state];
            this.syncTiles();
            this.updateLevel(this.elements.length);
            this.updateBlockState(step.state, 'current');
            this.setOutputInfo(step.state);

            if (this.stepIndex > 0) {
                this.addToTrail(step.state);
                this.permutations++;
                this.updateStats();
            }
        }

        this.stepIndex++;
        this.updateProgress();
        this.updateNextAction();
        this.updateControls();

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
        if (this.stepIndex >= this.steps.length) {
            this.reset();
        }
        this.playing = true;
        this.updateControls();
        this.runStep();
    }

    pause() {
        this.playing = false;
        this.updateControls();
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
        if (this.playBtn) {
            this.playBtn.textContent = this.playing ? 'Pause' : 'Play';
        }
    }

    reset() {
        this.pause();
        this.cancelPendingAnimation();
        this.stepIndex = this.steps.length > 0 ? 1 : 0;
        this.currentState = [...this.elements];
        this.permutations = 1;
        this.swaps = 0;
        this.phase = 'current';

        this.clearHighlight();
        this.syncTiles();
        this.resetTrail();

        this.updateStats();
        this.updateProgress();
        this.updateLevel(this.elements.length);
        this.updateBlockState(this.currentState, 'current');
        this.setOutputInfo(this.currentState);
        this.updateNextAction();
        this.updateControls();
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
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
