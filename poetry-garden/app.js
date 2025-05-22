class PoetryGarden {
    constructor() {
        this.canvas = document.getElementById('garden-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.plantedSeeds = new Map();
        this.growingPoems = new Set();
        this.harvestedPoems = [];
        this.draggedSeed = null;
        this.particles = [];
        
        this.initializeCanvas();
        this.setupEventListeners();
        this.startAnimationLoop();
        
        // Poetry generation templates and patterns
        this.poetryTemplates = {
            haiku: {
                syllables: [5, 7, 5],
                structure: 3
            },
            'free-verse': {
                lines: [2, 4],
                structure: 'flowing'
            },
            'rhyming': {
                structure: 'AABB',
                lines: 4
            },
            'minimalist': {
                lines: [1, 3],
                structure: 'sparse'
            }
        };
        
        this.poetryWords = {
            nature: {
                moonlight: ['silver', 'gentle', 'whispers', 'shadows', 'dreams', 'night'],
                ocean: ['waves', 'endless', 'deep', 'salt', 'horizon', 'blue'],
                forest: ['ancient', 'green', 'silence', 'trees', 'mystery', 'paths'],
                starlight: ['distant', 'bright', 'guide', 'eternal', 'dance', 'cosmos'],
                wind: ['gentle', 'carries', 'stories', 'freedom', 'touch', 'movement'],
                sunrise: ['golden', 'hope', 'awakening', 'warmth', 'new', 'light']
            },
            emotions: {
                longing: ['distant', 'heart', 'yearning', 'memory', 'reach', 'empty'],
                joy: ['laughter', 'bright', 'dancing', 'celebration', 'light', 'singing'],
                melancholy: ['autumn', 'rain', 'quiet', 'thoughtful', 'gray', 'solitude'],
                wonder: ['curious', 'magic', 'questions', 'amazement', 'discovery', 'awe'],
                peace: ['calm', 'stillness', 'harmony', 'breath', 'centered', 'quiet'],
                hope: ['tomorrow', 'rising', 'possibility', 'faith', 'growing', 'bloom']
            },
            abstract: {
                time: ['flowing', 'endless', 'moments', 'eternal', 'passing', 'memory'],
                memory: ['fading', 'precious', 'golden', 'whispers', 'holds', 'treasured'],
                dream: ['floating', 'colors', 'impossible', 'flight', 'wonder', 'magic'],
                silence: ['profound', 'speaks', 'empty', 'full', 'listening', 'space'],
                infinity: ['endless', 'circle', 'beyond', 'limitless', 'eternal', 'vast'],
                shadow: ['dancing', 'following', 'mystery', 'depth', 'contrast', 'hidden']
            }
        };
    }

    initializeCanvas() {
        const resizeCanvas = () => {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
        };
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Draw subtle background pattern
        this.drawBackground();
    }

    drawBackground() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Create a subtle grid pattern
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        
        for (let x = 0; x < width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        for (let y = 0; y < height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }

    setupEventListeners() {
        // Only setup drag and drop
        this.setupDragAndDrop();

        // Custom seed input
        document.getElementById('add-custom').addEventListener('click', () => this.addCustomSeed());
        document.getElementById('custom-word').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addCustomSeed();
        });

        // Controls
        document.getElementById('clear-garden').addEventListener('click', () => this.clearGarden());
        document.getElementById('harvest-poems').addEventListener('click', () => this.harvestAllPoems());
    }


    setupDragAndDrop() {
        // Make all current seeds draggable
        this.makeSeedsDraggable();

        // Garden drop zone
        const gardenPlot = document.querySelector('.garden-plot');
        
        gardenPlot.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            gardenPlot.classList.add('drag-over');
        });
        
        gardenPlot.addEventListener('dragenter', (e) => {
            e.preventDefault();
        });
        
        gardenPlot.addEventListener('dragleave', (e) => {
            gardenPlot.classList.remove('drag-over');
        });
        
        gardenPlot.addEventListener('drop', (e) => {
            e.preventDefault();
            gardenPlot.classList.remove('drag-over');
            
            const word = e.dataTransfer.getData('text/plain');
            if (word) {
                const rect = gardenPlot.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                this.plantSeed(word, x, y);
            }
        });
        
        // Fallback: mouse-based drag implementation
        this.setupMouseDrag();
    }

    makeSeedsDraggable() {
        const seeds = document.querySelectorAll('.seed');
        
        seeds.forEach((seed, index) => {
            if (seed.hasAttribute('data-drag-setup')) {
                return; // Already set up
            }
            
            seed.draggable = true;
            seed.style.cursor = 'grab';
            seed.setAttribute('data-drag-setup', 'true');
            
            seed.addEventListener('dragstart', (e) => {
                const word = e.target.dataset.word || e.target.textContent;
                e.dataTransfer.setData('text/plain', word);
                e.dataTransfer.setData('text', word);
                e.dataTransfer.effectAllowed = 'all';
                e.target.classList.add('dragging');
            });
            
            seed.addEventListener('dragend', (e) => {
                e.target.classList.remove('dragging');
            });
        });
    }

    setupMouseDrag() {
        let isDragging = false;
        let draggedWord = null;
        let dragElement = null;
        
        document.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('seed')) {
                isDragging = true;
                draggedWord = e.target.dataset.word || e.target.textContent;
                
                // Create visual drag element
                dragElement = e.target.cloneNode(true);
                dragElement.style.cssText = `
                    position: fixed;
                    pointer-events: none;
                    z-index: 1000;
                    opacity: 0.8;
                    transform: translate(${e.clientX - 20}px, ${e.clientY - 10}px) scale(0.9);
                    transition: none !important;
                    animation: none !important;
                    left: 0;
                    top: 0;
                `;
                document.body.appendChild(dragElement);
                
                e.preventDefault();
            }
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging && dragElement) {
                // Use transform for better performance
                dragElement.style.transform = `translate(${e.clientX - 20}px, ${e.clientY - 10}px) scale(0.9)`;
            }
        });
        
        document.addEventListener('mouseup', (e) => {
            if (isDragging && draggedWord) {
                // Check if we're over the garden
                const gardenPlot = document.querySelector('.garden-plot');
                const rect = gardenPlot.getBoundingClientRect();
                
                if (e.clientX >= rect.left && e.clientX <= rect.right &&
                    e.clientY >= rect.top && e.clientY <= rect.bottom) {
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    this.plantSeed(draggedWord, x, y);
                    this.createMagicalSparkles(x, y);
                }
                
                // Cleanup
                if (dragElement) {
                    document.body.removeChild(dragElement);
                    dragElement = null;
                }
                
                isDragging = false;
                draggedWord = null;
            }
        });
    }

    addCustomSeed() {
        const input = document.getElementById('custom-word');
        const word = input.value.trim();
        
        if (word) {
            // Create temporary seed element
            const tempSeed = document.createElement('span');
            tempSeed.className = 'seed';
            tempSeed.textContent = word;
            tempSeed.dataset.word = word;
            
            // Add to custom seeds area - find a better place to append
            const customArea = document.querySelector('.custom-seed');
            customArea.parentNode.insertBefore(tempSeed, customArea);
            
            // Make it draggable using our helper function
            this.makeSeedsDraggable();
            
            input.value = '';
            
            // Auto-remove after some time to keep UI clean
            setTimeout(() => {
                if (tempSeed.parentNode) {
                    tempSeed.parentNode.removeChild(tempSeed);
                }
            }, 60000);
        }
    }

    plantSeed(word, x, y) {
        const seedId = `seed-${Date.now()}-${Math.random()}`;
        
        // Create planted seed element
        const plantedSeed = document.createElement('div');
        plantedSeed.className = 'planted-seed';
        plantedSeed.textContent = word;
        plantedSeed.style.left = `${x - 30}px`;
        plantedSeed.style.top = `${y - 15}px`;
        plantedSeed.addEventListener('click', () => this.growPoetryFromSeed(seedId));
        
        document.querySelector('.planted-seeds').appendChild(plantedSeed);
        
        // Store seed data
        this.plantedSeeds.set(seedId, {
            word,
            x: x - 30,
            y: y - 15,
            element: plantedSeed,
            hasGrown: false
        });

        // Create planting particles
        this.createPlantingParticles(x, y);
        
        // Add hover suggestions for planted seeds
        this.addSeedSuggestions(plantedSeed, word);
        
        // Auto-grow after a delay
        setTimeout(() => {
            if (!this.plantedSeeds.get(seedId)?.hasGrown) {
                this.growPoetryFromSeed(seedId);
            }
        }, 2000 + Math.random() * 3000);
    }

    createPlantingParticles(x, y) {
        for (let i = 0; i < 8; i++) {
            const particle = {
                x: x + (Math.random() - 0.5) * 20,
                y: y + (Math.random() - 0.5) * 20,
                vx: (Math.random() - 0.5) * 4,
                vy: (Math.random() - 0.5) * 4,
                life: 1,
                decay: 0.02
            };
            this.particles.push(particle);
        }
    }

    growPoetryFromSeed(seedId) {
        const seed = this.plantedSeeds.get(seedId);
        if (!seed || seed.hasGrown) return;

        seed.hasGrown = true;
        seed.element.style.animation = 'none';

        const poem = this.generatePoetry(seed.word);
        this.displayGrowingPoem(poem, seed.x + 60, seed.y, seedId);
    }

    generatePoetry(seedWord) {
        const style = document.getElementById('poetry-style').value;
        const template = this.poetryTemplates[style];
        
        // Find word category and related words
        let relatedWords = [];
        let category = 'abstract';
        
        for (const [cat, words] of Object.entries(this.poetryWords)) {
            if (words[seedWord]) {
                relatedWords = words[seedWord];
                category = cat;
                break;
            }
        }
        
        // If not found, create generic related words
        if (relatedWords.length === 0) {
            relatedWords = ['gentle', 'mysterious', 'flowing', 'eternal', 'whispers', 'dreams'];
        }

        return this.createPoetryByStyle(style, seedWord, relatedWords, category);
    }

    createPoetryByStyle(style, seedWord, relatedWords, category) {
        const poems = {
            haiku: () => {
                const lines = [
                    `${this.capitalize(seedWord)} ${this.randomChoice(relatedWords)}`,
                    `${this.randomChoice(relatedWords)} ${this.randomChoice(relatedWords)} through the ${this.randomChoice(['night', 'day', 'moment', 'space'])}`,
                    `${this.randomChoice(relatedWords)} ${this.randomChoice(['remains', 'echoes', 'flows', 'whispers'])}`
                ];
                return lines;
            },
            'free-verse': () => {
                const lines = [
                    `In the ${seedWord} I find`,
                    `${this.randomChoice(relatedWords)} moments`,
                    `that ${this.randomChoice(relatedWords)} and ${this.randomChoice(relatedWords)}`,
                    `like ${this.randomChoice(['memories', 'dreams', 'whispers', 'shadows'])}`
                ];
                return lines;
            },
            'rhyming': () => {
                const rhymePairs = this.generateRhymingPairs();
                const lines = [
                    `${this.capitalize(seedWord)} ${this.randomChoice(relatedWords)} in the ${rhymePairs[0][0]}`,
                    `Bringing ${this.randomChoice(relatedWords)} to the ${rhymePairs[0][1]}`,
                    `Where ${this.randomChoice(relatedWords)} ${this.randomChoice(['dance', 'flow', 'whisper'])} ${rhymePairs[1][0]}`,
                    `And ${this.randomChoice(relatedWords)} ${this.randomChoice(['shine', 'glow', 'shimmer'])} ${rhymePairs[1][1]}`
                ];
                return lines;
            },
            'minimalist': () => {
                const lines = [
                    `${seedWord}.`,
                    `${this.randomChoice(relatedWords)}.`,
                    `${this.randomChoice(['silence', 'space', 'breath'])}.`
                ];
                return lines.slice(0, Math.random() > 0.5 ? 2 : 3);
            }
        };

        return poems[style] ? poems[style]() : poems['free-verse']();
    }

    generateRhymingPairs() {
        const rhymes = [
            ['night', 'light'], ['day', 'way'], ['sea', 'free'], ['sky', 'high'],
            ['dream', 'stream'], ['heart', 'start'], ['soul', 'whole'], ['mind', 'find']
        ];
        return [this.randomChoice(rhymes), this.randomChoice(rhymes)];
    }

    randomChoice(array) {
        return array[Math.floor(Math.random() * array.length)];
    }

    capitalize(word) {
        return word.charAt(0).toUpperCase() + word.slice(1);
    }

    displayGrowingPoem(lines, x, y, seedId) {
        const poemElement = document.createElement('div');
        poemElement.className = 'growing-poem';
        poemElement.style.left = `${Math.min(x, this.canvas.width - 250)}px`;
        poemElement.style.top = `${Math.min(y, this.canvas.height - 100)}px`;
        
        // Add lines with staggered animation
        lines.forEach((line, index) => {
            const lineElement = document.createElement('span');
            lineElement.className = 'poem-line';
            lineElement.textContent = line;
            lineElement.style.animationDelay = `${index * 0.5}s`;
            poemElement.appendChild(lineElement);
        });

        // Add harvest button
        const harvestBtn = document.createElement('button');
        harvestBtn.textContent = '🌸 Harvest';
        harvestBtn.style.cssText = `
            margin-top: 10px;
            padding: 5px 10px;
            border: none;
            border-radius: 5px;
            background: linear-gradient(45deg, #4facfe, #00f2fe);
            color: white;
            cursor: pointer;
            font-size: 0.8rem;
        `;
        harvestBtn.addEventListener('click', () => this.harvestPoem(lines, seedId, poemElement));
        poemElement.appendChild(harvestBtn);

        document.querySelector('.growing-poems').appendChild(poemElement);
        this.growingPoems.add(poemElement);

        // Auto-harvest after some time
        setTimeout(() => {
            if (document.body.contains(poemElement)) {
                this.harvestPoem(lines, seedId, poemElement);
            }
        }, 15000);
    }

    harvestPoem(lines, seedId, poemElement) {
        const poem = {
            text: lines.join('\n'),
            timestamp: new Date().toLocaleString(),
            seedWord: this.plantedSeeds.get(seedId)?.word || 'unknown',
            style: document.getElementById('poetry-style').value
        };

        this.harvestedPoems.push(poem);
        this.displayHarvestedPoem(poem);
        
        // Remove from garden with animation
        poemElement.style.animation = 'poem-grow 0.5s ease-in reverse';
        setTimeout(() => {
            if (poemElement.parentNode) {
                poemElement.parentNode.removeChild(poemElement);
            }
            this.growingPoems.delete(poemElement);
        }, 500);
    }

    displayHarvestedPoem(poem) {
        const harvestedContainer = document.getElementById('harvested-poems');
        const poemElement = document.createElement('div');
        poemElement.className = 'harvested-poem';
        
        poemElement.innerHTML = `
            <div class="poem-meta">
                ${poem.style} • from "${poem.seedWord}" • ${poem.timestamp}
            </div>
            <div class="poem-text">${poem.text.replace(/\n/g, '<br>')}</div>
        `;
        
        harvestedContainer.insertBefore(poemElement, harvestedContainer.firstChild);
    }

    harvestAllPoems() {
        const growingPoems = Array.from(this.growingPoems);
        growingPoems.forEach(poemElement => {
            const harvestBtn = poemElement.querySelector('button');
            if (harvestBtn) {
                harvestBtn.click();
            }
        });
    }

    clearGarden() {
        // Clear planted seeds
        document.querySelector('.planted-seeds').innerHTML = '';
        this.plantedSeeds.clear();
        
        // Clear growing poems
        document.querySelector('.growing-poems').innerHTML = '';
        this.growingPoems.clear();
        
        // Clear particles
        this.particles = [];
        
        // Redraw background
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawBackground();
    }

    startAnimationLoop() {
        const animate = () => {
            this.updateParticles();
            requestAnimationFrame(animate);
        };
        animate();
    }

    updateParticles() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawBackground();
        
        // Update and draw particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.life -= particle.decay;
            
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
                continue;
            }
            
            // Draw particle
            this.ctx.save();
            this.ctx.globalAlpha = particle.life;
            this.ctx.fillStyle = particle.color || '#4facfe';
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size || 2, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();
        }
    }

    createMagicalSparkles(x, y) {
        const colors = ['#f093fb', '#4facfe', '#00f2fe', '#f5576c', '#ffd700'];
        for (let i = 0; i < 12; i++) {
            const particle = {
                x: x + (Math.random() - 0.5) * 30,
                y: y + (Math.random() - 0.5) * 30,
                vx: (Math.random() - 0.5) * 6,
                vy: (Math.random() - 0.5) * 6,
                life: 1,
                decay: 0.015,
                color: colors[Math.floor(Math.random() * colors.length)],
                size: Math.random() * 3 + 1
            };
            this.particles.push(particle);
        }
    }

    addSeedSuggestions(plantedSeed, word) {
        let suggestionTooltip = null;
        
        // Find complementary words
        const suggestions = this.getComplementaryWords(word);
        
        plantedSeed.addEventListener('mouseenter', () => {
            suggestionTooltip = document.createElement('div');
            suggestionTooltip.className = 'suggestion-tooltip';
            suggestionTooltip.innerHTML = `
                <div class="suggestion-header">✨ Try pairing with:</div>
                <div class="suggestion-words">
                    ${suggestions.map(w => `<span class="suggestion-word">${w}</span>`).join('')}
                </div>
            `;
            
            const rect = plantedSeed.getBoundingClientRect();
            const gardenRect = document.querySelector('.garden-plot').getBoundingClientRect();
            
            suggestionTooltip.style.position = 'absolute';
            suggestionTooltip.style.left = `${rect.left - gardenRect.left + 50}px`;
            suggestionTooltip.style.top = `${rect.top - gardenRect.top - 10}px`;
            
            document.querySelector('.garden-plot').appendChild(suggestionTooltip);
        });
        
        plantedSeed.addEventListener('mouseleave', () => {
            if (suggestionTooltip) {
                suggestionTooltip.remove();
                suggestionTooltip = null;
            }
        });
    }

    getComplementaryWords(word) {
        const complementaryPairs = {
            moonlight: ['shadows', 'whispers', 'silver'],
            ocean: ['waves', 'horizon', 'depths'],
            forest: ['whispers', 'ancient', 'green'],
            starlight: ['cosmos', 'eternal', 'guide'],
            wind: ['freedom', 'stories', 'gentle'],
            sunrise: ['golden', 'hope', 'warmth'],
            longing: ['distance', 'heart', 'yearning'],
            joy: ['laughter', 'bright', 'celebration'],
            melancholy: ['rain', 'autumn', 'solitude'],
            wonder: ['magic', 'discovery', 'awe'],
            peace: ['stillness', 'harmony', 'calm'],
            hope: ['tomorrow', 'rising', 'bloom'],
            time: ['flowing', 'eternal', 'moments'],
            memory: ['precious', 'golden', 'treasured'],
            dream: ['floating', 'impossible', 'colors'],
            silence: ['profound', 'listening', 'space'],
            infinity: ['endless', 'limitless', 'vast'],
            shadow: ['mystery', 'contrast', 'hidden']
        };
        
        return complementaryPairs[word] || ['beauty', 'wonder', 'magic'];
    }
}

// Initialize the garden when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PoetryGarden();
});