class SortingVisualizer {
    constructor() {
        this.array = [];
        this.arraySize = 50;
        this.maxValue = 100;
        this.delay = 50;
        
        this.algorithmInfo = {
            bubble: {
                title: "Bubble Sort",
                description: "Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until no swaps are needed.",
                complexity: {
                    time: "O(n²)",
                    space: "O(1)",
                    best: "O(n)"
                }
            },
            selection: {
                title: "Selection Sort",
                description: "Selection Sort divides the input list into a sorted and an unsorted region. It repeatedly selects the smallest element from the unsorted region and adds it to the sorted region.",
                complexity: {
                    time: "O(n²)",
                    space: "O(1)",
                    best: "O(n²)"
                }
            },
            insertion: {
                title: "Insertion Sort",
                description: "Insertion Sort builds the final sorted array one item at a time. It takes each element from the input and inserts it into its correct position in the sorted portion of the array.",
                complexity: {
                    time: "O(n²)",
                    space: "O(1)",
                    best: "O(n)"
                }
            },
            quick: {
                title: "Quick Sort",
                description: "Quick Sort is a divide-and-conquer algorithm. It works by selecting a 'pivot' element and partitioning the array around it such that smaller elements go to the left and larger elements go to the right.",
                complexity: {
                    time: "O(n log n)",
                    space: "O(log n)",
                    best: "O(n log n)"
                }
            }
        };

        this.initializeButtons();
        this.initializeDelayControl();
        this.updateAlgorithmInfo();
        this.generateNewArray();
    }

    initializeButtons() {
        document.getElementById('newArrayBtn').addEventListener('click', () => this.generateNewArray());
        document.getElementById('sortBtn').addEventListener('click', () => this.sort());
        document.getElementById('algorithmSelect').addEventListener('change', () => this.updateAlgorithmInfo());
    }

    updateAlgorithmInfo() {
        const algorithm = document.getElementById('algorithmSelect').value;
        const info = this.algorithmInfo[algorithm];
        const explanationDiv = document.getElementById('algorithmExplanation');
        
        explanationDiv.innerHTML = `
            <h3>${info.title}</h3>
            <p>${info.description}</p>
            <div class="complexity">
                <p><strong>Time Complexity:</strong> ${info.complexity.time}</p>
                <p><strong>Space Complexity:</strong> ${info.complexity.space}</p>
                <p><strong>Best Case:</strong> ${info.complexity.best}</p>
            </div>
        `;
    }

    initializeDelayControl() {
        const delayRange = document.getElementById('delayRange');
        const delayValue = document.getElementById('delayValue');
        
        delayRange.addEventListener('input', () => {
            this.delay = parseInt(delayRange.value);
            delayValue.textContent = this.delay;
        });
    }

    generateNewArray() {
        this.array = Array.from({length: this.arraySize}, (_, i) => i + 1);
        this.shuffleArray();
        this.updateDisplay();
    }

    shuffleArray() {
        for (let i = this.array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.array[i], this.array[j]] = [this.array[j], this.array[i]];
        }
    }

    updateDisplay() {
        const container = document.getElementById('arrayContainer');
        container.innerHTML = '';
        
        const barWidth = Math.floor((container.clientWidth - (this.arraySize * 2)) / this.arraySize);
        
        this.array.forEach(value => {
            const bar = document.createElement('div');
            bar.className = 'array-bar';
            bar.style.height = `${(value / this.maxValue) * 100}%`;
            bar.style.width = `${barWidth}px`;
            container.appendChild(bar);
        });
    }

    async sort() {
        const algorithm = document.getElementById('algorithmSelect').value;
        const algorithms = {
            bubble: bubbleSort,
            selection: selectionSort,
            insertion: insertionSort,
            quick: quickSort
        };

        const sortFunction = algorithms[algorithm];
        await sortFunction(this.array, this.updateDisplay.bind(this), this.delay);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SortingVisualizer();
}); 