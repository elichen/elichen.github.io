// Initialize progress if not already set
if (!localStorage.getItem('puzzlesSolved')) {
    localStorage.setItem('puzzlesSolved', '0');
}
if (!localStorage.getItem('knowledgePoints')) {
    localStorage.setItem('knowledgePoints', '0');
}

// Update progress display
function updateProgress() {
    document.getElementById('puzzles-solved').textContent = localStorage.getItem('puzzlesSolved');
    document.getElementById('knowledge-points').textContent = localStorage.getItem('knowledgePoints');
}

// Call updateProgress when the page loads
document.addEventListener('DOMContentLoaded', updateProgress);

// AI Conceptle puzzle data
const puzzles = [
    {
        id: 1,
        domain: "Neural Networks",
        term: "NEURON",
        hint: "The basic computational unit of the brain and artificial neural networks.",
        explanation: "A neuron is the fundamental unit in neural networks, inspired by biological neurons. It receives inputs, processes them, and produces an output."
    },
    {
        id: 2,
        domain: "Machine Learning",
        term: "SIGMOID",
        hint: "An S-shaped activation function commonly used in neural networks.",
        explanation: "The sigmoid function maps any input to a value between 0 and 1, making it useful for binary classification problems and as an activation function in neural networks."
    },
    {
        id: 3,
        domain: "Natural Language Processing",
        term: "TOKENIZE",
        hint: "The process of breaking down text into smaller units for analysis.",
        explanation: "Tokenization is a fundamental step in NLP where text is divided into individual words, subwords, or characters to be processed by algorithms."
    },
    {
        id: 4,
        domain: "Computer Vision",
        term: "CONVOLUTION",
        hint: "A key operation in CNNs for feature extraction from images.",
        explanation: "Convolution involves sliding a small matrix (kernel) over an image to detect features, forming the basis of convolutional neural networks used in image processing."
    },
    {
        id: 5,
        domain: "Reinforcement Learning",
        term: "QLEARNING",
        hint: "A model-free algorithm for learning optimal action-selection policy.",
        explanation: "Q-learning is a reinforcement learning technique that learns the value of actions in states, allowing an agent to make optimal decisions without a model of the environment."
    },
    {
        id: 6,
        domain: "Optimization",
        term: "GRADIENT",
        hint: "The direction of steepest increase in a function, crucial for many optimization algorithms.",
        explanation: "Gradients are used in various optimization techniques, particularly in training neural networks through backpropagation to minimize the loss function."
    },
    {
        id: 7,
        domain: "Clustering",
        term: "KMEANS",
        hint: "An unsupervised learning algorithm that groups similar data points.",
        explanation: "K-means clustering partitions data into K clusters, each represented by the mean of its points, widely used for data segmentation and feature learning."
    },
    {
        id: 8,
        domain: "Dimensionality Reduction",
        term: "PCA",
        hint: "A technique for reducing the dimensionality of data while preserving its variance.",
        explanation: "Principal Component Analysis (PCA) is used to simplify complex datasets by transforming them into fewer dimensions which still retain most of the original information."
    },
    {
        id: 9,
        domain: "Deep Learning",
        term: "BACKPROP",
        hint: "The primary algorithm for training neural networks by adjusting weights.",
        explanation: "Backpropagation calculates the gradient of the loss function with respect to the network's weights, enabling effective training of deep neural networks."
    },
    {
        id: 10,
        domain: "Ethics in AI",
        term: "BIAS",
        hint: "A systematic error in AI systems that can lead to unfair outcomes.",
        explanation: "Bias in AI can result from skewed training data or flawed algorithms, leading to discriminatory or unfair decisions, a critical concern in AI ethics."
    }
];

let currentPuzzle;
let attempts;
const MAX_ATTEMPTS = 6;

function startNewGame() {
    currentPuzzle = puzzles[Math.floor(Math.random() * puzzles.length)];
    attempts = 0;
    updateUI();
}

function updateUI() {
    document.getElementById('domain').textContent = currentPuzzle.domain;
    document.getElementById('hint').textContent = currentPuzzle.hint;
    document.getElementById('attempts').textContent = `${attempts}/${MAX_ATTEMPTS}`;
    // Clear previous guesses and set up new guess slots
    // ... (implement this part)
}

function makeGuess() {
    const guess = document.getElementById('guess-input').value.toUpperCase();
    if (guess.length !== currentPuzzle.term.length) {
        alert(`Your guess must be ${currentPuzzle.term.length} letters long.`);
        return;
    }

    attempts++;
    const feedback = provideFeedback(guess);
    displayGuess(guess, feedback);

    if (guess === currentPuzzle.term) {
        endGame(true);
    } else if (attempts >= MAX_ATTEMPTS) {
        endGame(false);
    }

    updateUI();
}

function provideFeedback(guess) {
    // Implement Wordle-style feedback (correct letter & position, correct letter wrong position, incorrect)
    // ... (implement this part)
}

function displayGuess(guess, feedback) {
    // Display the guess and its feedback on the game board
    // ... (implement this part)
}

function endGame(isWin) {
    if (isWin) {
        alert(`Congratulations! You've guessed the term: ${currentPuzzle.term}`);
        updateProgressAfterSolve();
    } else {
        alert(`Game over. The correct term was: ${currentPuzzle.term}`);
    }
    document.getElementById('explanation').textContent = currentPuzzle.explanation;
}

function updateProgressAfterSolve() {
    let solved = parseInt(localStorage.getItem('puzzlesSolved'));
    let points = parseInt(localStorage.getItem('knowledgePoints'));
    
    solved++;
    points += 10;

    localStorage.setItem('puzzlesSolved', solved.toString());
    localStorage.setItem('knowledgePoints', points.toString());

    updateProgress();
}

// Start a new game when the page loads
document.addEventListener('DOMContentLoaded', startNewGame);
