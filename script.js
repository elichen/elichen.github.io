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

// Puzzle data
const puzzles = [
    {
        id: 1,
        title: "Neural Network Basics",
        description: "Complete the code for a simple perceptron",
        aiConcept: "A perceptron is the simplest form of a neural network, consisting of a single neuron. It takes multiple inputs, applies weights, and produces a binary output based on a threshold.",
        code: `
function perceptron(inputs, weights, bias) {
    let sum = bias;
    for (let i = 0; i < inputs.length; i++) {
        // TODO: Implement the weighted sum
    }
    // TODO: Implement the activation function (use step function: return 1 if sum > 0, else 0)
}
        `,
        solution: `
function perceptron(inputs, weights, bias) {
    let sum = bias;
    for (let i = 0; i < inputs.length; i++) {
        sum += inputs[i] * weights[i];
    }
    return sum > 0 ? 1 : 0;
}
        `
    },
    {
        id: 2,
        title: "Decision Trees",
        description: "Implement a simple decision function for a binary classification tree",
        aiConcept: "Decision trees are a type of supervised learning algorithm used for classification and regression. They work by making a series of decisions based on the features of the input data.",
        code: `
function decideNode(feature, threshold, leftDecision, rightDecision) {
    // TODO: Implement the decision function
    // If feature is less than or equal to threshold, return leftDecision
    // Otherwise, return rightDecision
}
        `,
        solution: `
function decideNode(feature, threshold, leftDecision, rightDecision) {
    return feature <= threshold ? leftDecision : rightDecision;
}
        `
    },
    {
        id: 3,
        title: "Natural Language Processing",
        description: "Create a function to count word frequencies in a text",
        aiConcept: "Word frequency analysis is a fundamental technique in NLP, used in various applications such as text classification, sentiment analysis, and topic modeling.",
        code: `
function wordFrequency(text) {
    // TODO: Implement word frequency counter
    // Return an object with words as keys and their frequencies as values
}
        `,
        solution: `
function wordFrequency(text) {
    const words = text.toLowerCase().match(/\\w+/g);
    return words.reduce((freq, word) => {
        freq[word] = (freq[word] || 0) + 1;
        return freq;
    }, {});
}
        `
    },
    {
        id: 4,
        title: "Computer Vision",
        description: "Write a function to invert colors in an image matrix",
        aiConcept: "Color inversion is a basic image processing technique. In computer vision, manipulating pixel values is fundamental to many algorithms and preprocessing steps.",
        code: `
function invertColors(imageMatrix) {
    // TODO: Implement color inversion
    // Assume imageMatrix is a 2D array of numbers between 0 and 255
    // Invert each pixel value (new_value = 255 - old_value)
}
        `,
        solution: `
function invertColors(imageMatrix) {
    return imageMatrix.map(row => row.map(pixel => 255 - pixel));
}
        `
    },
    {
        id: 5,
        title: "Clustering",
        description: "Implement a function to calculate Euclidean distance for k-means clustering",
        aiConcept: "K-means clustering is an unsupervised learning algorithm that groups similar data points together. Euclidean distance is commonly used to measure the similarity between data points in this algorithm.",
        code: `
function euclideanDistance(point1, point2) {
    // TODO: Implement Euclidean distance calculation
    // Assume point1 and point2 are arrays of numbers with the same length
}
        `,
        solution: `
function euclideanDistance(point1, point2) {
    return Math.sqrt(point1.reduce((sum, value, index) => {
        return sum + Math.pow(value - point2[index], 2);
    }, 0));
}
        `
    },
    {
        id: 6,
        title: "Genetic Algorithms",
        description: "Create a basic fitness function for a genetic algorithm",
        aiConcept: "Genetic algorithms are optimization techniques inspired by natural selection. The fitness function determines how well a solution solves the problem, guiding the evolution of solutions.",
        code: `
function fitnessFunctionBinaryString(binaryString) {
    // TODO: Implement a fitness function
    // Assume binaryString is a string of 0s and 1s
    // Return a fitness score (higher is better)
    // Hint: You could count the number of 1s in the string
}
        `,
        solution: `
function fitnessFunctionBinaryString(binaryString) {
    return binaryString.split('1').length - 1;
}
        `
    },
    {
        id: 7,
        title: "Reinforcement Learning",
        description: "Implement a simple Q-learning update function",
        aiConcept: "Q-learning is a model-free reinforcement learning algorithm. It learns to make decisions by computing the expected utility of actions in a given state.",
        code: `
function qLearningUpdate(oldValue, reward, nextMaxQ, learningRate, discountFactor) {
    // TODO: Implement the Q-learning update formula
    // Q(s,a) = (1-α) * Q(s,a) + α * (r + γ * max(Q(s',a')))
    // Where α is the learning rate and γ is the discount factor
}
        `,
        solution: `
function qLearningUpdate(oldValue, reward, nextMaxQ, learningRate, discountFactor) {
    return (1 - learningRate) * oldValue + learningRate * (reward + discountFactor * nextMaxQ);
}
        `
    },
    {
        id: 8,
        title: "Data Preprocessing",
        description: "Normalize an array of numbers using min-max scaling",
        aiConcept: "Data normalization is a crucial preprocessing step in many machine learning algorithms. Min-max scaling is a common method that scales features to a fixed range, typically between 0 and 1.",
        code: `
function minMaxNormalization(numbers) {
    // TODO: Implement min-max normalization
    // Formula: (x - min(x)) / (max(x) - min(x))
}
        `,
        solution: `
function minMaxNormalization(numbers) {
    const min = Math.min(...numbers);
    const max = Math.max(...numbers);
    return numbers.map(x => (x - min) / (max - min));
}
        `
    },
    {
        id: 9,
        title: "Backpropagation",
        description: "Implement the derivative of the sigmoid activation function",
        aiConcept: "Backpropagation is a key algorithm in training neural networks. The sigmoid function is a common activation function, and its derivative is used in calculating gradients during backpropagation.",
        code: `
function sigmoidDerivative(x) {
    // TODO: Implement the derivative of the sigmoid function
    // Hint: The derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
        `,
        solution: `
function sigmoidDerivative(x) {
    const sigX = sigmoid(x);
    return sigX * (1 - sigX);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
        `
    },
    {
        id: 10,
        title: "Convolutional Neural Networks",
        description: "Implement a simple 2D convolution operation",
        aiConcept: "Convolution is a fundamental operation in convolutional neural networks (CNNs). It involves sliding a kernel over an input matrix to produce a feature map, which helps in detecting features in images.",
        code: `
function convolution2D(image, kernel) {
    // TODO: Implement 2D convolution
    // Assume 'image' and 'kernel' are 2D arrays
    // You can assume the kernel is 3x3 for simplicity
    // Return the convolved image
}
        `,
        solution: `
function convolution2D(image, kernel) {
    const result = [];
    for (let i = 1; i < image.length - 1; i++) {
        const row = [];
        for (let j = 1; j < image[0].length - 1; j++) {
            let sum = 0;
            for (let ki = -1; ki <= 1; ki++) {
                for (let kj = -1; kj <= 1; kj++) {
                    sum += image[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }
            row.push(sum);
        }
        result.push(row);
    }
    return result;
}
        `
    }
];

// Function to load a puzzle (to be implemented in puzzle.html)
function loadPuzzle(id) {
    const puzzle = puzzles.find(p => p.id === id);
    // TODO: Implement puzzle loading logic
}

// Function to check the solution (to be implemented in puzzle.html)
function checkSolution() {
    // TODO: Implement solution checking logic
}

// Function to update progress after solving a puzzle
function updateProgressAfterSolve() {
    let solved = parseInt(localStorage.getItem('puzzlesSolved'));
    let points = parseInt(localStorage.getItem('knowledgePoints'));
    
    solved++;
    points += 10;

    localStorage.setItem('puzzlesSolved', solved.toString());
    localStorage.setItem('knowledgePoints', points.toString());

    updateProgress();
}
