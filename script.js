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
        aiConcept: "A perceptron is the simplest form of a neural network, consisting of a single neuron.",
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
    // Add more puzzles here...
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
