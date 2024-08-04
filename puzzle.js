let currentPuzzle;
let editor;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize CodeMirror editor
    editor = CodeMirror(document.getElementById("code-editor"), {
        mode: "javascript",
        theme: "default",
        lineNumbers: true,
        autofocus: true
    });

    // Load a random puzzle
    loadRandomPuzzle();
});

function loadRandomPuzzle() {
    const randomIndex = Math.floor(Math.random() * puzzles.length);
    currentPuzzle = puzzles[randomIndex];
    displayPuzzle(currentPuzzle);
}

function displayPuzzle(puzzle) {
    document.getElementById("puzzle-title").textContent = puzzle.title;
    document.getElementById("puzzle-description").textContent = puzzle.description;
    document.getElementById("ai-explanation").textContent = puzzle.aiConcept;
    editor.setValue(puzzle.code);
}

function checkSolution() {
    const userSolution = editor.getValue();
    const resultMessage = document.getElementById("result-message");

    // Simple string comparison - in a real-world scenario, you'd want more robust checking
    if (userSolution.replace(/\s/g, '') === currentPuzzle.solution.replace(/\s/g, '')) {
        resultMessage.textContent = "Congratulations! Your solution is correct.";
        resultMessage.style.color = "green";
        updateProgressAfterSolve();
    } else {
        resultMessage.textContent = "Your solution is not quite right. Try again!";
        resultMessage.style.color = "red";
    }
}

// Add this function to the existing script.js file
function updateProgressAfterSolve() {
    let solved = parseInt(localStorage.getItem('puzzlesSolved'));
    let points = parseInt(localStorage.getItem('knowledgePoints'));
    
    solved++;
    points += 10;

    localStorage.setItem('puzzlesSolved', solved.toString());
    localStorage.setItem('knowledgePoints', points.toString());

    // If we're on the index page, update the progress display
    const puzzlesSolvedElement = document.getElementById('puzzles-solved');
    const knowledgePointsElement = document.getElementById('knowledge-points');
    
    if (puzzlesSolvedElement && knowledgePointsElement) {
        puzzlesSolvedElement.textContent = solved;
        knowledgePointsElement.textContent = points;
    }
}
