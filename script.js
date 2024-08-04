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

let currentPuzzle;
let attempts;
const MAX_ATTEMPTS = 6;
let remainingPuzzles = [...allPuzzles];

function startNewGame() {
    if (remainingPuzzles.length === 0) {
        remainingPuzzles = [...allPuzzles];
    }
    const randomIndex = Math.floor(Math.random() * remainingPuzzles.length);
    currentPuzzle = remainingPuzzles[randomIndex];
    remainingPuzzles.splice(randomIndex, 1);

    attempts = 0;
    document.getElementById('guess-board').innerHTML = '';
    document.getElementById('explanation-container').style.display = 'none';
    document.getElementById('guess-input').value = '';
    document.getElementById('guess-input').disabled = false;
    
    // Remove any existing "Next Puzzle" button
    const existingNextButton = document.getElementById('next-puzzle-button');
    if (existingNextButton) {
        existingNextButton.remove();
    }

    updateUI();
    displayAnswerLength();
}

function updateUI() {
    document.getElementById('domain').textContent = currentPuzzle.domain;
    document.getElementById('hint').textContent = currentPuzzle.hint;
    document.getElementById('attempts').textContent = `${attempts}/${MAX_ATTEMPTS}`;
}

function displayAnswerLength() {
    const answerLength = document.getElementById('answer-length');
    answerLength.innerHTML = '';
    for (let i = 0; i < currentPuzzle.term.length; i++) {
        const box = document.createElement('div');
        box.className = 'answer-box';
        answerLength.appendChild(box);
    }
}

function makeGuess() {
    const guessInput = document.getElementById('guess-input');
    const guess = guessInput.value.toUpperCase();
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

    guessInput.value = '';
    updateUI();
}

function provideFeedback(guess) {
    const feedback = [];
    const termLetters = [...currentPuzzle.term];
    
    // First pass: mark correct letters
    for (let i = 0; i < guess.length; i++) {
        if (guess[i] === termLetters[i]) {
            feedback[i] = 'correct';
            termLetters[i] = null;
        }
    }
    
    // Second pass: mark present letters
    for (let i = 0; i < guess.length; i++) {
        if (feedback[i]) continue;
        const index = termLetters.indexOf(guess[i]);
        if (index !== -1) {
            feedback[i] = 'present';
            termLetters[index] = null;
        } else {
            feedback[i] = 'absent';
        }
    }
    
    return feedback;
}

function displayGuess(guess, feedback) {
    const guessBoard = document.getElementById('guess-board');
    const guessRow = document.createElement('div');
    guessRow.className = 'guess-row';
    
    for (let i = 0; i < guess.length; i++) {
        const letterBox = document.createElement('div');
        letterBox.className = `guess-letter ${feedback[i]}`;
        letterBox.textContent = guess[i];
        guessRow.appendChild(letterBox);
    }
    
    guessBoard.appendChild(guessRow);
}

function endGame(isWin) {
    const guessInput = document.getElementById('guess-input');
    guessInput.disabled = true;
    
    if (isWin) {
        const points = calculatePoints(attempts);
        alert(`Congratulations! You've guessed the term: ${currentPuzzle.term}\nYou earned ${points} points!`);
        updateProgressAfterSolve(points);
    } else {
        alert(`Game over. The correct term was: ${currentPuzzle.term}`);
    }
    
    document.getElementById('explanation-container').style.display = 'block';
    document.getElementById('explanation').textContent = currentPuzzle.explanation;
    
    // Add a "Next Puzzle" button if it doesn't exist
    if (!document.getElementById('next-puzzle-button')) {
        const nextButton = document.createElement('button');
        nextButton.textContent = 'Next Puzzle';
        nextButton.id = 'next-puzzle-button';
        nextButton.onclick = startNewGame;
        document.getElementById('game-container').appendChild(nextButton);
    }
}

function calculatePoints(attempts) {
    // Award points based on the number of attempts
    switch (attempts) {
        case 1: return 60;
        case 2: return 50;
        case 3: return 40;
        case 4: return 30;
        case 5: return 20;
        case 6: return 10;
        default: return 0;
    }
}

function updateProgressAfterSolve(points) {
    let solved = parseInt(localStorage.getItem('puzzlesSolved'));
    let totalPoints = parseInt(localStorage.getItem('knowledgePoints'));
    
    solved++;
    totalPoints += points;

    localStorage.setItem('puzzlesSolved', solved.toString());
    localStorage.setItem('knowledgePoints', totalPoints.toString());

    updateProgress();
}

// Start a new game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    startNewGame();
    updateProgress();
});
