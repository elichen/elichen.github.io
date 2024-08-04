// Initialize progress if not already set
if (!localStorage.getItem('totalPuzzlesSolved')) {
    localStorage.setItem('totalPuzzlesSolved', '0');
}
if (!localStorage.getItem('totalPoints')) {
    localStorage.setItem('totalPoints', '0');
}

let currentPuzzle;
let attempts;
const MAX_ATTEMPTS = 6;
let currentRoundPuzzles = [];
let currentPuzzleIndex = 0;
let roundScore = 0;

function updateProgress() {
    document.getElementById('total-puzzles-solved').textContent = localStorage.getItem('totalPuzzlesSolved');
    document.getElementById('total-points').textContent = localStorage.getItem('totalPoints');
}

function startNewRound() {
    // Reset round-specific variables
    currentRoundPuzzles = getRandomPuzzles(5);
    currentPuzzleIndex = 0;
    roundScore = 0;
    
    // Update UI for new round
    document.getElementById('round-progress').textContent = `Puzzle 1 of 5`;
    document.getElementById('round-score').textContent = '0';
    
    startNewPuzzle();
}

function getRandomPuzzles(count) {
    let shuffled = [...allPuzzles].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}

function startNewPuzzle() {
    currentPuzzle = currentRoundPuzzles[currentPuzzleIndex];
    attempts = 0;
    document.getElementById('guess-board').innerHTML = '';
    document.getElementById('explanation-container').style.display = 'none';
    document.getElementById('guess-input').value = '';
    document.getElementById('guess-input').disabled = false;
    
    updateUI();
    displayAnswerLength();
}

function updateUI() {
    document.getElementById('domain').textContent = currentPuzzle.domain;
    document.getElementById('hint').textContent = currentPuzzle.hint;
    document.getElementById('attempts').textContent = `${attempts}/${MAX_ATTEMPTS}`;
    document.getElementById('round-progress').textContent = `Puzzle ${currentPuzzleIndex + 1} of 5`;
    document.getElementById('round-score').textContent = roundScore;
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
        endPuzzle(true);
    } else if (attempts >= MAX_ATTEMPTS) {
        endPuzzle(false);
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

function endPuzzle(isWin) {
    const guessInput = document.getElementById('guess-input');
    guessInput.disabled = true;
    
    if (isWin) {
        const points = calculatePoints(attempts);
        roundScore += points;
        alert(`Correct! You've guessed the term: ${currentPuzzle.term}\nYou earned ${points} points!`);
    } else {
        alert(`Game over. The correct term was: ${currentPuzzle.term}`);
    }
    
    document.getElementById('explanation-container').style.display = 'block';
    document.getElementById('explanation').textContent = currentPuzzle.explanation;
    
    // Show Next Puzzle button
    showNextPuzzleButton();
}

function showNextPuzzleButton() {
    const nextButton = document.createElement('button');
    nextButton.textContent = 'Next Puzzle';
    nextButton.onclick = moveToNextPuzzle;
    nextButton.id = 'next-puzzle-button';
    document.getElementById('game-container').appendChild(nextButton);
}

function moveToNextPuzzle() {
    // Remove the Next Puzzle button
    const nextButton = document.getElementById('next-puzzle-button');
    if (nextButton) {
        nextButton.remove();
    }

    currentPuzzleIndex++;
    if (currentPuzzleIndex < currentRoundPuzzles.length) {
        startNewPuzzle();
    } else {
        endRound();
    }
}

function calculatePoints(attempts) {
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

function endRound() {
    let totalSolved = parseInt(localStorage.getItem('totalPuzzlesSolved'));
    let totalPoints = parseInt(localStorage.getItem('totalPoints'));
    
    totalSolved += 5;
    totalPoints += roundScore;

    localStorage.setItem('totalPuzzlesSolved', totalSolved.toString());
    localStorage.setItem('totalPoints', totalPoints.toString());

    if (roundScore === 300) {  // Perfect score: 60 points * 5 puzzles
        setTimeout(() => {
            alert("Perfect score! You've mastered these AI concepts. Enjoy a special bonus animation!");
            startBonusAnimation();
        }, 1000);
    } else {
        alert(`Round complete! Your score: ${roundScore} out of 300 possible points.`);
    }

    document.getElementById('restart-button').style.display = 'block';
}

function restartGame() {
    document.getElementById('restart-button').style.display = 'none';
    startNewRound();
}

// Start a new round when the page loads
document.addEventListener('DOMContentLoaded', () => {
    startNewRound();
});
