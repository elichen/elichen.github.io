const instructions = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const status = document.getElementById('status');
const gameArea = document.getElementById('game-area');
const playerMarblesDisplay = document.getElementById('player-marbles');
const opponentMarblesDisplay = document.getElementById('opponent-marbles');
const betAmountInput = document.getElementById('bet-amount');
const guessButtons = document.querySelectorAll('.guess-btn');
const opponentHand = document.querySelector('.closed-hand');

let playerMarbles = 10;
let opponentMarbles = 10;
let gameRunning = false;
let opponentHolding = 0;

function updateDisplay() {
    playerMarblesDisplay.textContent = playerMarbles;
    opponentMarblesDisplay.textContent = opponentMarbles;
    betAmountInput.max = Math.min(playerMarbles, opponentMarbles);
    if (parseInt(betAmountInput.value) > betAmountInput.max) {
        betAmountInput.value = betAmountInput.max;
    }
}

function generateOpponentHand() {
    opponentHolding = Math.floor(Math.random() * Math.min(opponentMarbles, 5)) + 1;
    opponentHand.textContent = '?';
    opponentHand.classList.remove('revealed');
}

function revealHand() {
    opponentHand.textContent = opponentHolding;
    opponentHand.classList.add('revealed');

    const marbleDisplay = document.createElement('div');
    marbleDisplay.id = 'marble-display';
    for (let i = 0; i < opponentHolding; i++) {
        const marble = document.createElement('div');
        marble.className = 'marble';
        marbleDisplay.appendChild(marble);
    }
    opponentHand.appendChild(marbleDisplay);
}

function makeGuess(guess) {
    if (!gameRunning) return;

    const betAmount = parseInt(betAmountInput.value);
    if (betAmount < 1 || betAmount > playerMarbles) return;

    guessButtons.forEach(btn => btn.disabled = true);

    revealHand();

    const isOdd = opponentHolding % 2 === 1;
    const correct = (guess === 'odd' && isOdd) || (guess === 'even' && !isOdd);

    setTimeout(() => {
        if (correct) {
            playerMarbles += betAmount;
            opponentMarbles -= betAmount;
            status.innerHTML = `<span style="color: #2ecc71">Correct! ${opponentHolding} is ${isOdd ? 'odd' : 'even'}!</span><br>You won ${betAmount} marble(s)!`;
        } else {
            playerMarbles -= betAmount;
            opponentMarbles += betAmount;
            status.innerHTML = `<span style="color: #e74c3c">Wrong! ${opponentHolding} is ${isOdd ? 'odd' : 'even'}!</span><br>You lost ${betAmount} marble(s)!`;
        }

        updateDisplay();

        if (playerMarbles <= 0) {
            endGame(false);
        } else if (opponentMarbles <= 0) {
            endGame(true);
        } else {
            setTimeout(() => {
                generateOpponentHand();
                guessButtons.forEach(btn => btn.disabled = false);
                status.textContent = 'Make your next guess!';
            }, 1500);
        }
    }, 1000);
}

function startGame() {
    instructions.style.display = 'none';
    gameArea.style.display = 'flex';
    playerMarbles = 10;
    opponentMarbles = 10;
    gameRunning = true;
    updateDisplay();
    generateOpponentHand();
    guessButtons.forEach(btn => btn.disabled = false);
    status.textContent = 'Guess: Odd or Even?';
}

function endGame(win) {
    gameRunning = false;
    guessButtons.forEach(btn => btn.disabled = true);

    if (win) {
        status.innerHTML = '<span style="color: #2ecc71; font-size: 24px;">You Win!</span><br>You collected all the marbles!';
    } else {
        status.innerHTML = '<span style="color: #e74c3c; font-size: 24px;">You Lose!</span><br>You ran out of marbles!';
    }

    setTimeout(() => {
        instructions.style.display = 'block';
        gameArea.style.display = 'none';
        startButton.textContent = 'Play Again';
    }, 2000);
}

startButton.addEventListener('click', startGame);
guessButtons.forEach(btn => {
    btn.addEventListener('click', () => makeGuess(btn.dataset.guess));
});
