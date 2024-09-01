const emojis = ['🍎', '🍋', '🍒', '🍇', '🍊'];
const reels = Array.from(document.querySelectorAll('.reel'));
const spinButton = document.getElementById('spin-button');
const balanceElement = document.getElementById('balance').querySelector('span');
const winMessageElement = document.getElementById('win-message');

let balance = 10;
let spinning = false;

spinButton.addEventListener('click', spin);

function spin() {
    if (spinning || balance < 1) {
        return;
    }

    spinning = true;
    balance -= 1;
    updateBalance();
    winMessageElement.textContent = '';

    const spinDuration = 2000; // 2 seconds
    const fps = 30; // Frames per second
    const totalFrames = (spinDuration / 1000) * fps;

    let frame = 0;
    const spinInterval = setInterval(() => {
        reels.forEach((reel) => {
            const randomIndex = Math.floor(Math.random() * emojis.length);
            reel.textContent = emojis[randomIndex];
        });

        frame++;

        if (frame >= totalFrames) {
            clearInterval(spinInterval);
            checkWin();
            spinning = false;
            spinButton.disabled = false;
        }
    }, 1000 / fps);

    spinButton.disabled = true;
}

function checkWin() {
    const results = reels.map(reel => reel.textContent);

    if (results[0] === results[1] && results[1] === results[2]) {
        if (results[0] === '🍒') {
            balance += 10;
            displayWinMessage('Jackpot! You won $10!');
        } else {
            balance += 5;
            displayWinMessage('You won $5! (3 matching)');
        }
    } else if (results[0] === results[1] || results[1] === results[2] || results[0] === results[2]) {
        balance += 2;
        displayWinMessage('You won $2! (2 matching)');
    } else {
        displayWinMessage('No win this time. Try again!');
    }

    updateBalance();
}

function displayWinMessage(message) {
    winMessageElement.textContent = message;
}

function updateBalance() {
    balanceElement.textContent = balance;
}