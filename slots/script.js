const emojis = ['🍎', '🍋', '🍒', '🍇', '🍊'];
const reels = Array.from(document.querySelectorAll('.reel'));
const spinButton = document.getElementById('spin-button');
const balanceElement = document.getElementById('balance').querySelector('span');
const winMessageElement = document.getElementById('win-message');

let balance = 10;
let spinning = false;

spinButton.addEventListener('click', spin);

// Initialize reels
reels.forEach(initializeReel);

function initializeReel(reel) {
    // Create a long strip of emojis
    const reelStrip = [...emojis, ...emojis, ...emojis, ...emojis, ...emojis];
    reelStrip.forEach(emoji => {
        const div = document.createElement('div');
        div.textContent = emoji;
        reel.appendChild(div);
    });
}

function spin() {
    if (spinning || balance < 1) return;

    spinning = true;
    balance -= 1;
    updateBalance();
    winMessageElement.textContent = '';
    spinButton.disabled = true;

    const spinDuration = 3000; // 3 seconds

    reels.forEach((reel, index) => {
        // Remove previous spin classes
        reel.classList.remove('spin-animation', 'stop-animation');
        
        // Trigger reflow
        void reel.offsetWidth;
        
        // Add spin animation with delay
        reel.style.animationDelay = `${index * 0.2}s`;
        reel.classList.add('spin-animation');
    });

    // Stop the reels and check for win
    setTimeout(() => {
        reels.forEach((reel, index) => {
            reel.classList.remove('spin-animation');
            reel.classList.add('stop-animation');
            
            // Randomize final position
            const finalPosition = -Math.floor(Math.random() * emojis.length) * 80;
            reel.style.transform = `translateY(${finalPosition}px)`;
        });

        setTimeout(checkWin, 200);
    }, spinDuration);
}

function checkWin() {
    const results = reels.map(reel => {
        const transform = getComputedStyle(reel).getPropertyValue('transform');
        const matrix = new DOMMatrix(transform);
        const currentPosition = Math.abs(matrix.m42);
        const visibleIndex = Math.round(currentPosition / 80) % emojis.length;
        return emojis[visibleIndex];
    });

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
    spinning = false;
    spinButton.disabled = false;
}

function displayWinMessage(message) {
    winMessageElement.textContent = message;
}

function updateBalance() {
    balanceElement.textContent = balance;
}