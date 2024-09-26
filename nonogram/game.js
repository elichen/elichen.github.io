let gameGrid, solution, emoji;

function initGame() {
    const { grid, emoji: randomEmoji } = createEmojiGrid();
    gameGrid = Array(10).fill().map(() => Array(10).fill(0));
    solution = grid;
    emoji = randomEmoji;

    const gridContainer = document.getElementById('grid-container');
    gridContainer.innerHTML = '';

    for (let y = 0; y < 10; y++) {
        for (let x = 0; x < 10; x++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.x = x;
            cell.dataset.y = y;
            cell.addEventListener('click', toggleCell);
            cell.addEventListener('contextmenu', markCell);
            gridContainer.appendChild(cell);
        }
    }

    generateHints();

    // Add this line to reset the emoji overlay
    document.getElementById('emoji-overlay').style.opacity = '0';
}

function toggleCell(event) {
    const x = parseInt(event.target.dataset.x);
    const y = parseInt(event.target.dataset.y);
    if (!event.target.classList.contains('marked')) {
        gameGrid[y][x] = 1 - gameGrid[y][x];
        event.target.classList.toggle('filled');
    }
}

function markCell(event) {
    event.preventDefault(); // Prevent the default context menu
    const cell = event.target;
    if (!cell.classList.contains('filled')) {
        cell.classList.toggle('marked');
    }
}

function generateHints() {
    const rowHints = document.getElementById('row-hints');
    const colHints = document.getElementById('col-hints');
    rowHints.innerHTML = '';
    colHints.innerHTML = '';

    for (let i = 0; i < 10; i++) {
        const rowHint = document.createElement('div');
        rowHint.className = 'hint';
        rowHint.textContent = getHint(solution[i]);
        rowHints.appendChild(rowHint);

        const colHint = document.createElement('div');
        colHint.className = 'col-hint';
        const hintNumbers = getHint(solution.map(row => row[i])).split(' ');
        colHint.innerHTML = hintNumbers.map(num => `<span>${num}</span>`).join('');
        colHints.appendChild(colHint);
    }
}

function getHint(line) {
    const hint = [];
    let count = 0;
    for (const cell of line) {
        if (cell === 1) {
            count++;
        } else if (count > 0) {
            hint.push(count);
            count = 0;
        }
    }
    if (count > 0) {
        hint.push(count);
    }
    return hint.join(' ') || '0';
}

function checkSolution() {
    const message = document.getElementById('message');
    if (JSON.stringify(gameGrid) === JSON.stringify(solution)) {
        message.textContent = `Congratulations! You solved it!`;
        revealSolution();
        showEmojiOverlay();
    } else {
        message.textContent = 'Not quite right. Keep trying!';
    }
}

function revealSolution() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, index) => {
        const x = index % 10;
        const y = Math.floor(index / 10);
        if (solution[y][x] === 1) {
            cell.classList.add('filled');
        } else {
            cell.classList.remove('filled');
        }
    });
}

function showEmojiOverlay() {
    const emojiOverlay = document.getElementById('emoji-overlay');
    emojiOverlay.textContent = emoji;
    emojiOverlay.style.opacity = '0.5';
}

function cheatSolution() {
    gameGrid = JSON.parse(JSON.stringify(solution));
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, index) => {
        const x = index % 10;
        const y = Math.floor(index / 10);
        if (solution[y][x] === 1) {
            cell.classList.add('filled');
        } else {
            cell.classList.remove('filled');
        }
    });
    checkSolution();
    showEmojiOverlay();
}

document.getElementById('check-solution').addEventListener('click', checkSolution);
document.getElementById('cheat-button').addEventListener('click', cheatSolution);
document.getElementById('grid-container').addEventListener('contextmenu', (e) => e.preventDefault());

initGame();