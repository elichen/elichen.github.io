const puzzleGrid = document.getElementById('puzzle-grid');
const moveCount = document.getElementById('move-count');
const nextImageButton = document.getElementById('next-image');

let currentImage;
let tiles = [];
let moves = 0;

function initGame() {
    currentImage = Math.floor(Math.random() * 10) + 1;
    tiles = [1, 2, 3, 4, 5, 6, 7, 8, 0];
    moves = 0;
    shuffleTiles();
    renderPuzzle();
    updateMoveCount();
    nextImageButton.style.display = 'none';
}

function shuffleTiles() {
    for (let i = tiles.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [tiles[i], tiles[j]] = [tiles[j], tiles[i]];
    }
}

function renderPuzzle() {
    puzzleGrid.innerHTML = '';
    for (let i = 0; i < 9; i++) {
        const tile = document.createElement('div');
        tile.className = 'puzzle-tile';
        if (tiles[i] !== 0) {
            tile.style.backgroundImage = `url(${currentImage}.png)`;
            tile.style.backgroundPosition = `${((tiles[i] - 1) % 3) * 50}% ${Math.floor((tiles[i] - 1) / 3) * 50}%`;
        }
        tile.addEventListener('click', () => moveTile(i));
        puzzleGrid.appendChild(tile);
    }
}

function moveTile(index) {
    const emptyIndex = tiles.indexOf(0);
    if (isAdjacent(index, emptyIndex)) {
        [tiles[index], tiles[emptyIndex]] = [tiles[emptyIndex], tiles[index]];
        moves++;
        renderPuzzle();
        updateMoveCount();
        checkWin();
    }
}

function isAdjacent(index1, index2) {
    const row1 = Math.floor(index1 / 3);
    const col1 = index1 % 3;
    const row2 = Math.floor(index2 / 3);
    const col2 = index2 % 3;
    return Math.abs(row1 - row2) + Math.abs(col1 - col2) === 1;
}

function updateMoveCount() {
    moveCount.textContent = moves;
}

function checkWin() {
    if (tiles.every((tile, index) => tile === (index + 1) % 9)) {
        console.log("Puzzle solved! Displaying full image...");
        nextImageButton.style.display = 'inline-block';
        
        // Get the dimensions of the puzzle grid
        const gridRect = puzzleGrid.getBoundingClientRect();
        console.log("Puzzle grid dimensions:", gridRect.width, gridRect.height);

        // Clear the puzzle grid
        puzzleGrid.innerHTML = '';

        // Create the full image element
        const fullImage = document.createElement('img');
        fullImage.src = `${currentImage}.png`;
        fullImage.alt = "Complete puzzle";
        fullImage.className = 'full-image';
        
        // Set explicit dimensions
        fullImage.style.width = `${gridRect.width}px`;
        fullImage.style.height = `${gridRect.height}px`;
        fullImage.style.objectFit = 'cover';

        console.log("Full image dimensions set to:", fullImage.style.width, fullImage.style.height);

        // Add the image to the grid after a short delay
        setTimeout(() => {
            puzzleGrid.appendChild(fullImage);
            console.log("Full image appended to the grid");
        }, 50);
    }
}

nextImageButton.addEventListener('click', initGame);

initGame();