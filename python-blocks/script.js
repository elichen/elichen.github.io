let currentPuzzle = 0;
let score = 0;
let currentRound = 1;
const totalRounds = 3;
let selectedPuzzles = [];

function initGame() {
    score = 0;
    currentRound = 1;
    selectedPuzzles = selectRandomPuzzles(totalRounds);
    updateGameInfo();
    loadPuzzle();
    document.getElementById('check-solution').addEventListener('click', checkSolution);
    document.getElementById('next-puzzle').addEventListener('click', nextPuzzle);
    document.getElementById('hint').addEventListener('click', getHint);
}

function selectRandomPuzzles(count) {
    const shuffled = [...puzzleData].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}

function updateGameInfo() {
    document.getElementById('current-round').textContent = currentRound;
    document.getElementById('total-rounds').textContent = totalRounds;
    document.getElementById('score').textContent = score;
}

function loadPuzzle() {
    const puzzle = selectedPuzzles[currentRound - 1];
    document.getElementById('puzzle-objective').textContent = puzzle.objective;
    
    const codeBlocksContainer = document.getElementById('code-blocks');
    codeBlocksContainer.innerHTML = '';
    
    // Randomize the order of code blocks
    const randomizedBlocks = [...puzzle.codeBlocks].sort(() => 0.5 - Math.random());
    
    randomizedBlocks.forEach((block, index) => {
        const blockElement = document.createElement('div');
        blockElement.className = 'code-block';
        blockElement.textContent = block;
        blockElement.draggable = true;
        blockElement.id = `block-${index}`;
        blockElement.addEventListener('dragstart', drag);
        codeBlocksContainer.appendChild(blockElement);
    });

    const solutionArea = document.getElementById('solution-area');
    solutionArea.innerHTML = '';
    for (let i = 0; i < puzzle.solution.length; i++) {
        const dropZone = document.createElement('div');
        dropZone.className = 'drop-zone';
        dropZone.addEventListener('dragover', allowDrop);
        dropZone.addEventListener('drop', drop);
        solutionArea.appendChild(dropZone);
    }
}

function drag(ev) {
    ev.dataTransfer.setData("text/plain", ev.target.id);
}

function allowDrop(ev) {
    ev.preventDefault();
}

function drop(ev) {
    ev.preventDefault();
    const data = ev.dataTransfer.getData("text/plain");
    const draggedElement = document.getElementById(data);
    
    if (ev.target.className === 'drop-zone') {
        if (ev.target.firstChild) {
            // If drop zone is occupied, swap elements
            const occupyingElement = ev.target.firstChild;
            draggedElement.parentNode.appendChild(occupyingElement);
        }
        ev.target.appendChild(draggedElement);
    } else if (ev.target.className === 'code-block' && ev.target.parentNode.className === 'drop-zone') {
        // If dropping on another code block in a drop zone, swap them
        const targetParent = ev.target.parentNode;
        const draggedParent = draggedElement.parentNode;
        targetParent.appendChild(draggedElement);
        draggedParent.appendChild(ev.target);
    }
}

function checkSolution() {
    const puzzle = selectedPuzzles[currentRound - 1];
    const solutionArea = document.getElementById('solution-area');
    const userSolution = Array.from(solutionArea.children).map(zone => 
        zone.firstChild ? puzzle.codeBlocks.indexOf(zone.firstChild.textContent) : -1
    );
    
    if (JSON.stringify(userSolution) === JSON.stringify(puzzle.solution)) {
        score += 10;
        document.getElementById('feedback-text').textContent = "Correct! Well done!";
        document.getElementById('next-puzzle').style.display = 'inline-block';
        showExplanation();
    } else {
        document.getElementById('feedback-text').textContent = "Not quite right. Try again!";
    }
    updateGameInfo();
}

function nextPuzzle() {
    currentRound++;
    if (currentRound > totalRounds) {
        endGame();
        return;
    }
    document.getElementById('next-puzzle').style.display = 'none';
    document.getElementById('feedback-text').textContent = '';
    document.getElementById('explanation').style.display = 'none';
    loadPuzzle();
    updateGameInfo();
}

function getHint() {
    const puzzle = selectedPuzzles[currentRound - 1];
    document.getElementById('feedback-text').textContent = puzzle.hint;
}

function showExplanation() {
    const puzzle = selectedPuzzles[currentRound - 1];
    const explanationSection = document.getElementById('explanation');
    explanationSection.style.display = 'block';
    document.getElementById('explanation-text').textContent = puzzle.explanation;
}

function endGame() {
    const gameContainer = document.getElementById('game-container');
    let endMessage = `
        <h2>Game Over!</h2>
        <p>Your final score: ${score}</p>
    `;
    
    if (score === totalRounds * 10) {
        endMessage += `
            <p>Perfect score! Enjoy this bonus animation:</p>
            <div id="game-of-life-3d"></div>
        `;
        gameContainer.innerHTML = endMessage;
        initGameOfLife3D();
    } else {
        endMessage += `<button onclick="initGame()">Play Again</button>`;
        gameContainer.innerHTML = endMessage;
    }
}

// Initialize the game when the page loads
window.onload = initGame;