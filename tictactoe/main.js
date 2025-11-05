const game = new TicTacToeGame();
const agent = new PPOAgent();

let gameActive = false;

function updateGameStatus(message) {
  const statusElement = document.getElementById('gameStatus');
  if (statusElement) {
    statusElement.textContent = message;
  }
}

async function playGame() {
  // Randomly decide who goes first
  const aiStarts = Math.random() < 0.5;

  // AI plays as X, Human plays as O
  game.reset(false, aiStarts);
  game.render(false);
  gameActive = true;

  if (aiStarts) {
    updateGameStatus("AI goes first as X");
    // Small delay so user can see who goes first
    await new Promise(resolve => setTimeout(resolve, 500));
  } else {
    updateGameStatus("You go first! Click a square to play as O");
  }

  while (!game.gameOver && gameActive) {
    if (game.currentPlayer === 1) {
      // AI's turn (X)
      updateGameStatus("AI is thinking...");
      await new Promise(resolve => setTimeout(resolve, 300)); // Small delay for UX

      const state = game.getState(1);
      const validMoves = game.getValidMoves();
      const action = agent.act(state, validMoves);
      game.makeMove(action);
      game.render(false);

      if (game.gameOver) break;
      updateGameStatus("Your turn! Click a square to play as O");
    } else {
      // Human's turn (O)
      const cells = document.querySelectorAll('.cell');
      const action = await waitForHumanMove(cells);
      game.makeMove(action);
      game.render(false);
    }
  }

  // Show result
  if (game.gameOver) {
    if (game.checkWin(1)) {
      updateGameStatus("AI wins! (X)");
    } else if (game.checkWin(-1)) {
      updateGameStatus("You win! (O)");
    } else {
      updateGameStatus("It's a draw!");
    }

    // Auto restart after 2 seconds
    await new Promise(resolve => setTimeout(resolve, 2000));
    if (gameActive) {
      playGame();
    }
  }
}

function waitForHumanMove(cells) {
  return new Promise((resolve) => {
    const clickHandler = (event) => {
      const index = Array.from(cells).indexOf(event.target);
      if (game.isValidMove(index)) {
        cells.forEach(cell => cell.removeEventListener('click', clickHandler));
        resolve(index);
      }
    };
    cells.forEach(cell => cell.addEventListener('click', clickHandler));
  });
}

async function init() {
  // Load the PPO model
  const modelLoaded = await agent.loadModel();

  if (modelLoaded) {
    console.log('PPO model loaded successfully');
    // Start the game
    playGame();
  } else {
    updateGameStatus('⚠️ Model not loaded - Please train a model first');
  }
}

init();