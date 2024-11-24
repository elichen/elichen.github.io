const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization(1000);

let isTraining = true;
let episodeCount = 0;
let testGamesPlayed = 0;
let testGamesWon = 0;
let gameLoopTimeout = null;

function toggleMode() {
  isTraining = !isTraining;
  document.getElementById('modeButton').textContent = isTraining ? 'Switch to Test Mode' : 'Switch to Train Mode';
  if (!isTraining) {
    testGamesPlayed = 0;
    testGamesWon = 0;
    updateWinPercentage();
  }
  resetGame();
}

function updateWinPercentage() {
  const winPercentage = (testGamesWon / testGamesPlayed * 100).toFixed(2);
  document.getElementById('winPercentage').textContent = `Win Percentage: ${winPercentage}%`;
}

function getRandomStartingPlayer() {
  return Math.random() < 0.5 ? 1 : 2;
}

async function runEpisode() {
  const agentStarts = getRandomStartingPlayer() === 1;
  game.reset(isTraining, agentStarts);
  let gameResult = 0;

  if (!isTraining) {
    game.render(isTraining);
  }

  while (!game.gameOver) {
    const currentPlayerState = game.getState();
    const validMoves = game.getValidMoves();
    let action;

    if (!isTraining && game.currentPlayer === 2) {
      action = await waitForHumanMove();
    } else {
      tf.tidy(() => {
        action = agent.act(currentPlayerState, isTraining, validMoves);
      });
    }

    // Store pre-move state and action
    const preState = game.getState();
    game.makeMove(action);
    
    // Get reward and next state from current player's perspective
    const reward = game.getReward(game.currentPlayer === 1 ? 2 : 1); // Get reward for player who just moved
    const nextState = game.getState(game.currentPlayer === 1 ? 2 : 1); // Get next state from that player's perspective

    // Store experience for the player who just moved
    if (isTraining) {
      agent.remember(preState, action, reward, nextState, game.gameOver);
    }

    if (!isTraining) {
      game.render(isTraining);
    }
  }

  game.render(isTraining);

  episodeCount++;
  if (isTraining) {
    agent.decayEpsilon();
    const loss = await agent.replay();
    
    // Determine game result for visualization
    if (game.checkWin(1)) {
      gameResult = 1;
    } else if (game.checkWin(-1)) {
      gameResult = -1;
    }
  } else {
    testGamesPlayed++;
    if (game.checkWin(1)) testGamesWon++;
    updateWinPercentage();
  }
  
  visualization.updateChart(episodeCount, agent.epsilon, null, gameResult);

  if (isTraining || !isTraining) {
    if (!isTraining) {
      await showNewGameMessage();
    }
    if (gameLoopTimeout) {
      clearTimeout(gameLoopTimeout);
    }
    gameLoopTimeout = setTimeout(runEpisode, isTraining ? 0 : 1000);
  }
}

function waitForHumanMove() {
  return new Promise((resolve) => {
    const cells = document.querySelectorAll('.cell');
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

function resetGame() {
  if (gameLoopTimeout) {
    clearTimeout(gameLoopTimeout);
    gameLoopTimeout = null;
  }
  game.reset(isTraining);
  if (!isTraining) {
    runEpisode();
  } else {
    gameLoopTimeout = setTimeout(runEpisode, 0);
  }
}

async function init() {
  visualization.createChart();
  document.getElementById('modeButton').addEventListener('click', toggleMode);
  
  const winPercentageElement = document.createElement('div');
  winPercentageElement.id = 'winPercentage';
  winPercentageElement.className = 'stats';
  const container = document.querySelector('.container');
  container.insertBefore(winPercentageElement, document.getElementById('chart'));
  
  gameLoopTimeout = setTimeout(runEpisode, 0);
}

init();

function showNewGameMessage() {
  return new Promise((resolve) => {
    const messageElement = document.createElement('div');
    messageElement.textContent = 'New game starting...';
    messageElement.style.fontSize = '20px';
    messageElement.style.marginTop = '10px';
    game.container.appendChild(messageElement);
    
    setTimeout(() => {
      messageElement.remove();
      game.render(false);
      resolve();
    }, 1000);
  });
}
