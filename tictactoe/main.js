const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization(1000);

let isTraining = true;
let episodeCount = 0;
let testGamesPlayed = 0;
let testGamesWon = 0;
let opponentDifficulty = 5; // Default difficulty
let isHumanOpponent = false;
let gameLoopTimeout = null; // Add this line to keep track of the game loop

function toggleMode() {
  isTraining = !isTraining;
  document.getElementById('modeButton').textContent = isTraining ? 'Switch to Test Mode' : 'Switch to Train Mode';
  if (!isTraining) {
    // Reset test statistics when entering test mode
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

function getOpponentMove(game) {
  const randomThreshold = opponentDifficulty / 10;
  return Math.random() < randomThreshold ? game.findOptimalMove() : game.findRandomMove();
}

function getRandomStartingPlayer() {
  return Math.random() < 0.5 ? 1 : 2;
}

async function runEpisode() {
  const agentStarts = getRandomStartingPlayer() === 1;
  game.reset(isTraining, agentStarts);
  let totalReward = 0;
  let moveCount = 0;
  let loss = null;
  let gameResult = 0; // 0 for draw, 1 for win, -1 for loss

  // Render the initial game state for human opponent mode
  if (!isTraining && isHumanOpponent) {
    game.render(isTraining);
  }

  while (!game.gameOver) {
    const state = game.getState();
    const validMoves = game.getValidMoves();
    let action, nextState;
    let reward = 0;

    if (game.currentPlayer === 1) {
      // Agent's turn (always X)
      tf.tidy(() => {
        action = agent.act(state, isTraining, validMoves);
      });

      game.makeMove(action);
      moveCount++;
      
      // Check if the game is over after agent's move
      if (game.gameOver) {
        if (game.isDraw()) {
          reward = 0.5;  // Draw
          console.log(`Game ended in a tie after ${moveCount} moves.`);
        } else {
          reward = 1;  // Win
          console.log(`Agent won in ${moveCount} moves!`);
          if (!isTraining) {
            testGamesWon++;
          }
        }
      }
    } else {
      // Opponent's turn (always O)
      if (isHumanOpponent && !isTraining) {
        // Wait for human move
        action = await waitForHumanMove();
      } else {
        action = getOpponentMove(game);
      }
      game.makeMove(action);
      moveCount++;
      
      // Evaluate the result after opponent's move
      if (game.gameOver) {
        if (game.isDraw()) {
          reward = 0.5;  // Draw
          console.log(`Game ended in a tie after ${moveCount} moves.`);
        } else {
          reward = -1;  // Loss
          console.log(`Agent lost after ${moveCount} moves.`);
        }
      }
    }

    nextState = game.getState();

    // Store the experience after each move
    if (game.currentPlayer === 2) {  // Store after agent's move (when it's opponent's turn)
      agent.remember(state, action, reward, nextState, game.gameOver);
    }

    totalReward += reward;

    if (!isTraining) {
      game.render(isTraining);
      if (!isHumanOpponent) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
  }

  game.render(isTraining); // Render the final game state

  episodeCount++;
  if (isTraining) {
    agent.decayEpsilon();
    loss = await agent.replay();
    
    // Determine game result for visualization
    if (totalReward > 0) {
      gameResult = 1; // Win
    } else if (totalReward < 0) {
      gameResult = -1; // Loss
    }
  } else {
    testGamesPlayed++;
    updateWinPercentage();
  }
  
  // Update the chart with the game result
  visualization.updateChart(episodeCount, agent.epsilon, loss, gameResult);

  // Update this part at the end of runEpisode
  if (isTraining || !isTraining) {
    if (!isTraining && isHumanOpponent) {
      await showNewGameMessage();
    }
    // Clear any existing timeout before setting a new one
    if (gameLoopTimeout) {
      clearTimeout(gameLoopTimeout);
    }
    gameLoopTimeout = setTimeout(runEpisode, isTraining ? 0 : 1000);
  }
}

function updateDifficulty(value) {
  opponentDifficulty = value;
  document.getElementById('difficultyValue').textContent = value;
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
  // Clear any existing game loop before resetting
  if (gameLoopTimeout) {
    clearTimeout(gameLoopTimeout);
    gameLoopTimeout = null;
  }
  game.reset(isTraining);
  if (!isTraining) {
    runEpisode();
  } else {
    // If switching to training mode, start a new episode
    gameLoopTimeout = setTimeout(runEpisode, 0);
  }
}

async function init() {
  visualization.createChart();
  document.getElementById('modeButton').addEventListener('click', toggleMode);
  
  // Create the win percentage element and insert it into the container
  const winPercentageElement = document.createElement('div');
  winPercentageElement.id = 'winPercentage';
  winPercentageElement.className = 'stats';
  const container = document.querySelector('.container');
  container.insertBefore(winPercentageElement, document.getElementById('chart'));
  
  document.getElementById('humanOpponentCheckbox').addEventListener('change', (event) => {
    isHumanOpponent = event.target.checked;
    if (!isTraining) {
      resetGame();
    }
  });
  
  // Start the initial episode
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
      game.render(false); // Render the new game state immediately after removing the message
      resolve();
    }, 1000);
  });
}
