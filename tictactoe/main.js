const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization(1000);

let isTraining = true;
let episodeCount = 0;
let stopTraining = false;

function toggleMode() {
  isTraining = !isTraining;
  stopTraining = isTraining ? false : true;
  episodeCount = 0;
  document.getElementById('modeButton').textContent = isTraining ? 'Switch to Test Mode' : 'Switch to Train Mode';
  if (!isTraining) {
    game.clearDisplay(); // Only clear the display when switching to test mode
  }
  runEpisode();
}

async function runEpisode() {
  game.reset(isTraining);
  let totalReward = 0;
  let moveCount = 0;

  while (!game.gameOver) {
    const state = game.getState();
    let action, validMove, reward, invalid;

    if (game.currentPlayer === 1) {  // AI agent's turn
      action = agent.act(state, isTraining);
      validMove = game.makeMove(action);
      
      if (!validMove) {
        console.log("Invalid move by agent. Penalizing.");
        invalid = true;
        game.gameOver = true;  // End the game on invalid move
      }
    } else {  // Opponent's turn
      // Use epsilon to decide between optimal and random move
      action = Math.random() < agent.epsilon ? game.findRandomMove() : game.findOptimalMove();
      if (action === -1) {
        console.error("Opponent failed to find a valid move");
        break;  // Exit the game loop if no valid move found
      }
      validMove = game.makeMove(action);
      if (!validMove) {
        console.error("Opponent's move was invalid");
        break;  // Exit the game loop if the move was invalid
      }
    }

    moveCount++;
    const nextState = game.getState();

    if (game.gameOver) {
      if (invalid) {
        reward = -10; // Increase penalty for invalid moves
      } else if (game.isDraw()) {
        reward = 0.5;  // Draw
      } else {
        reward = game.currentPlayer === 0 ? 1 : -1;  // Assign higher rewards
        console.log("Game over! currentPlayer:", game.currentPlayer, " moves:", moveCount)
      }
    } else {
      reward = 0;  // Game not over yet
    }

    totalReward += reward
    agent.remember(state, action, reward, nextState, game.gameOver);

    if (!isTraining) {
      game.render(isTraining);
      // Add a small delay to visualize moves in test mode
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    if (stopTraining && isTraining) {
      break;
    }

    // Safety check to prevent infinite loops
    if (moveCount > 9) {
      console.error("Game exceeded maximum moves");
      game.gameOver = true;
    }
  }

  game.render(isTraining); // Render the final game state

  episodeCount++;
  if (isTraining) {
    agent.decayEpsilon(); // Decay epsilon after each episode
    await agent.replay(); // Perform replay after each episode
  }
  visualization.updateChart(episodeCount, totalReward, agent.epsilon);

  // Continue training indefinitely or run test episodes
  setTimeout(runEpisode, isTraining ? 0 : 1000); // Delay between episodes in test mode
}

async function init() {
  visualization.createChart();
  document.getElementById('modeButton').addEventListener('click', toggleMode);
  await runEpisode();
}

init();