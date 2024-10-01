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
    let action, validMove, reward, nextState, invalid;

    // AI agent's turn
    action = agent.act(state, isTraining);
    validMove = game.makeMove(action);
    moveCount++;
    
    if (!validMove) {
      console.log("Invalid move by agent. Penalizing.");
      invalid = true;
      game.gameOver = true;  // End the game on invalid move
      reward = -1; // Penalty for invalid move
    } else {
      // Check if the game is over after agent's move
      if (game.gameOver) {
        if (game.isDraw()) {
          reward = 0.5;  // Draw
          console.log(`Game ended in a tie after ${moveCount} moves.`);
        } else {
          reward = 1;  // Win
          console.log(`Agent won in ${moveCount} moves!`);
        }
      } else {
        // Opponent's turn
        const opponentAction = Math.random() < 0.1 ? game.findRandomMove() : game.findOptimalMove();
        if (opponentAction === -1) {
          console.error("Opponent failed to find a valid move");
          break;
        }
        game.makeMove(opponentAction);
        
        // Evaluate the result after opponent's move
        if (game.gameOver) {
          if (game.isDraw()) {
            reward = 0.5;  // Draw
            console.log(`Game ended in a tie after ${moveCount + 1} moves.`);
          } else {
            reward = -1;  // Loss
          }
        } else {
          // Small positive reward for making a valid move
          reward = 0.1;  // Game continues
        }
      }
    }

    nextState = game.getState();

    // Store the experience with the result of both moves
    agent.remember(state, action, reward, nextState, game.gameOver);

    totalReward += reward;

    if (!isTraining) {
      game.render(isTraining);
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