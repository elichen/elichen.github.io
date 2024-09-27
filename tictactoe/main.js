const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization();

let isTraining = true;
let episodeCount = 0;
const maxEpisodes = 1000;
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
    let state, action, validMove;

    if (game.currentPlayer === 1) {  // AI agent's turn
      state = game.getState();
      action = agent.act(state, isTraining);
      validMove = game.makeMove(action);
    } else {  // Optimal opponent's turn
      action = game.findOptimalMove();
      validMove = game.makeMove(action);
    }

    if (validMove) {
      moveCount++;
      const nextState = game.getState();
      let reward;
      if (game.gameOver) {
        if (game.isDraw()) {
          reward = 0;  // Draw
        } else {
          reward = game.currentPlayer === 1 ? -1 : 1;  // Win/Loss
        }
      } else {
        reward = 0;  // Game not over yet
      }
      totalReward += reward;

      if (game.currentPlayer === -1) {  // Only remember and replay after the agent's move
        agent.remember(state, action, reward, nextState, game.gameOver);
        if (isTraining) {
          await agent.replay();
        }
      }
    }

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
      game.gameOver = true;
    }
  }

  game.render(isTraining); // Render the final game state

  episodeCount++;
  visualization.updateChart(episodeCount, totalReward, agent.epsilon);

  if ((isTraining && episodeCount < maxEpisodes) || !isTraining) {
    setTimeout(runEpisode, isTraining ? 0 : 1000); // Delay between episodes in test mode
  }
}

async function init() {
  visualization.createChart();
  document.getElementById('modeButton').addEventListener('click', toggleMode);
  await runEpisode();
}

init();