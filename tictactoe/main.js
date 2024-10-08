const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization(1000);

let isTraining = true;
let episodeCount = 0;

function toggleMode() {
  isTraining = !isTraining;
  document.getElementById('modeButton').textContent = isTraining ? 'Switch to Test Mode' : 'Switch to Train Mode';
}

async function runEpisode() {
  game.reset(isTraining);
  let totalReward = 0;
  let moveCount = 0;
  let loss = null;

  while (!game.gameOver) {
    const state = game.getState();
    let action, validMove, nextState, invalid;
    let reward = 0;

    tf.tidy(() => {
      // AI agent's turn
      action = agent.act(state, isTraining);
    });

    validMove = game.makeMove(action);
    moveCount++;
    
    if (!validMove) {
      console.log("Invalid move by agent. Debugging information:");
      console.log("Attempted move:", action);
      console.log("Current board state:", state);
      console.log("Valid moves:", game.getValidMoves());
      
      invalid = true;
      game.gameOver = true;  // End the game on invalid move
      reward = -10; // Penalty for invalid move
    } else {
      // Check if the game is over after agent's move
      if (game.gameOver) {
        if (game.isDraw()) {
          reward = 0;  // Draw
          console.log(`Game ended in a tie after ${moveCount} moves.`);
        } else {
          reward = 1;  // Win
          console.log(`Agent won in ${moveCount} moves!`);
        }
      } else {
        // Opponent's turn
        const opponentAction = Math.random() < 0.5 ? game.findRandomMove() : game.findOptimalMove();
        game.makeMove(opponentAction);
        
        // Evaluate the result after opponent's move
        if (game.gameOver) {
          if (game.isDraw()) {
            reward = 0;  // Draw
            console.log(`Game ended in a tie after ${moveCount + 1} moves.`);
          } else {
            reward = -1;  // Loss
          }
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
  }

  game.render(isTraining); // Render the final game state

  episodeCount++;
  if (isTraining) {
    agent.decayEpsilon(); // Decay epsilon after each episode
    loss = await agent.replay(); // Perform replay after each episode and get the loss
  }
  visualization.updateChart(episodeCount, agent.epsilon, loss);

  // Continue training indefinitely or run test episodes
  setTimeout(runEpisode, isTraining ? 0 : 1000); // Delay between episodes in test mode
}

async function init() {
  visualization.createChart();
  document.getElementById('modeButton').addEventListener('click', toggleMode);
  await runEpisode();
}

init();