const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization(1000);

let isTraining = true;
let episodeCount = 0;
let testGamesPlayed = 0;
let testGamesWon = 0;

function toggleMode() {
  isTraining = !isTraining;
  document.getElementById('modeButton').textContent = isTraining ? 'Switch to Test Mode' : 'Switch to Train Mode';
  if (!isTraining) {
    // Reset test statistics when entering test mode
    testGamesPlayed = 0;
    testGamesWon = 0;
    updateWinPercentage();
  }
}

function updateWinPercentage() {
  const winPercentage = (testGamesWon / testGamesPlayed * 100).toFixed(2);
  document.getElementById('winPercentage').textContent = `Win Percentage: ${winPercentage}%`;
}

async function runEpisode() {
  game.reset(isTraining);
  let totalReward = 0;
  let moveCount = 0;
  let loss = null;
  let gameResult = 0; // 0 for draw, 1 for win, -1 for loss

  while (!game.gameOver) {
    const state = game.getState();
    const validMoves = game.getValidMoves();  // Get valid moves
    let action, nextState;
    let reward = 0;

    tf.tidy(() => {
      // Pass the valid moves to the agent
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
    } else {
      // Opponent's turn
      const opponentAction = Math.random() < .5 ? game.findRandomMove() : game.findOptimalMove();
      game.makeMove(opponentAction);
      
      // Evaluate the result after opponent's move
      if (game.gameOver) {
        if (game.isDraw()) {
          reward = 0.5;  // Draw
          console.log(`Game ended in a tie after ${moveCount + 1} moves.`);
        } else {
          reward = -1;  // Loss
          console.log(`Agent lost after ${moveCount + 1} moves.`);
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
    
    // Determine game result for visualization
    if (totalReward > 0) {
      gameResult = 1; // Win
    } else if (totalReward < 0) {
      gameResult = -1; // Loss
    }
    // Draw (gameResult = 0) is already set by default
  } else {
    testGamesPlayed++;
    updateWinPercentage();
  }
  
  // Update the chart with the game result
  visualization.updateChart(episodeCount, agent.epsilon, loss, gameResult);

  // Continue training indefinitely or run test episodes
  setTimeout(runEpisode, isTraining ? 0 : 1000); // Delay between episodes in test mode
}

async function init() {
  visualization.createChart();
  document.getElementById('modeButton').addEventListener('click', toggleMode);
  
  // Create a new element to display win percentage
  const winPercentageElement = document.createElement('div');
  winPercentageElement.id = 'winPercentage';
  document.body.insertBefore(winPercentageElement, document.getElementById('chart'));
  
  await runEpisode();
}

init();
