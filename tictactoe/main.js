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
  runEpisode();
}

async function runEpisode() {
  game.reset();
  let totalReward = 0;
  let moveCount = 0;

  while (!game.gameOver) {
    const state = game.getState();
    const action = agent.act(state, isTraining);
    console.log(`Move ${moveCount + 1}: Player ${game.currentPlayer}, Action: ${action}`);
    
    const validMove = game.makeMove(action);

    if (validMove) {
      moveCount++;
      const nextState = game.getState();
      const reward = game.gameOver ? (game.currentPlayer === 1 ? -1 : 1) : 0;
      totalReward += reward;

      agent.remember(state, action, reward, nextState, game.gameOver);

      if (isTraining) {
        await agent.replay();
      }
    } else {
      console.log(`Invalid move: ${action}`);
      // In test mode, if an invalid move is made, choose a random valid move
      if (!isTraining) {
        const validMoves = game.getValidMoves();
        const randomValidMove = validMoves[Math.floor(Math.random() * validMoves.length)];
        console.log(`Choosing random valid move: ${randomValidMove}`);
        game.makeMove(randomValidMove);
        moveCount++;
      }
    }

    game.render(); // Always render the game state

    if (!isTraining) {
      // Add a small delay to visualize moves in test mode
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    if (stopTraining && isTraining) {
      break;
    }

    // Safety check to prevent infinite loops
    if (moveCount > 9) {
      console.log("Game exceeded maximum moves. Forcing end.");
      game.gameOver = true;
    }
  }

  console.log(`Episode ended. Total moves: ${moveCount}`);

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