const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization();

let isTraining = true;
let episodeCount = 0;
let testGamesPlayed = 0;
let testGamesWon = 0;
let gameLoopTimeout = null;

class EvaluationManager {
    constructor(evaluationFrequency = 100) {
        this.evaluationFrequency = evaluationFrequency;
        this.evaluationResults = [];
    }

    async evaluateAgainstOptimal(agent, game, numGames = 10) {
        let totalReward = 0;
        
        for (let i = 0; i < numGames; i++) {
            game.reset(true);
            let gameReward = 0;
            
            while (!game.gameOver) {
                // Agent's turn
                const state = game.getState();
                const validMoves = game.getValidMoves();
                const action = agent.act(state, false, validMoves); // false for no exploration
                game.makeMove(action);
                
                if (game.gameOver) {
                    gameReward = game.getReward(1);
                    break;
                }
                
                // Optimal player's turn
                const optimalMove = game.getBestMove();
                game.makeMove(optimalMove);
                
                if (game.gameOver) {
                    gameReward = game.getReward(1);
                }
            }
            
            totalReward += gameReward;
        }
        
        return totalReward / numGames;
    }
}

const evaluationManager = new EvaluationManager(100);

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
  let episodeLoss = null;
  let finalReward = 0;

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
    const reward = game.getReward(game.currentPlayer === 1 ? 2 : 1);
    finalReward = reward;
    const nextState = game.getState(game.currentPlayer === 1 ? 2 : 1);

    // Store experience and potentially train
    if (isTraining) {
      const loss = await agent.remember(preState, action, reward, nextState, game.gameOver);
      if (loss !== null) {
        episodeLoss = loss;
      }
    }

    if (!isTraining) {
      game.render(isTraining);
    }

    if (isTraining && episodeCount % evaluationManager.evaluationFrequency === 0) {
        const evaluationReward = await evaluationManager.evaluateAgainstOptimal(agent, game);
        visualization.updateStats(evaluationReward);
    }
  }

  game.render(isTraining);

  episodeCount++;
  if (isTraining) {
    agent.decayEpsilon();
    
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
  
  visualization.updateStats(finalReward);

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
  visualization.updateStats();
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
