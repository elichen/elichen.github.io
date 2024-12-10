const game = new TicTacToeGame();
const agent = new DQNAgent();
const visualization = new Visualization();

let isTraining = true;
let episodeCount = 0;
let testGamesPlayed = 0;
let testGamesWon = 0;
let gameLoopTimeout = null;

class EvaluationManager {
  constructor(evaluationFrequency = 100, numGames = 10) {
      this.evaluationFrequency = evaluationFrequency;
      this.numGames = numGames;
      this.evaluationResults = [];
  }

  async evaluateAgainstOptimal(agent, game) {
      let losses = 0;
      
      for (let i = 0; i < this.numGames; i++) {
          game.reset(true);
          
          while (!game.gameOver) {
              // Agent (player 1) moves
              const state = game.getState();
              const validMoves = game.getValidMoves();
              const action = agent.act(state, false, validMoves);
              game.makeMove(action);
              if (game.gameOver) break;
              
              // Optimal (player 2, -1) moves
              const optimalMove = game.getBestMove(-1);
              game.makeMove(optimalMove);
          }
          
          // Check if agent lost (O wins)
          if (game.checkWin(-1)) {
              losses++;
          }
      }
      
      // Losing rate = fraction of games lost
      const losingRate = losses / this.numGames;
      return losingRate; 
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
  const agentPlayer = agentStarts ? 1 : 2;
  const agentIsX = (agentPlayer === 1);

  game.reset(isTraining, agentStarts);

  if (!isTraining) {
    game.render(isTraining);
  }

  let episodeLoss = null;

  while (!game.gameOver) {
    // Always consider the "X perspective" state
    const stateFromX = game.getState(1);
    const validMovesFromX = game.getValidMoves();

    // If the agent is O, invert the perspective
    let inputState = stateFromX;
    let inputValidMoves = validMovesFromX;
    if (!agentIsX) {
      inputState = stateFromX.map(cell => -cell);
      inputValidMoves = inputState.reduce((acc, cell, idx) => {
        if (cell === 0) acc.push(idx);
        return acc;
      }, []);
    }

    let action;
    if (!isTraining && game.currentPlayer === 2) {
      // Human move if in test mode and current player is O
      const cells = document.querySelectorAll('.cell');
      action = await waitForHumanMove(cells);
    } else {
      // Agent move: same perspective logic applies in test mode as in training mode
      tf.tidy(() => {
        action = agent.act(inputState, isTraining, inputValidMoves);
      });
    }

    // Execute the move in the environment
    game.makeMove(action);

    // Compute reward from X’s perspective
    const rewardFromX = game.getReward(1);
    const done = game.gameOver;
    const nextStateFromX = game.getState(1);

    // Adjust reward and next state if agent is O
    let finalReward, finalNextState;
    if (agentIsX) {
      finalReward = rewardFromX;
      finalNextState = nextStateFromX;
    } else {
      finalNextState = nextStateFromX.map(cell => -cell);
      // Invert reward:
      // X win (1) => agent loses => -1
      // X lose (-1) => agent wins => +1
      // Draw (0.5) stays 0.5
      if (rewardFromX === 1) {
        finalReward = -1;
      } else if (rewardFromX === -1) {
        finalReward = 1;
      } else {
        finalReward = rewardFromX;
      }
    }

    if (isTraining) {
      // During training, store experience and train
      const preStateForAgent = agentIsX ? stateFromX : stateFromX.map(cell => -cell);
      const loss = await agent.remember(preStateForAgent, action, finalReward, finalNextState, done);
      if (loss !== null) {
        episodeLoss = loss;
      }
    }

    if (!isTraining) {
      game.render(isTraining);
    }

    // Evaluate periodically in training mode
    if (isTraining && episodeCount % evaluationManager.evaluationFrequency === 0) {
      const losingRate = await evaluationManager.evaluateAgainstOptimal(agent, game);
      visualization.updateStats(losingRate);
    }
  }

  episodeCount++;
  if (isTraining) {
    agent.decayEpsilon();
  }

  if (!isTraining) {
    await showNewGameMessage();
  }

  if (gameLoopTimeout) {
    clearTimeout(gameLoopTimeout);
  }
  gameLoopTimeout = setTimeout(runEpisode, isTraining ? 0 : 1000);
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
