const ROWS = 6;
const COLUMNS = 7;
const CONNECT = 4;
const PLAYER = 1;
const AI = 2;

const DIFFICULTY_SETTINGS = [
  { label: "Casual", depth: 2, randomness: 0.65, timeBudget: 130 },
  { label: "Challenger", depth: 5, randomness: 0.1, timeBudget: 320 },
  { label: "Grandmaster", depth: 7, randomness: 0, timeBudget: 720 }
];

const boardElement = document.getElementById("board");
const slotGridElement = document.getElementById("slot-grid");
const discLayerElement = document.getElementById("disc-layer");
const statusElement = document.getElementById("status");
const resetButton = document.getElementById("reset-button");
const difficultyButtons = Array.from(document.querySelectorAll(".difficulty__option"));

let boardState = createMatrix(0);
let currentPlayer = PLAYER;
let gameActive = true;
let difficultyIndex = 0;
let boardMetrics = { cellSize: 0 };
let aiMoveTimeout = null;

const perf = window.performance || { now: () => Date.now() };

const previewDisc = document.createElement("div");
previewDisc.className = "disc preview";
discLayerElement.appendChild(previewDisc);

initialize();

function initialize() {
  buildSlots();
  updateBoardMetrics();
  bindEvents();
  startNewGame();
  requestAnimationFrame(updateBoardMetrics);
}

function buildSlots() {
  const fragment = document.createDocumentFragment();
  for (let r = 0; r < ROWS; r += 1) {
    for (let c = 0; c < COLUMNS; c += 1) {
      const slot = document.createElement("div");
      slot.className = "slot";
      slot.dataset.row = String(r);
      slot.dataset.col = String(c);
      fragment.appendChild(slot);
    }
  }
  slotGridElement.appendChild(fragment);
}

function bindEvents() {
  boardElement.addEventListener("click", handleBoardClick);
  boardElement.addEventListener("mousemove", handleBoardHover);
  boardElement.addEventListener("mouseleave", hidePreview);
  window.addEventListener("resize", () => {
    updateBoardMetrics();
  });

  resetButton.addEventListener("click", () => {
    startNewGame();
  });

  difficultyButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.level) {
        setDifficulty(Number(button.dataset.level));
      }
    });
  });
}

function createMatrix(fillValue) {
  return Array.from({ length: ROWS }, () => Array(COLUMNS).fill(fillValue));
}

function clearDiscs() {
  discLayerElement.querySelectorAll(".disc:not(.preview)").forEach((disc) => disc.remove());
  discLayerElement.appendChild(previewDisc);
}

function startNewGame() {
  if (aiMoveTimeout) {
    clearTimeout(aiMoveTimeout);
    aiMoveTimeout = null;
  }
  boardState = createMatrix(0);
  clearDiscs();
  currentPlayer = PLAYER;
  gameActive = true;
  updateStatus("Your move.");
  if (boardMetrics.cellSize === 0) {
    updateBoardMetrics();
  }
  showPreviewForColumn(3);
}

function setDifficulty(index) {
  if (difficultyIndex === index) {
    return;
  }
  difficultyIndex = index;
  difficultyButtons.forEach((btn, idx) => {
    const active = idx === index;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
  startNewGame();
}

function readBoardMetrics() {
  const gridRect = slotGridElement.getBoundingClientRect();
  const cellSize = gridRect.width / COLUMNS;
  return {
    cellSize: Number.isFinite(cellSize) ? cellSize : 0
  };
}

function updateBoardMetrics() {
  boardMetrics = readBoardMetrics();
}

function handleBoardClick(event) {
  if (!gameActive || currentPlayer !== PLAYER) {
    return;
  }
  const column = columnFromEvent(event);
  if (column < 0) {
    return;
  }
  if (findOpenRow(boardState, column) === -1) {
    bouncePreview();
    return;
  }
  applyMove(column, PLAYER);
}

function columnFromEvent(event) {
  if (boardMetrics.cellSize <= 0) {
    updateBoardMetrics();
  }
  const rect = slotGridElement.getBoundingClientRect();
  const x = event.clientX - rect.left;
  if (x < 0) {
    return -1;
  }
  if (x > rect.width) {
    return -1;
  }
  return Math.floor(x / boardMetrics.cellSize);
}

function handleBoardHover(event) {
  if (!gameActive || currentPlayer !== PLAYER) {
    hidePreview();
    return;
  }
  const column = columnFromEvent(event);
  if (column < 0) {
    hidePreview();
    return;
  }
  if (findOpenRow(boardState, column) === -1) {
    hidePreview();
    return;
  }
  showPreviewForColumn(column);
}

function showPreviewForColumn(column) {
  previewDisc.style.setProperty("--col", column);
  previewDisc.classList.add("is-visible");
}

function hidePreview() {
  previewDisc.classList.remove("is-visible");
}

function bouncePreview() {
  previewDisc.classList.add("bounce");
  setTimeout(() => {
    previewDisc.classList.remove("bounce");
  }, 250);
}

function findOpenRow(board, column) {
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    if (board[row][column] === 0) {
      return row;
    }
  }
  return -1;
}

function applyMove(column, piece) {
  const row = findOpenRow(boardState, column);
  if (row === -1) {
    return;
  }

  boardState[row][column] = piece;
  spawnDisc(row, column, piece);

  const winSequence = computeWinSequence(boardState, piece);
  if (winSequence) {
    concludeGame(piece, winSequence);
    return;
  }

  if (isBoardFull(boardState)) {
    concludeDraw();
    return;
  }

  currentPlayer = piece === PLAYER ? AI : PLAYER;
  if (currentPlayer === AI) {
    updateStatus("AI is thinking…");
    hidePreview();
    scheduleAiTurn();
  } else {
    updateStatus("Your move.");
    showPreviewForColumn(column);
  }

}

function spawnDisc(row, column, piece) {
  const disc = document.createElement("div");
  disc.className = `disc ${piece === PLAYER ? "player" : "ai"}`;
  disc.dataset.row = String(row);
  disc.dataset.col = String(column);
  disc.style.setProperty("--row", row);
  disc.style.setProperty("--col", column);
  discLayerElement.appendChild(disc);
  requestAnimationFrame(() => {
    disc.classList.add("dropping");
  });
}

function scheduleAiTurn() {
  const settings = DIFFICULTY_SETTINGS[difficultyIndex];
  aiMoveTimeout = setTimeout(() => {
    aiMoveTimeout = null;
    const column = chooseAiColumn(boardState, settings);
    if (column === null || column === undefined) {
      return;
    }
    applyMove(column, AI);
  }, 360);
}

function chooseAiColumn(board, settings) {
  const validColumns = getValidColumns(board);
  if (validColumns.length === 0) {
    return null;
  }

  // Secure immediate wins or blocks before deeper search.
  for (const column of validColumns) {
    const row = findOpenRow(board, column);
    if (row === -1) {
      continue;
    }
    board[row][column] = AI;
    const winning = hasConnect(board, AI);
    board[row][column] = 0;
    if (winning) {
      return column;
    }
  }

  for (const column of validColumns) {
    const row = findOpenRow(board, column);
    if (row === -1) {
      continue;
    }
    board[row][column] = PLAYER;
    const opponentWinning = hasConnect(board, PLAYER);
    board[row][column] = 0;
    if (opponentWinning) {
      return column;
    }
  }

  if (settings.randomness > 0 && Math.random() < settings.randomness) {
    return validColumns[Math.floor(Math.random() * validColumns.length)];
  }

  const deadline = perf.now() + settings.timeBudget;
  let bestScore = -Infinity;
  let chosenColumn = validColumns[0];

  const orderedColumns = orderColumns(validColumns);
  for (const column of orderedColumns) {
    const row = findOpenRow(board, column);
    if (row === -1) {
      continue;
    }
    board[row][column] = AI;
    const result = minimax(board, settings.depth - 1, -Infinity, Infinity, false, deadline);
    board[row][column] = 0;
    let score = result.score;
    if (score > bestScore) {
      bestScore = score;
      chosenColumn = column;
    }
    if (perf.now() > deadline && bestScore > 0) {
      break;
    }
  }

  return chosenColumn;
}

function minimax(board, depth, alpha, beta, maximizingPlayer, deadline) {
  const validColumns = getValidColumns(board);
  const aiWins = hasConnect(board, AI);
  const playerWins = hasConnect(board, PLAYER);
  const noMoreMoves = validColumns.length === 0;

  if (depth === 0 || aiWins || playerWins || noMoreMoves || perf.now() > deadline) {
    if (aiWins) {
      return { score: 100000 + depth };
    }
    if (playerWins) {
      return { score: -100000 - depth };
    }
    if (noMoreMoves) {
      return { score: 0 };
    }
    return { score: evaluateBoard(board) };
  }

  const orderedColumns = orderColumns(validColumns);

  if (maximizingPlayer) {
    let value = -Infinity;
    for (const column of orderedColumns) {
      const row = findOpenRow(board, column);
      if (row === -1) {
        continue;
      }
      board[row][column] = AI;
      const { score } = minimax(board, depth - 1, alpha, beta, false, deadline);
      board[row][column] = 0;
      value = Math.max(value, score);
      alpha = Math.max(alpha, value);
      if (alpha >= beta) {
        break;
      }
      if (perf.now() > deadline) {
        break;
      }
    }
    return { score: value };
  }

  let value = Infinity;
  for (const column of orderedColumns) {
    const row = findOpenRow(board, column);
    if (row === -1) {
      continue;
    }
    board[row][column] = PLAYER;
    const { score } = minimax(board, depth - 1, alpha, beta, true, deadline);
    board[row][column] = 0;
    value = Math.min(value, score);
    beta = Math.min(beta, value);
    if (beta <= alpha) {
      break;
    }
    if (perf.now() > deadline) {
      break;
    }
  }
  return { score: value };
}

function orderColumns(columns) {
  const center = Math.floor(COLUMNS / 2);
  return [...columns].sort((a, b) => Math.abs(a - center) - Math.abs(b - center));
}

function getValidColumns(board) {
  const columns = [];
  for (let column = 0; column < COLUMNS; column += 1) {
    if (board[0][column] === 0) {
      columns.push(column);
    }
  }
  return columns;
}

function evaluateBoard(board) {
  let score = 0;
  const centerColumn = Math.floor(COLUMNS / 2);
  let centerCount = 0;
  for (let row = 0; row < ROWS; row += 1) {
    if (board[row][centerColumn] === AI) {
      centerCount += 1;
    }
  }
  score += centerCount * 3;

  score += evaluateDirection(board, 0, 1);
  score += evaluateDirection(board, 1, 0);
  score += evaluateDirection(board, 1, 1);
  score += evaluateDirection(board, 1, -1);

  return score;
}

function evaluateDirection(board, deltaRow, deltaColumn) {
  let score = 0;
  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLUMNS; column += 1) {
      const window = [];
      for (let offset = 0; offset < CONNECT; offset += 1) {
        const r = row + deltaRow * offset;
        const c = column + deltaColumn * offset;
        if (r < 0 || r >= ROWS || c < 0 || c >= COLUMNS) {
          window.length = 0;
          break;
        }
        window.push(board[r][c]);
      }
      if (window.length === CONNECT) {
        score += scoreWindow(window);
      }
    }
  }
  return score;
}

function scoreWindow(window) {
  let aiCount = 0;
  let playerCount = 0;
  let emptyCount = 0;

  for (const cell of window) {
    if (cell === AI) aiCount += 1;
    else if (cell === PLAYER) playerCount += 1;
    else emptyCount += 1;
  }

  let score = 0;

  if (aiCount === 4) score += 120000;
  else if (aiCount === 3 && emptyCount === 1) score += 160;
  else if (aiCount === 2 && emptyCount === 2) score += 24;
  else if (aiCount === 1 && emptyCount === 3) score += 4;

  if (playerCount === 3 && emptyCount === 1) score -= 190;
  if (playerCount === 4) score -= 140000;

  return score;
}

function hasConnect(board, piece) {
  const directions = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1]
  ];

  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLUMNS; column += 1) {
      if (board[row][column] !== piece) {
        continue;
      }
      for (const [deltaRow, deltaColumn] of directions) {
        let inLine = 1;
        for (let offset = 1; offset < CONNECT; offset += 1) {
          const r = row + deltaRow * offset;
          const c = column + deltaColumn * offset;
          if (r < 0 || r >= ROWS || c < 0 || c >= COLUMNS) {
            break;
          }
          if (board[r][c] !== piece) {
            break;
          }
          inLine += 1;
        }
        if (inLine === CONNECT) {
          return true;
        }
      }
    }
  }

  return false;
}

function computeWinSequence(board, piece) {
  const directions = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1]
  ];

  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLUMNS; column += 1) {
      if (board[row][column] !== piece) {
        continue;
      }
      for (const [deltaRow, deltaColumn] of directions) {
        const sequence = [[row, column]];
        for (let offset = 1; offset < CONNECT; offset += 1) {
          const r = row + deltaRow * offset;
          const c = column + deltaColumn * offset;
          if (r < 0 || r >= ROWS || c < 0 || c >= COLUMNS) {
            break;
          }
          if (board[r][c] !== piece) {
            break;
          }
          sequence.push([r, c]);
        }
        if (sequence.length === CONNECT) {
          return sequence;
        }
      }
    }
  }
  return null;
}

function isBoardFull(board) {
  return board[0].every((value) => value !== 0);
}

function concludeGame(winner, sequence) {
  gameActive = false;
  hidePreview();
  const winnerIsPlayer = winner === PLAYER;
  updateStatus(winnerIsPlayer ? "You win!" : victoryLineForAI());

  const highlight = new Set(sequence.map(([r, c]) => `${r}-${c}`));
  discLayerElement.querySelectorAll(".disc:not(.preview)").forEach((disc) => {
    const key = `${disc.dataset.row}-${disc.dataset.col}`;
    if (highlight.has(key)) {
      disc.classList.add("winning");
      disc.classList.remove("faded");
    } else {
      disc.classList.add("faded");
    }
  });
}

function concludeDraw() {
  gameActive = false;
  hidePreview();
  updateStatus("Stalemate. Balanced minds.");
}

function updateStatus(text) {
  statusElement.textContent = text;
}

function victoryLineForAI() {
  switch (difficultyIndex) {
    case 0:
      return "AI wins. Brush up and try again.";
    case 1:
      return "AI holds the line. Think two moves ahead.";
    default:
      return "Grandmaster AI prevails. Study the patterns.";
  }
}
