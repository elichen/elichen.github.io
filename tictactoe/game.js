class TicTacToeGame {
  constructor() {
    this.container = document.getElementById('game-container');
    this.reset()
  }

  reset(isTraining = false, agentStarts = true) {
    this.board = new Array(9).fill(0);
    this.currentPlayer = agentStarts ? 1 : 2;
    this.gameOver = false;
    if (!isTraining) {
      this.clearDisplay();
    }
  }

  isValidMove(position) {
    return this.board[position] === 0;
  }

  makeMove(position) {
    if (this.isValidMove(position)) {
      this.board[position] = this.currentPlayer === 1 ? 1 : -1;
      this.checkGameOver();
      this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
      return true;
    }
    return false;
  }

  getState(forPlayer = this.currentPlayer) {
    if (forPlayer === 1) {
      return [...this.board];
    } else {
      return this.board.map(x => -x);
    }
  }

  getReward(forPlayer = this.currentPlayer) {
    if (!this.gameOver) return 0;
    
    if (this.isDraw()) return 0.5;
    
    const winner = this.checkWin(1) ? 1 : (this.checkWin(-1) ? 2 : 0);
    if (winner === 0) return 0.5;
    
    return winner === forPlayer ? 1 : -1;
  }

  checkGameOver() {
    if (this.checkWin(1) || this.checkWin(-1)) {
      this.gameOver = true;
      return;
    }

    if (!this.board.includes(0)) {
      this.gameOver = true;
    }
  }

  clearDisplay() {
    this.container.innerHTML = '';
  }

  render(isTraining = false) {
    if (isTraining && !this.gameOver) {
      return;
    }

    this.clearDisplay();
    const boardElement = document.createElement('div');
    boardElement.className = 'tic-tac-toe-board';
    
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.textContent = this.getCellContent(this.board[i]);
      boardElement.appendChild(cell);
    }

    this.container.appendChild(boardElement);

    if (this.gameOver) {
      const gameOverMessage = document.createElement('div');
      gameOverMessage.className = 'game-over-message';
      gameOverMessage.textContent = this.getGameOverMessage();
      this.container.appendChild(gameOverMessage);
    }
  }

  getCellContent(value) {
    if (value === 1) return 'X';
    if (value === -1) return 'O';
    return '';
  }

  getGameOverMessage() {
    if (this.checkWin(1)) {
      return "X wins!";
    } else if (this.checkWin(-1)) {
      return "O wins!";
    } else if (!this.board.includes(0)) {
      return "It's a draw!";
    } else {
      return "Game in progress";
    }
  }

  getValidMoves() {
    return this.board.reduce((validMoves, cell, index) => {
      if (cell === 0) validMoves.push(index);
      return validMoves;
    }, []);
  }

  isDraw() {
    return this.gameOver && !this.board.includes(0);
  }

  checkWin(player) {
    const winPatterns = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6] // Diagonals
    ];

    for (const pattern of winPatterns) {
      const [a, b, c] = pattern;
      if (this.board[a] === player && this.board[b] === player && this.board[c] === player) {
        return true;
      }
    }
    return false;
  }

  minimax(board, player, depth = 0) {
    const availableMoves = this.getValidMoves();
    
    // Base cases
    if (this.checkWin(1)) return 10 - depth;
    if (this.checkWin(-1)) return depth - 10;
    if (availableMoves.length === 0) return 0;
    
    if (player === 1) {
      let bestScore = -Infinity;
      for (let move of availableMoves) {
        board[move] = player;
        let score = this.minimax(board, -1, depth + 1);
        board[move] = 0;
        bestScore = Math.max(score, bestScore);
      }
      return bestScore;
    } else {
      let bestScore = Infinity;
      for (let move of availableMoves) {
        board[move] = player;
        let score = this.minimax(board, 1, depth + 1);
        board[move] = 0;
        bestScore = Math.min(score, bestScore);
      }
      return bestScore;
    }
  }

  getBestMove(player = 1) {
    const availableMoves = this.getValidMoves();
    let bestScore = player === 1 ? -Infinity : Infinity;
    let bestMove = availableMoves[0];
    
    for (let move of availableMoves) {
        this.board[move] = player;
        let score = this.minimax(this.board, -player);
        this.board[move] = 0;
        
        if (player === 1 ? score > bestScore : score < bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    
    return bestMove;
  }
}
