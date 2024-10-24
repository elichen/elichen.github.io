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

  checkGameOver() {
    if (this.checkWin(1) || this.checkWin(-1)) {
      this.gameOver = true;
      return;
    }

    if (!this.board.includes(0)) {
      this.gameOver = true;
    }
  }

  getState() {
    return this.board;
  }

  clearDisplay() {
    this.container.innerHTML = '';
  }

  render(isTraining = false) {
    if (isTraining && !this.gameOver) {
      // Don't render intermediate states during training
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
      return "Agent (X) wins!";
    } else if (this.checkWin(-1)) {
      return "Opponent (O) wins!";
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

  findOptimalMove() {
    const currentPlayer = this.currentPlayer === 1 ? 1 : -1;
    const opponent = -currentPlayer;

    // If center is empty, take it
    if (this.board[4] === 0) return 4;

    // Check for winning move
    for (let i = 0; i < 9; i++) {
      if (this.board[i] === 0) {
        this.board[i] = currentPlayer;
        if (this.checkWin(currentPlayer)) {
          this.board[i] = 0;
          return i;
        }
        this.board[i] = 0;
      }
    }

    // Check for blocking opponent's winning move
    for (let i = 0; i < 9; i++) {
      if (this.board[i] === 0) {
        this.board[i] = opponent;
        if (this.checkWin(opponent)) {
          this.board[i] = 0;
          return i;
        }
        this.board[i] = 0;
      }
    }

    // Create a fork or block opponent's fork
    const forkMove = this.findForkMove(currentPlayer) || this.findForkMove(opponent);
    if (forkMove !== -1) return forkMove;

    // Take corners if available
    const corners = [0, 2, 6, 8];
    const availableCorners = corners.filter(i => this.board[i] === 0);
    if (availableCorners.length > 0) {
      return availableCorners[Math.floor(Math.random() * availableCorners.length)];
    }

    // Take any available side
    const sides = [1, 3, 5, 7];
    const availableSides = sides.filter(i => this.board[i] === 0);
    if (availableSides.length > 0) {
      return availableSides[Math.floor(Math.random() * availableSides.length)];
    }

    // No moves available (shouldn't happen in a normal game)
    return -1;
  }

  findForkMove(player) {
    for (let i = 0; i < 9; i++) {
      if (this.board[i] === 0) {
        this.board[i] = player;
        let winningMoves = 0;
        for (let j = 0; j < 9; j++) {
          if (this.board[j] === 0) {
            this.board[j] = player;
            if (this.checkWin(player)) {
              winningMoves++;
            }
            this.board[j] = 0;
          }
        }
        this.board[i] = 0;
        if (winningMoves >= 2) {
          return i;
        }
      }
    }
    return -1;
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

  // Add a new method to find a random valid move
  findRandomMove() {
    const validMoves = this.getValidMoves();
    if (validMoves.length === 0) return -1;
    return validMoves[Math.floor(Math.random() * validMoves.length)];
  }
}
