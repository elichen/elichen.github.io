class TicTacToeGame {
  constructor() {
    this.board = new Array(9).fill(0);
    this.currentPlayer = 1;
    this.gameOver = false;
    this.container = document.getElementById('game-container');
  }

  reset() {
    this.board = new Array(9).fill(0);
    this.currentPlayer = 1;
    this.gameOver = false;
    this.render();
  }

  makeMove(position) {
    if (this.board[position] === 0 && !this.gameOver) {
      this.board[position] = this.currentPlayer;
      this.checkGameOver();
      this.currentPlayer = -this.currentPlayer;
      this.render();
      return true;
    }
    return false;
  }

  checkGameOver() {
    // Check rows, columns, and diagonals for a win
    const winPatterns = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6] // Diagonals
    ];

    for (const pattern of winPatterns) {
      const [a, b, c] = pattern;
      if (this.board[a] !== 0 && this.board[a] === this.board[b] && this.board[a] === this.board[c]) {
        this.gameOver = true;
        return;
      }
    }

    if (!this.board.includes(0)) {
      this.gameOver = true;
    }
  }

  getState() {
    return this.board;
  }

  render() {
    this.container.innerHTML = '';
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
    if (this.board.includes(0)) {
      return `Player ${this.getCellContent(-this.currentPlayer)} wins!`;
    } else {
      return "It's a draw!";
    }
  }

  getValidMoves() {
    return this.board.reduce((validMoves, cell, index) => {
      if (cell === 0) validMoves.push(index);
      return validMoves;
    }, []);
  }
}