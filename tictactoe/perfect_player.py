"""
Rule-based perfect Tic-Tac-Toe player
No recursion, no minimax - just simple priority rules
"""

import numpy as np


class PerfectPlayer:
    """
    Implements perfect Tic-Tac-Toe play using rule-based strategy
    Much faster than minimax - no recursion needed!
    """

    def __init__(self, player_symbol=1):
        """
        player_symbol: 1 for X, -1 for O
        """
        self.player = player_symbol
        self.opponent = -player_symbol

        # Win patterns for checking
        self.win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        # Corner and side positions for strategy
        self.corners = [0, 2, 6, 8]
        self.sides = [1, 3, 5, 7]
        self.center = 4

    def get_move(self, board):
        """
        Get the best move using rule-based strategy
        board: numpy array of 9 elements (1=X, -1=O, 0=empty)
        Returns: position (0-8) for best move
        """
        # 1. Win if possible
        move = self._find_winning_move(board, self.player)
        if move is not None:
            return move

        # 2. Block opponent's win
        move = self._find_winning_move(board, self.opponent)
        if move is not None:
            return move

        # 3. Create fork (two ways to win)
        move = self._find_fork(board, self.player)
        if move is not None:
            return move

        # 4. Block opponent's fork
        move = self._block_fork(board)
        if move is not None:
            return move

        # 5. Play center
        if board[self.center] == 0:
            return self.center

        # 6. Play opposite corner from opponent
        move = self._opposite_corner(board)
        if move is not None:
            return move

        # 7. Play empty corner
        for corner in self.corners:
            if board[corner] == 0:
                return corner

        # 8. Play empty side
        for side in self.sides:
            if board[side] == 0:
                return side

        # Should never reach here in a valid game
        valid_moves = np.where(board == 0)[0]
        return valid_moves[0] if len(valid_moves) > 0 else None

    def _find_winning_move(self, board, player):
        """Find a move that wins immediately"""
        for pattern in self.win_patterns:
            values = [board[i] for i in pattern]
            if values.count(player) == 2 and values.count(0) == 1:
                # Two of player's pieces and one empty
                for i in pattern:
                    if board[i] == 0:
                        return i
        return None

    def _find_fork(self, board, player):
        """Find a move that creates two winning threats"""
        for move in np.where(board == 0)[0]:
            # Try the move
            test_board = board.copy()
            test_board[move] = player

            # Count winning threats
            threats = 0
            for pattern in self.win_patterns:
                values = [test_board[i] for i in pattern]
                if values.count(player) == 2 and values.count(0) == 1:
                    threats += 1

            if threats >= 2:
                return move
        return None

    def _block_fork(self, board):
        """Block opponent's fork attempts"""
        # Check if opponent can create a fork
        opponent_fork = self._find_fork(board, self.opponent)

        if opponent_fork is not None:
            # Try to force opponent to defend instead
            # Look for a move that creates a threat
            for move in np.where(board == 0)[0]:
                test_board = board.copy()
                test_board[move] = self.player

                # Check if this creates a winning threat
                for pattern in self.win_patterns:
                    values = [test_board[i] for i in pattern]
                    if values.count(self.player) == 2 and values.count(0) == 1:
                        # Make sure opponent's block doesn't create their fork
                        for i in pattern:
                            if test_board[i] == 0:
                                block_board = test_board.copy()
                                block_board[i] = self.opponent
                                if self._find_fork(block_board, self.opponent) is None:
                                    return move

            # If no safe attacking move, just block the fork
            return opponent_fork
        return None

    def _opposite_corner(self, board):
        """Play opposite corner from opponent"""
        opposite_pairs = [(0, 8), (2, 6)]

        for corner1, corner2 in opposite_pairs:
            if board[corner1] == self.opponent and board[corner2] == 0:
                return corner2
            if board[corner2] == self.opponent and board[corner1] == 0:
                return corner1
        return None


def test_perfect_player():
    """Test the perfect player"""
    player = PerfectPlayer(player_symbol=-1)  # Play as O

    # Test scenario 1: Block win
    board = np.array([1, 1, 0, 0, -1, 0, 0, 0, 0], dtype=float)
    move = player.get_move(board)
    print(f"Block win test: Should play 2, got {move}")
    assert move == 2

    # Test scenario 2: Take win
    board = np.array([-1, -1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    move = player.get_move(board)
    print(f"Take win test: Should play 2, got {move}")
    assert move == 2

    # Test scenario 3: Take center
    board = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    move = player.get_move(board)
    print(f"Take center test: Should play 4, got {move}")
    assert move == 4

    print("All tests passed!")


if __name__ == "__main__":
    test_perfect_player()

    print("\nSpeed comparison:")
    import time

    # Time 1000 moves
    player = PerfectPlayer()
    board = np.zeros(9)

    start = time.time()
    for _ in range(1000):
        move = player.get_move(board)
    end = time.time()

    print(f"1000 moves in {end-start:.4f} seconds")
    print(f"Average: {(end-start)/1000*1000:.2f} ms per move")