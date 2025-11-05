"""
TicTacToe Environment for PPO Training with Perfect/Random Opponents
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from perfect_player import PerfectPlayer


class TicTacToeEnv(gym.Env):
    """TicTacToe environment with configurable opponents"""

    def __init__(self, opponent_type='mixed', perfect_ratio=0.7):
        super().__init__()

        # 9 positions on the board
        self.action_space = spaces.Discrete(9)
        # Board state: 9 positions, each can be 0 (empty), 1 (agent), -1 (opponent)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)

        # Opponent configuration
        self.opponent_type = opponent_type
        self.perfect_ratio = perfect_ratio  # For mixed opponent

        # Initialize board
        self.board = None
        self.done = False

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        self.board = np.zeros(9, dtype=np.float32)
        self.done = False

        # Randomly decide who goes first
        if random.random() < 0.5:
            # Opponent goes first
            self._opponent_move()

        return self.board.copy(), {}

    def step(self, action):
        """Execute agent's action"""
        if self.done:
            return self.board.copy(), 0.0, True, False, {}

        # Check if action is valid
        if self.board[action] != 0:
            # Invalid move - big penalty and end episode
            return self.board.copy(), -10.0, True, False, {"invalid_move": True}

        # Make agent's move (agent is 1)
        self.board[action] = 1

        # Check if agent won
        if self._check_winner(1):
            return self.board.copy(), 1.0, True, False, {"winner": "agent"}

        # Check for draw
        if np.all(self.board != 0):
            # Draw is actually good against perfect opponent!
            return self.board.copy(), 0.5, True, False, {"draw": True}

        # Opponent's turn
        self._opponent_move()

        # Check if opponent won
        if self._check_winner(-1):
            return self.board.copy(), -1.0, True, False, {"winner": "opponent"}

        # Check for draw after opponent move
        if np.all(self.board != 0):
            return self.board.copy(), 0.5, True, False, {"draw": True}

        # Game continues
        return self.board.copy(), 0.0, False, False, {}

    def _opponent_move(self):
        """Make opponent move based on opponent type"""
        available = np.where(self.board == 0)[0]
        if len(available) == 0:
            return

        # Determine opponent type for this move
        if self.opponent_type == 'mixed':
            # Mix of perfect and random
            use_perfect = random.random() < self.perfect_ratio
        elif self.opponent_type == 'perfect':
            use_perfect = True
        elif self.opponent_type == 'random':
            use_perfect = False
        else:  # self_play or other
            # For now, just use random for self_play
            use_perfect = False

        if use_perfect:
            # Use fast PerfectPlayer instead of slow minimax
            if not hasattr(self, 'perfect_player'):
                self.perfect_player = PerfectPlayer(player_symbol=-1)  # Opponent is -1
            move = self.perfect_player.get_move(self.board)
        else:
            # Random player
            move = random.choice(available)

        self.board[move] = -1

    def _check_winner(self, player):
        """Check if a player has won"""
        return self._check_winner_static(self.board, player)

    def _check_winner_static(self, board, player):
        """Static version of winner check for minimax"""
        # Check rows
        for i in range(0, 9, 3):
            if all(board[i:i+3] == player):
                return True

        # Check columns
        for i in range(3):
            if all(board[i::3] == player):
                return True

        # Check diagonals
        if all(board[[0, 4, 8]] == player):
            return True
        if all(board[[2, 4, 6]] == player):
            return True

        return False

    def action_masks(self):
        """Return valid actions mask for MaskablePPO"""
        # Valid moves are where board is empty (0)
        return self.board == 0

    def render(self):
        """Render the board"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for i in range(0, 9, 3):
            print(' '.join(symbols[int(x)] for x in self.board[i:i+3]))
        print()