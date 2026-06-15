/*
 * UnicornEngine — a pure, DOM-free tic-tac-toe engine.
 *
 * Works in the browser as a global (window.UnicornEngine) and in Node for
 * testing (module.exports at the bottom). No DOM, no audio, no side effects.
 *
 * Marks are plain strings. The UI calls a unicorn 'U' and a rainbow 'R', but
 * the engine treats them as opaque player tokens, so it stays renderer-agnostic.
 */
(function (root) {
  'use strict';

  var WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6],            // diagonals
  ];

  /** A fresh empty board: 9 nulls. */
  function emptyBoard() {
    return [null, null, null, null, null, null, null, null, null];
  }

  /** Indices of empty squares, left to right, top to bottom. */
  function availableMoves(board) {
    var moves = [];
    for (var i = 0; i < 9; i++) {
      if (!board[i]) moves.push(i);
    }
    return moves;
  }

  /**
   * The winning line if one exists, else null.
   * Returns the line as an array of three indices so the UI can highlight it.
   */
  function winningLine(board) {
    for (var i = 0; i < WIN_LINES.length; i++) {
      var a = WIN_LINES[i][0], b = WIN_LINES[i][1], c = WIN_LINES[i][2];
      if (board[a] && board[a] === board[b] && board[a] === board[c]) {
        return WIN_LINES[i];
      }
    }
    return null;
  }

  /** The winning mark ('U' / 'R') or null. */
  function winner(board) {
    var line = winningLine(board);
    return line ? board[line[0]] : null;
  }

  function isFull(board) {
    for (var i = 0; i < 9; i++) {
      if (!board[i]) return false;
    }
    return true;
  }

  function isGameOver(board) {
    return !!winner(board) || isFull(board);
  }

  function opponent(mark) {
    return mark === 'U' ? 'R' : 'U';
  }

  /**
   * Perfect minimax with depth-aware scoring: faster wins and slower losses
   * score better, so the AI presses an advantage and stalls when behind.
   * `ai` is the maximizing player. Returns { score, move }.
   *
   * Alpha-beta pruning keeps the full 9! search instant in the browser.
   */
  function minimax(board, ai, toMove, depth, alpha, beta) {
    var win = winner(board);
    if (win === ai) return { score: 10 - depth, move: -1 };
    if (win === opponent(ai)) return { score: depth - 10, move: -1 };
    if (isFull(board)) return { score: 0, move: -1 };

    var moves = availableMoves(board);
    var maximizing = toMove === ai;
    var best = { score: maximizing ? -Infinity : Infinity, move: moves[0] };

    for (var i = 0; i < moves.length; i++) {
      var idx = moves[i];
      board[idx] = toMove;
      var result = minimax(board, ai, opponent(toMove), depth + 1, alpha, beta);
      board[idx] = null;

      if (maximizing) {
        if (result.score > best.score) { best.score = result.score; best.move = idx; }
        if (best.score > alpha) alpha = best.score;
      } else {
        if (result.score < best.score) { best.score = result.score; best.move = idx; }
        if (best.score < beta) beta = best.score;
      }
      if (beta <= alpha) break;
    }
    return best;
  }

  var MOVE_ORDER = [4, 0, 2, 6, 8, 1, 3, 5, 7];

  function moveRank(move) {
    for (var i = 0; i < MOVE_ORDER.length; i++) {
      if (MOVE_ORDER[i] === move) return i;
    }
    return MOVE_ORDER.length;
  }

  /**
   * All optimal moves for `mark`, sorted by human tic-tac-toe priorities:
   * center, corners, then edges. Every returned move has the same minimax score,
   * so choosing among them remains perfect play.
   */
  function bestMoves(board, mark) {
    if (isGameOver(board)) return [];
    var moves = availableMoves(board);
    var bestScore = -Infinity;
    var result = [];
    for (var i = 0; i < moves.length; i++) {
      var idx = moves[i];
      var next = board.slice();
      next[idx] = mark;
      var score = minimax(next, mark, opponent(mark), 1, -Infinity, Infinity).score;
      if (score > bestScore) {
        bestScore = score;
        result = [idx];
      } else if (score === bestScore) {
        result.push(idx);
      }
    }
    result.sort(function (a, b) { return moveRank(a) - moveRank(b); });
    return result;
  }

  /** The optimal move for `mark`, or -1 if the board is finished. */
  function bestMove(board, mark) {
    var moves = bestMoves(board, mark);
    return moves.length ? moves[0] : -1;
  }

  /** A uniformly random legal move, or -1 if none. Accepts an optional RNG. */
  function randomMove(board, rng) {
    var moves = availableMoves(board);
    if (!moves.length) return -1;
    var r = (rng || Math.random)();
    return moves[Math.min(moves.length - 1, Math.floor(r * moves.length))];
  }

  /**
   * One-ply heuristic: take an immediate win, else block the opponent's
   * immediate win, else null. Used to make "medium" feel competent without
   * being unbeatable.
   */
  function tacticalMove(board, mark) {
    var moves = availableMoves(board);
    var i;
    for (i = 0; i < moves.length; i++) {
      var t = board.slice(); t[moves[i]] = mark;
      if (winner(t) === mark) return moves[i];
    }
    var foe = opponent(mark);
    for (i = 0; i < moves.length; i++) {
      var b = board.slice(); b[moves[i]] = foe;
      if (winner(b) === foe) return moves[i];
    }
    return null;
  }

  /**
   * Difficulty profiles. `optimal` is the chance the AI plays a perfect move
   * on any given turn; the rest of the time it plays a tactical-or-random move.
   *  - easy   : almost always casual, very beatable, kid-friendly.
   *  - medium : mixes perfect and tactical play, beatable with care.
   *  - hard   : always perfect, provably unbeatable.
   */
  var DIFFICULTY = {
    easy:   { optimal: 0.0,  tactical: 0.35 },
    medium: { optimal: 0.55, tactical: 0.85 },
    hard:   { optimal: 1.0,  tactical: 1.0 },
  };

  /**
   * Choose a move for `mark` at the given difficulty.
   * `rng` is an optional deterministic random source for tests.
   */
  function chooseMove(board, mark, difficulty, rng) {
    if (isGameOver(board)) return -1;
    var profile = DIFFICULTY[difficulty] || DIFFICULTY.hard;
    var random = rng || Math.random;

    if (profile.optimal >= 1 || random() < profile.optimal) {
      var perfect = bestMoves(board, mark);
      if (!perfect.length) return -1;
      if (difficulty === 'hard' && perfect.length > 1) {
        return perfect[Math.min(perfect.length - 1, Math.floor(random() * perfect.length))];
      }
      return perfect[0];
    }
    if (profile.tactical >= 1 || random() < profile.tactical) {
      var tactical = tacticalMove(board, mark);
      if (tactical !== null) return tactical;
    }
    return randomMove(board, random);
  }

  /**
   * A tiny immutable game-state helper for the UI. Each method that changes
   * state returns a new state object, so the renderer can diff cleanly.
   */
  function createGame(firstPlayer) {
    function build(board, toMove) {
      var line = winningLine(board);
      var win = line ? board[line[0]] : null;
      return {
        board: board,
        toMove: toMove,
        winner: win,
        winningLine: line,
        isTie: !win && isFull(board),
        isOver: !!win || isFull(board),
        play: function (index) {
          if (board[index] || win || isFull(board)) return this;
          var next = board.slice();
          next[index] = toMove;
          return build(next, opponent(toMove));
        },
      };
    }
    return build(emptyBoard(), firstPlayer || 'U');
  }

  var UnicornEngine = {
    WIN_LINES: WIN_LINES,
    DIFFICULTY: DIFFICULTY,
    emptyBoard: emptyBoard,
    availableMoves: availableMoves,
    winningLine: winningLine,
    winner: winner,
    isFull: isFull,
    isGameOver: isGameOver,
    opponent: opponent,
    bestMove: bestMove,
    bestMoves: bestMoves,
    randomMove: randomMove,
    tacticalMove: tacticalMove,
    chooseMove: chooseMove,
    createGame: createGame,
  };

  root.UnicornEngine = UnicornEngine;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnicornEngine;
  }
})(typeof window !== 'undefined' ? window : this);
