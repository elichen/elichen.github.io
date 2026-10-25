// One Atari game played the way BBF was trained and evaluated (Dopamine's AtariPreprocessing + BBF's runner):
// each agent step repeats the action for 4 frames and sums the reward; the observation is the max of the
// last two frames' grayscale, area-resized to 84x84; the agent sees the last 4 observations. Losing a life
// clears that stack (the agent was trained with life loss as a terminal). New games start with 0-29 no-ops.

import { makeObserver } from './preprocess.js';

const W = 160, H = 210, FRAME_SKIP = 4, MAX_NOOPS = 30, MAX_MOVES = 27000;  // BBF's cap: 30 minutes a game

export class Game {
  constructor(ALE, rom, { seed = 0, epsilon = 0.001, maxNoops = MAX_NOOPS, sound = false } = {}) {
    this.ale = new ALE.ALEInterface();
    this.ale.setInt('random_seed', seed);
    this.ale.setFloat('repeat_action_probability', 0.0);  // Atari 100k: no sticky actions
    this.ale.setInt('max_num_frames_per_episode', 100000);
    if (sound) this.ale.setBool('sound', true);  // played by SDL through WebAudio (see app.js)
    this.ale.loadROM(rom);
    this.actions = [...this.ale.getMinimalActionSet()];
    this.epsilon = epsilon;
    this.maxNoops = maxNoops;
    this.observe = makeObserver();
    this.screens = [new Uint8Array(W * H), new Uint8Array(W * H)];
    this.obs = new Uint8Array(84 * 84);
    this.state = new Uint8Array(84 * 84 * 4);  // [84, 84, 4], newest frame last
    this.rng = mulberry32(seed + 1);
    this.score = 0;
    this.reset();
  }

  // frees the emulator's wasm memory
  dispose() { this.ale.delete(); }


  grayscale(out) {
    out.set(this.ale.getScreenGrayscale());
  }

  // AtariPreprocessing.reset
  envReset() {
    this.ale.resetGame();
    this.lives = this.ale.lives();
    this.grayscale(this.screens[0]);
    this.screens[1].fill(0);
    return this.observe(this.screens[0], this.screens[1], this.obs);
  }

  // AtariPreprocessing.step: returns { reward, terminal } and leaves the observation in this.obs
  envStep(actionIndex) {
    const a = this.actions[actionIndex];
    let reward = 0, terminal = false, gameOver = false;
    for (let t = 0; t < FRAME_SKIP; t++) {
      reward += this.ale.act(a);
      this.onFrame?.();
      gameOver = this.ale.gameOver() || this.ale.gameTruncated();
      const lives = this.ale.lives();
      terminal = gameOver || lives < this.lives;
      this.lives = lives;
      if (t >= FRAME_SKIP - 2) this.grayscale(this.screens[t - (FRAME_SKIP - 2)]);
      if (terminal) break;
    }
    this.observe(this.screens[0], this.screens[1], this.obs);
    this.gameOver = gameOver;
    return { reward, terminal };
  }

  push(obs) {
    const s = this.state;
    for (let i = 0; i < 84 * 84; i++) {
      const j = i * 4;
      s[j] = s[j + 1]; s[j + 1] = s[j + 2]; s[j + 2] = s[j + 3]; s[j + 3] = obs[i];
    }
  }

  // BBF's _initialize_episode: reset, then a random number of no-ops
  reset() {
    this.envReset();
    const noops = Math.floor(this.rng() * this.maxNoops);
    for (let i = 0; i < noops; i++) {
      if (this.envStep(0).terminal) this.envReset();
    }
    this.state.fill(0);
    this.push(this.obs);
    this.score = 0;
    this.moves = 0;
    this.gameOver = false;
  }

  // Epsilon-greedy on the policy's Q values (eval epsilon 0.001), as BBF evaluates.
  chooseAction(q) {
    if (this.rng() <= this.epsilon) return Math.floor(this.rng() * this.actions.length);
    let best = 0;
    for (let a = 1; a < q.length; a++) if (q[a] > q[best]) best = a;
    return best;
  }

  // One agent step with an already chosen action. Starts a new game after game over.
  step(actionIndex) {
    const { reward, terminal } = this.envStep(actionIndex);
    this.score += reward;
    this.moves++;
    const ended = this.gameOver || this.moves >= MAX_MOVES;
    if (ended) {
      this.lastScore = this.score;
      this.reset();
    } else {
      if (terminal) this.state.fill(0);
      this.push(this.obs);
    }
    return { reward, terminal, ended };
  }
}

function mulberry32(a) {
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
