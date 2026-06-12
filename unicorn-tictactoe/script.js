// Rainbow Unicorn Tic-Tac-Toe
// You are the unicorn 🦄, your friend (or the Rainbow computer) is 🌈.

const UNICORN = '🦄';
const RAINBOW = '🌈';
const RAINBOW_COLORS = ['#ff6b6b', '#ffa94d', '#ffe066', '#69db7c', '#74c0fc', '#b197fc', '#ff9ff3'];
const WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
];

let board = Array(9).fill(null);
let turn = UNICORN;
let gameOver = false;
let vsComputer = true;

const cells = [...document.querySelectorAll('.cell')];
const statusEl = document.getElementById('status');
const modeBtn = document.getElementById('mode-btn');
const restartBtn = document.getElementById('restart-btn');

// --- Dancing rainbow title ---

const titleEl = document.getElementById('title');
const titleText = '🦄 Rainbow Tic-Tac-Toe 🌈';
[...titleText].forEach((ch, i) => {
    const span = document.createElement('span');
    span.textContent = ch === ' ' ? ' ' : ch;
    span.style.color = RAINBOW_COLORS[i % RAINBOW_COLORS.length];
    span.style.animationDelay = `${(i * 0.06).toFixed(2)}s`;
    titleEl.appendChild(span);
});

// --- Happy sounds (WebAudio, no files needed) ---

let audioCtx = null;

function beep(freq, duration, delay = 0) {
    if (!audioCtx) return;
    const t = audioCtx.currentTime + delay;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'triangle';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.15, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t);
    osc.stop(t + duration);
}

function placeSound() {
    beep(660, 0.12);
    beep(880, 0.12, 0.08);
}

function winSound() {
    [523, 659, 784, 1047, 1319].forEach((f, i) => beep(f, 0.2, i * 0.09));
}

function tieSound() {
    beep(440, 0.2);
    beep(392, 0.3, 0.18);
}

// --- Game logic ---

function findWin(b) {
    for (const line of WIN_LINES) {
        const [a, c, d] = line;
        if (b[a] && b[a] === b[c] && b[a] === b[d]) return line;
    }
    return null;
}

function setStatus(text) {
    statusEl.textContent = text;
}

function place(i, who) {
    board[i] = who;
    const piece = document.createElement('span');
    piece.className = 'piece';
    piece.textContent = who;
    cells[i].appendChild(piece);
    cells[i].classList.add('taken');
    placeSound();
}

function endGame(winLine) {
    gameOver = true;
    if (winLine) {
        const winner = board[winLine[0]];
        winLine.forEach(i => cells[i].classList.add('winner'));
        if (winner === UNICORN) {
            setStatus('🦄 The unicorn wins! Yay! 🎉');
        } else if (vsComputer) {
            setStatus('🌈 Rainbow wins! Try again! 💜');
        } else {
            setStatus('🌈 The rainbow wins! Yay! 🎉');
        }
        winSound();
        confettiBurst();
    } else {
        setStatus('Everybody wins! It\'s a tie! 🦄💜🌈');
        tieSound();
    }
}

function afterMove() {
    const winLine = findWin(board);
    if (winLine || board.every(Boolean)) {
        endGame(winLine);
        return true;
    }
    return false;
}

function computerMove() {
    // Win if possible, block if needed, otherwise pick a random sparkle spot.
    const empty = board.map((v, i) => (v ? null : i)).filter(v => v !== null);
    let choice = null;
    for (const who of [RAINBOW, UNICORN]) {
        for (const i of empty) {
            const copy = [...board];
            copy[i] = who;
            if (findWin(copy)) { choice = i; break; }
        }
        if (choice !== null) break;
    }
    if (choice === null) choice = empty[Math.floor(Math.random() * empty.length)];
    place(choice, RAINBOW);
    if (!afterMove()) {
        turn = UNICORN;
        setStatus('Your turn, little unicorn! 🦄');
    }
}

cells.forEach(cell => {
    cell.addEventListener('click', () => {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const i = Number(cell.dataset.i);
        if (gameOver || board[i]) return;
        if (vsComputer && turn !== UNICORN) return;

        place(i, turn);
        if (afterMove()) return;

        if (vsComputer) {
            turn = RAINBOW;
            setStatus('Rainbow is thinking... 🌈✨');
            setTimeout(computerMove, 450);
        } else {
            turn = turn === UNICORN ? RAINBOW : UNICORN;
            setStatus(turn === UNICORN ? 'Unicorn\'s turn! 🦄' : 'Rainbow\'s turn! 🌈');
        }
    });
});

function restart() {
    board = Array(9).fill(null);
    turn = UNICORN;
    gameOver = false;
    cells.forEach(c => {
        c.textContent = '';
        c.classList.remove('taken', 'winner');
    });
    setStatus(vsComputer ? 'Your turn, little unicorn! 🦄' : 'Unicorn goes first! 🦄');
}

restartBtn.addEventListener('click', restart);

modeBtn.addEventListener('click', () => {
    vsComputer = !vsComputer;
    modeBtn.textContent = vsComputer
        ? 'Playing: You 🦄 vs Rainbow 🌈'
        : 'Playing: 2 Friends 🦄🌈';
    restart();
});

// --- Fast flying unicorns and rainbows in the sky ---

const sky = document.getElementById('sky');
const FLYER_EMOJI = ['🦄', '🌈', '⭐', '💜', '✨', '🦄', '🌈'];
const flyers = [];

for (let i = 0; i < 14; i++) {
    const el = document.createElement('div');
    el.className = 'flyer';
    el.textContent = FLYER_EMOJI[i % FLYER_EMOJI.length];
    el.style.fontSize = `${24 + Math.random() * 40}px`;
    sky.appendChild(el);
    flyers.push({
        el,
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        // Fast! The kid asked for very fast.
        vx: (Math.random() < 0.5 ? -1 : 1) * (250 + Math.random() * 350),
        vy: (Math.random() - 0.5) * 250,
        spin: (Math.random() - 0.5) * 720,
        angle: 0,
    });
}

let lastTime = performance.now();

function flyLoop(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    const w = window.innerWidth;
    const h = window.innerHeight;

    for (const f of flyers) {
        f.x += f.vx * dt;
        f.y += f.vy * dt;
        f.angle += f.spin * dt;
        if (f.x < -80) f.x = w + 80;
        if (f.x > w + 80) f.x = -80;
        if (f.y < -80) f.y = h + 80;
        if (f.y > h + 80) f.y = -80;
        f.el.style.transform =
            `translate(${f.x}px, ${f.y}px) rotate(${f.angle}deg) scaleX(${f.vx < 0 ? -1 : 1})`;
    }
    requestAnimationFrame(flyLoop);
}

requestAnimationFrame(flyLoop);

// --- Confetti when somebody wins ---

function confettiBurst() {
    const emoji = ['🦄', '🌈', '⭐', '💜', '✨', '🎉', '💖'];
    for (let i = 0; i < 60; i++) {
        const piece = document.createElement('div');
        piece.className = 'confetti';
        piece.textContent = emoji[Math.floor(Math.random() * emoji.length)];
        document.body.appendChild(piece);

        const startX = window.innerWidth / 2;
        const startY = window.innerHeight / 2;
        const angle = Math.random() * Math.PI * 2;
        const speed = 300 + Math.random() * 700;
        let x = startX, y = startY;
        let vx = Math.cos(angle) * speed;
        let vy = Math.sin(angle) * speed - 300;
        let rot = Math.random() * 360;
        const spin = (Math.random() - 0.5) * 1440;
        let prev = performance.now();

        function fall(now) {
            const dt = Math.min((now - prev) / 1000, 0.05);
            prev = now;
            vy += 1200 * dt;
            x += vx * dt;
            y += vy * dt;
            rot += spin * dt;
            piece.style.transform = `translate(${x}px, ${y}px) rotate(${rot}deg)`;
            if (y < window.innerHeight + 60) {
                requestAnimationFrame(fall);
            } else {
                piece.remove();
            }
        }
        requestAnimationFrame(fall);
    }
}
