const bridge = document.getElementById('bridge');
const bridgeContainer = document.getElementById('bridge-container');
const instructions = document.getElementById('instructions');
const startButton = document.getElementById('start-button');
const status = document.getElementById('status');
const choiceArea = document.getElementById('choice-area');
const choiceButtons = document.querySelectorAll('.choice-btn');
const currentStepDisplay = document.getElementById('current-step');
const totalStepsDisplay = document.getElementById('total-steps');

const TOTAL_STEPS = 18;
let gameRunning = false;
let currentStep = 0;
let safePanels = [];

function generateBridge() {
    bridge.innerHTML = '';
    safePanels = [];

    for (let i = TOTAL_STEPS - 1; i >= 0; i--) {
        const row = document.createElement('div');
        row.className = 'glass-row';
        row.dataset.step = i;

        const safeIsLeft = Math.random() < 0.5;
        safePanels[i] = safeIsLeft ? 'left' : 'right';

        const leftPanel = document.createElement('div');
        leftPanel.className = 'glass-panel disabled';
        leftPanel.dataset.side = 'left';
        leftPanel.dataset.step = i;

        const rightPanel = document.createElement('div');
        rightPanel.className = 'glass-panel disabled';
        rightPanel.dataset.side = 'right';
        rightPanel.dataset.step = i;

        row.appendChild(leftPanel);
        row.appendChild(rightPanel);
        bridge.appendChild(row);
    }

    highlightCurrentStep();
}

function highlightCurrentStep() {
    document.querySelectorAll('.glass-panel').forEach(panel => {
        panel.classList.remove('current');
        panel.classList.add('disabled');
    });

    if (currentStep < TOTAL_STEPS) {
        const currentPanels = document.querySelectorAll(`.glass-panel[data-step="${currentStep}"]`);
        currentPanels.forEach(panel => {
            panel.classList.add('current');
            panel.classList.remove('disabled');
        });
    }
}

function makeChoice(choice) {
    if (!gameRunning || currentStep >= TOTAL_STEPS) return;

    choiceButtons.forEach(btn => btn.disabled = true);

    const panels = document.querySelectorAll(`.glass-panel[data-step="${currentStep}"]`);
    const chosenPanel = document.querySelector(`.glass-panel[data-step="${currentStep}"][data-side="${choice}"]`);
    const otherPanel = document.querySelector(`.glass-panel[data-step="${currentStep}"][data-side="${choice === 'left' ? 'right' : 'left'}"]`);

    const isSafe = safePanels[currentStep] === choice;

    if (isSafe) {
        chosenPanel.classList.add('safe');
        chosenPanel.classList.remove('current');
        currentStep++;
        currentStepDisplay.textContent = currentStep;

        if (currentStep >= TOTAL_STEPS) {
            endGame(true);
        } else {
            status.innerHTML = '<span style="color: #2ecc71">Safe! Keep going!</span>';
            setTimeout(() => {
                highlightCurrentStep();
                choiceButtons.forEach(btn => btn.disabled = false);
                scrollToCurrentStep();
            }, 500);
        }
    } else {
        chosenPanel.classList.add('broken');
        chosenPanel.classList.remove('current');
        otherPanel.classList.add('safe');
        endGame(false);
    }
}

function scrollToCurrentStep() {
    const currentRow = document.querySelector(`.glass-row[data-step="${currentStep}"]`);
    if (currentRow) {
        currentRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function startGame() {
    instructions.style.display = 'none';
    bridgeContainer.style.display = 'flex';
    choiceArea.style.display = 'flex';
    currentStep = 0;
    currentStepDisplay.textContent = '0';
    totalStepsDisplay.textContent = TOTAL_STEPS;
    gameRunning = true;
    status.textContent = 'Choose: Left or Right?';

    generateBridge();
    choiceButtons.forEach(btn => btn.disabled = false);

    setTimeout(() => {
        const startPlatform = document.getElementById('start-platform');
        startPlatform.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
}

function endGame(win) {
    gameRunning = false;
    choiceButtons.forEach(btn => btn.disabled = true);

    if (win) {
        status.innerHTML = '<span style="color: #2ecc71; font-size: 24px">You Win!</span><br>You crossed the Glass Bridge!';
    } else {
        status.innerHTML = '<span style="color: #e74c3c; font-size: 24px">You Lose!</span><br>The glass shattered!';
    }

    setTimeout(() => {
        instructions.style.display = 'block';
        bridgeContainer.style.display = 'none';
        choiceArea.style.display = 'none';
        startButton.textContent = 'Play Again';
    }, 2500);
}

startButton.addEventListener('click', startGame);

choiceButtons.forEach(btn => {
    btn.addEventListener('click', () => makeChoice(btn.dataset.choice));
});

document.querySelectorAll('.glass-panel').forEach(panel => {
    panel.addEventListener('click', () => {
        if (!panel.classList.contains('disabled') && gameRunning) {
            makeChoice(panel.dataset.side);
        }
    });
});

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('glass-panel') && !e.target.classList.contains('disabled') && gameRunning) {
        makeChoice(e.target.dataset.side);
    }
});
