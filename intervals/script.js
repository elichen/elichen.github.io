const INTERVALS = [
    'Perfect Unison', 'Minor Second', 'Major Second', 'Minor Third', 'Major Third',
    'Perfect Fourth', 'Tritone', 'Perfect Fifth', 'Minor Sixth', 'Major Sixth',
    'Minor Seventh', 'Major Seventh', 'Perfect Octave'
];

const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

let currentInterval;
let score = 0;
let gameMode = 'identify';

const staffContainer = document.getElementById('staff-container');
const pianoContainer = document.getElementById('piano-container');
const playIntervalButton = document.getElementById('play-interval');
const gameModeSelect = document.getElementById('game-mode');
const feedbackElement = document.getElementById('feedback');
const scoreElement = document.getElementById('score-value');
const intervalSelect = document.getElementById('interval-select');

const VF = Vex.Flow;
let renderer, context, stave;

let audioContext;

function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function initGame() {
    createPiano();
    setGameMode();
    initializeStaff();
    newRound();
}

function createPiano() {
    pianoContainer.innerHTML = ''; // Clear existing piano keys
    for (let i = 0; i < 13; i++) {
        const key = document.createElement('div');
        key.classList.add('piano-key');
        key.dataset.note = i;
        if ([1, 3, 6, 8, 10].includes(i % 12)) {
            key.classList.add('black');
        }
        key.addEventListener('click', () => handleNoteClick(i));
        pianoContainer.appendChild(key);
    }
}

function setGameMode() {
    gameMode = gameModeSelect.value;
    intervalSelect.style.display = 'none';
    staffContainer.style.display = 'block';
    
    if (gameMode === 'identify' || gameMode === 'ear-training') {
        intervalSelect.style.display = 'inline-block';
    }
    
    if (gameMode === 'ear-training') {
        staffContainer.style.display = 'none';
    }
    
    newRound();
}

function newRound() {
    currentInterval = Math.floor(Math.random() * INTERVALS.length);
    
    if (gameMode === 'identify') {
        displayInterval();
    } else if (gameMode === 'ear-training') {
        displayEarTrainingPrompt();
    } else if (gameMode === 'create') {
        displayCreatePrompt();
    }
    
    intervalSelect.value = ''; // Reset selection
}

function displayInterval() {
    staffContainer.innerHTML = '<div id="vexflow-staff"></div>';
    
    const { context: newContext, stave: newStave } = initializeStaff();
    context = newContext;
    stave = newStave;
    const startNote = 'c/5'; // Middle C (C4)
    const endNote = getNoteForInterval(startNote, currentInterval);
    drawNotes(startNote, endNote);
    
    // Update the instruction text instead of creating a new element
    const instructionText = document.getElementById('instruction-text');
    instructionText.textContent = 'Click "Play Interval" to hear the interval, then select your answer from the dropdown.';
}

function displayEarTrainingPrompt() {
    staffContainer.innerHTML = '';
    const promptElement = document.createElement('p');
    promptElement.textContent = 'Click "Play Interval" to hear the interval, then select your answer from the dropdown.';
    staffContainer.appendChild(promptElement);
}

function displayCreatePrompt() {
    const startNote = Math.floor(Math.random() * 12);
    staffContainer.innerHTML = `<p>Create a ${INTERVALS[currentInterval]} starting from ${NOTES[startNote]}</p>`;
}

function handleNoteClick(note) {
    if (gameMode === 'create') {
        checkCreatedInterval(note);
    } else if (gameMode === 'identify') {
        checkIdentifiedInterval();
    }
}

function checkCreatedInterval(endNote) {
    const startNote = parseInt(staffContainer.querySelector('p').textContent.split(' ').pop());
    const createdInterval = (endNote - startNote + 12) % 12;
    
    if (createdInterval === currentInterval) {
        updateScore(true);
        feedbackElement.textContent = 'Correct!';
    } else {
        updateScore(false);
        feedbackElement.textContent = 'Incorrect. Try again!';
    }
    
    setTimeout(newRound, 1500);
}

function checkIdentifiedInterval() {
    const selectedInterval = parseInt(intervalSelect.value);
    if (selectedInterval === currentInterval) {
        updateScore(true);
        feedbackElement.textContent = 'Correct!';
    } else {
        updateScore(false);
        feedbackElement.textContent = `Incorrect. The correct interval was ${INTERVALS[currentInterval]}.`;
    }
    setTimeout(newRound, 1500);
}

function playInterval() {
    initAudioContext();
    const synth = new Tone.Synth().toDestination();
    const now = Tone.now();
    const startNote = 'C4';
    const endNote = Tone.Frequency(startNote).transpose(currentInterval).toNote();
    
    synth.triggerAttackRelease(startNote, '0.5s', now);
    synth.triggerAttackRelease(endNote, '0.5s', now + 0.75);
    
    if (gameMode === 'ear-training') {
        feedbackElement.textContent = 'Listen carefully and select your answer.';
    }
}

function updateScore(isCorrect) {
    score += isCorrect ? 10 : -5;
    scoreElement.textContent = score;
}

function initializeStaff() {
    const container = document.getElementById("staff-container");
    const div = document.getElementById("vexflow-staff");
    
    // Clear previous content
    div.innerHTML = '';
    
    // Create renderer with full container width
    renderer = new VF.Renderer(div, VF.Renderer.Backends.SVG);
    renderer.resize(container.clientWidth, 140);
    
    context = renderer.getContext();
    context.setFont("Arial", 10, "").setBackgroundFillStyle("#eed");
    
    // Calculate center position
    const centerX = (container.clientWidth - 250) / 2;
    
    stave = new VF.Stave(centerX, 40, 250);
    stave.addClef("treble").addTimeSignature("4/4");
    stave.setContext(context).draw();
    
    return { context, stave };
}

function drawNotes(startNote, endNote) {
    if (!context || !stave) {
        console.error('VexFlow context or stave not initialized');
        return;
    }
    context.clear();
    stave.setContext(context).draw();

    const notes = [
        new VF.StaveNote({ clef: "treble", keys: [startNote], duration: "q" }),
        new VF.StaveNote({ clef: "treble", keys: [endNote], duration: "q" })
    ];

    const voice = new VF.Voice({ num_beats: 2, beat_value: 4 });
    voice.addTickables(notes);

    new VF.Formatter().joinVoices([voice]).format([voice], 200);
    voice.draw(context, stave);
}

function getNoteForInterval(startNote, interval) {
    const noteNames = ['c', 'd', 'e', 'f', 'g', 'a', 'b'];
    const startIndex = noteNames.indexOf(startNote[0]);
    const octave = parseInt(startNote[2]);
    
    let endIndex = (startIndex + interval) % 7;
    let endOctave = octave + Math.floor((startIndex + interval) / 7);
    
    // Ensure we stay within one octave above the starting note
    if (endOctave > octave + 1) {
        endOctave = octave + 1;
    }
    
    return `${noteNames[endIndex]}/${endOctave}`;
}

function populateIntervalSelect() {
    INTERVALS.forEach((interval, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = interval;
        intervalSelect.appendChild(option);
    });
}

gameModeSelect.addEventListener('change', setGameMode);
playIntervalButton.addEventListener('click', playInterval);

// Add event listeners to initialize audio context on user interaction
document.addEventListener('click', initAudioContext);
document.addEventListener('keydown', initAudioContext);

intervalSelect.addEventListener('change', checkIdentifiedInterval);

// Call this function when initializing the game
populateIntervalSelect();

initGame();