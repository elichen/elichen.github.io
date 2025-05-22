console.log('app.js: Script execution started.');

// Global instance for the Magenta.js player and MusicVAE
let player;
let music_vae_instance;
let toneStarted = false;
let tf;

let currentBPM = 120;
let currentVAEStepDuration = (60 / currentBPM) / 4;

// Global variables for loop state
let loopIsEnabled = false;
let sequenceToLoop = null;
let titleToLoop = "";

const playerCallback = {
  run: (note) => {
    const cell = document.querySelector(`.grid-cell[data-pitch='${note.pitch}'][data-time='${note.quantizedStartStep}']`);
    if (cell) {
      cell.classList.add('playing');
      const stepDuration = currentVAEStepDuration || (60 / 120) / 4; // Fallback, should be currentVAEStepDuration
      const durationMs = (note.quantizedEndStep - note.quantizedStartStep) * stepDuration * 1000;
      setTimeout(() => {
        cell.classList.remove('playing');
      }, durationMs);
    }
  },
  stop: () => {
    document.querySelectorAll('.grid-cell.playing').forEach(cell => cell.classList.remove('playing'));
    // This callback is triggered when player.stop() is called or a sequence ends.
    // Loop logic is handled in playSequence's promise.
  }
};

function initializePlayer() {
    if (player) return;
    try {
        if (typeof mm !== 'undefined' && mm.Player) {
            player = new mm.Player(false, playerCallback);
            console.log("Magenta.js Player initialized successfully using mm.Player.");
        } else {
            console.error("mm.Player not found.");
            alert("Error: Music Player component (mm.Player) not found.");
        }
    } catch (e) {
        console.error("Error initializing Magenta.js Player (mm.Player):", e);
        alert("Error initializing music player.");
    }
}

async function ensureToneStarted() {
    if (!toneStarted) {
        try {
            if (typeof mm !== 'undefined' && mm.Player && mm.Player.tone && typeof mm.Player.tone.start === 'function') {
                 if (mm.Player.tone.context && mm.Player.tone.context.state !== 'running') {
                    await mm.Player.tone.start();
                    toneStarted = true;
                    console.log('AudioContext started via mm.Player.tone.start()');
                 } else if (mm.Player.tone.context && mm.Player.tone.context.state === 'running') {
                    toneStarted = true; // Already running
                 }
            } else if (typeof Tone !== 'undefined' && typeof Tone.start === 'function') {
                 if (Tone.context && Tone.context.state !== 'running') {
                    await Tone.start();
                    toneStarted = true;
                    console.log('AudioContext started via global Tone.start()');
                 } else if (Tone.context && Tone.context.state === 'running') {
                    toneStarted = true; // Already running
                 }
            } else {
                console.log("No explicit Tone.start() found. Player will handle AudioContext on first play.");
            }
        } catch (e) {
            console.error("Error trying to start Tone.js AudioContext:", e);
        }
    }
}

const testSequence = {
  notes: [ { pitch: 60, startTime: 0.0, endTime: 0.5, velocity: 80, quantizedStartStep: 0, quantizedEndStep: 4 } ],
  totalTime: 0.5,
  quantizationInfo: {stepsPerQuarter: 4},
  tempos: [{ time: 0, qpm: 120 }] // Add default tempo
};

function initializePlayTestNoteButton() {
    const btn = document.getElementById('play-test-note-btn');
    if (btn) {
      btn.addEventListener('click', async () => {
        await ensureToneStarted();
        if (!player) initializePlayer();
        if (!player) { console.error("Player not init for test note."); alert("Player not init."); return; }
        playTestNote();
      });
    } else { console.error("Play Test Note button not found."); }
}

function playTestNote() {
    if (!player) { console.error("Player not available for test note."); return; }

    // Explicitly stop any ongoing loop and the player before playing test note
    loopIsEnabled = false;
    sequenceToLoop = null;
    titleToLoop = "";
    if (player.isPlaying()) {
        player.stop();
    }
    // Ensure testSequence has the currentBPM if you want it to match UI tempo
    const currentTestSequence = mm.sequences.clone(testSequence);
    currentTestSequence.tempos = [{ time: 0, qpm: currentBPM }];


    try {
        console.log("Playing test sequence with mm.Player:", currentTestSequence);
        player.start(currentTestSequence)
            .then(() => console.log("Test playback finished."))
            .catch(e => { console.error("Error during test playback:", e); alert("Error playing test sound."); });
    } catch (e) {
        console.error("Error calling player.start() for test sequence:", e); alert("Error initiating test playback.");
    }
}

window.addEventListener('load', async () => {
    console.log('app.js: window.load event fired.');

    if (typeof mm === 'undefined') {
        console.error("Magenta.js UMD bundle NOT found.");
        alert("Critical Error: Magenta.js library (mm) not loaded.");
        document.getElementById('generate-variation-btn').disabled = true;
        return;
    }
    console.log("Magenta.js UMD bundle found:", mm);

    if (typeof mm.tf !== 'undefined') {
        tf = mm.tf;
        console.log('app.js: TensorFlow.js (mm.tf) object FOUND.');
    } else if (typeof window.tf !== 'undefined') {
        tf = window.tf;
        console.log('app.js: Global TensorFlow.js (tf) object FOUND.');
    } else {
        console.error('app.js: TensorFlow.js (tf or mm.tf) IS UNDEFINED.');
        alert("Critical Error: TensorFlow.js is not available.");
        return;
    }

    if (tf && tf.version && tf.version.tfjs) {
        console.log('TensorFlow.js version being used:', tf.version.tfjs);
        try {
            tf.tensor([1, 2, 3, 4]).print();
            console.log('app.js: TensorFlow.js basic test successful.');
        } catch (e) {
            console.error('app.js: Error during TensorFlow.js test:', e);
        }
    } else {
         console.error('app.js: TensorFlow.js object is invalid.');
         alert("Critical Error: TensorFlow.js object is invalid.");
         return;
    }

    initializePlayer();
    initializeGrid();
    await initializeMusicVAE();
    initializeTempoControls();
    initializeClearGridButton();
    initializePlayTestNoteButton();
    initializePlayUserLoopButton();
    initializeGenerateVariationButton();
    console.log('app.js: All initializations complete.');
});


function initializeTempoControls() {
    const slider = document.getElementById('tempo-slider');
    const display = document.getElementById('tempo-value');
    if (slider && display) {
        slider.addEventListener('input', () => {
            currentBPM = parseInt(slider.value, 10);
            display.textContent = currentBPM;
            currentVAEStepDuration = (60 / currentBPM) / 4;
            console.log(`Tempo: ${currentBPM} BPM. VAE Step Duration: ${currentVAEStepDuration}s`);

            // If a loop is active and tempo changes, restart the loop with new tempo
            // This is optional; without it, the loop continues at old tempo until next natural loop point.
            if (loopIsEnabled && sequenceToLoop && player.isPlaying()) {
                console.log("Tempo changed during active loop. Restarting loop with new tempo.");
                // Update the tempo of the sequenceToLoop
                const newTempoSequence = mm.sequences.clone(sequenceToLoop);
                newTempoSequence.tempos = [{ time: 0, qpm: currentBPM }];
                playSequence(newTempoSequence, titleToLoop, true);
            }
        });
    } else { console.error("Tempo controls not found."); }
}

function initializeClearGridButton() {
    const btn = document.getElementById('clear-grid-btn');
    if (btn) {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.grid-cell.active').forEach(cell => cell.classList.remove('active'));
            console.log("Grid cleared.");

            // Stop playback and disable looping if grid is cleared
            if (player && player.isPlaying()) {
                loopIsEnabled = false;
                sequenceToLoop = null;
                titleToLoop = "";
                player.stop();
                console.log("Playback stopped due to grid clear.");
            }
        });
    } else { console.error("Clear Grid button not found."); }
}

const VAE_CHECKPOINT_URL = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2';

async function initializeMusicVAE() {
  const generateBtn = document.getElementById('generate-variation-btn');
  try {
    if (typeof mm !== 'undefined' && mm.MusicVAE) {
        music_vae_instance = new mm.MusicVAE(VAE_CHECKPOINT_URL);
    } else {
        console.error("mm.MusicVAE constructor not found.");
        if (generateBtn) generateBtn.disabled = true; return;
    }
    await music_vae_instance.initialize();
    console.log('MusicVAE (mm.MusicVAE) initialized successfully.');
    if (generateBtn) generateBtn.disabled = false;
  } catch (error) {
    console.error('Failed to initialize MusicVAE (mm.MusicVAE):', error);
    if (generateBtn) generateBtn.disabled = true;
  }
}

function initializeGrid() {
    const cells = document.querySelectorAll('.grid-cell');
    cells.forEach(cell => cell.addEventListener('click', () => cell.classList.toggle('active')));
    currentVAEStepDuration = (60 / currentBPM) / 4;
    console.log(`Initialized ${cells.length} grid cells. Initial step duration: ${currentVAEStepDuration}s.`);
}

function gridToNoteSequence() {
    const notes = [];
    const activeCells = document.querySelectorAll('.grid-cell.active');
    let maxEndTime = 0;
    const stepsPerQuarter = 4;

    activeCells.forEach(cell => {
        const pitch = parseInt(cell.dataset.pitch, 10);
        const timeStep = parseInt(cell.dataset.time, 10);
        const startTime = timeStep * currentVAEStepDuration;
        const endTime = startTime + currentVAEStepDuration;
        notes.push({ pitch, startTime, endTime, quantizedStartStep: timeStep, quantizedEndStep: timeStep + 1, velocity: 80 });
        if (endTime > maxEndTime) maxEndTime = endTime;
    });

    if (notes.length === 0) return null;

    return {
        notes: notes,
        totalTime: maxEndTime,
        quantizationInfo: { stepsPerQuarter: stepsPerQuarter },
        tempos: [{ time: 0, qpm: currentBPM }]
    };
}

function initializePlayUserLoopButton() {
    const btn = document.getElementById('play-user-loop-btn');
    if (btn) {
        btn.addEventListener('click', async () => {
            await ensureToneStarted();
            if (!player) initializePlayer();
            if (!player) { console.error("Player not init for user loop."); alert("Player not init."); return; }
            const userSeq = gridToNoteSequence();
            if (!userSeq || userSeq.notes.length === 0) { alert("Select notes on the grid to play your loop."); return; }
            playSequence(userSeq, "User Loop", true); // Play with looping enabled
        });
    } else { console.error("Play User Loop button not found."); }
}

function playSequence(sequenceToPlay, title = "Sequence", shouldLoop = false) {
    if (!player) {
        console.error(`Player not init. Cannot play ${title}.`);
        alert("Player not initialized.");
        return;
    }

    if (player.isPlaying()) {
        player.stop(); // Stop current playback. This also cancels any pending loop from a previous call.
    }

    loopIsEnabled = shouldLoop; // Set looping state for the *new* sequence
    if (shouldLoop) {
        sequenceToLoop = mm.sequences.clone(sequenceToPlay); // Store a clone for looping
        titleToLoop = title;
    } else {
        sequenceToLoop = null;
        titleToLoop = "";
    }

    // Ensure the sequence has up-to-date tempo information
    if (!sequenceToPlay.tempos || sequenceToPlay.tempos.length === 0 || sequenceToPlay.tempos[0].qpm !== currentBPM) {
        console.log(`Updating tempo for "${title}" to ${currentBPM} BPM.`);
        sequenceToPlay.tempos = [{ time: 0, qpm: currentBPM }];
    }
    if (!sequenceToPlay.quantizationInfo || !sequenceToPlay.quantizationInfo.stepsPerQuarter) {
        sequenceToPlay.quantizationInfo = { ...sequenceToPlay.quantizationInfo, stepsPerQuarter: 4 };
    }

    console.log(`Playing ${title}${shouldLoop ? ' (looping enabled)' : ''}:`, JSON.parse(JSON.stringify(sequenceToPlay)));

    player.start(sequenceToPlay)
        .then(() => {
            console.log(`${title} playback finished.`);
            // Check if looping is still enabled for *this specific sequence type* and player is stopped
            if (loopIsEnabled && sequenceToLoop && player.getPlayState() === 'stopped') {
                console.log(`Looping ${titleToLoop}...`);
                // Use a timeout to prevent potential call stack issues and give a brief pause
                setTimeout(() => {
                    // Re-check loopIsEnabled as another action might have changed it during the timeout
                    if (loopIsEnabled && sequenceToLoop) {
                        // Ensure the sequenceToLoop has the most current BPM for the next iteration
                        const loopNextIter = mm.sequences.clone(sequenceToLoop);
                        loopNextIter.tempos = [{ time: 0, qpm: currentBPM }];
                        playSequence(loopNextIter, titleToLoop, true);
                    }
                }, 100); // 100ms delay, adjust as needed
            }
        })
        .catch(e => {
            console.error(`Error during ${title} playback:`, e);
            alert(`Error playing ${title}.`);
            loopIsEnabled = false; // Disable looping on error
            sequenceToLoop = null;
            titleToLoop = "";
        });
}

function initializeGenerateVariationButton() {
    const btn = document.getElementById('generate-variation-btn');
    if (btn) {
        btn.addEventListener('click', async () => {
            await ensureToneStarted();
            if (!player) initializePlayer();
            if (!music_vae_instance || !music_vae_instance.isInitialized()) {
                alert("AI model is not ready."); return;
            }
            let userSeq = gridToNoteSequence();
            if (!userSeq || userSeq.notes.length === 0) {
                alert("Create a melody to generate a variation."); return;
            }

            if (typeof mm === 'undefined' || !mm.sequences || typeof mm.sequences.clone !== 'function' || typeof mm.sequences.quantizeNoteSequence !== 'function') {
                console.error("Magenta.js sequence utility functions not available!");
                alert("Sequence processing function not loaded."); return;
            }

            const STEPS_PER_BAR = 16;
            const STEPS_PER_QUARTER_FOR_VAE = 4;

            if (!userSeq.quantizationInfo) {
                userSeq.quantizationInfo = { stepsPerQuarter: STEPS_PER_QUARTER_FOR_VAE };
            }
            if (!userSeq.tempos) {
                userSeq.tempos = [{ time: 0, qpm: currentBPM }];
            }

            let inputSequenceForVAE = mm.sequences.quantizeNoteSequence(
                mm.sequences.clone(userSeq),
                STEPS_PER_QUARTER_FOR_VAE
            );
            inputSequenceForVAE.totalQuantizedSteps = STEPS_PER_BAR;

            console.log("Prepared quantized input for VAE:", JSON.parse(JSON.stringify(inputSequenceForVAE)));

            try {
                const genSequences = await music_vae_instance.sample(1, 0.7, null, STEPS_PER_QUARTER_FOR_VAE, inputSequenceForVAE);
                if (genSequences && genSequences.length > 0) {
                    let generatedSeq = genSequences[0];
                    generatedSeq.tempos = [{ time: 0, qpm: currentBPM }];
                    if (!generatedSeq.quantizationInfo || generatedSeq.quantizationInfo.stepsPerQuarter !== STEPS_PER_QUARTER_FOR_VAE) {
                        generatedSeq.quantizationInfo = { stepsPerQuarter: STEPS_PER_QUARTER_FOR_VAE };
                    }
                    console.log("Sanitized sequence for player:", JSON.parse(JSON.stringify(generatedSeq)));
                    playSequence(generatedSeq, "Generated Variation", true); // Play with looping enabled
                } else {
                    alert("AI could not generate a variation.");
                }
            } catch (error) {
                console.error("Error during MusicVAE sampling:", error);
                alert("Error generating music variation.");
            }
        });
    } else { console.error("Generate Variation button not found."); }
}