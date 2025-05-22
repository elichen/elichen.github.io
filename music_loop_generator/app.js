console.log('app.js: Script execution started.');

// Global instance for the Magenta.js player and MusicVAE
let player;
let music_vae_instance; // Renamed to avoid confusion if 'music_vae' is ever part of 'mm'
let toneStarted = false;
let tf; // Will hold the TensorFlow.js instance, expected from mm.tf or global

let currentBPM = 120;
let currentVAEStepDuration = (60 / currentBPM) / 4;

const playerCallback = {
  run: (note) => {
    const cell = document.querySelector(`.grid-cell[data-pitch='${note.pitch}'][data-time='${note.quantizedStartStep}']`);
    if (cell) {
      cell.classList.add('playing');
      const stepDuration = currentVAEStepDuration || (60 / 120) / 4;
      const durationMs = (note.quantizedEndStep - note.quantizedStartStep) * stepDuration * 1000;
      setTimeout(() => {
        cell.classList.remove('playing');
      }, durationMs);
    }
  },
  stop: () => {
    document.querySelectorAll('.grid-cell.playing').forEach(cell => cell.classList.remove('playing'));
  }
};

function initializePlayer() {
    if (player) return;
    try {
        // Use 'mm' as the global Magenta object, as per official docs
        if (typeof mm !== 'undefined' && mm.Player) {
            player = new mm.Player(false, playerCallback);
            console.log("Magenta.js Player initialized successfully using mm.Player.");
        } else {
            console.error("mm.Player not found. Magenta.js UMD bundle might not be loaded correctly or 'mm' global is not set.");
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
            // Tone.js is bundled, mm.Player usually handles starting it.
            // For explicit start or if other Tone features are used directly:
            // mm.Player.tone might be the Tone object, or it might be started implicitly.
            // The docs don't show an explicit mm.Tone.start().
            // Let's rely on the player to start it, or a user gesture triggering play.
            // If direct Tone access is needed, one might need to see how mm exposes Tone.
            if (typeof mm !== 'undefined' && mm.Player && mm.Player.tone && mm.Player.tone.start) { // Check if mm.Player.tone is a thing
                 await mm.Player.tone.start();
                 toneStarted = true;
                 console.log('AudioContext started via mm.Player.tone.start()');
            } else if (typeof Tone !== 'undefined' && Tone.start) { // Check global Tone as a fallback
                 await Tone.start();
                 toneStarted = true;
                 console.log('AudioContext started via global Tone.start()');
            }
             else {
                console.log("No explicit Tone.start() found on 'mm'. Player will handle AudioContext on first play.");
                // We can try to ensure it's started before any sound by creating a dummy player action
                // or assume the first button click that plays sound will handle it.
                // For now, let the first sound-playing action (like playTestNote) trigger it.
            }
        } catch (e) {
            console.error("Error trying to start Tone.js AudioContext:", e);
            // Don't alert here as it might be normal for it to start on first play.
        }
    }
}

const testSequence = {
  notes: [ { pitch: 60, startTime: 0.0, endTime: 0.5, velocity: 80, quantizedStartStep: 0, quantizedEndStep: 4 } ],
  totalTime: 0.5,
  quantizationInfo: {stepsPerQuarter: 4}
};

function initializePlayTestNoteButton() {
    const btn = document.getElementById('play-test-note-btn');
    if (btn) {
      btn.addEventListener('click', async () => {
        await ensureToneStarted(); // Try to start Tone explicitly
        if (!player) initializePlayer();
        if (!player) { console.error("Player not init for test note."); alert("Player not init."); return; }
        // Ensure Tone is really started before playing by interacting with the player
        if (!player.isPlaying() && player.getPlayState() === 'stopped' && !toneStarted) {
            try {
                // A bit of a hack: start and immediately stop a silent sequence to ensure Tone context is running
                // This is only if ensureToneStarted didn't set toneStarted = true.
                console.log("Attempting to ensure AudioContext is started by briefly starting player.");
                await mm.Player.tone.start(); // More direct if available
                toneStarted = true;
            } catch (e_tone) {
                 console.warn("Could not explicitly start Tone before test note, relying on player.start()", e_tone);
            }
        }
        playTestNote();
      });
    } else { console.error("Play Test Note button not found."); }
}

function playTestNote() {
    if (!player) { console.error("Player not available for test note."); return; }
    if (player.isPlaying()) player.stop();
    try {
        console.log("Playing test sequence with mm.Player:", testSequence);
        player.start(testSequence)
            .then(() => console.log("Test playback finished."))
            .catch(e => { console.error("Error during test playback:", e); alert("Error playing test sound."); });
    } catch (e) {
        console.error("Error calling player.start() for test sequence:", e); alert("Error initiating test playback.");
    }
}

window.addEventListener('load', async () => {
    console.log('app.js: window.load event fired.');

    console.log("app.js: Checking for Magenta.js UMD bundle (expects global 'mm')...");
    console.log("app.js: Value of window.mm:", window.mm);
    console.log("app.js: typeof window.mm:", typeof window.mm);

    if (typeof mm === 'undefined') {
        console.error("Magenta.js UMD bundle NOT found (global 'mm' is undefined).");
        alert("Critical Error: Magenta.js library (mm) not loaded. App cannot function.");
        const generateBtn = document.getElementById('generate-variation-btn');
        if(generateBtn) generateBtn.disabled = true;
        return;
    } else {
        console.log("Magenta.js UMD bundle found (global 'mm' object exists):", mm);

        // TensorFlow.js should be bundled and available via mm.tf or global tf
        if (typeof mm.tf !== 'undefined') {
            tf = mm.tf; // Prioritize mm.tf
            console.log('app.js: TensorFlow.js (mm.tf) object FOUND.');
        } else if (typeof window.tf !== 'undefined') {
            tf = window.tf; // Fallback to global tf if mm.tf is not there (less likely for UMD)
            console.log('app.js: Global TensorFlow.js (tf) object FOUND (used as fallback).');
        } else {
            console.error('app.js: TensorFlow.js (tf or mm.tf) IS UNDEFINED within Magenta.js bundle.');
            alert("Critical Error: TensorFlow.js is not available. App will fail.");
            return;
        }

        // Test tf instance
        if (tf && tf.version && tf.version.tfjs) {
            console.log('TensorFlow.js version being used:', tf.version.tfjs);
            try {
                tf.tensor([1, 2, 3, 4]).print();
                console.log('app.js: TensorFlow.js basic test successful.');
            } catch (e) {
                console.error('app.js: Error during TensorFlow.js test:', e);
                alert("TFJS test error.");
            }
        } else {
             console.error('app.js: TensorFlow.js object is invalid or version is missing.');
             alert("Critical Error: TensorFlow.js object is invalid.");
             return;
        }

        // Check other Magenta components under 'mm'
        if (typeof mm.Player !== 'function') {
            console.warn("'mm.Player' is NOT a function.");
        }
        if (typeof mm.MusicVAE !== 'function') {
            console.warn("'mm.MusicVAE' is NOT a function.");
        }
        if (typeof mm.sequences !== 'object' || typeof mm.sequences.clone !== 'function') {
            console.warn("'mm.sequences.clone' is NOT available.");
        }
    }

    console.log('app.js: All essential libraries appear loaded. Proceeding with initializations.');

    initializePlayer();
    initializeGrid();
    await initializeMusicVAE();
    initializeTempoControls();
    initializeClearGridButton();
    initializePlayTestNoteButton();
    initializePlayUserLoopButton();
    initializeGenerateVariationButton();
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
        });
    } else { console.error("Tempo controls not found."); }
}

function initializeClearGridButton() {
    const btn = document.getElementById('clear-grid-btn');
    if (btn) {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.grid-cell.active').forEach(cell => cell.classList.remove('active'));
            console.log("Grid cleared.");
        });
    } else { console.error("Clear Grid button not found."); }
}

const VAE_CHECKPOINT_URL = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2';

async function initializeMusicVAE() {
  try {
    if (typeof mm !== 'undefined' && mm.MusicVAE) {
        music_vae_instance = new mm.MusicVAE(VAE_CHECKPOINT_URL);
    } else {
        console.error("mm.MusicVAE constructor not found.");
        alert("VAE component (mm.MusicVAE) not found.");
        document.getElementById('generate-variation-btn').disabled = true; return;
    }
    await music_vae_instance.initialize();
    console.log('MusicVAE (mm.MusicVAE) initialized successfully.');
    document.getElementById('generate-variation-btn').disabled = false;
  } catch (error) {
    console.error('Failed to initialize MusicVAE (mm.MusicVAE):', error);
    alert('Failed to load AI model.');
    document.getElementById('generate-variation-btn').disabled = true;
  }
}

function initializeGrid() {
    const cells = document.querySelectorAll('.grid-cell');
    cells.forEach(cell => cell.addEventListener('click', () => cell.classList.toggle('active')));
    console.log(`Initialized ${cells.length} grid cells. Step duration: ${currentVAEStepDuration}s.`);
}

function gridToNoteSequence() {
    const notes = [];
    const activeCells = document.querySelectorAll('.grid-cell.active');
    let maxEndTime = 0;
    activeCells.forEach(cell => {
        const pitch = parseInt(cell.dataset.pitch, 10);
        const timeStep = parseInt(cell.dataset.time, 10);
        const startTime = timeStep * currentVAEStepDuration;
        const endTime = startTime + currentVAEStepDuration;
        notes.push({ pitch, startTime, endTime, quantizedStartStep: timeStep, quantizedEndStep: timeStep + 1, velocity: 80 });
        if (endTime > maxEndTime) maxEndTime = endTime;
    });
    if (notes.length === 0) return null;
    return { notes, totalTime: maxEndTime, quantizationInfo: { stepsPerQuarter: 4 } };
}

function initializePlayUserLoopButton() {
    const btn = document.getElementById('play-user-loop-btn');
    if (btn) {
        btn.addEventListener('click', async () => {
            await ensureToneStarted();
            if (!player) initializePlayer();
            if (!player) { console.error("Player not init for user loop."); alert("Player not init."); return; }
            const userSeq = gridToNoteSequence();
            if (!userSeq || userSeq.notes.length === 0) { alert("Select notes to play loop."); return; }
            playSequence(userSeq, "User Loop");
        });
    } else { console.error("Play User Loop button not found."); }
}

function playSequence(sequenceToPlay, title = "Sequence") {
    if (!player) { console.error(`Player not init. Cannot play ${title}.`); return; }
    if (player.isPlaying()) player.stop();
    try {
        player.start(sequenceToPlay)
            .then(() => console.log(`${title} playback finished.`))
            .catch(e => { console.error(`Error during ${title} playback:`, e); alert(`Error playing ${title}.`); });
    } catch (e) {
        console.error(`Error calling player.start() for ${title}:`, e); alert(`Error initiating ${title} playback.`);
    }
}

function initializeGenerateVariationButton() {
    const btn = document.getElementById('generate-variation-btn');
    if (btn) {
        btn.addEventListener('click', async () => {
            await ensureToneStarted();
            if (!player) initializePlayer();
            if (!music_vae_instance || !music_vae_instance.isInitialized()) {
                alert("AI model not ready."); return;
            }
            let userSeq = gridToNoteSequence();
            if (!userSeq || userSeq.notes.length === 0) {
                alert("Create a melody to generate a variation."); return;
            }

            if (typeof mm === 'undefined' || !mm.sequences || typeof mm.sequences.clone !== 'function') {
                console.error("mm.sequences.clone function not available!");
                alert("Sequence processing function (clone) not loaded."); return;
            }
            let inputSequenceForVAE = mm.sequences.clone(userSeq);

            if (inputSequenceForVAE.notes.length > 0 && inputSequenceForVAE.totalTime <= 8 * currentVAEStepDuration) {
                const originalNotes = mm.sequences.clone(inputSequenceForVAE).notes;
                const timeOffset = inputSequenceForVAE.totalTime > 0 ? inputSequenceForVAE.totalTime : 8 * currentVAEStepDuration;
                originalNotes.forEach(note => {
                    note.startTime += timeOffset;
                    note.endTime += timeOffset;
                    if (note.quantizedStartStep !== undefined) {
                        note.quantizedStartStep += 8;
                        note.quantizedEndStep += 8;
                    }
                });
                inputSequenceForVAE.notes.push(...originalNotes);
            }
            
            if (inputSequenceForVAE.notes.length > 0) {
                 inputSequenceForVAE.totalTime = 16 * currentVAEStepDuration;
            } else {
                alert("Cannot process an empty sequence."); return;
            }

            console.log("Prepared input for VAE (16 steps):", inputSequenceForVAE);

            try {
                const genSequences = await music_vae_instance.sample(1, 0.7, null, 4, inputSequenceForVAE);
                if (genSequences && genSequences.length > 0) {
                    playSequence(genSequences[0], "Generated Variation");
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