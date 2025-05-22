console.log('app.js: Script execution started.');

// Initialize Magenta.js Player
// Global instance for the Magenta.js player and MusicVAE
let player;
let music_vae;
let toneStarted = false; // To track if Tone.js AudioContext has been started

// Player callback for visual feedback
const playerCallback = {
  run: (note) => {
    const cell = document.querySelector(`.grid-cell[data-pitch='${note.pitch}'][data-time='${note.quantizedStartStep}']`);
    if (cell) {
      cell.classList.add('playing');
      const durationMs = (note.quantizedEndStep - note.quantizedStartStep) * currentVAEStepDuration * 1000;
      setTimeout(() => {
        cell.classList.remove('playing');
      }, durationMs);
    }
  },
  stop: () => {
    document.querySelectorAll('.grid-cell.playing').forEach(cell => cell.classList.remove('playing'));
  }
};

// Function to initialize the Magenta Player
function initializePlayer() {
    if (player) return; // Already initialized

    try {
        // When using the UMD bundle, Magenta components are under magenta.music
        if (magenta && magenta.music && magenta.music.Player) {
            player = new magenta.music.Player(false, playerCallback);
            console.log("Magenta.js Player initialized successfully using magenta.music.Player.");
        } else {
            console.error("magenta.music.Player not found. Magenta.js UMD bundle might not be loaded correctly.");
            alert("Error: Music Player component (magenta.music.Player) not found. Playback will not work.");
        }
    } catch (e) {
        console.error("Error initializing Magenta.js Player:", e);
        alert("Error initializing music player. Check console for details.");
    }
}


// Function to ensure Tone.js is started
async function ensureToneStarted() {
    if (!toneStarted) {
        try {
            // Tone may be available globally or under magenta.music.Tone
            if (typeof Tone !== 'undefined' && Tone.start) {
                await Tone.start();
            } else if (magenta && magenta.music && magenta.music.Tone && magenta.music.Tone.start) {
                await magenta.music.Tone.start();
            } else {
                console.warn("Tone.start() not found. Audio playback might not work on first interaction.");
            }
            toneStarted = true;
            console.log('AudioContext started via ensureToneStarted()');
        } catch (e) {
            console.error("Error starting Tone.js AudioContext:", e);
            alert("Could not start audio. Please interact with the page again or refresh.");
        }
    }
}


// Test NoteSequence (will not have visual feedback unless quantizedStartStep is added)
const testSequence = {
  notes: [
    // Add quantizedStartStep and quantizedEndStep for potential feedback
    { pitch: 60, startTime: 0.0, endTime: 0.5, velocity: 80, quantizedStartStep: 0, quantizedEndStep: 4 } 
  ],
  totalTime: 0.5,
  quantizationInfo: {stepsPerQuarter: 4} // Assuming 4 steps per quarter for test seq
};

// Event listener for the "Play Test Note" button
function initializePlayTestNoteButton() {
    const playTestNoteButton = document.getElementById('play-test-note-btn');
    if (playTestNoteButton) {
      playTestNoteButton.addEventListener('click', async () => {
        await ensureToneStarted();
        if (!player) {
            initializePlayer(); // Attempt to initialize if not already
            if(!player) {
                console.error("Player is not initialized for test note.");
                alert("Error: Music player is not initialized. Check console.");
                return;
            }
        }
        playTestNote();
      });
    } else {
        console.error("Play Test Note button not found.");
    }
}

function playTestNote() {
    if (!player) {
        console.error("Cannot play test note, player not available.");
        return;
    }
    if (player.isPlaying()) {
        console.log("Player is currently playing. Stopping before starting test sequence.");
        player.stop();
    }
    try {
        console.log("Playing test sequence:", testSequence);
        player.start(testSequence)
            .then(() => console.log("Test playback finished."))
            .catch(e => {
                console.error("Error during test playback:", e);
                alert("Error playing test sound. Check console.");
            });
    } catch (e) {
        console.error("Error calling player.start() for test sequence:", e);
        alert("Error initiating test playback. Check console.");
    }
}

// Load event: setup everything
window.addEventListener('load', async () => { // Make the load listener async
    console.log('app.js: window.load event fired.');

    // TFJS Check
    console.log('app.js: Checking for TensorFlow.js (tf) object...');
    if (typeof tf !== 'undefined' && typeof tf.version !== 'undefined' && tf.version.tfjs) { // More robust check
      console.log('app.js: TensorFlow.js (tf) IS defined.');
      try {
        console.log('app.js: Attempting to use TensorFlow.js...');
        console.log('TensorFlow.js version:', tf.version.tfjs);
        tf.tensor([1, 2, 3, 4]).print(); // tf.print() is not a function, .print() is a method on a tensor
        console.log('app.js: TensorFlow.js basic test successful.');
      } catch (e) {
        console.error('app.js: Error during TensorFlow.js test:', e);
        alert("app.js: Error during TensorFlow.js test. Check console. App might not work.");
      }
    } else {
      console.error('app.js: TensorFlow.js (tf) IS UNDEFINED or version is missing.');
      if (typeof tf === 'undefined') {
        console.error('Reason: tf object itself is undefined.');
      } else {
        console.error('Reason: tf object is defined, but tf.version or tf.version.tfjs is missing.');
      }
      alert("app.js: Critical Error: TensorFlow.js (tf) is not loaded correctly. App will likely fail.");
    }
    
    console.log('app.js: Music Loop Generator app loaded, proceeding with initializations.');
    // ... rest of the initializations as before

    // Verify Magenta UMD bundle
    if (typeof magenta !== 'undefined') {
        console.log("Magenta global object found:", magenta);
        if (magenta.music) {
            console.log("magenta.music namespace found:", magenta.music);
        } else {
            console.warn("magenta.music namespace NOT found. UMD bundle might be incomplete or problematic.");
        }
    } else {
        console.error("Magenta global object NOT found. UMD script might have failed to load or execute.");
        alert("Critical Error: Magenta.js library not loaded. The application cannot function. Please check your internet connection and refresh.");
        return; // Stop further initialization if Magenta is not available
    }

    initializePlayer(); // Initialize player on load after Magenta is confirmed
    initializeGrid();
    // initializeMusicVAE is async, ensure it's handled correctly if it needs to be awaited
    // or if subsequent initializations depend on it.
    await initializeMusicVAE(); // Initialize MusicVAE model on load
    initializeTempoControls();
    initializeClearGridButton();
    initializePlayTestNoteButton();
    initializePlayUserLoopButton();
    initializeGenerateVariationButton();
});

// Tempo and Duration
let currentBPM = 120;
let currentVAEStepDuration = (60 / currentBPM) / 4;


function initializeTempoControls() {
    const tempoSlider = document.getElementById('tempo-slider');
    const tempoValueDisplay = document.getElementById('tempo-value');
    if (tempoSlider && tempoValueDisplay) {
        tempoSlider.addEventListener('input', () => {
            currentBPM = parseInt(tempoSlider.value, 10);
            tempoValueDisplay.textContent = currentBPM;
            currentVAEStepDuration = (60 / currentBPM) / 4;
            console.log(`Tempo changed to ${currentBPM} BPM. VAE Step Duration: ${currentVAEStepDuration}s`);
        });
    } else {
        console.error("Tempo control elements not found.");
    }
}

function initializeClearGridButton() {
    const clearGridButton = document.getElementById('clear-grid-btn');
    if (clearGridButton) {
        clearGridButton.addEventListener('click', () => {
            document.querySelectorAll('.grid-cell.active').forEach(cell => cell.classList.remove('active'));
            console.log("Grid cleared.");
        });
    } else {
        console.error("Clear Grid button not found.");
    }
}

const VAE_CHECKPOINT_URL = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2';

async function initializeMusicVAE() {
  try {
    if (magenta && magenta.music && magenta.music.MusicVAE) {
        music_vae = new magenta.music.MusicVAE(VAE_CHECKPOINT_URL);
    } else {
        console.error("magenta.music.MusicVAE constructor not found. Ensure Magenta.js UMD bundle is loaded correctly.");
        alert("Error: Music VAE component (magenta.music.MusicVAE) not found. Generation will not work.");
        document.getElementById('generate-variation-btn').disabled = true;
        return;
    }
    await music_vae.initialize();
    console.log('MusicVAE initialized successfully.');
    document.getElementById('generate-variation-btn').disabled = false;
  } catch (error) {
    console.error('Failed to initialize MusicVAE:', error);
    alert('Failed to load AI model for music generation. Please refresh and try again.');
    document.getElementById('generate-variation-btn').disabled = true;
  }
}

function initializeGrid() {
    const cells = document.querySelectorAll('.grid-cell');
    cells.forEach(cell => {
        cell.addEventListener('click', () => cell.classList.toggle('active'));
    });
    console.log(`Initialized ${cells.length} grid cells. Each step duration for VAE: ${currentVAEStepDuration}s.`);
}

function gridToNoteSequence() {
    const notes = [];
    const activeCells = document.querySelectorAll('.grid-cell.active');
    let maxEndTime = 0;

    activeCells.forEach(cell => {
        const pitch = parseInt(cell.dataset.pitch, 10);
        const timeStep = parseInt(cell.dataset.time, 10); // This is 0-7

        // For VAE, startTime and endTime should be in terms of currentVAEStepDuration
        const startTime = timeStep * currentVAEStepDuration;
        const endTime = startTime + currentVAEStepDuration;

        notes.push({
            pitch: pitch,
            startTime: startTime,
            endTime: endTime,
            quantizedStartStep: timeStep, // Important for Magenta.js processing
            quantizedEndStep: timeStep + 1, // Important for Magenta.js processing
            velocity: 80
        });

        if (endTime > maxEndTime) {
            maxEndTime = endTime;
        }
    });

    if (notes.length === 0) {
        return null;
    }
    // The totalTime for the sequence from the grid will be 8 * currentVAEStepDuration
    return {
        notes: notes,
        totalTime: maxEndTime, 
        quantizationInfo: { stepsPerQuarter: 4 } 
    };
}

function initializePlayUserLoopButton() {
    const playUserLoopButton = document.getElementById('play-user-loop-btn');
    if (playUserLoopButton) {
        playUserLoopButton.addEventListener('click', async () => {
            await ensureToneStarted();
            if (!player) {
                initializePlayer();
                if(!player) {
                    console.error("Player is not initialized for user loop playback.");
                    alert("Error: Music player is not initialized. Check console.");
                    return;
                }
            }
            const userSequence = gridToNoteSequence();
            if (!userSequence || userSequence.notes.length === 0) {
                console.log("No notes selected in the grid.");
                alert("Please select some notes in the grid to play a loop.");
                return;
            }
            playSequence(userSequence, "User Loop");
        });
    } else {
        console.error("Play User Loop button not found.");
    }
}


function playSequence(sequence, title = "Sequence") {
    if (!player) {
        console.error(`Player not initialized. Cannot play ${title}.`);
        alert(`Player not initialized. Cannot play ${title}.`);
        return;
    }
    if (player.isPlaying()) {
        console.log(`Player is currently playing. Stopping before starting ${title}.`);
        player.stop();
    }
    try {
        console.log(`Playing ${title}:`, sequence);
        player.start(sequence)
            .then(() => console.log(`${title} playback finished.`))
            .catch(e => {
                console.error(`Error during ${title} playback:`, e);
                alert(`Error playing ${title}. Check console.`);
            });
    } catch (e) {
        console.error(`Error calling player.start() for ${title}:`, e);
        alert(`Error initiating ${title} playback. Check console.`);
    }
}

function initializeGenerateVariationButton() {
    const generateVariationButton = document.getElementById('generate-variation-btn');
    if (generateVariationButton) {
        generateVariationButton.addEventListener('click', async () => {
            await ensureToneStarted();
            if (!player) initializePlayer(); // Ensure player is ready for playback
            if (!music_vae || !music_vae.isInitialized()) {
                console.error("MusicVAE is not initialized for variation generation.");
                alert("AI Music Generation model is not ready. Please wait or try refreshing.");
                return;
            }

            let userSequence = gridToNoteSequence();
            if (!userSequence || userSequence.notes.length === 0) {
                alert("Please create a short melody in the grid first to generate a variation.");
                return;
            }

            // Use magenta.music.sequences for clone and potentially quantize if needed
            // The gridToNoteSequence should already provide a quantized sequence.
            let inputSequenceForVAE = magenta.music.sequences.clone(userSequence);

            if (inputSequenceForVAE.notes.length > 0 && inputSequenceForVAE.totalTime <= 8 * currentVAEStepDuration) {
                const originalNotes = JSON.parse(JSON.stringify(inputSequenceForVAE.notes));
                const timeOffset = inputSequenceForVAE.totalTime;
                originalNotes.forEach(note => {
                    note.startTime += timeOffset;
                    note.endTime += timeOffset;
                    if (note.quantizedStartStep !== undefined) {
                        note.quantizedStartStep += 8;
                        note.quantizedEndStep += 8;
                    }
                });
                inputSequenceForVAE.notes.push(...originalNotes);
                inputSequenceForVAE.totalTime += timeOffset;
            }
            
            if (inputSequenceForVAE.notes.length > 0) {
                 inputSequenceForVAE.totalTime = 16 * currentVAEStepDuration;
            } else {
                alert("Cannot process an empty sequence for variation.");
                return;
            }
            console.log("Prepared input sequence for VAE (16 steps):", inputSequenceForVAE);

            try {
                const generatedSequences = await music_vae.sample(1, 0.7, null, inputSequenceForVAE, 16);
                if (generatedSequences && generatedSequences.length > 0) {
                    const variation = generatedSequences[0];
                    console.log("Generated variation:", variation);
                    playSequence(variation, "Generated Variation");
                } else {
                    console.warn("MusicVAE returned no sequences.");
                    alert("The AI could not generate a variation for this melody.");
                }
            } catch (error) {
                console.error("Error during MusicVAE sampling:", error);
                alert("An error occurred while generating the music variation. Check console.");
            }
        });
    } else {
        console.error("Generate Variation button not found.");
    }
}
