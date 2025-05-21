console.log("Music Loop Generator app loaded");

// Initialize Magenta.js Player
// Magenta.js is typically available as `mm` when loaded from the CDN.
// However, since we are loading individual modules (core, music_vae, player),
// the Player might be available directly under `core` or as `magenta.Player`.
// For robustness, let's try to find it, or assume it's globally available via `magenta.Player` or `core.Player`
let player;

const playerCallback = {
  run: (note) => {
    // This callback is called shortly BEFORE a note is played.
    // Our grid data-time is 0-7. Note's quantizedStartStep should map directly.
    const cell = document.querySelector(`.grid-cell[data-pitch='${note.pitch}'][data-time='${note.quantizedStartStep}']`);
    if (cell) {
      cell.classList.add('playing');
      // Schedule removal of the class after note duration
      // Note duration is (quantizedEndStep - quantizedStartStep) * VAE_STEP_DURATION
      const durationMs = (note.quantizedEndStep - note.quantizedStartStep) * currentVAEStepDuration * 1000;
      setTimeout(() => {
        cell.classList.remove('playing');
      }, durationMs);
    }
  },
  stop: () => {
    // Cleanup any remaining highlights if needed
    document.querySelectorAll('.grid-cell.playing').forEach(cell => cell.classList.remove('playing'));
  }
};

try {
    if (typeof core !== 'undefined' && core.Player) {
        player = new core.Player(false, playerCallback);
    } else if (typeof magenta !== 'undefined' && magenta.Player) {
        player = new magenta.Player(false, playerCallback);
    } else if (typeof mm !== 'undefined' && mm.Player) { // Check for mm.Player as well
        player = new mm.Player(false, playerCallback);
        console.log("Player initialized using mm.Player with callback.");
    }
    else {
        console.error("Magenta.js Player not found. Ensure Magenta.js is loaded correctly.");
    }
} catch (e) {
    console.error("Error initializing Magenta.js Player with callback:", e);
    // Fallback to basic player if callback version fails
    if (typeof core !== 'undefined' && core.Player) player = new core.Player();
    else if (typeof magenta !== 'undefined' && magenta.Player) player = new magenta.Player();
    else if (typeof mm !== 'undefined' && mm.Player) player = new mm.Player();
}


// Test NoteSequence (will not have visual feedback unless quantizedStartStep is added)
const testSequence = {
  notes: [
    { pitch: 60, startTime: 0.0, endTime: 0.5, velocity: 80 } // C4 note
  ],
  totalTime: 0.5
};

// Event listener for the "Play Test Note" button
const playTestNoteButton = document.getElementById('play-test-note-btn');

if (playTestNoteButton) {
  playTestNoteButton.addEventListener('click', () => {
    if (!player) {
        console.error("Player is not initialized.");
        alert("Error: Music player is not initialized. Check console for details.");
        return;
    }

    // Resume AudioContext on user gesture
    // Magenta's player usually handles this, but it's good practice.
    // For Tone.js (which Magenta.js Player uses under the hood), Tone.start() is used.
    if (typeof Tone !== 'undefined' && Tone.context && Tone.context.state !== 'running') {
        Tone.start().then(() => {
            console.log("AudioContext started");
            playNote();
        }).catch(e => {
            console.error("Error starting AudioContext:", e);
            alert("Could not start audio. Please interact with the page and try again.");
        });
    } else {
        playNote();
    }
  });
} else {
    console.error("Play Test Note button not found.");
}

function playNote() {
    if (player.isPlaying()) {
        console.log("Player is currently playing. Stopping before starting new sequence.");
        player.stop();
    }

    try {
        console.log("Playing test sequence:", testSequence);
        player.start(testSequence)
            .then(() => {
                console.log("Playback finished.");
            })
            .catch(e => {
                console.error("Error during playback:", e);
                alert("Error playing sound. Check console for details.");
            });
    } catch (e) {
        console.error("Error calling player.start():", e);
        alert("Error initiating playback. Check console for details.");
    }
}

// A simple check to see if Magenta objects are loaded
window.addEventListener('load', () => {
    initializeGrid();

    if (typeof core !== 'undefined') {
        console.log('Magenta Core loaded:', core);
    } else {
        console.warn('Magenta Core (core) is not defined on window load.');
    }
    if (typeof music_vae !== 'undefined') {
        console.log('Magenta MusicVAE loaded:', music_vae);
    } else {
        console.warn('Magenta MusicVAE (music_vae) is not defined on window load.');
    }
    if (typeof mm !== 'undefined') { // mm is the common global for full bundle
        console.log('Magenta (mm) loaded:', mm);
        if (!player && mm.Player) { // If initial player setup failed, try with mm.Player with callback
             try {
                player = new mm.Player(false, playerCallback);
                console.log("Player initialized using mm.Player with callback");
             } catch (e) {
                console.error("Error initializing mm.Player with callback", e);
             }
        }
    } else {
        console.warn('Magenta (mm) is not defined on window load. This is expected if loading individual modules.');
    }
     if (!player) {
        console.error("Player could not be initialized. Playback will not work.");
        alert("Music player components could not be loaded. Please ensure you are connected to the internet and try refreshing. Check the console for more details.");
    }

    initializeMusicVAE(); // Initialize MusicVAE model on load
    initializeTempoControls();
    initializeClearGridButton();
});

// Tempo and Duration
let currentBPM = 120;
// VAE_STEP_DURATION will now be currentVAEStepDuration and updated by tempo changes
// A 16th note duration = (60 / BPM) / 4
let currentVAEStepDuration = (60 / currentBPM) / 4;


const tempoSlider = document.getElementById('tempo-slider');
const tempoValueDisplay = document.getElementById('tempo-value');

function initializeTempoControls() {
    if (tempoSlider && tempoValueDisplay) {
        tempoSlider.addEventListener('input', () => {
            currentBPM = parseInt(tempoSlider.value, 10);
            tempoValueDisplay.textContent = currentBPM;
            currentVAEStepDuration = (60 / currentBPM) / 4;
            console.log(`Tempo changed to ${currentBPM} BPM. VAE Step Duration: ${currentVAEStepDuration}s`);
            // Note: Magenta Player doesn't have a global setTempo that affects already playing or future sequences' internal timing directly
            // without re-calculating their note start/end times.
            // The change in currentVAEStepDuration will affect new sequences created by gridToNoteSequence
            // and how their playback is visualized.
        });
    } else {
        console.error("Tempo control elements not found.");
    }
}

const clearGridButton = document.getElementById('clear-grid-btn');

function initializeClearGridButton() {
    if (clearGridButton) {
        clearGridButton.addEventListener('click', () => {
            const activeCells = document.querySelectorAll('.grid-cell.active');
            activeCells.forEach(cell => {
                cell.classList.remove('active');
            });
            console.log("Grid cleared.");
        });
    } else {
        console.error("Clear Grid button not found.");
    }
}


let music_vae;
const VAE_CHECKPOINT_URL = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2';

async function initializeMusicVAE() {
  try {
    // Check if mm (Magenta global) is available, otherwise use magenta.MusicVAE
    if (typeof mm !== 'undefined' && mm.MusicVAE) {
        music_vae = new mm.MusicVAE(VAE_CHECKPOINT_URL);
    } else if (typeof magenta !== 'undefined' && magenta.MusicVAE) {
        music_vae = new magenta.MusicVAE(VAE_CHECKPOINT_URL);
    } else {
        console.error("Magenta MusicVAE constructor not found. Ensure Magenta.js is loaded correctly.");
        alert("Error: Music VAE components could not be loaded. Playback and generation might not work.");
        return;
    }
    await music_vae.initialize();
    console.log('MusicVAE initialized.');
    document.getElementById('generate-variation-btn').disabled = false;
  } catch (error) {
    console.error('Failed to initialize MusicVAE:', error);
    alert('Failed to load AI model for music generation. Please refresh and try again.');
  }
}

function initializeGrid() {
    const cells = document.querySelectorAll('.grid-cell');
    cells.forEach(cell => {
        cell.addEventListener('click', () => {
            cell.classList.toggle('active');
        });
    });
    // Update console log to reflect VAE step duration if necessary for clarity
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
        totalTime: maxEndTime, // This would be 8 * currentVAEStepDuration if the last step is active
        quantizationInfo: { stepsPerQuarter: 4 } // Standard for many Magenta models
    };
}

const playUserLoopButton = document.getElementById('play-user-loop-btn');

if (playUserLoopButton) {
    playUserLoopButton.addEventListener('click', () => {
        if (!player) {
            console.error("Player is not initialized for user loop playback.");
            alert("Error: Music player is not initialized. Check console for details.");
            return;
        }

        const userSequence = gridToNoteSequence();

        if (!userSequence || userSequence.notes.length === 0) {
            console.log("No notes selected in the grid.");
            alert("Please select some notes in the grid to play a loop.");
            return;
        }

        // Resume AudioContext on user gesture (similar to test note playback)
        if (typeof Tone !== 'undefined' && Tone.context && Tone.context.state !== 'running') {
            Tone.start().then(() => {
                console.log("AudioContext started for user loop.");
                playUserNotes(userSequence);
            }).catch(e => {
                console.error("Error starting AudioContext for user loop:", e);
                alert("Could not start audio. Please interact with the page and try again.");
            });
        } else {
            playUserNotes(userSequence);
        }
    });
} else {
    console.error("Play User Loop button not found.");
}

function playGeneratedNotes(sequence, title = "Generated Sequence") {
    if (!player) {
        console.error("Player not initialized for generated notes.");
        alert("Player not initialized. Cannot play generated notes.");
        return;
    }
    if (player.isPlaying()) {
        console.log(`Player is currently playing. Stopping before starting ${title}.`);
        player.stop();
    }

    try {
        console.log(`Playing ${title}:`, sequence);
        player.start(sequence)
            .then(() => {
                console.log(`${title} playback finished.`);
            })
            .catch(e => {
                console.error(`Error during ${title} playback:`, e);
                alert(`Error playing ${title}. Check console.`);
            });
    } catch (e) {
        console.error(`Error calling player.start() for ${title}:`, e);
        alert(`Error initiating ${title} playback. Check console.`);
    }
}


function playUserNotes(sequence) {
    if (player.isPlaying()) {
        console.log("Player is currently playing. Stopping before starting user sequence.");
        player.stop();
    }

    try {
        console.log("Playing user sequence:", sequence);
        player.start(sequence)
            .then(() => {
                console.log("User loop playback finished.");
            })
            .catch(e => {
                console.error("Error during user loop playback:", e);
                alert("Error playing user loop. Check console for details.");
            });
    } catch (e) {
        console.error("Error calling player.start() for user loop:", e);
        alert("Error initiating user loop playback. Check console for details.");
    }
}

const generateVariationButton = document.getElementById('generate-variation-btn');

if (generateVariationButton) {
    generateVariationButton.addEventListener('click', async () => {
        if (!music_vae || !music_vae.isInitialized()) {
            console.error("MusicVAE is not initialized.");
            alert("AI Music Generation model is not ready. Please wait or try refreshing.");
            return;
        }

        let userSequence = gridToNoteSequence();

        if (!userSequence || userSequence.notes.length === 0) {
            alert("Please create a short melody in the grid first to generate a variation.");
            return;
        }

        // The model mel_4bar_small_q2 expects sequences of 16 steps (1 bar) or multiples.
        // Our grid provides 8 steps. We need to adapt this.
        // Option 1: Pad the sequence to 16 steps.
        // Option 2: Repeat the sequence to make it 16 steps.

        // Let's try repeating the 8-step sequence to make it 16 steps.
        // First, ensure the sequence is quantized correctly.
        // The gridToNoteSequence already creates steps that are equivalent to 16th notes.
        // So, quantization is implicitly handled by VAE_STEP_DURATION and quantizedStartStep/EndStep.

        let inputSequenceForVAE = mm.sequences.clone(userSequence);

        // If the sequence is 8 steps long (totalTime = 8 * currentVAEStepDuration), duplicate it.
        if (inputSequenceForVAE.notes.length > 0 && inputSequenceForVAE.totalTime <= 8 * currentVAEStepDuration) {
            const originalNotes = JSON.parse(JSON.stringify(inputSequenceForVAE.notes)); // Deep clone
            const timeOffset = inputSequenceForVAE.totalTime; // Should be 8 * currentVAEStepDuration

            originalNotes.forEach(note => {
                note.startTime += timeOffset;
                note.endTime += timeOffset;
                if (note.quantizedStartStep !== undefined) { // Check if properties exist
                    note.quantizedStartStep += 8; // Shift by 8 steps
                    note.quantizedEndStep += 8;
                }
            });
            inputSequenceForVAE.notes.push(...originalNotes);
            inputSequenceForVAE.totalTime += timeOffset; // Now it's 16 steps long
        }
        
        // Ensure totalTime is correctly set for the 16 steps
        if (inputSequenceForVAE.notes.length > 0) {
             inputSequenceForVAE.totalTime = 16 * currentVAEStepDuration;
        } else { // Handle empty sequence after potential modification
            alert("Cannot process an empty sequence for variation.");
            return;
        }


        console.log("Prepared input sequence for VAE (16 steps):", inputSequenceForVAE);

        try {
            // Temperature: 0.5 for some variation but not too random.
            // The third argument (chord progression) is null for this melody model.
            // The fourth argument is the number of bars to fill (not needed if providing sequence).
            // The fifth argument is the source sequence.
            const generatedSequences = await music_vae.sample(1, 0.7, null, inputSequenceForVAE, 16); // 1 seq, temp 0.7, no chords, use inputSeq, 16 steps per bar

            if (generatedSequences && generatedSequences.length > 0) {
                const variation = generatedSequences[0];
                console.log("Generated variation:", variation);

                // Before playing, ensure the generated sequence is also quantized or structured as expected by the player
                // The VAE output should be playable directly.
                playGeneratedNotes(variation, "Generated Variation");

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
