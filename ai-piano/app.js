// Magic Piano - Genie Mode
// 8 buttons control 88 keys with contour preservation
// Inspired by Piano Genie: https://magenta.tensorflow.org/pianogenie

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Keyboard mapping: keys 1-8 map to genie buttons
const KEY_TO_BUTTON = {
    'Digit1': 1, 'Digit2': 2, 'Digit3': 3, 'Digit4': 4,
    'Digit5': 5, 'Digit6': 6, 'Digit7': 7, 'Digit8': 8,
    'Numpad1': 1, 'Numpad2': 2, 'Numpad3': 3, 'Numpad4': 4,
    'Numpad5': 5, 'Numpad6': 6, 'Numpad7': 7, 'Numpad8': 8
};

// MelodyRNN encoding constants
const MIN_PITCH = 48;  // C3
const MAX_PITCH = 84;  // C6
const PITCH_RANGE = MAX_PITCH - MIN_PITCH;  // 36 semitones (3 octaves)
const FIRST_PITCH_INDEX = 2;  // Indices 0,1 are NO_EVENT and NOTE_OFF

// State
let melodyRNN = null;
let player = null;
const temperature = 0.95;  // Balanced creativity
let isGenerating = false;

// Genie state - track last button and note for contour preservation
let lastButton = null;
let lastPitch = null;

// Chord support - buffer buttons pressed within a short window
let chordBuffer = [];
let chordTimeout = null;
const CHORD_WINDOW_MS = 50;

// Melody state
let currentSequence = null;
const STEPS_PER_QUARTER = 4;
const QUARTERS_PER_TAP = 0.5;
const MAX_CONTEXT = 20;

// ==================== INIT ====================

async function initModel() {
    const status = document.getElementById('status');
    const overlay = document.getElementById('init-overlay');

    try {
        melodyRNN = new mm.MusicRNN(
            'https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/melody_rnn'
        );
        await melodyRNN.initialize();

        player = new mm.SoundFontPlayer(
            'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus'
        );

        status.textContent = 'Press 1-8 to play';
        overlay.classList.add('hidden');

        console.log('MelodyRNN initialized');
    } catch (err) {
        console.error('Failed to load model:', err);
        status.textContent = 'Failed to load. Try refreshing.';
    }
}

// ==================== SEQUENCE HELPERS ====================

function createEmptySequence() {
    return {
        notes: [],
        totalQuantizedSteps: 0,
        quantizationInfo: { stepsPerQuarter: STEPS_PER_QUARTER }
    };
}

function midiToNoteName(midi) {
    const note = NOTE_NAMES[midi % 12];
    const octave = Math.floor(midi / 12) - 1;
    return note + octave;
}

function indexToMidi(index) {
    return index - FIRST_PITCH_INDEX + MIN_PITCH;
}

function midiToIndex(midi) {
    return midi - MIN_PITCH + FIRST_PITCH_INDEX;
}

// ==================== CONTOUR-PRESERVING NOTE GENERATION ====================

// Map button (1-8) to a target pitch range
// Button 1 = lowest range, Button 8 = highest range
function getButtonPitchRange(button) {
    // Divide pitch range into 8 overlapping regions
    const regionSize = PITCH_RANGE / 4;  // ~9 semitones per region
    const center = MIN_PITCH + ((button - 1) / 7) * PITCH_RANGE;

    return {
        min: Math.max(MIN_PITCH, Math.floor(center - regionSize / 2)),
        max: Math.min(MAX_PITCH, Math.ceil(center + regionSize / 2)),
        center: Math.round(center)
    };
}

// Apply contour constraint: if ascending buttons, note must not descend (and vice versa)
function applyContourConstraint(button, pitchRange) {
    if (lastButton === null || lastPitch === null) {
        return pitchRange;  // No constraint on first note
    }

    const constrained = { ...pitchRange };

    if (button > lastButton) {
        // Ascending button press - note should be >= lastPitch
        constrained.min = Math.max(constrained.min, lastPitch);
    } else if (button < lastButton) {
        // Descending button press - note should be <= lastPitch
        constrained.max = Math.min(constrained.max, lastPitch);
    }
    // If same button, allow some variation around lastPitch

    // Ensure valid range
    if (constrained.min > constrained.max) {
        // Constraint impossible - use boundary
        if (button > lastButton) {
            constrained.min = constrained.max = Math.min(MAX_PITCH, lastPitch + 1);
        } else {
            constrained.min = constrained.max = Math.max(MIN_PITCH, lastPitch - 1);
        }
    }

    return constrained;
}

async function generateGenieNote(button) {
    if (isGenerating || !melodyRNN) return null;

    isGenerating = true;
    const status = document.getElementById('status');

    try {
        // Get target range for this button
        let pitchRange = getButtonPitchRange(button);

        // Apply contour constraint
        pitchRange = applyContourConstraint(button, pitchRange);

        console.log(`Button ${button}: range [${pitchRange.min}-${pitchRange.max}], last: ${lastPitch}`);

        // Initialize sequence if needed
        if (!currentSequence || currentSequence.notes.length === 0) {
            currentSequence = createEmptySequence();
            const startPitch = pitchRange.center;
            addNoteToSequence(startPitch);
            lastButton = button;
            lastPitch = startPitch;
            isGenerating = false;
            return startPitch;
        }

        status.textContent = 'AI thinking...';

        // Get probability distribution from RNN
        const result = await melodyRNN.continueSequenceAndReturnProbabilities(
            currentSequence,
            1,
            temperature
        );

        const probs = result.probs[0];

        // Build candidates within the constrained pitch range
        const candidates = [];
        for (let midi = pitchRange.min; midi <= pitchRange.max; midi++) {
            const idx = midiToIndex(midi);
            if (idx >= FIRST_PITCH_INDEX && idx < probs.length) {
                // Weight by distance from button's center pitch
                const distFromCenter = Math.abs(midi - pitchRange.center);
                const centerWeight = Math.exp(-distFromCenter * distFromCenter / 50);
                candidates.push({
                    midi,
                    prob: probs[idx] * centerWeight
                });
            }
        }

        // Sample from candidates
        let selectedMidi;
        if (candidates.length === 0) {
            // Fallback if no candidates
            selectedMidi = pitchRange.center;
        } else {
            const totalProb = candidates.reduce((sum, c) => sum + c.prob, 0);
            if (totalProb === 0) {
                selectedMidi = candidates[Math.floor(candidates.length / 2)].midi;
            } else {
                let r = Math.random() * totalProb;
                for (const c of candidates) {
                    r -= c.prob;
                    if (r <= 0) {
                        selectedMidi = c.midi;
                        break;
                    }
                }
                if (selectedMidi === undefined) {
                    selectedMidi = candidates[candidates.length - 1].midi;
                }
            }
        }

        addNoteToSequence(selectedMidi);
        lastButton = button;
        lastPitch = selectedMidi;

        status.textContent = 'Ready! Press 1-8 to play';
        isGenerating = false;
        return selectedMidi;

    } catch (err) {
        console.error('Generation error:', err);
        status.textContent = 'Ready! Press 1-8 to play';
        isGenerating = false;
        return null;
    }
}

// Generate multiple notes for a chord
async function generateGenieChord(buttons) {
    if (isGenerating || !melodyRNN) return [];

    isGenerating = true;
    const status = document.getElementById('status');
    const generatedNotes = [];

    try {
        // Sort buttons to process in order (for consistent contour)
        buttons.sort((a, b) => a - b);

        // Initialize sequence if needed
        if (!currentSequence || currentSequence.notes.length === 0) {
            currentSequence = createEmptySequence();
            // Generate each note in the chord
            for (const button of buttons) {
                const pitchRange = getButtonPitchRange(button);
                const pitch = pitchRange.center;
                generatedNotes.push(pitch);
                addNoteToSequence(pitch);
            }
            // Track the highest button/pitch for next contour
            lastButton = buttons[buttons.length - 1];
            lastPitch = Math.max(...generatedNotes);
            isGenerating = false;
            return generatedNotes;
        }

        status.textContent = 'AI thinking...';

        // Get probability distribution from RNN (single call)
        const result = await melodyRNN.continueSequenceAndReturnProbabilities(
            currentSequence,
            1,
            temperature
        );
        const probs = result.probs[0];

        // Generate a note for each button
        let currentLastButton = lastButton;
        let currentLastPitch = lastPitch;

        for (const button of buttons) {
            // Get target range for this button
            let pitchRange = getButtonPitchRange(button);

            // Apply contour constraint based on current state
            if (currentLastButton !== null && currentLastPitch !== null) {
                const constrained = { ...pitchRange };
                if (button > currentLastButton) {
                    constrained.min = Math.max(constrained.min, currentLastPitch);
                } else if (button < currentLastButton) {
                    constrained.max = Math.min(constrained.max, currentLastPitch);
                }
                if (constrained.min > constrained.max) {
                    if (button > currentLastButton) {
                        constrained.min = constrained.max = Math.min(MAX_PITCH, currentLastPitch + 1);
                    } else {
                        constrained.min = constrained.max = Math.max(MIN_PITCH, currentLastPitch - 1);
                    }
                }
                pitchRange = constrained;
            }

            // Build candidates
            const candidates = [];
            for (let midi = pitchRange.min; midi <= pitchRange.max; midi++) {
                const idx = midiToIndex(midi);
                if (idx >= FIRST_PITCH_INDEX && idx < probs.length) {
                    const distFromCenter = Math.abs(midi - pitchRange.center);
                    const centerWeight = Math.exp(-distFromCenter * distFromCenter / 50);
                    candidates.push({ midi, prob: probs[idx] * centerWeight });
                }
            }

            // Sample
            let selectedMidi;
            if (candidates.length === 0) {
                selectedMidi = pitchRange.center;
            } else {
                const totalProb = candidates.reduce((sum, c) => sum + c.prob, 0);
                if (totalProb === 0) {
                    selectedMidi = candidates[Math.floor(candidates.length / 2)].midi;
                } else {
                    let r = Math.random() * totalProb;
                    for (const c of candidates) {
                        r -= c.prob;
                        if (r <= 0) {
                            selectedMidi = c.midi;
                            break;
                        }
                    }
                    if (selectedMidi === undefined) {
                        selectedMidi = candidates[candidates.length - 1].midi;
                    }
                }
            }

            generatedNotes.push(selectedMidi);
            addNoteToSequence(selectedMidi);
            currentLastButton = button;
            currentLastPitch = selectedMidi;
        }

        // Update state with the highest note in chord for next contour
        lastButton = buttons[buttons.length - 1];
        lastPitch = Math.max(...generatedNotes);

        status.textContent = 'Ready! Press 1-8 to play';
        isGenerating = false;
        return generatedNotes;

    } catch (err) {
        console.error('Chord generation error:', err);
        status.textContent = 'Ready! Press 1-8 to play';
        isGenerating = false;
        return [];
    }
}

function addNoteToSequence(pitch) {
    const stepsPerNote = STEPS_PER_QUARTER * QUARTERS_PER_TAP;

    currentSequence.notes.push({
        pitch: pitch,
        quantizedStartStep: currentSequence.totalQuantizedSteps,
        quantizedEndStep: currentSequence.totalQuantizedSteps + stepsPerNote
    });

    currentSequence.totalQuantizedSteps += stepsPerNote;

    // Trim old notes
    if (currentSequence.notes.length > MAX_CONTEXT) {
        const removeCount = currentSequence.notes.length - MAX_CONTEXT;
        currentSequence.notes.splice(0, removeCount);

        let step = 0;
        for (const note of currentSequence.notes) {
            note.quantizedStartStep = step;
            note.quantizedEndStep = step + stepsPerNote;
            step += stepsPerNote;
        }
        currentSequence.totalQuantizedSteps = step;
    }
}

// ==================== AUDIO ====================

async function playNotes(midiNotes) {
    if (!player) return;

    if (!Array.isArray(midiNotes)) {
        midiNotes = [midiNotes];
    }

    const noteSeq = {
        notes: midiNotes.map(midi => ({ pitch: midi, startTime: 0, endTime: 0.4 })),
        totalTime: 0.4
    };

    try {
        if (player.isPlaying()) {
            player.stop();
        }
        await player.loadSamples(noteSeq);
        player.start(noteSeq);
    } catch (err) {
        console.error('Playback error:', err);
    }
}

// ==================== HANDLERS ====================

function handleGenieButton(button) {
    console.log('Genie button:', button);
    animateButton(button);

    // Add to chord buffer (avoid duplicates)
    if (!chordBuffer.includes(button)) {
        chordBuffer.push(button);
    }

    // Reset the chord window timer
    if (chordTimeout) {
        clearTimeout(chordTimeout);
    }

    // After CHORD_WINDOW_MS of no new buttons, process the chord
    chordTimeout = setTimeout(() => {
        processChordBuffer();
    }, CHORD_WINDOW_MS);
}

async function processChordBuffer() {
    if (chordBuffer.length === 0) return;

    const buttons = [...chordBuffer];
    chordBuffer = [];
    chordTimeout = null;

    console.log('Processing buttons:', buttons);

    let notes;
    if (buttons.length === 1) {
        const midi = await generateGenieNote(buttons[0]);
        notes = midi !== null ? [midi] : [];
    } else {
        notes = await generateGenieChord(buttons);
    }

    if (notes.length > 0) {
        playNotes(notes);
        const noteNames = notes.map(midiToNoteName);
        showNote(noteNames.join(' '));
    }
}

// ==================== UI ====================

function showNote(noteName) {
    const display = document.getElementById('note-display');
    display.innerHTML = `<span class="current-note">${noteName}</span>`;
}

function animateButton(button) {
    const btn = document.querySelector(`.genie-btn[data-btn="${button}"]`);
    if (btn) {
        btn.classList.add('active');
        setTimeout(() => btn.classList.remove('active'), 150);
    }
}

// ==================== SETUP ====================

document.addEventListener('DOMContentLoaded', () => {
    initModel();

    // Keyboard handlers
    document.addEventListener('keydown', (e) => {
        if (e.repeat) return;

        const button = KEY_TO_BUTTON[e.code];
        if (button !== undefined) {
            e.preventDefault();
            handleGenieButton(button);
        }
    });

    // On-screen button clicks
    document.querySelectorAll('.genie-btn').forEach(btn => {
        btn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const button = parseInt(btn.dataset.btn);
            handleGenieButton(button);
        });
        btn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const button = parseInt(btn.dataset.btn);
            handleGenieButton(button);
        });
    });
});
