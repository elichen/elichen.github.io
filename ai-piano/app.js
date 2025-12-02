// Magic Piano - Neural Net melody with keyboard hints
// Press keys to hint pitch class - RNN picks the best note

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Two-octave keyboard mapping to MIDI notes
// Lower octave (C3-B3): white=ZXCVBNM, black=SDGHJ
// Upper octave (C4-B4): white=QWERTYU, black=23567
const KEY_TO_MIDI = {
    // Lower octave (C3 = 48)
    'KeyZ': 48,  // C3
    'KeyS': 49,  // C#3
    'KeyX': 50,  // D3
    'KeyD': 51,  // D#3
    'KeyC': 52,  // E3
    'KeyV': 53,  // F3
    'KeyG': 54,  // F#3
    'KeyB': 55,  // G3
    'KeyH': 56,  // G#3
    'KeyN': 57,  // A3
    'KeyJ': 58,  // A#3
    'KeyM': 59,  // B3
    // Upper octave (C4 = 60)
    'KeyQ': 60,  // C4
    'Digit2': 61,  // C#4
    'KeyW': 62,  // D4
    'Digit3': 63,  // D#4
    'KeyE': 64,  // E4
    'KeyR': 65,  // F4
    'Digit5': 66,  // F#4
    'KeyT': 67,  // G4
    'Digit6': 68,  // G#4
    'KeyY': 69,  // A4
    'Digit7': 70,  // A#4
    'KeyU': 71   // B4
};

// MelodyRNN encoding constants
const MIN_PITCH = 48;  // C3
const MAX_PITCH = 84;  // C6
const FIRST_PITCH_INDEX = 2;  // Indices 0,1 are NO_EVENT and NOTE_OFF
const BIAS_STRENGTH = 2;  // How strongly to bias toward hint (higher = stronger)

// State
let melodyRNN = null;
let player = null;
let temperature = 1.2;
let isGenerating = false;

// Melody state
let currentSequence = null;
const STEPS_PER_QUARTER = 4;
const QUARTERS_PER_TAP = 0.5;
const MAX_CONTEXT = 20;

// ==================== INIT ====================

async function initModel() {
    const status = document.getElementById('status');
    const loading = document.getElementById('loading');
    const startBtn = document.getElementById('start-btn');

    status.textContent = 'Loading MelodyRNN...';

    try {
        melodyRNN = new mm.MusicRNN(
            'https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/melody_rnn'
        );
        await melodyRNN.initialize();

        player = new mm.SoundFontPlayer(
            'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus'
        );

        status.textContent = 'Ready! Press keys to play';
        loading.style.display = 'none';
        startBtn.style.display = 'block';

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

// Convert RNN output index to MIDI pitch
function indexToMidi(index) {
    return index - FIRST_PITCH_INDEX + MIN_PITCH;
}

// Convert MIDI pitch to RNN index
function midiToIndex(midi) {
    return midi - MIN_PITCH + FIRST_PITCH_INDEX;
}

// Calculate pitch class distance (wraps around octave)
function pitchClassDistance(pc1, pc2) {
    const diff = Math.abs(pc1 - pc2);
    return Math.min(diff, 12 - diff);
}

// Weighted random sampling from candidates
function weightedSample(candidates) {
    if (!candidates || candidates.length === 0) {
        console.error('weightedSample: no candidates!');
        return null;
    }

    const totalProb = candidates.reduce((sum, c) => sum + c.prob, 0);
    if (totalProb === 0) {
        console.warn('weightedSample: all probs are 0, picking first');
        return candidates[0].midi;
    }

    let r = Math.random() * totalProb;

    for (const c of candidates) {
        r -= c.prob;
        if (r <= 0) return c.midi;
    }
    return candidates[candidates.length - 1].midi;
}

// ==================== RNN-BIASED NOTE GENERATION ====================

async function generateNote(hintMidi = null) {
    console.log('generateNote called, hintMidi:', hintMidi, 'isGenerating:', isGenerating);

    if (isGenerating) {
        console.log('Skipped: already generating');
        return null;
    }
    if (!melodyRNN) {
        console.log('Skipped: melodyRNN not ready');
        return null;
    }

    isGenerating = true;
    const status = document.getElementById('status');

    try {
        // Initialize sequence if needed
        if (!currentSequence || currentSequence.notes.length === 0) {
            currentSequence = createEmptySequence();
            // Start with the hinted note or middle C
            const startPitch = hintMidi !== null ? hintMidi : 60;
            addNoteToSequence(startPitch);
            isGenerating = false;
            return startPitch;
        }

        status.textContent = 'AI thinking...';

        // Get probability distribution from RNN
        const result = await melodyRNN.continueSequenceAndReturnProbabilities(
            currentSequence,
            1,  // Generate 1 step
            temperature
        );

        // Get the probability array (already Float32Array, no .data() needed)
        const probs = result.probs[0];

        let selectedMidi;

        if (hintMidi !== null) {
            // Soft bias: weight all notes by distance from hint
            // Gaussian-like falloff: weight = exp(-distance² / (2 * sigma²))
            const sigma = 12 / BIAS_STRENGTH;  // Controls spread
            const candidates = [];

            for (let i = FIRST_PITCH_INDEX; i < probs.length; i++) {
                const midi = indexToMidi(i);
                if (midi >= MIN_PITCH && midi <= MAX_PITCH) {
                    const distance = Math.abs(midi - hintMidi);
                    // Gaussian weight - notes near hint get boosted
                    const weight = Math.exp(-(distance * distance) / (2 * sigma * sigma));
                    const biasedProb = probs[i] * weight;

                    candidates.push({ midi, prob: biasedProb, origProb: probs[i], distance, weight });
                }
            }

            // Sort by biased prob to see top candidates
            const sorted = [...candidates].sort((a, b) => b.prob - a.prob);
            console.log('Top 5 biased candidates:', sorted.slice(0, 5));
            console.log('Probs array length:', probs.length);

            // Sample from biased distribution
            selectedMidi = weightedSample(candidates);
            console.log('Selected MIDI:', selectedMidi);
        } else {
            // No hint - use the RNN's own choice from the returned sequence
            const continued = await result.sequence;
            if (continued.notes && continued.notes.length > 0) {
                selectedMidi = continued.notes[continued.notes.length - 1].pitch;
            } else {
                // Fallback
                const lastNote = currentSequence.notes[currentSequence.notes.length - 1];
                selectedMidi = lastNote.pitch + (Math.random() < 0.5 ? 1 : -1);
            }
        }

        // Check if we got a valid note
        if (selectedMidi === null || selectedMidi === undefined) {
            console.error('No valid MIDI selected');
            status.textContent = 'Ready! Press keys to play';
            isGenerating = false;
            return null;
        }

        // Clamp to valid range
        selectedMidi = Math.max(MIN_PITCH, Math.min(MAX_PITCH, selectedMidi));

        addNoteToSequence(selectedMidi);
        status.textContent = 'Ready! Press keys to play';
        isGenerating = false;
        console.log('Returning MIDI:', selectedMidi);
        return selectedMidi;

    } catch (err) {
        console.error('Generation error:', err);
        status.textContent = 'Ready! Press keys to play';
        isGenerating = false;
        return null;
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

    // Trim old notes to keep generation fast
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

async function playNote(midi) {
    console.log('playNote called, midi:', midi);
    if (!player) {
        console.error('playNote: no player!');
        return;
    }

    const noteSeq = {
        notes: [{ pitch: midi, startTime: 0, endTime: 0.4 }],
        totalTime: 0.4
    };

    try {
        if (player.isPlaying()) {
            console.log('Stopping previous playback');
            player.stop();
        }
        console.log('Loading samples...');
        await player.loadSamples(noteSeq);
        console.log('Starting playback');
        player.start(noteSeq);
    } catch (err) {
        console.error('Playback error:', err);
    }
}

// ==================== HANDLERS ====================

async function handleKeyPress(hintMidi) {
    console.log('handleKeyPress:', hintMidi);
    animateKey(hintMidi);

    const midi = await generateNote(hintMidi);
    console.log('generateNote returned:', midi);
    if (midi === null) {
        console.log('midi is null, skipping playback');
        return;
    }

    playNote(midi);
    showNote(midiToNoteName(midi));
    addToHistory(midi);
}

// ==================== UI ====================

function showNote(noteName) {
    const display = document.getElementById('note-display');
    display.innerHTML = `<span class="current-note">${noteName}</span>`;
}

function addToHistory(midi) {
    const history = document.getElementById('note-history');
    const pitchClass = midi % 12;
    const noteName = NOTE_NAMES[pitchClass];

    const el = document.createElement('div');
    el.className = 'history-note';
    el.dataset.pitch = pitchClass;
    el.textContent = noteName;
    history.appendChild(el);

    while (history.children.length > 16) {
        history.removeChild(history.firstChild);
    }
}

function animateKey(midi) {
    const key = document.querySelector(`.piano-key[data-midi="${midi}"]`);
    if (key) {
        key.classList.add('active');
        setTimeout(() => key.classList.remove('active'), 150);
    }
}

function resetMelody() {
    currentSequence = createEmptySequence();
    document.getElementById('note-history').innerHTML = '';
    document.getElementById('note-display').innerHTML = '';
    document.getElementById('status').textContent = 'Melody reset - press keys to play!';
}

// ==================== SETUP ====================

document.addEventListener('DOMContentLoaded', () => {
    initModel();

    const overlay = document.getElementById('init-overlay');
    const startBtn = document.getElementById('start-btn');
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('temp-value');
    const resetBtn = document.getElementById('reset-btn');

    startBtn.addEventListener('click', () => {
        overlay.classList.add('hidden');
    });

    tempSlider.addEventListener('input', (e) => {
        temperature = parseFloat(e.target.value);
        tempValue.textContent = temperature.toFixed(1);
    });

    resetBtn.addEventListener('click', resetMelody);

    // Keyboard handlers
    document.addEventListener('keydown', (e) => {
        if (e.repeat) return;

        const midi = KEY_TO_MIDI[e.code];
        if (midi !== undefined) {
            e.preventDefault();
            handleKeyPress(midi);
        }
    });

    // On-screen piano clicks
    document.querySelectorAll('.piano-key').forEach(key => {
        key.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const midi = parseInt(key.dataset.midi);
            handleKeyPress(midi);
        });
        key.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const midi = parseInt(key.dataset.midi);
            handleKeyPress(midi);
        });
    });
});
