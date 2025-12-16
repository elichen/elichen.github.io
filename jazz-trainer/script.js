// Jazz Changes Trainer
// Music theory constants and utilities

const NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const ENHARMONICS = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
};

// Guitar tuning (standard) - from high E to low E
const GUITAR_STRINGS = ['E', 'B', 'G', 'D', 'A', 'E'];
const STRING_MIDI_BASE = [64, 59, 55, 50, 45, 40]; // MIDI note numbers for open strings

const NUM_FRETS = 15;

// Chord types with intervals (semitones from root)
const CHORD_TYPES = {
    'maj7': { intervals: [0, 4, 7, 11], tensions: [2, 6, 9], avoid: [5] },
    '7': { intervals: [0, 4, 7, 10], tensions: [2, 6, 9], avoid: [5] },  // dominant
    'm7': { intervals: [0, 3, 7, 10], tensions: [2, 5, 9], avoid: [] },
    'm7b5': { intervals: [0, 3, 6, 10], tensions: [2, 5, 8], avoid: [] },
    'dim7': { intervals: [0, 3, 6, 9], tensions: [], avoid: [] },
    '7alt': { intervals: [0, 4, 7, 10], tensions: [1, 3, 6, 8], avoid: [] },
    '7#11': { intervals: [0, 4, 7, 10], tensions: [2, 6, 9], avoid: [] },
};

// Scale patterns (semitones from root)
const SCALES = {
    'ionian': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'melodic-minor': [0, 2, 3, 5, 7, 9, 11],
    'altered': [0, 1, 3, 4, 6, 8, 10],
    'lydian-dominant': [0, 2, 4, 6, 7, 9, 10],
};

// Chord-scale relationships
const CHORD_SCALES = {
    'maj7': 'ionian',
    '7': 'mixolydian',
    'm7': 'dorian',
    'm7b5': 'locrian',
    'dim7': 'locrian',  // simplified
    '7alt': 'altered',
    '7#11': 'lydian-dominant',
};

// Common progressions
const PROGRESSIONS = {
    'ii-V-I-major': [
        { root: 1, type: 'm7', function: 'ii' },
        { root: 6, type: '7', function: 'V' },
        { root: 0, type: 'maj7', function: 'I' },
    ],
    'ii-V-I-minor': [
        { root: 1, type: 'm7b5', function: 'ii°' },
        { root: 6, type: '7alt', function: 'V' },
        { root: 0, type: 'm7', function: 'i' },
    ],
    'I-vi-ii-V': [
        { root: 0, type: 'maj7', function: 'I' },
        { root: 8, type: 'm7', function: 'vi' },
        { root: 1, type: 'm7', function: 'ii' },
        { root: 6, type: '7', function: 'V' },
    ],
    'blues': [
        { root: 0, type: '7', function: 'I' },
        { root: 0, type: '7', function: 'I' },
        { root: 0, type: '7', function: 'I' },
        { root: 0, type: '7', function: 'I' },
        { root: 5, type: '7', function: 'IV' },
        { root: 5, type: '7', function: 'IV' },
        { root: 0, type: '7', function: 'I' },
        { root: 0, type: '7', function: 'I' },
        { root: 7, type: '7', function: 'V' },
        { root: 5, type: '7', function: 'IV' },
        { root: 0, type: '7', function: 'I' },
        { root: 7, type: '7', function: 'V' },
    ],
    'autumn-leaves': [
        { root: 0, type: 'm7', function: 'ii' },
        { root: 5, type: '7', function: 'V' },
        { root: 10, type: 'maj7', function: 'I' },
        { root: 3, type: 'maj7', function: 'IV' },
        { root: 8, type: 'm7b5', function: 'vii°' },
        { root: 1, type: '7alt', function: 'III' },
        { root: 5, type: 'm7', function: 'vi' },
        { root: 5, type: 'm7', function: 'vi' },
    ],
};

// Practice tips for each progression type
const TIPS = {
    'ii-V-I-major': '<strong>ii-V-I Tip:</strong> Focus on the guide tones (3rds and 7ths). Notice how the 7th of one chord resolves down by half-step to the 3rd of the next chord.',
    'ii-V-I-minor': '<strong>Minor ii-V-i Tip:</strong> The V chord often uses the altered scale. Try targeting the b9 and #9 for tension.',
    'I-vi-ii-V': '<strong>Rhythm Changes Tip:</strong> This progression moves quickly. Practice connecting chord tones with chromatic approach notes.',
    'blues': '<strong>Blues Tip:</strong> Mix major and minor pentatonic. The "blue notes" (b3, b5, b7) add character over any chord.',
    'autumn-leaves': '<strong>Autumn Leaves Tip:</strong> This is in G minor. The relative major (Bb) gives you two tonal centers to play with.',
};

// State
let state = {
    currentKey: 'C',
    currentProgression: 'ii-V-I-major',
    currentChordIndex: 0,
    tempo: 120,
    beatsPerChord: 4,
    isPlaying: false,
    player: null,
    playbackInterval: null,
    audioContext: null,
    isLoading: false,
};

// Utility functions
function noteIndex(note) {
    const normalized = ENHARMONICS[note] || note;
    return NOTES.indexOf(normalized);
}

function transposedNote(rootNote, semitones) {
    const rootIndex = noteIndex(rootNote);
    const newIndex = (rootIndex + semitones + 12) % 12;
    return NOTES[newIndex];
}

function noteToMidi(noteName, octave) {
    const noteIdx = noteIndex(noteName);
    return 12 + (octave + 1) * 12 + noteIdx;
}

function getChordNotes(rootNote, chordType) {
    const chord = CHORD_TYPES[chordType];
    return chord.intervals.map(interval => transposedNote(rootNote, interval));
}

function getScaleNotes(rootNote, scaleName) {
    const scale = SCALES[scaleName];
    return scale.map(interval => transposedNote(rootNote, interval));
}

function getNoteOnFret(stringIndex, fret) {
    const openNote = GUITAR_STRINGS[stringIndex];
    return transposedNote(openNote, fret);
}

function getNoteCategory(note, chordRoot, chordType) {
    const chord = CHORD_TYPES[chordType];
    const noteIdx = noteIndex(note);
    const rootIdx = noteIndex(chordRoot);
    const interval = (noteIdx - rootIdx + 12) % 12;

    if (interval === 0) return 'root';
    if (chord.intervals.includes(interval)) return 'chord-tone';
    if (chord.tensions.includes(interval)) return 'tension';
    if (chord.avoid.includes(interval)) return 'avoid';

    // Check if it's in the scale
    const scaleName = CHORD_SCALES[chordType];
    const scale = SCALES[scaleName];
    if (scale.includes(interval)) return 'scale-tone';

    return 'hidden';
}

// DOM Elements
const fretboardEl = document.getElementById('fretboard');
const fretNumbersEl = document.getElementById('fretNumbers');
const currentChordEl = document.getElementById('currentChord');
const chordFunctionEl = document.getElementById('chordFunction');
const scaleNameEl = document.getElementById('scaleName');
const progressionDisplayEl = document.getElementById('progressionDisplay');
const progressionSelectEl = document.getElementById('progressionSelect');
const keySelectEl = document.getElementById('keySelect');
const tempoSliderEl = document.getElementById('tempoSlider');
const tempoValueEl = document.getElementById('tempoValue');
const beatsSelectEl = document.getElementById('beatsSelect');
const playBtnEl = document.getElementById('playBtn');
const stopBtnEl = document.getElementById('stopBtn');
const currentTipEl = document.getElementById('currentTip');

// Theory display elements
const theoryRootEl = document.getElementById('theoryRoot');
const theory3rdEl = document.getElementById('theory3rd');
const theory5thEl = document.getElementById('theory5th');
const theory7thEl = document.getElementById('theory7th');
const theory9thEl = document.getElementById('theory9th');
const theory11thEl = document.getElementById('theory11th');
const theory13thEl = document.getElementById('theory13th');

// Build fretboard
function buildFretboard() {
    fretboardEl.innerHTML = '';
    fretNumbersEl.innerHTML = '';

    // Create strings (high E to low E)
    for (let stringIdx = 0; stringIdx < 6; stringIdx++) {
        const stringEl = document.createElement('div');
        stringEl.className = 'string';

        // Open note
        const openNoteEl = document.createElement('div');
        openNoteEl.className = 'open-note';
        openNoteEl.textContent = GUITAR_STRINGS[stringIdx];
        stringEl.appendChild(openNoteEl);

        // Frets
        for (let fret = 1; fret <= NUM_FRETS; fret++) {
            const fretEl = document.createElement('div');
            fretEl.className = 'fret';

            const noteEl = document.createElement('div');
            noteEl.className = 'note hidden';
            noteEl.dataset.string = stringIdx;
            noteEl.dataset.fret = fret;

            const note = getNoteOnFret(stringIdx, fret);
            noteEl.textContent = note;
            noteEl.dataset.note = note;

            // Click and touch to play note
            noteEl.addEventListener('click', () => playNote(stringIdx, fret));
            noteEl.addEventListener('touchstart', (e) => {
                e.preventDefault();
                playNote(stringIdx, fret);
            }, { passive: false });

            fretEl.appendChild(noteEl);
            stringEl.appendChild(fretEl);
        }

        fretboardEl.appendChild(stringEl);
    }

    // Fret numbers
    const emptyEl = document.createElement('div');
    emptyEl.className = 'fret-number';
    fretNumbersEl.appendChild(emptyEl);

    const markers = [3, 5, 7, 9, 12, 15];
    for (let fret = 1; fret <= NUM_FRETS; fret++) {
        const fretNumEl = document.createElement('div');
        fretNumEl.className = 'fret-number' + (markers.includes(fret) ? ' marker' : '');
        fretNumEl.textContent = fret;
        fretNumbersEl.appendChild(fretNumEl);
    }
}

// Update fretboard highlighting
function updateFretboard() {
    const progression = PROGRESSIONS[state.currentProgression];
    const chordInfo = progression[state.currentChordIndex];
    const chordRoot = transposedNote(state.currentKey, chordInfo.root);
    const chordType = chordInfo.type;

    // Update all notes on fretboard
    const notes = fretboardEl.querySelectorAll('.note');
    notes.forEach(noteEl => {
        const note = noteEl.dataset.note;
        const category = getNoteCategory(note, chordRoot, chordType);

        noteEl.className = 'note ' + category;
    });

    // Update open string indicators too
    const openNotes = fretboardEl.querySelectorAll('.open-note');
    openNotes.forEach((openNoteEl, idx) => {
        const note = GUITAR_STRINGS[idx];
        const category = getNoteCategory(note, chordRoot, chordType);
        openNoteEl.className = 'open-note';
        if (category === 'root') {
            openNoteEl.style.color = 'var(--root-color)';
            openNoteEl.style.fontWeight = '700';
        } else if (category === 'chord-tone') {
            openNoteEl.style.color = 'var(--chord-tone-color)';
            openNoteEl.style.fontWeight = '600';
        } else {
            openNoteEl.style.color = 'var(--text-secondary)';
            openNoteEl.style.fontWeight = '600';
        }
    });
}

// Update chord display
function updateChordDisplay() {
    const progression = PROGRESSIONS[state.currentProgression];
    const chordInfo = progression[state.currentChordIndex];
    const chordRoot = transposedNote(state.currentKey, chordInfo.root);
    const chordType = chordInfo.type;
    const scaleName = CHORD_SCALES[chordType];

    // Chord name
    currentChordEl.textContent = chordRoot + chordType;
    chordFunctionEl.textContent = chordInfo.function;

    // Scale name
    const scaleDisplayNames = {
        'ionian': 'Ionian (Major)',
        'dorian': 'Dorian',
        'phrygian': 'Phrygian',
        'lydian': 'Lydian',
        'mixolydian': 'Mixolydian',
        'aeolian': 'Aeolian (Natural Minor)',
        'locrian': 'Locrian',
        'altered': 'Altered',
        'lydian-dominant': 'Lydian Dominant',
    };
    scaleNameEl.textContent = chordRoot + ' ' + scaleDisplayNames[scaleName];

    // Theory display - chord tones and tensions
    const chord = CHORD_TYPES[chordType];
    theoryRootEl.textContent = chordRoot;
    theory3rdEl.textContent = transposedNote(chordRoot, chord.intervals[1]);
    theory5thEl.textContent = transposedNote(chordRoot, chord.intervals[2]);
    theory7thEl.textContent = transposedNote(chordRoot, chord.intervals[3]);

    // Tensions (9, 11, 13)
    if (chord.tensions.length >= 1) {
        theory9thEl.textContent = transposedNote(chordRoot, chord.tensions[0]);
    }
    if (chord.tensions.length >= 2) {
        theory11thEl.textContent = transposedNote(chordRoot, chord.tensions[1]);
    }
    if (chord.tensions.length >= 3) {
        theory13thEl.textContent = transposedNote(chordRoot, chord.tensions[2]);
    }
}

// Build progression display
function buildProgressionDisplay() {
    progressionDisplayEl.innerHTML = '';

    const progression = PROGRESSIONS[state.currentProgression];
    progression.forEach((chordInfo, index) => {
        const chordRoot = transposedNote(state.currentKey, chordInfo.root);
        const chordEl = document.createElement('div');
        chordEl.className = 'progression-chord' + (index === state.currentChordIndex ? ' active' : '');
        chordEl.textContent = chordRoot + chordInfo.type;
        chordEl.addEventListener('click', () => {
            state.currentChordIndex = index;
            updateAll();
        });
        progressionDisplayEl.appendChild(chordEl);
    });

    // Update tip
    currentTipEl.innerHTML = '<p>' + TIPS[state.currentProgression] + '</p>';
}

// Update all displays
function updateAll() {
    buildProgressionDisplay();
    updateChordDisplay();
    updateFretboard();
}

// Audio - Magenta SoundFont Player
const SOUNDFONT_URL = 'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus';

async function initAudio() {
    if (state.player) return;
    if (state.isLoading) return;

    state.isLoading = true;
    playBtnEl.textContent = 'Loading...';
    playBtnEl.disabled = true;

    try {
        // Create audio context
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Initialize Magenta SoundFont player
        // Program 26 = Electric Guitar (jazz), Program 25 = Acoustic Guitar (steel)
        state.player = new core.SoundFontPlayer(SOUNDFONT_URL, state.audioContext, undefined, undefined, {
            run: (note) => {},
            stop: () => {}
        });

        // Load the soundfont
        await state.player.loadSamples({
            notes: [
                // Preload a range of guitar notes
                ...Array.from({ length: 49 }, (_, i) => ({
                    pitch: 40 + i, // E2 to E6
                    program: 26,   // Jazz Guitar
                    velocity: 80
                }))
            ]
        });

        state.isLoading = false;
        playBtnEl.textContent = 'Play';
        playBtnEl.disabled = false;
    } catch (e) {
        console.error('Failed to load soundfont:', e);
        state.isLoading = false;
        playBtnEl.textContent = 'Play';
        playBtnEl.disabled = false;
    }
}

function playChord(chordRoot, chordType, durationSec) {
    if (!state.player || !state.audioContext) return;

    const chord = CHORD_TYPES[chordType];
    const rootMidi = noteToMidi(chordRoot, 3); // Root in octave 3

    // Create a nice jazz voicing
    // Drop 2 voicing style: Root, 5th, 7th, 3rd (from low to high)
    const voicing = [
        rootMidi,           // Root
        rootMidi + chord.intervals[2],  // 5th
        rootMidi + chord.intervals[3],  // 7th
        rootMidi + chord.intervals[1] + 12,  // 3rd (up an octave)
    ];

    // Add slight strum effect (stagger note starts)
    const now = state.audioContext.currentTime;
    const strumDelay = 0.02; // 20ms between notes for strum feel

    voicing.forEach((pitch, idx) => {
        const noteStart = now + (idx * strumDelay);
        const noteEnd = now + durationSec - 0.1;

        state.player.playNoteDown({
            pitch: pitch,
            velocity: 70 + Math.random() * 20, // Slight velocity variation
            program: 26, // Jazz Guitar
            startTime: noteStart,
            endTime: noteEnd
        });
    });
}

function playNote(stringIndex, fret) {
    // Initialize audio on first interaction
    if (!state.player) {
        initAudio().then(() => {
            if (state.player) {
                playSingleNote(stringIndex, fret);
            }
        });
        return;
    }
    playSingleNote(stringIndex, fret);
}

function playSingleNote(stringIndex, fret) {
    if (!state.player || !state.audioContext) return;

    const midiNote = STRING_MIDI_BASE[stringIndex] + fret;
    const now = state.audioContext.currentTime;

    state.player.playNoteDown({
        pitch: midiNote,
        velocity: 80,
        program: 26, // Jazz Guitar
        startTime: now,
        endTime: now + 0.8
    });
}

function startPlayback() {
    if (state.isPlaying) return;
    if (!state.player) return;

    state.isPlaying = true;
    playBtnEl.textContent = 'Playing...';
    playBtnEl.disabled = true;

    const progression = PROGRESSIONS[state.currentProgression];
    let beatCount = 0;

    // Calculate interval in ms
    const beatDurationMs = (60 / state.tempo) * 1000;
    const chordDurationSec = (60 / state.tempo) * state.beatsPerChord;

    // Play first chord immediately
    state.currentChordIndex = 0;
    updateAll();
    const firstChordInfo = progression[0];
    const firstChordRoot = transposedNote(state.currentKey, firstChordInfo.root);
    playChord(firstChordRoot, firstChordInfo.type, chordDurationSec);
    beatCount = 1;

    state.playbackInterval = setInterval(() => {
        const chordIndex = Math.floor(beatCount / state.beatsPerChord) % progression.length;
        const beatInChord = beatCount % state.beatsPerChord;

        // Play chord on first beat of each chord
        if (beatInChord === 0) {
            state.currentChordIndex = chordIndex;
            updateAll();

            const chordInfo = progression[chordIndex];
            const chordRoot = transposedNote(state.currentKey, chordInfo.root);
            playChord(chordRoot, chordInfo.type, chordDurationSec);
        }

        beatCount++;
    }, beatDurationMs);
}

function stopPlayback() {
    state.isPlaying = false;
    playBtnEl.textContent = 'Play';
    playBtnEl.disabled = false;

    if (state.playbackInterval) {
        clearInterval(state.playbackInterval);
        state.playbackInterval = null;
    }

    // Stop all sounds
    if (state.player) {
        state.player.stop();
    }
}

// Event listeners
progressionSelectEl.addEventListener('change', (e) => {
    state.currentProgression = e.target.value;
    state.currentChordIndex = 0;
    updateAll();
});

keySelectEl.addEventListener('change', (e) => {
    state.currentKey = e.target.value;
    updateAll();
});

tempoSliderEl.addEventListener('input', (e) => {
    state.tempo = parseInt(e.target.value);
    tempoValueEl.textContent = state.tempo;

    // If playing, restart with new tempo
    if (state.isPlaying) {
        stopPlayback();
        startPlayback();
    }
});

beatsSelectEl.addEventListener('change', (e) => {
    state.beatsPerChord = parseInt(e.target.value);

    // If playing, restart with new beats
    if (state.isPlaying) {
        stopPlayback();
        startPlayback();
    }
});

playBtnEl.addEventListener('click', async () => {
    if (!state.player) {
        await initAudio();
    }

    // Resume audio context if suspended (browser autoplay policy)
    if (state.audioContext && state.audioContext.state === 'suspended') {
        await state.audioContext.resume();
    }

    startPlayback();
});

stopBtnEl.addEventListener('click', stopPlayback);

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        if (state.isPlaying) {
            stopPlayback();
        } else {
            if (!state.player) {
                initAudio().then(() => {
                    if (state.audioContext && state.audioContext.state === 'suspended') {
                        state.audioContext.resume();
                    }
                    startPlayback();
                });
            } else {
                if (state.audioContext && state.audioContext.state === 'suspended') {
                    state.audioContext.resume();
                }
                startPlayback();
            }
        }
    }
    if (e.code === 'ArrowRight') {
        const progression = PROGRESSIONS[state.currentProgression];
        state.currentChordIndex = (state.currentChordIndex + 1) % progression.length;
        updateAll();
    }
    if (e.code === 'ArrowLeft') {
        const progression = PROGRESSIONS[state.currentProgression];
        state.currentChordIndex = (state.currentChordIndex - 1 + progression.length) % progression.length;
        updateAll();
    }
});

// Touch swipe gestures for chord navigation
let touchStartX = 0;
let touchStartY = 0;
const SWIPE_THRESHOLD = 50;

document.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
}, { passive: true });

document.addEventListener('touchend', (e) => {
    if (!e.changedTouches.length) return;

    const touchEndX = e.changedTouches[0].clientX;
    const touchEndY = e.changedTouches[0].clientY;
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;

    // Only trigger swipe if horizontal movement is greater than vertical
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > SWIPE_THRESHOLD) {
        const progression = PROGRESSIONS[state.currentProgression];
        if (deltaX < 0) {
            // Swipe left - next chord
            state.currentChordIndex = (state.currentChordIndex + 1) % progression.length;
        } else {
            // Swipe right - previous chord
            state.currentChordIndex = (state.currentChordIndex - 1 + progression.length) % progression.length;
        }
        updateAll();
    }
}, { passive: true });

// Prevent double-tap zoom on buttons
document.querySelectorAll('.btn, .progression-chord, select').forEach(el => {
    el.addEventListener('touchend', (e) => {
        e.preventDefault();
        el.click();
    }, { passive: false });
});

// Initialize
buildFretboard();
updateAll();
