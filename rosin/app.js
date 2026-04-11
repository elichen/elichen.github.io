const STRING_ORDER = ["G", "D", "A", "E"];
const STRING_COLORS = {
    G: "#6a7f55",
    D: "#9b6f37",
    A: "#b34e36",
    E: "#d8bc72"
};
const POSITION_LABELS = ["Open", "Low", "Mid", "High"];
const VIOLIN_STRINGS = [
    { name: "G", midi: 55 },
    { name: "D", midi: 62 },
    { name: "A", midi: 69 },
    { name: "E", midi: 76 }
];
const NOTE_TO_SEMITONE = {
    C: 0,
    D: 2,
    E: 4,
    F: 5,
    G: 7,
    A: 9,
    B: 11
};
const RAIL_BEAT_WIDTH = 78;
const RAIL_MIN_Y = 26;
const RAIL_MAX_Y = 158;
const POSE_SAMPLE_MS = 55;

const SONG_LIBRARY = {
    ode: {
        title: "Ode to Joy",
        composer: "Ludwig van Beethoven",
        prompt: "Shape the four-note climbs as one breath and keep the bow moving through the longer landings.",
        tempo: 88,
        notes: [
            ["B4", 1], ["B4", 1], ["C5", 1], ["D5", 1],
            ["D5", 1], ["C5", 1], ["B4", 1], ["A4", 1],
            ["G4", 1], ["G4", 1], ["A4", 1], ["B4", 1],
            ["B4", 1.5], ["A4", 0.5], ["A4", 2],
            ["B4", 1], ["B4", 1], ["C5", 1], ["D5", 1],
            ["D5", 1], ["C5", 1], ["B4", 1], ["A4", 1],
            ["G4", 1], ["G4", 1], ["A4", 1], ["B4", 1],
            ["A4", 1.5], ["G4", 0.5], ["G4", 2],
            ["A4", 1], ["A4", 1], ["B4", 1], ["G4", 1],
            ["A4", 1], ["B4", 0.5], ["C5", 0.5], ["B4", 1], ["G4", 1],
            ["A4", 1], ["B4", 0.5], ["C5", 0.5], ["B4", 1], ["A4", 1],
            ["G4", 1], ["A4", 1], ["D5", 2],
            ["B4", 1], ["B4", 1], ["C5", 1], ["D5", 1],
            ["D5", 1], ["C5", 1], ["B4", 1], ["A4", 1],
            ["G4", 1], ["G4", 1], ["A4", 1], ["B4", 1],
            ["A4", 1.5], ["G4", 0.5], ["G4", 2]
        ]
    },
    twinkle: {
        title: "Twinkle, Twinkle",
        composer: "Traditional",
        prompt: "This one is about even bow lengths. Let the rail simplify the tune while you focus on smooth, unhurried gestures.",
        tempo: 84,
        notes: [
            ["A4", 1], ["A4", 1], ["E5", 1], ["E5", 1],
            ["F#5", 1], ["F#5", 1], ["E5", 2],
            ["D5", 1], ["D5", 1], ["C#5", 1], ["C#5", 1],
            ["B4", 1], ["B4", 1], ["A4", 2],
            ["E5", 1], ["E5", 1], ["D5", 1], ["D5", 1],
            ["C#5", 1], ["C#5", 1], ["B4", 2],
            ["E5", 1], ["E5", 1], ["D5", 1], ["D5", 1],
            ["C#5", 1], ["C#5", 1], ["B4", 2],
            ["A4", 1], ["A4", 1], ["E5", 1], ["E5", 1],
            ["F#5", 1], ["F#5", 1], ["E5", 2],
            ["D5", 1], ["D5", 1], ["C#5", 1], ["C#5", 1],
            ["B4", 1], ["B4", 1], ["A4", 2]
        ]
    },
    minuet: {
        title: "Minuet in G",
        composer: "Christian Petzold",
        prompt: "Use lighter strokes here. The rail shows the dance contour, but the phrase still needs lift on every turn.",
        tempo: 96,
        notes: [
            ["D5", 1], ["G4", 0.5], ["A4", 0.5], ["B4", 0.5], ["C5", 0.5],
            ["D5", 1], ["G4", 1], ["G4", 1],
            ["E5", 1], ["C5", 0.5], ["D5", 0.5], ["E5", 0.5], ["F#5", 0.5],
            ["G5", 1], ["G4", 1], ["G4", 1],
            ["C5", 1], ["D5", 0.5], ["C5", 0.5], ["B4", 0.5], ["A4", 0.5],
            ["B4", 1], ["C5", 0.5], ["B4", 0.5], ["A4", 0.5], ["G4", 0.5],
            ["F#4", 1], ["G4", 0.5], ["A4", 0.5], ["B4", 0.5], ["G4", 0.5],
            ["A4", 3],
            ["D5", 1], ["G4", 0.5], ["A4", 0.5], ["B4", 0.5], ["C5", 0.5],
            ["D5", 1], ["G4", 1], ["G4", 1],
            ["E5", 1], ["C5", 0.5], ["D5", 0.5], ["E5", 0.5], ["F#5", 0.5],
            ["G5", 1], ["G4", 1], ["G4", 1],
            ["C5", 1], ["D5", 0.5], ["C5", 0.5], ["B4", 0.5], ["A4", 0.5],
            ["B4", 1], ["C5", 0.5], ["B4", 0.5], ["A4", 0.5], ["G4", 0.5],
            ["A4", 1], ["B4", 0.5], ["A4", 0.5], ["G4", 0.5], ["F#4", 0.5],
            ["G4", 3],
            ["B5", 1], ["G5", 0.5], ["A5", 0.5], ["B5", 0.5], ["G5", 0.5],
            ["A5", 1], ["D5", 0.5], ["E5", 0.5], ["F#5", 0.5], ["D5", 0.5],
            ["G5", 1], ["E5", 0.5], ["F#5", 0.5], ["G5", 0.5], ["D5", 0.5],
            ["C#5", 1], ["B4", 0.5], ["C#5", 0.5], ["A4", 1],
            ["A4", 0.5], ["B4", 0.5], ["C#5", 0.5], ["D5", 0.5], ["E5", 0.5], ["F#5", 0.5],
            ["G5", 1], ["F#5", 1], ["E5", 1],
            ["F#5", 1], ["A4", 1], ["C#5", 1],
            ["D5", 3],
            ["D5", 1], ["G4", 0.5], ["F#4", 0.5], ["G4", 1],
            ["E5", 1], ["G4", 0.5], ["F#4", 0.5], ["G4", 1],
            ["D5", 1], ["C5", 1], ["B4", 1],
            ["A4", 0.5], ["G4", 0.5], ["F#4", 0.5], ["G4", 0.5], ["A4", 1],
            ["D4", 0.5], ["E4", 0.5], ["F#4", 0.5], ["G4", 0.5], ["A4", 0.5], ["B4", 0.5],
            ["C5", 1], ["B4", 1], ["A4", 1],
            ["B4", 0.5], ["D5", 0.5], ["G4", 1], ["F#4", 1],
            ["G4", 3]
        ]
    }
};

const elements = {
    songSelect: document.getElementById("songSelect"),
    bootButton: document.getElementById("bootButton"),
    countdownOverlay: document.getElementById("countdownOverlay"),
    trackingOverlay: document.getElementById("trackingOverlay"),
    guideStage: document.getElementById("guideStage"),
    ghostViolin: document.getElementById("ghostViolin"),
    ghostBow: document.getElementById("ghostBow"),
    ghostStop: document.getElementById("ghostStop"),
    ghostTracker: document.getElementById("ghostTracker"),
    video: document.getElementById("video"),
    cameraStatus: document.getElementById("cameraStatus"),
    noteLabel: document.getElementById("noteLabel"),
    stringLabel: document.getElementById("stringLabel"),
    positionLabel: document.getElementById("positionLabel"),
    railViewport: document.getElementById("railViewport"),
    railTrack: document.getElementById("railTrack")
};

const trackingCtx = elements.trackingOverlay.getContext("2d");

const state = {
    booted: false,
    booting: false,
    detector: null,
    stream: null,
    poseAvailable: false,
    cameraError: "",
    poseLoopActive: false,
    renderLoopActive: false,
    lastPoseSampleAt: 0,
    lastFrameAt: performance.now(),
    transportBeat: 0,
    isPlaying: false,
    startedAt: 0,
    lastBeatTick: -1,
    countdownToken: 0,
    countdownActive: false,
    selectedSongId: "ode",
    currentPiece: null,
    flourish: 0,
    pose: {
        points: {},
        lastPoseAt: 0,
        trackingConfidence: 0,
        bowEnergy: 0,
        posture: 0,
        stringIndex: 1,
        positionIndex: 0,
        stringLabel: "D",
        positionLabel: "Open",
        bowDirection: 1,
        previousRightWrist: null
    },
    stringAccuracy: 0,
    positionAccuracy: 0,
    audio: null
};

class ViolinAudioEngine {
    constructor() {
        this.ctx = null;
        this.master = null;
        this.bodyGain = null;
        this.guideGain = null;
        this.noiseGain = null;
        this.filter = null;
        this.guideFilter = null;
        this.bodyOsc1 = null;
        this.bodyOsc2 = null;
        this.guideOsc = null;
        this.vibratoOsc = null;
        this.vibratoDepth1 = null;
        this.vibratoDepth2 = null;
        this.vibratoDepthGuide = null;
        this.ready = false;
    }

    async init() {
        if (this.ready) {
            return;
        }

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioContextClass();

        this.master = this.ctx.createGain();
        this.master.gain.value = 0.92;
        this.master.connect(this.ctx.destination);

        this.filter = this.ctx.createBiquadFilter();
        this.filter.type = "lowpass";
        this.filter.frequency.value = 1600;
        this.filter.Q.value = 1.05;

        this.guideFilter = this.ctx.createBiquadFilter();
        this.guideFilter.type = "lowpass";
        this.guideFilter.frequency.value = 900;
        this.guideFilter.Q.value = 0.6;

        this.bodyGain = this.ctx.createGain();
        this.bodyGain.gain.value = 0;
        this.bodyGain.connect(this.filter);
        this.filter.connect(this.master);

        this.guideGain = this.ctx.createGain();
        this.guideGain.gain.value = 0;
        this.guideGain.connect(this.guideFilter);
        this.guideFilter.connect(this.master);

        this.noiseGain = this.ctx.createGain();
        this.noiseGain.gain.value = 0;
        const noiseFilter = this.ctx.createBiquadFilter();
        noiseFilter.type = "bandpass";
        noiseFilter.frequency.value = 2400;
        noiseFilter.Q.value = 1.6;
        this.noiseGain.connect(noiseFilter);
        noiseFilter.connect(this.master);

        this.bodyOsc1 = this.ctx.createOscillator();
        this.bodyOsc1.type = "sawtooth";
        this.bodyOsc2 = this.ctx.createOscillator();
        this.bodyOsc2.type = "triangle";
        this.bodyOsc2.detune.value = 7;
        this.guideOsc = this.ctx.createOscillator();
        this.guideOsc.type = "sine";

        this.vibratoOsc = this.ctx.createOscillator();
        this.vibratoOsc.type = "sine";
        this.vibratoOsc.frequency.value = 5.4;
        this.vibratoDepth1 = this.ctx.createGain();
        this.vibratoDepth2 = this.ctx.createGain();
        this.vibratoDepthGuide = this.ctx.createGain();
        this.vibratoDepth1.gain.value = 0;
        this.vibratoDepth2.gain.value = 0;
        this.vibratoDepthGuide.gain.value = 0;
        this.vibratoOsc.connect(this.vibratoDepth1);
        this.vibratoOsc.connect(this.vibratoDepth2);
        this.vibratoOsc.connect(this.vibratoDepthGuide);
        this.vibratoDepth1.connect(this.bodyOsc1.frequency);
        this.vibratoDepth2.connect(this.bodyOsc2.frequency);
        this.vibratoDepthGuide.connect(this.guideOsc.frequency);

        this.bodyOsc1.connect(this.bodyGain);
        this.bodyOsc2.connect(this.bodyGain);
        this.guideOsc.connect(this.guideGain);

        const noiseSource = this.ctx.createBufferSource();
        noiseSource.buffer = this.createNoiseBuffer();
        noiseSource.loop = true;
        noiseSource.connect(this.noiseGain);

        this.bodyOsc1.start();
        this.bodyOsc2.start();
        this.guideOsc.start();
        this.vibratoOsc.start();
        noiseSource.start();

        this.ready = true;
    }

    createNoiseBuffer() {
        const buffer = this.ctx.createBuffer(1, this.ctx.sampleRate * 2, this.ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < data.length; i += 1) {
            data[i] = (Math.random() * 2 - 1) * 0.42;
        }
        return buffer;
    }

    async resume() {
        if (this.ctx && this.ctx.state !== "running") {
            await this.ctx.resume();
        }
    }

    update({ midi, bowEnergy, guideLevel, brightness, vibrato }) {
        if (!this.ready) {
            return;
        }

        const now = this.ctx.currentTime;
        if (midi == null) {
            this.bodyGain.gain.setTargetAtTime(0, now, 0.06);
            this.guideGain.gain.setTargetAtTime(0, now, 0.08);
            this.noiseGain.gain.setTargetAtTime(0, now, 0.04);
            return;
        }

        const freq = 440 * Math.pow(2, (midi - 69) / 12);
        this.bodyOsc1.frequency.setTargetAtTime(freq, now, 0.018);
        this.bodyOsc2.frequency.setTargetAtTime(freq * 1.0024, now, 0.018);

        this.filter.frequency.setTargetAtTime(1150 + brightness * 2700, now, 0.08);

        this.vibratoDepth1.gain.setTargetAtTime(0.5 + vibrato * 3.2, now, 0.08);
        this.vibratoDepth2.gain.setTargetAtTime(0.6 + vibrato * 4.4, now, 0.08);
        this.vibratoDepthGuide.gain.setTargetAtTime(0, now, 0.08);

        this.bodyGain.gain.setTargetAtTime(bowEnergy * 0.25, now, 0.045);
        this.guideGain.gain.setTargetAtTime(0, now, 0.06);
        this.noiseGain.gain.setTargetAtTime(Math.max(0, bowEnergy - 0.08) * 0.014, now, 0.04);
    }

    tick(isDownbeat) {
        if (!this.ready) {
            return;
        }

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        const now = this.ctx.currentTime;

        osc.type = "triangle";
        osc.frequency.value = isDownbeat ? 920 : 660;
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(isDownbeat ? 0.035 : 0.022, now + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

        osc.connect(gain);
        gain.connect(this.master);
        osc.start(now);
        osc.stop(now + 0.14);
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function smoothStep(edge0, edge1, value) {
    const t = clamp((value - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
}

function noteNameToMidi(noteName) {
    const match = /^([A-G])([#b]?)(\d)$/.exec(noteName);
    if (!match) {
        throw new Error(`Invalid note name: ${noteName}`);
    }

    const [, letter, accidental, octaveString] = match;
    let semitone = NOTE_TO_SEMITONE[letter];
    if (accidental === "#") {
        semitone += 1;
    } else if (accidental === "b") {
        semitone -= 1;
    }

    const octave = Number(octaveString);
    return 12 + (octave * 12) + semitone;
}

function deriveTargetFromMidi(midi) {
    const candidates = VIOLIN_STRINGS.filter((stringInfo) => midi >= stringInfo.midi && midi - stringInfo.midi <= 12);
    const best = (candidates.length ? candidates : [VIOLIN_STRINGS[0]])
        .slice()
        .sort((a, b) => (midi - a.midi) - (midi - b.midi))[0];
    const delta = midi - best.midi;

    let positionIndex = 0;
    if (delta <= 1) {
        positionIndex = 0;
    } else if (delta <= 4) {
        positionIndex = 1;
    } else if (delta <= 7) {
        positionIndex = 2;
    } else {
        positionIndex = 3;
    }

    return {
        string: best.name,
        positionIndex,
        position: POSITION_LABELS[positionIndex]
    };
}

function createPiece(songId) {
    const song = SONG_LIBRARY[songId];
    let beatCursor = 0;
    const events = song.notes.map(([noteName, duration], index) => {
        const midi = noteNameToMidi(noteName);
        const target = deriveTargetFromMidi(midi);
        const event = {
            id: `${songId}-${index}`,
            noteName,
            midi,
            duration,
            startBeat: beatCursor,
            endBeat: beatCursor + duration,
            string: target.string,
            positionIndex: target.positionIndex,
            position: target.position,
            coverage: 0,
            playedSeconds: 0,
            element: null
        };
        beatCursor += duration;
        return event;
    });

    const midiValues = events.map((event) => event.midi);
    return {
        id: songId,
        title: song.title,
        composer: song.composer,
        prompt: song.prompt,
        tempo: song.tempo,
        totalBeats: beatCursor,
        minMidi: Math.min(...midiValues),
        maxMidi: Math.max(...midiValues),
        events
    };
}

function getActiveEvent() {
    if (!state.currentPiece) {
        return null;
    }

    return state.currentPiece.events.find((event) => state.transportBeat >= event.startBeat && state.transportBeat < event.endBeat) || null;
}

function getEventAtBeat(beat) {
    return state.currentPiece.events.find((event) => beat >= event.startBeat && beat < event.endBeat) || null;
}

function buildSongSelect() {
    elements.songSelect.innerHTML = "";
    Object.entries(SONG_LIBRARY).forEach(([id, song]) => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = song.title;
        elements.songSelect.appendChild(option);
    });
}

function setTransportBeat(nextBeat) {
    state.transportBeat = clamp(nextBeat, 0, state.currentPiece.totalBeats);
}

function applySong(songId) {
    state.selectedSongId = songId;
    state.currentPiece = createPiece(songId);
    state.tempo = state.currentPiece.tempo;
    elements.songSelect.value = songId;
    resetPerformanceState();
    buildRail();
    updateTargetCard(getEventAtBeat(0));
    updateGuideVisual(getEventAtBeat(0));
}

function resetPerformanceState() {
    if (!state.currentPiece) {
        return;
    }

    cancelCountdown();
    state.isPlaying = false;
    state.startedAt = 0;
    state.lastBeatTick = -1;
    setTransportBeat(0);
    state.flourish = 0;
    state.stringAccuracy = 0;
    state.positionAccuracy = 0;
    state.currentPiece.events.forEach((event) => {
        event.coverage = 0;
        event.playedSeconds = 0;
    });
    updateRail();
    if (state.audio) {
        state.audio.update({ midi: null, bowEnergy: 0, guideLevel: 0, brightness: 0, vibrato: 0 });
    }
}

function buildRail() {
    elements.railTrack.innerHTML = "";
    const piece = state.currentPiece;
    const noteRange = Math.max(1, piece.maxMidi - piece.minMidi);
    const width = piece.totalBeats * RAIL_BEAT_WIDTH + 320;

    elements.railTrack.style.width = `${width}px`;

    piece.events.forEach((event) => {
        const block = document.createElement("div");
        const yRatio = (piece.maxMidi - event.midi) / noteRange;
        const top = RAIL_MIN_Y + yRatio * (RAIL_MAX_Y - RAIL_MIN_Y);
        const widthPx = Math.max(48, event.duration * RAIL_BEAT_WIDTH - 10);

        block.className = "note-block";
        block.style.setProperty("--note-color", STRING_COLORS[event.string]);
        block.style.left = `${event.startBeat * RAIL_BEAT_WIDTH + 160}px`;
        block.style.top = `${top}px`;
        block.style.width = `${widthPx}px`;
        block.innerHTML = `<span class="note-block__name">${event.noteName}</span><span class="note-block__string">${event.string}</span>`;
        event.element = block;
        elements.railTrack.appendChild(block);
    });
}

function updateTargetCard(event) {
    if (!event) {
        elements.noteLabel.textContent = "-";
        elements.stringLabel.textContent = "String -";
        elements.positionLabel.textContent = "Position -";
        return;
    }

    elements.noteLabel.textContent = event.noteName;
    elements.stringLabel.textContent = `${event.string} string`;
    elements.positionLabel.textContent = `${event.position} position`;
}

function mapVideoPointToStage(point) {
    const videoWidth = elements.video.videoWidth || elements.trackingOverlay.width;
    const videoHeight = elements.video.videoHeight || elements.trackingOverlay.height;
    const stageWidth = elements.guideStage.clientWidth;
    const stageHeight = elements.guideStage.clientHeight;

    if (!point || !videoWidth || !videoHeight || !stageWidth || !stageHeight) {
        return null;
    }

    const scale = Math.max(stageWidth / videoWidth, stageHeight / videoHeight);
    const renderedWidth = videoWidth * scale;
    const renderedHeight = videoHeight * scale;
    const offsetX = (stageWidth - renderedWidth) / 2;
    const offsetY = (stageHeight - renderedHeight) / 2;

    return {
        x: stageWidth - (point.x * scale + offsetX),
        y: point.y * scale + offsetY
    };
}

function updateGuideVisual(event, now = performance.now()) {
    const targetString = event?.string || state.pose.stringLabel || "A";
    const targetPositionIndex = event?.positionIndex ?? state.pose.positionIndex ?? 0;
    const trackingLive = state.poseAvailable && state.pose.trackingConfidence >= 0.22;
    const liveString = trackingLive ? state.pose.stringLabel : targetString;
    const liveStringIndex = Math.max(0, STRING_ORDER.indexOf(liveString));
    const livePositionIndex = trackingLive ? state.pose.positionIndex : targetPositionIndex;
    const idleWave = Math.sin(now / 180);
    const bowDirection = trackingLive ? state.pose.bowDirection : Math.sign(idleWave) || 1;
    const bowAmplitude = trackingLive
        ? 8 + state.pose.bowEnergy * 46
        : (state.isPlaying ? 12 + state.pose.bowEnergy * 32 : 6);
    const bowShift = bowDirection * bowAmplitude;
    const bowTilt = -19 + liveStringIndex * 4 + bowDirection * (trackingLive ? 2.2 : 1.4);
    const stopPosition = 48 + targetPositionIndex * 6.5;
    const trackerLeft = 48 + livePositionIndex * 6.5;
    const trackerTop = 84 + liveStringIndex * 8;
    const stageWidth = elements.guideStage.clientWidth || window.innerWidth;
    const stageHeight = elements.guideStage.clientHeight || window.innerHeight;
    const fallbackScale = clamp(stageWidth / 1500, 0.68, 0.96);
    let anchor = { x: stageWidth * 0.43, y: stageHeight * 0.56 };
    let neckEnd = { x: anchor.x + 340 * fallbackScale, y: anchor.y - 54 * fallbackScale };
    let violinRotation = -12;
    let violinScale = fallbackScale;
    let bowAngle = -9 + bowDirection * 1.2;
    let bowScale = clamp(fallbackScale * 0.92, 0.58, 1);

    if (trackingLive) {
        const leftShoulder = state.pose.points.left_shoulder;
        const rightShoulder = state.pose.points.right_shoulder;
        const leftWrist = state.pose.points.left_wrist;
        const rightWrist = state.pose.points.right_wrist;
        if (leftShoulder && rightShoulder) {
            const anchorRaw = {
                x: lerp(leftShoulder.x, rightShoulder.x, 0.18),
                y: lerp(leftShoulder.y, rightShoulder.y, 0.34)
            };
            const fallbackNeckRaw = {
                x: leftShoulder.x + Math.max(36, distance(leftShoulder, rightShoulder) * 1.25),
                y: leftShoulder.y - 12
            };
            const neckRaw = leftWrist
                ? {
                    x: lerp(leftShoulder.x, leftWrist.x, 0.82),
                    y: lerp(leftShoulder.y, leftWrist.y, 0.82)
                }
                : fallbackNeckRaw;
            const mappedAnchor = mapVideoPointToStage(anchorRaw);
            const mappedNeck = mapVideoPointToStage(neckRaw);
            if (mappedAnchor && mappedNeck) {
                anchor = mappedAnchor;
                neckEnd = mappedNeck;
                violinRotation = Math.atan2(neckEnd.y - anchor.y, neckEnd.x - anchor.x) * (180 / Math.PI);
                violinScale = clamp(distance(anchor, neckEnd) / 360, 0.54, 1.08);
            }

            if (rightWrist) {
                const mappedRightWrist = mapVideoPointToStage(rightWrist);
                if (mappedRightWrist) {
                    const bowTarget = {
                        x: lerp(anchor.x, neckEnd.x, 0.24),
                        y: lerp(anchor.y, neckEnd.y, 0.24) + (liveStringIndex - 1.5) * 10 * violinScale
                    };
                    bowAngle = Math.atan2(bowTarget.y - mappedRightWrist.y, bowTarget.x - mappedRightWrist.x) * (180 / Math.PI);
                    bowScale = clamp(distance(mappedRightWrist, bowTarget) / 604, 0.42, 1.08);
                    const bowX = mappedRightWrist.x - 36 * bowScale;
                    const bowY = mappedRightWrist.y - 6 * bowScale;
                    elements.ghostBow.style.transform = `translate(${bowX.toFixed(1)}px, ${bowY.toFixed(1)}px) rotate(${bowAngle.toFixed(1)}deg) scale(${bowScale.toFixed(3)})`;
                }
            }
        }
    }

    const violinX = anchor.x - 124 * violinScale;
    const violinY = anchor.y - 96 * violinScale;
    elements.ghostViolin.style.transform = `translate(${violinX.toFixed(1)}px, ${violinY.toFixed(1)}px) rotate(${violinRotation.toFixed(1)}deg) scale(${violinScale.toFixed(3)})`;

    if (!(trackingLive && state.pose.points.right_wrist)) {
        const bowX = violinX - 22 * violinScale + bowShift;
        const bowY = violinY + 128 * violinScale;
        elements.ghostBow.style.transform = `translate(${bowX.toFixed(1)}px, ${bowY.toFixed(1)}px) rotate(${(violinRotation + bowTilt).toFixed(1)}deg) scale(${bowScale.toFixed(3)})`;
    }

    elements.guideStage.dataset.string = targetString;
    elements.guideStage.dataset.visible = trackingLive ? "true" : "false";
    elements.guideStage.style.setProperty("--accent-color", STRING_COLORS[targetString] || STRING_COLORS.A);
    elements.guideStage.style.setProperty("--stop-position", `${stopPosition.toFixed(1)}%`);
    elements.guideStage.style.setProperty("--tracker-left", `${trackerLeft.toFixed(1)}%`);
    elements.guideStage.style.setProperty("--tracker-top", `${trackerTop.toFixed(1)}px`);
    elements.guideStage.style.setProperty("--tracker-opacity", trackingLive ? "1" : "0.18");
}

function updateRail() {
    const playheadOffset = elements.railViewport.clientWidth * 0.31;
    const translate = playheadOffset - (state.transportBeat * RAIL_BEAT_WIDTH + 160);
    elements.railTrack.style.transform = `translateX(${translate}px)`;

    state.currentPiece.events.forEach((event) => {
        if (!event.element) {
            return;
        }

        event.element.style.setProperty("--fill", event.coverage.toFixed(3));
        event.element.classList.toggle("active", state.transportBeat >= event.startBeat && state.transportBeat < event.endBeat);
        event.element.classList.toggle("done", state.transportBeat >= event.endBeat);
    });
}

function syncTrackingOverlaySize() {
    const width = elements.video.videoWidth || elements.video.clientWidth;
    const height = elements.video.videoHeight || elements.video.clientHeight;
    if (!width || !height) {
        return;
    }

    if (elements.trackingOverlay.width !== width) {
        elements.trackingOverlay.width = width;
    }
    if (elements.trackingOverlay.height !== height) {
        elements.trackingOverlay.height = height;
    }
}

function drawTrackingPoint(point, radius, fillStyle, strokeStyle = "rgba(255, 247, 232, 0.78)") {
    trackingCtx.beginPath();
    trackingCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    trackingCtx.fillStyle = fillStyle;
    trackingCtx.fill();
    trackingCtx.lineWidth = Math.max(1.5, radius * 0.22);
    trackingCtx.strokeStyle = strokeStyle;
    trackingCtx.stroke();
}

function drawTrackingOverlay() {
    syncTrackingOverlaySize();
    const { width, height } = elements.trackingOverlay;
    trackingCtx.clearRect(0, 0, width, height);

    if (!state.poseAvailable) {
        return;
    }

    const { left_wrist: leftWrist, right_wrist: rightWrist } = state.pose.points;
    const activeEvent = getActiveEvent();
    const accent = STRING_COLORS[activeEvent?.string || state.pose.stringLabel] || STRING_COLORS.A;

    if (leftWrist) {
        drawTrackingPoint(leftWrist, 10, accent);
    }
    if (rightWrist) {
        drawTrackingPoint(rightWrist, 10, "rgba(255, 242, 215, 0.86)");
    }
}

async function setupCamera() {
    state.stream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    });

    elements.video.srcObject = state.stream;

    await new Promise((resolve) => {
        elements.video.onloadedmetadata = () => {
            elements.video.play();
            syncTrackingOverlaySize();
            resolve();
        };
    });
}

async function setupDetector() {
    const { tf, poseDetection } = window;
    await tf.setBackend("webgl");
    await tf.ready();
    state.detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
        }
    );
}

function blendPoint(previous, next) {
    if (!next) {
        return previous ? { ...previous, score: previous.score * 0.88 } : null;
    }

    if (!previous) {
        return { ...next };
    }

    return {
        x: lerp(previous.x, next.x, 0.36),
        y: lerp(previous.y, next.y, 0.36),
        score: lerp(previous.score, next.score, 0.48)
    };
}

function extractNamedKeypoints(pose) {
    const named = {};
    (pose?.keypoints || []).forEach((keypoint) => {
        const key = keypoint.name || keypoint.part;
        if (key && keypoint.score > 0.18) {
            named[key] = { x: keypoint.x, y: keypoint.y, score: keypoint.score };
        }
    });
    return named;
}

function processPose(poses) {
    const pose = poses[0];
    const named = extractNamedKeypoints(pose);
    const trackedNames = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"];

    trackedNames.forEach((name) => {
        state.pose.points[name] = blendPoint(state.pose.points[name], named[name] || null);
    });

    const leftShoulder = state.pose.points.left_shoulder;
    const rightShoulder = state.pose.points.right_shoulder;
    const leftWrist = state.pose.points.left_wrist;
    const rightWrist = state.pose.points.right_wrist;

    state.pose.lastPoseAt = performance.now();

    if (!(leftShoulder && rightShoulder)) {
        state.pose.trackingConfidence = 0;
        state.pose.bowEnergy = lerp(state.pose.bowEnergy, 0, 0.3);
        state.pose.posture = lerp(state.pose.posture, 0, 0.25);
        return;
    }

    const shoulderSpan = Math.max(30, distance(leftShoulder, rightShoulder));
    const trackingConfidence = [leftShoulder, rightShoulder, leftWrist, rightWrist]
        .filter(Boolean)
        .reduce((sum, point) => sum + point.score, 0) / 4;

    let stringIndex = state.pose.stringIndex;
    let positionIndex = state.pose.positionIndex;
    let posture = state.pose.posture;

    if (leftWrist) {
        const relativeLift = clamp((leftShoulder.y - leftWrist.y) / (shoulderSpan * 1.1), -0.55, 1.1);
        const stringFloat = clamp((relativeLift + 0.24) / 1.12, 0, 1) * 3;
        stringIndex = clamp(Math.round(stringFloat), 0, 3);

        const leftReach = clamp((distance(leftShoulder, leftWrist) / shoulderSpan - 0.22) / 0.95, 0, 1);
        positionIndex = clamp(Math.floor(leftReach * 4.0), 0, 3);

        const holdDistance = distance(leftShoulder, leftWrist) / shoulderSpan;
        const holdBand = 1 - clamp(Math.abs(holdDistance - 0.68) / 0.72, 0, 1);
        const heightBand = 1 - clamp(Math.abs((leftWrist.y - leftShoulder.y) / shoulderSpan - 0.08) / 0.95, 0, 1);
        posture = clamp(0.08 + 0.54 * holdBand + 0.38 * heightBand, 0, 1);
    } else {
        posture = lerp(posture, 0, 0.2);
    }

    let bowEnergy = state.pose.bowEnergy;
    let bowDirection = state.pose.bowDirection;
    if (rightWrist) {
        const previousRight = state.pose.previousRightWrist;
        if (previousRight) {
            const dt = Math.max(0.016, (state.pose.lastPoseAt - previousRight.t) / 1000);
            const normDx = (rightWrist.x - previousRight.x) / (shoulderSpan * dt);
            const normDy = (rightWrist.y - previousRight.y) / (shoulderSpan * dt);
            const horizontalBias = Math.abs(normDx) / (Math.abs(normDx) + Math.abs(normDy) + 0.001);
            const laneCenter = lerp(leftShoulder.y, rightShoulder.y, 0.42);
            const laneScore = 1 - clamp(Math.abs((rightWrist.y - laneCenter) / (shoulderSpan * 1.25)), 0, 1);
            const energyRaw = smoothStep(0.4, 2.6, Math.abs(normDx)) * (0.22 + horizontalBias * 0.78) * (0.26 + laneScore * 0.74);
            bowEnergy = lerp(bowEnergy, energyRaw, 0.42);
            if (Math.abs(normDx) > 0.08) {
                bowDirection = normDx >= 0 ? 1 : -1;
            }
        } else {
            bowEnergy = lerp(bowEnergy, 0, 0.18);
        }

        state.pose.previousRightWrist = { x: rightWrist.x, y: rightWrist.y, t: state.pose.lastPoseAt };
    } else {
        bowEnergy = lerp(bowEnergy, 0, 0.24);
        state.pose.previousRightWrist = null;
    }

    state.pose.trackingConfidence = trackingConfidence;
    state.pose.bowEnergy = clamp(bowEnergy, 0, 1);
    state.pose.posture = clamp(posture * trackingConfidence, 0, 1);
    state.pose.stringIndex = stringIndex;
    state.pose.positionIndex = positionIndex;
    state.pose.stringLabel = STRING_ORDER[stringIndex];
    state.pose.positionLabel = POSITION_LABELS[positionIndex];
    state.pose.bowDirection = bowDirection;
}

async function poseLoop() {
    if (!state.poseLoopActive) {
        return;
    }

    const now = performance.now();
    if (state.detector && elements.video.readyState >= 2 && now - state.lastPoseSampleAt >= POSE_SAMPLE_MS) {
        state.lastPoseSampleAt = now;
        try {
            const poses = await state.detector.estimatePoses(elements.video, { maxPoses: 1, flipHorizontal: false });
            processPose(poses);
        } catch (error) {
            console.error("Pose estimation failed", error);
        }
    }

    window.requestAnimationFrame(poseLoop);
}

function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function cancelCountdown() {
    state.countdownToken += 1;
    state.countdownActive = false;
    elements.countdownOverlay.textContent = "";
    elements.countdownOverlay.classList.add("hidden");
}

async function runCountdown(seconds = 3) {
    const token = ++state.countdownToken;
    state.countdownActive = true;
    elements.countdownOverlay.classList.remove("hidden");

    for (let count = seconds; count >= 1; count -= 1) {
        if (token !== state.countdownToken) {
            return false;
        }

        elements.countdownOverlay.textContent = String(count);
        elements.cameraStatus.textContent = `Starting in ${count}...`;
        state.audio?.tick(count === 1);
        await sleep(700);
    }

    if (token !== state.countdownToken) {
        return false;
    }

    elements.countdownOverlay.textContent = "Play";
    elements.cameraStatus.textContent = "Follow the rail and wake each note with the bowing motion.";
    await sleep(260);

    if (token !== state.countdownToken) {
        return false;
    }

    state.countdownActive = false;
    elements.countdownOverlay.textContent = "";
    elements.countdownOverlay.classList.add("hidden");
    return true;
}

function updateCoaching(event) {
    if (!state.booted) {
        elements.cameraStatus.textContent = "Press Start once to allow camera and sound. A short countdown will follow.";
        return;
    }

    if (state.booting || state.countdownActive) {
        return;
    }

    if (!state.poseAvailable) {
        if (event) {
            elements.cameraStatus.textContent = "Camera unavailable. Guided playback is running, but pose tracking is off. Allow camera access and reload to use body control.";
        } else {
            elements.cameraStatus.textContent = "Camera unavailable. Change the piece any time.";
        }
        return;
    }

    if (state.pose.trackingConfidence < 0.22) {
        elements.cameraStatus.textContent = "Step back until both shoulders and wrists are visible. Front-facing light helps the tracker lock on.";
        return;
    }

    if (!event) {
        elements.cameraStatus.textContent = "Listening for your pose. Change the piece any time.";
        return;
    }

    if (state.pose.posture < 0.24) {
        elements.cameraStatus.textContent = `Raise your left wrist toward your collarbone and let it hover there. That gives the tracker a stable ${event.string}-string silhouette.`;
        return;
    }

    if (state.stringAccuracy < 0.55) {
        const direction = state.pose.stringIndex < STRING_ORDER.indexOf(event.string) ? "higher" : "lower";
        elements.cameraStatus.textContent = `Slide the left wrist ${direction} to find the ${event.string} string. The rail pitch is fixed; the string color still matters.`;
        return;
    }

    if (state.positionAccuracy < 0.55) {
        const direction = state.pose.positionIndex < event.positionIndex ? "farther from" : "closer to";
        elements.cameraStatus.textContent = `Reach ${direction} your shoulder to land in ${event.position.toLowerCase()} position.`;
        return;
    }

    if (state.pose.bowEnergy < 0.18) {
        elements.cameraStatus.textContent = "Sweep the right hand side to side like a slow paint stroke. The note only blooms when the bow moves.";
        return;
    }

    if (state.pose.bowEnergy > 0.72) {
        elements.cameraStatus.textContent = "That is the feel. Keep the bow long and even, then let the next target pull your left hand into place.";
        return;
    }

    elements.cameraStatus.textContent = state.currentPiece.prompt;
}

function stepPerformance(now) {
    const dt = Math.min(0.05, (now - state.lastFrameAt) / 1000);
    state.lastFrameAt = now;

    if (state.isPlaying) {
        const msPerBeat = 60000 / state.tempo;
        setTransportBeat((now - state.startedAt) / msPerBeat);

        if (state.transportBeat >= state.currentPiece.totalBeats) {
            resetLoop(now);
        } else {
            const beatNumber = Math.floor(state.transportBeat);
            if (beatNumber !== state.lastBeatTick) {
                state.audio?.tick(beatNumber % 4 === 0);
                state.lastBeatTick = beatNumber;
            }
        }
    }

    const event = getActiveEvent();
    updateTargetCard(event);

    if (event) {
        const targetStringIndex = STRING_ORDER.indexOf(event.string);
        const stringDistance = Math.abs(state.pose.stringIndex - targetStringIndex);
        const positionDistance = Math.abs(state.pose.positionIndex - event.positionIndex);
        state.stringAccuracy = clamp(1 - stringDistance / 3, 0, 1);
        state.positionAccuracy = clamp(1 - positionDistance / 3, 0, 1);

        const engagement = state.pose.posture * (0.18 + state.pose.bowEnergy * 0.82);
        const phraseQuality = engagement * (0.52 * state.stringAccuracy + 0.28 * state.positionAccuracy + 0.2);
        const noteDurationSeconds = (event.duration * 60) / state.tempo;

        if (state.poseAvailable) {
            event.playedSeconds = clamp(event.playedSeconds + phraseQuality * dt, 0, noteDurationSeconds);
            event.coverage = clamp(event.playedSeconds / noteDurationSeconds, 0, 1);

            const flourishDelta = (state.pose.bowEnergy * state.pose.posture * state.stringAccuracy * (0.5 + 0.5 * state.positionAccuracy) - 0.12) * 30 * dt;
            state.flourish = clamp(state.flourish + flourishDelta, 0, 100);
        } else {
            const phase = clamp((state.transportBeat - event.startBeat) / Math.max(event.duration, 0.01), 0, 1);
            event.playedSeconds = phase * noteDurationSeconds;
            event.coverage = phase;
            state.flourish = lerp(state.flourish, 22, 0.08);
        }
    } else {
        state.stringAccuracy = lerp(state.stringAccuracy, 0, 0.2);
        state.positionAccuracy = lerp(state.positionAccuracy, 0, 0.2);
        state.flourish = clamp(state.flourish - dt * 10, 0, 100);
    }

    const brightness = clamp(0.22 + 0.48 * state.stringAccuracy + 0.3 * (state.flourish / 100), 0, 1);
    const motionLevel = state.isPlaying && event
        ? state.pose.bowEnergy * state.pose.posture * (0.34 + 0.66 * state.stringAccuracy)
        : 0;
    const bodyLevel = state.isPlaying && event
        ? (state.poseAvailable ? motionLevel : 0.12)
        : 0;
    const guideLevel = state.isPlaying && event
        ? (state.poseAvailable ? 0.015 + (1 - bodyLevel) * 0.028 : 0.05)
        : 0;
    const vibrato = clamp(0.12 + state.flourish / 120, 0, 1);

    state.audio?.update({
        midi: state.isPlaying && event ? event.midi : null,
        bowEnergy: bodyLevel,
        guideLevel,
        brightness,
        vibrato
    });

    drawTrackingOverlay();
    updateGuideVisual(event, now);
    updateRail();
    updateCoaching(event);
}

function renderLoop(now) {
    stepPerformance(now);
    window.requestAnimationFrame(renderLoop);
}

function resetLoop(now = performance.now()) {
    state.currentPiece.events.forEach((event) => {
        event.coverage = 0;
        event.playedSeconds = 0;
    });
    state.flourish = 0;
    setTransportBeat(0);
    state.startedAt = now;
    state.lastBeatTick = -1;
}

async function boot() {
    if (state.booting || state.booted) {
        return;
    }

    state.booting = true;
    elements.bootButton.disabled = true;
    elements.cameraStatus.textContent = "Warming the violin engine...";

    try {
        state.audio = new ViolinAudioEngine();
        await state.audio.init();
        await state.audio.resume();
        state.poseAvailable = false;
        state.cameraError = "";

        try {
            elements.cameraStatus.textContent = "Opening camera...";
            await setupCamera();
            elements.cameraStatus.textContent = "Loading MoveNet pose detector...";
            await setupDetector();
            state.poseAvailable = true;
        } catch (error) {
            console.warn("Camera or pose setup failed", error);
            state.cameraError = error.message;
            state.pose.trackingConfidence = 0;
            state.pose.bowEnergy = 0;
            state.pose.posture = 0;
            state.pose.previousRightWrist = null;
        }

        state.booted = true;
        elements.cameraStatus.textContent = state.poseAvailable
            ? "Camera live. Starting countdown..."
            : "Camera unavailable. Starting guided playback without pose tracking.";
        elements.bootButton.classList.add("hidden");

        if (state.poseAvailable && !state.poseLoopActive) {
            state.poseLoopActive = true;
            window.requestAnimationFrame(poseLoop);
        }

        if (!state.renderLoopActive) {
            state.renderLoopActive = true;
            window.requestAnimationFrame(renderLoop);
        }
    } catch (error) {
        console.error(error);
        elements.cameraStatus.textContent = `Setup failed: ${error.message}`;
        elements.bootButton.disabled = false;
    } finally {
        state.booting = false;
    }
}

async function startPlayback() {
    if (!state.booted) {
        return;
    }

    cancelCountdown();
    await state.audio.resume();
    resetLoop();
    const canStart = await runCountdown(3);
    if (!canStart) {
        return;
    }

    state.startedAt = performance.now();
    state.lastBeatTick = -1;
    state.isPlaying = true;
    elements.cameraStatus.textContent = state.poseAvailable
        ? "Follow the highlighted note blocks. The rail fixes pitch; your motion shapes the phrase."
        : "Guided playback is running. Allow camera access and reload to turn body motion back on.";
}

async function startExperience() {
    if (state.booting) {
        return;
    }

    if (!state.booted) {
        await boot();
    }

    if (state.booted && !state.isPlaying) {
        await startPlayback();
    }
}

function attachEvents() {
    elements.bootButton.addEventListener("click", startExperience);

    elements.songSelect.addEventListener("change", (event) => {
        applySong(event.target.value);
        if (state.booted) {
            startPlayback();
        }
    });

    window.addEventListener("resize", syncTrackingOverlaySize);
}

function init() {
    buildSongSelect();
    applySong(state.selectedSongId);
    attachEvents();
}

init();
