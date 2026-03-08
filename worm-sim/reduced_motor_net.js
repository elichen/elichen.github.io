const NUM_MUSCLES = 96;
const SEGMENT_COUNT = 24;
const RESTING_VOLTAGE_MV = -56;
const ACTIVE_RANGE_MV = 78;

const DEFAULT_PARAMS = {
  forwardDrive: 0.68,
  turnBias: 0,
  wavePeriod: 1.15
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function hashString(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function toVoltage(level) {
  return RESTING_VOLTAGE_MV + clamp(level, 0, 1.1) * ACTIVE_RANGE_MV;
}

function classifyNeuron(name) {
  const sideMatch = name.match(/(DL|DR|VL|VR|L|R)$/);
  const side = sideMatch ? sideMatch[1] : '';
  const bodyMatch = name.match(/^DA(\d+)/);

  let group = 'relay';
  if (/^(AVB|PVC|RID|DVA)/.test(name)) {
    group = 'forward';
  } else if (/^(AVA|AVE)/.test(name)) {
    group = 'reverse';
  } else if (/^RIM/.test(name)) {
    group = 'turn';
  } else if (/^(SMD|SMB|RMD|RME|SAB|SAA|SIA|RIV)/.test(name)) {
    group = 'headMotor';
  } else if (/^DA\d+/.test(name)) {
    group = 'bodyMotor';
  } else if (/^(AWA|AWC|ASK|AIY|AIA|AIZ|AIB|ALA|ALN|PHA|URY|RIS|RIB|DVB|DVC|PLM|PVN)/.test(name)) {
    group = 'sensory';
  }

  return {
    name,
    group,
    side,
    phase: (hashString(name) % 628) / 100,
    bodyIndex: bodyMatch ? clamp(Number(bodyMatch[1]) - 1, 0, 3) : 0
  };
}

function leftBias(side, turnBias) {
  if (side.includes('L')) {
    return clamp(turnBias, -0.4, 0.4);
  }
  if (side.includes('R')) {
    return clamp(-turnBias, -0.4, 0.4);
  }
  return 0;
}

function dorsalSignal(segmentDorsal, segmentVentral, side) {
  if (side.includes('D')) {
    return segmentDorsal;
  }
  if (side.includes('V')) {
    return segmentVentral;
  }
  return 0.5 * (segmentDorsal + segmentVentral);
}

export class ReducedMotorNet {
  constructor(neuronNames) {
    this.neuronDefs = neuronNames.map(classifyNeuron);
    this.neuronVoltages = new Float32Array(neuronNames.length);
    this.muscleActivations = new Float32Array(NUM_MUSCLES);
    this.segmentDorsal = new Float32Array(SEGMENT_COUNT);
    this.segmentVentral = new Float32Array(SEGMENT_COUNT);
    this.params = { ...DEFAULT_PARAMS };
    this.time = 0;
    this.headPhase = 0;
    this.stepCount = 0;
    this.reset();
  }

  reset() {
    this.time = 0;
    this.headPhase = 0;
    this.stepCount = 0;
    this.segmentDorsal.fill(0.24);
    this.segmentVentral.fill(0.24);
    this.muscleActivations.fill(0.24);
    this.neuronVoltages.fill(RESTING_VOLTAGE_MV);
  }

  setParams(nextParams = {}) {
    if (Number.isFinite(nextParams.forwardDrive)) {
      this.params.forwardDrive = clamp(nextParams.forwardDrive, 0.2, 1.1);
    }
    if (Number.isFinite(nextParams.turnBias)) {
      this.params.turnBias = clamp(nextParams.turnBias, -0.45, 0.45);
    }
    if (Number.isFinite(nextParams.wavePeriod)) {
      this.params.wavePeriod = clamp(nextParams.wavePeriod, 0.65, 2.4);
    }
  }

  getOutputs() {
    return {
      muscles: this.muscleActivations,
      neuronVoltages: this.neuronVoltages,
      time: this.time,
      stepCount: this.stepCount
    };
  }

  step(deltaTime) {
    const dt = clamp(deltaTime, 1 / 480, 1 / 30);
    const drive = this.params.forwardDrive;
    const turnBias = this.params.turnBias;
    const omega = (Math.PI * 2) / this.params.wavePeriod;
    const response = 1 - Math.exp(-dt * (9 + drive * 5));
    const baseTonic = 0.12 + drive * 0.24;
    const waveGain = 0.28 + drive * 0.4;

    this.time += dt;
    this.stepCount += 1;
    this.headPhase += dt * omega;

    for (let segment = 0; segment < SEGMENT_COUNT; segment++) {
      const lag = 0.36 + (1 - drive) * 0.14;
      const phase = this.headPhase - segment * lag;
      const wave = Math.sin(phase);
      const envelope = 0.98 - segment * 0.016;
      const turnEnvelope = 1 - segment / (SEGMENT_COUNT * 1.3);
      const neighborD = segment > 0 ? this.segmentDorsal[segment - 1] : 0.3;
      const neighborV = segment > 0 ? this.segmentVentral[segment - 1] : 0.3;

      let targetDorsal = baseTonic + waveGain * envelope * (0.5 + 0.5 * wave);
      let targetVentral = baseTonic + waveGain * envelope * (0.5 - 0.5 * wave);

      targetDorsal += Math.max(0, turnBias) * 0.18 * turnEnvelope;
      targetVentral += Math.max(0, -turnBias) * 0.18 * turnEnvelope;

      targetDorsal = targetDorsal * 0.76 + neighborD * 0.24 - this.segmentVentral[segment] * 0.08;
      targetVentral = targetVentral * 0.76 + neighborV * 0.24 - this.segmentDorsal[segment] * 0.08;

      this.segmentDorsal[segment] += (clamp(targetDorsal, 0.04, 1) - this.segmentDorsal[segment]) * response;
      this.segmentVentral[segment] += (clamp(targetVentral, 0.04, 1) - this.segmentVentral[segment]) * response;

      const dorsal = clamp(this.segmentDorsal[segment], 0, 1);
      const ventral = clamp(this.segmentVentral[segment], 0, 1);

      this.muscleActivations[segment] = dorsal;
      this.muscleActivations[segment + 24] = ventral;
      this.muscleActivations[segment + 48] = dorsal;
      this.muscleActivations[segment + 72] = ventral;
    }

    const headDorsal = this.segmentDorsal[2];
    const headVentral = this.segmentVentral[2];
    const midDorsal = this.segmentDorsal[10];
    const midVentral = this.segmentVentral[10];
    const tailDorsal = this.segmentDorsal[20];
    const tailVentral = this.segmentVentral[20];
    const forwardState = clamp(0.18 + drive * 0.82 + 0.05 * Math.sin(this.time * 0.9), 0, 1.05);
    const reverseState = clamp(0.42 - drive * 0.34 + 0.08 * Math.max(0, Math.abs(turnBias) - 0.18), 0.04, 0.45);
    const turnState = clamp(Math.abs(turnBias) * 1.8, 0, 0.95);

    for (let index = 0; index < this.neuronDefs.length; index++) {
      const neuron = this.neuronDefs[index];
      const sideTurn = leftBias(neuron.side, turnBias);
      let targetLevel = 0.08;

      if (neuron.group === 'forward') {
        targetLevel = forwardState * 0.74 + 0.16 + sideTurn * 0.12;
      } else if (neuron.group === 'reverse') {
        targetLevel = reverseState * 0.82 + 0.08 - sideTurn * 0.08;
      } else if (neuron.group === 'turn') {
        targetLevel = 0.18 + turnState * 0.6 + sideTurn * 0.2;
      } else if (neuron.group === 'headMotor') {
        const headSignal = dorsalSignal(headDorsal, headVentral, neuron.side);
        targetLevel = 0.18 + headSignal * 0.78 + sideTurn * 0.12;
      } else if (neuron.group === 'bodyMotor') {
        const bodySegment = clamp(18 + neuron.bodyIndex * 2, 0, SEGMENT_COUNT - 1);
        const bodySignal = this.segmentDorsal[bodySegment] * 0.82 + this.segmentVentral[bodySegment] * 0.18;
        targetLevel = 0.14 + bodySignal * 0.78 + drive * 0.08;
      } else if (neuron.group === 'sensory') {
        const mixedSignal = 0.22 * forwardState + 0.16 * turnState + 0.12 * Math.sin(this.time * 0.55 + neuron.phase);
        targetLevel = 0.12 + mixedSignal;
      } else {
        const relaySignal = 0.18 * forwardState + 0.14 * dorsalSignal(midDorsal, midVentral, neuron.side) + 0.08 * Math.sin(this.time * 0.7 + neuron.phase);
        targetLevel = 0.11 + relaySignal;
      }

      if (neuron.name === 'DVB' || neuron.name === 'DVC') {
        targetLevel = 0.12 + 0.36 * dorsalSignal(tailDorsal, tailVentral, neuron.side) + 0.22 * reverseState;
      }

      const targetVoltage = toVoltage(targetLevel);
      const voltageResponse = 1 - Math.exp(-dt * (7 + forwardState * 4));
      this.neuronVoltages[index] += (targetVoltage - this.neuronVoltages[index]) * voltageResponse;
    }
  }
}
