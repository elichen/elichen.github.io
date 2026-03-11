const state = {
  data: null,
  sampleIndex: 0,
  view: 'truth',
  selectedNodeId: null,
  activeStepId: 'truth',
};

const svgNS = 'http://www.w3.org/2000/svg';

const refs = {
  gnnF1: document.getElementById('gnn-f1'),
  mlpF1: document.getElementById('mlp-f1'),
  subsetGap: document.getElementById('subset-gap'),
  overallGap: document.getElementById('overall-gap'),
  hardGap: document.getElementById('hard-gap'),
  sampleCount: document.getElementById('sample-count'),
  figureBlock: document.getElementById('primary-figure'),
  figureNotes: document.getElementById('figure-notes'),
  scenarioTitle: document.getElementById('scenario-title'),
  scenarioCaption: document.getElementById('scenario-caption'),
  graphSvg: document.getElementById('graph-svg'),
  scenarioList: document.getElementById('scenario-list'),
  nodeName: document.getElementById('node-name'),
  nodeSummary: document.getElementById('node-summary'),
  truthPill: document.getElementById('truth-pill'),
  gnnPill: document.getElementById('gnn-pill'),
  mlpPill: document.getElementById('mlp-pill'),
  incomingPressure: document.getElementById('incoming-pressure'),
  incomingRelief: document.getElementById('incoming-relief'),
  strongestSource: document.getElementById('strongest-source'),
  socialPressure: document.getElementById('social-pressure'),
  traitBars: document.getElementById('trait-bars'),
  contributors: document.getElementById('contributors'),
  historyChart: document.getElementById('history-chart'),
  viewToggle: document.getElementById('view-toggle'),
  prevSample: document.getElementById('prev-sample'),
  nextSample: document.getElementById('next-sample'),
  shuffleSample: document.getElementById('shuffle-sample'),
  figureSteps: Array.from(document.querySelectorAll('.figure-step')),
};

const calmColor = '#78d9b2';
const dangerColor = '#ff7a59';
const supportColor = '#ffc46b';

const figureNoteConfigs = [
  { id: 'dominant', label: 'Floor holder', view: 'pressure', focusMode: 'dominant', stepId: 'pressure' },
  { id: 'target', label: 'Pressure sink', view: 'truth', focusMode: 'target', stepId: 'truth' },
  { id: 'contrast', label: 'Where graph helps', view: 'gnn', focusMode: 'contrast', stepId: 'graph' },
  { id: 'anchor', label: 'Repair hub', view: 'pressure', focusMode: 'anchor', stepId: null },
];

async function loadData() {
  const response = await fetch('./data/experiment-data.json');
  if (!response.ok) {
    throw new Error(`Failed to load data: ${response.status}`);
  }
  return response.json();
}

function formatMetric(value) {
  return value.toFixed(3);
}

function formatDelta(value) {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(3)}`;
}

function formatCompact(value) {
  return value.toFixed(2);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function hexToRgb(hex) {
  const normalized = hex.replace('#', '');
  return {
    r: parseInt(normalized.slice(0, 2), 16),
    g: parseInt(normalized.slice(2, 4), 16),
    b: parseInt(normalized.slice(4, 6), 16),
  };
}

function rgbToHex({ r, g, b }) {
  return `#${[r, g, b].map((channel) => channel.toString(16).padStart(2, '0')).join('')}`;
}

function lerpColor(a, b, amount) {
  const left = hexToRgb(a);
  const right = hexToRgb(b);
  const t = clamp(amount, 0, 1);
  return rgbToHex({
    r: Math.round(left.r + (right.r - left.r) * t),
    g: Math.round(left.g + (right.g - left.g) * t),
    b: Math.round(left.b + (right.b - left.b) * t),
  });
}

function svgElement(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function getCurrentSample() {
  return state.data.samples[state.sampleIndex];
}

function sampleLabel(sample) {
  return `Conversation ${sample.id.replace(/^graph-/, '')}`;
}

function getStepElement(stepId) {
  return refs.figureSteps.find((step) => step.dataset.step === stepId) ?? null;
}

function findNodeByName(sample, name) {
  return sample.nodes.find((node) => node.name === name) ?? null;
}

function getSampleInsights(sample) {
  const dominant = findNodeByName(sample, sample.dominant_name)
    ?? sample.nodes.reduce((best, node) => node.floor_control > best.floor_control ? node : best, sample.nodes[0]);
  const anchor = findNodeByName(sample, sample.anchor_name)
    ?? sample.nodes.reduce((best, node) => node.incoming_relief > best.incoming_relief ? node : best, sample.nodes[0]);
  const target = sample.nodes.reduce((best, node) => node.incoming_pressure > best.incoming_pressure ? node : best, sample.nodes[0]);
  const contrastPool = sample.nodes
    .filter((node) => node.gnn_prediction === node.label && node.mlp_prediction !== node.label)
    .sort((left, right) => {
      const leftGap = Math.abs(left.gnn_probability - left.mlp_probability) + left.incoming_pressure * 0.25;
      const rightGap = Math.abs(right.gnn_probability - right.mlp_probability) + right.incoming_pressure * 0.25;
      return rightGap - leftGap;
    });
  const contrast = contrastPool[0]
    ?? sample.nodes
      .slice()
      .sort((left, right) => {
        const leftGap = Math.abs(left.gnn_probability - left.mlp_probability);
        const rightGap = Math.abs(right.gnn_probability - right.mlp_probability);
        return rightGap - leftGap;
      })[0]
    ?? target;

  return { dominant, anchor, target, contrast };
}

function getFocusNodeForMode(sample, focusMode) {
  const insights = getSampleInsights(sample);
  return insights[focusMode] ?? insights.contrast ?? sample.nodes[0];
}

function getTopContributorNode(node, sample) {
  const sourceId = node.contributors[0]?.source;
  return sourceId == null ? null : sample.nodes.find((item) => item.id === sourceId) ?? null;
}

function getFocusEdge(sample, focusMode, focusNode) {
  if (!focusNode) {
    return null;
  }

  if (focusMode === 'dominant') {
    return sample.edges
      .filter((edge) => edge.source === focusNode.id)
      .slice()
      .sort((left, right) => right.pressure - left.pressure)[0] ?? null;
  }

  if (focusMode === 'anchor') {
    return sample.edges
      .filter((edge) => edge.target === focusNode.id)
      .slice()
      .sort((left, right) => right.support - left.support)[0] ?? null;
  }

  const sourceId = focusNode.contributors[0]?.source;
  if (sourceId == null) {
    return null;
  }
  return sample.edges.find((edge) => edge.source === sourceId && edge.target === focusNode.id) ?? null;
}

function getCurrentFigureFocus(sample) {
  if (!state.activeStepId) {
    return { node: null, edge: null, focusMode: null };
  }

  const activeStep = getStepElement(state.activeStepId);
  const focusMode = activeStep?.dataset.focusMode ?? 'contrast';
  const node = getFocusNodeForMode(sample, focusMode);
  return {
    node,
    edge: getFocusEdge(sample, focusMode, node),
    focusMode,
  };
}

function composeScenarioSummary(sample, short = false) {
  const overloadedCount = sample.nodes.filter((node) => node.label === 1).length;
  if (short) {
    return `Dominant speaker ${sample.dominant_name}; repair centered on ${sample.anchor_name}; ${overloadedCount} of ${sample.nodes.length} above threshold.`;
  }
  return `${sample.dominant_name} holds the floor hardest. Repair is most concentrated around ${sample.anchor_name}. ${overloadedCount} of ${sample.nodes.length} participants cross the overload threshold.`;
}

function getDisplayColor(node) {
  if (state.view === 'truth') {
    return node.label ? dangerColor : calmColor;
  }
  if (state.view === 'gnn') {
    return lerpColor(calmColor, dangerColor, node.gnn_probability);
  }
  if (state.view === 'mlp') {
    return lerpColor(calmColor, dangerColor, node.mlp_probability);
  }
  return lerpColor('#274655', dangerColor, node.social_pressure);
}

function computeLayout(nodes, width, height) {
  const positions = new Map();
  const dominant = nodes.reduce((best, node) => node.social_pressure > best.social_pressure ? node : best, nodes[0]);
  const others = nodes.filter((node) => node.id !== dominant.id)
    .sort((left, right) => right.social_pressure - left.social_pressure);

  const centerX = width * 0.52;
  const centerY = height * 0.46;
  positions.set(dominant.id, { x: centerX, y: centerY });

  const radiusX = width * 0.34;
  const radiusY = height * 0.28;
  others.forEach((node, index) => {
    const angle = -Math.PI / 2 + (index / Math.max(1, others.length)) * Math.PI * 2;
    const rScale = 0.88 + (1 - node.social_pressure) * 0.18 + (index % 2) * 0.04;
    positions.set(node.id, {
      x: centerX + Math.cos(angle) * radiusX * rScale,
      y: centerY + Math.sin(angle) * radiusY * rScale,
    });
  });

  return positions;
}

function chooseDefaultNode(sample) {
  return getFocusNodeForMode(sample, 'contrast').id;
}

function predictionLabel(value) {
  return value ? 'Overloaded' : 'Stable';
}

function scenarioTitle(sample) {
  if (state.activeStepId === 'truth') {
    return `${sampleLabel(sample)}: where overload lands`;
  }
  if (state.activeStepId === 'pressure') {
    return `${sampleLabel(sample)}: how pressure spreads`;
  }
  if (state.activeStepId === 'traits') {
    return `${sampleLabel(sample)}: what traits can see`;
  }
  if (state.activeStepId === 'graph') {
    return `${sampleLabel(sample)}: what the graph adds`;
  }
  return sampleLabel(sample);
}

function manualScenarioCaption(sample) {
  if (state.view === 'truth') {
    return `${composeScenarioSummary(sample)} Meeting intensity ${formatCompact(sample.meeting_intensity)}.`;
  }
  if (state.view === 'gnn') {
    return 'Node color shows the GNN prediction for overload probability.';
  }
  if (state.view === 'mlp') {
    return 'Node color shows the trait-only baseline prediction.';
  }
  return 'Node color shows social pressure; thicker arrows mark the strongest pressure links.';
}

function scenarioCaption(sample) {
  const insights = getSampleInsights(sample);
  const contrastSource = getTopContributorNode(insights.contrast, sample);

  if (state.activeStepId === 'truth') {
    return `${insights.target.name} receives the heaviest incoming pressure in this room. The red nodes are the participants who cross the threshold in the final label.`;
  }
  if (state.activeStepId === 'pressure') {
    return `${insights.dominant.name} holds the floor hardest, while ${insights.target.name} absorbs the densest incoming pressure. The highlighted arrow marks the strongest pressure route in this case.`;
  }
  if (state.activeStepId === 'traits') {
    return `The trait-only model reads ${insights.contrast.name} as ${predictionLabel(insights.contrast.mlp_prediction).toLowerCase()} from node features alone. It cannot follow the rerouting happening around them.`;
  }
  if (state.activeStepId === 'graph') {
    return contrastSource
      ? `Once the graph is visible, ${insights.contrast.name} is read through the room around them. The strongest pressure into that node comes from ${contrastSource.name}.`
      : `Once the graph is visible, ${insights.contrast.name} is read through the room around them rather than from traits alone.`;
  }
  return manualScenarioCaption(sample);
}

function syncFigureSteps() {
  refs.figureSteps.forEach((step) => {
    step.classList.toggle('is-active', step.dataset.step === state.activeStepId);
  });
}

function activateStep(stepId) {
  const step = getStepElement(stepId);
  if (!step) {
    return;
  }

  const sample = getCurrentSample();
  state.activeStepId = stepId;
  state.view = step.dataset.view;
  state.selectedNodeId = getFocusNodeForMode(sample, step.dataset.focusMode).id;
  renderScenario();
}

function nodeSummary(node, sample) {
  const truthText = node.label ? 'is above the overload threshold' : 'remains below the overload threshold';
  const dominantSource = node.contributors[0] ? sample.nodes.find((item) => item.id === node.contributors[0].source) : null;
  const dominantText = dominantSource
    ? `Strongest pressure source: ${dominantSource.name}.`
    : 'No single source dominates the pressure pattern.';
  return `${node.name} ${truthText}. ${dominantText} Incoming pressure ${formatCompact(node.incoming_pressure)}; repair support ${formatCompact(node.incoming_relief)}.`;
}

function contributorRows(node, sample) {
  if (!node.contributors.length) {
    refs.contributors.innerHTML = '<p class="contributors-empty">This participant is relatively insulated in this scenario. No single speaker contributes enough pressure to stand out.</p>';
    return;
  }

  refs.contributors.innerHTML = '';
  node.contributors.forEach((contributor) => {
    const sourceNode = sample.nodes.find((item) => item.id === contributor.source);
    const row = document.createElement('div');
    row.className = 'contributor-row';
    row.innerHTML = `
      <div>
        <div class="contributor-name">${sourceNode.name}</div>
        <div class="contributor-value">floor ${formatCompact(sourceNode.floor_control)} · support ${formatCompact(sourceNode.endorsement)}</div>
      </div>
      <div class="contributor-value">${formatCompact(contributor.weight)}</div>
    `;
    refs.contributors.appendChild(row);
  });
}

function renderTraitBars(node) {
  refs.traitBars.innerHTML = '';
  state.data.meta.node_features.forEach((feature) => {
    const value = node.features[feature.key];
    const row = document.createElement('div');
    row.className = 'trait-row';
    row.innerHTML = `
      <label>${feature.label}</label>
      <div class="bar-track"><div class="bar-fill" style="width:${(value * 100).toFixed(1)}%"></div></div>
      <div class="trait-value">${formatCompact(value)}</div>
    `;
    refs.traitBars.appendChild(row);
  });
}

function renderFigureNotes(sample) {
  const insights = getSampleInsights(sample);
  const notes = {
    dominant: {
      node: insights.dominant,
      copy: `${insights.dominant.name} sets the pace here with the highest floor control and the widest outward pressure.`,
    },
    target: {
      node: insights.target,
      copy: `${insights.target.name} takes the heaviest incoming pressure in this room and is the clearest pressure sink in the graph.`,
    },
    contrast: {
      node: insights.contrast,
      copy: `${insights.contrast.name} is the clearest place to compare the graph view against the trait-only baseline.`,
    },
    anchor: {
      node: insights.anchor,
      copy: `${insights.anchor.name} receives the most repair attempts, which changes how pressure redistributes nearby.`,
    },
  };

  refs.figureNotes.innerHTML = '';
  figureNoteConfigs.forEach((config) => {
    const note = notes[config.id];
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `figure-note${state.selectedNodeId === note.node.id ? ' is-active' : ''}`;
    button.innerHTML = `
      <span class="figure-note-label">${config.label}</span>
      <strong>${note.node.name}</strong>
      <p>${note.copy}</p>
    `;
    button.addEventListener('click', () => {
      state.view = config.view;
      state.selectedNodeId = note.node.id;
      state.activeStepId = config.stepId;
      renderScenario();
    });
    refs.figureNotes.appendChild(button);
  });
}

function updateNodePanel(sample, nodeId) {
  const node = sample.nodes.find((item) => item.id === nodeId);
  if (!node) {
    return;
  }

  refs.nodeName.textContent = node.name;
  refs.nodeSummary.textContent = nodeSummary(node, sample);
  refs.truthPill.textContent = `Truth: ${node.label ? 'Overloaded' : 'Stable'}`;
  refs.gnnPill.textContent = `GNN: ${node.gnn_prediction ? 'Overloaded' : 'Stable'} (${formatCompact(node.gnn_probability)})`;
  refs.mlpPill.textContent = `Trait-only: ${node.mlp_prediction ? 'Overloaded' : 'Stable'} (${formatCompact(node.mlp_probability)})`;
  refs.incomingPressure.textContent = formatCompact(node.incoming_pressure);
  refs.incomingRelief.textContent = formatCompact(node.incoming_relief);
  const topSource = getTopContributorNode(node, sample);
  refs.strongestSource.textContent = topSource
    ? `${topSource.name} · ${formatCompact(node.strongest_source)}`
    : 'None';
  refs.socialPressure.textContent = formatCompact(node.social_pressure);
  renderTraitBars(node);
  contributorRows(node, sample);
}

function edgeCurve(source, target, magnitude) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const mx = (source.x + target.x) / 2;
  const my = (source.y + target.y) / 2;
  const normalX = -dy;
  const normalY = dx;
  const normalLength = Math.hypot(normalX, normalY) || 1;
  const bend = 18 + magnitude * 22;
  const cx = mx + (normalX / normalLength) * bend;
  const cy = my + (normalY / normalLength) * bend;
  return `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`;
}

function markerDefinitions() {
  const defs = svgElement('defs');
  const pressureMarker = svgElement('marker', {
    id: 'arrow-pressure',
    viewBox: '0 0 10 10',
    refX: '9',
    refY: '5',
    markerWidth: '7',
    markerHeight: '7',
    orient: 'auto-start-reverse',
  });
  pressureMarker.appendChild(svgElement('path', {
    d: 'M 0 0 L 10 5 L 0 10 z',
    class: 'edge-arrow',
  }));
  defs.appendChild(pressureMarker);
  return defs;
}

function renderGraph(sample) {
  const svg = refs.graphSvg;
  svg.innerHTML = '';
  svg.appendChild(markerDefinitions());

  const width = 920;
  const height = 620;
  const layout = computeLayout(sample.nodes, width, height);
  const figureFocus = getCurrentFigureFocus(sample);
  const focusEdge = figureFocus.edge;
  const focusNodeId = figureFocus.node?.id ?? null;

  svg.appendChild(svgElement('ellipse', {
    cx: width * 0.52,
    cy: height * 0.46,
    rx: width * 0.32,
    ry: height * 0.26,
    class: 'graph-ring',
  }));

  const pressureEdges = sample.edges.slice().sort((left, right) => right.pressure - left.pressure).slice(0, 10);
  const supportEdges = sample.edges
    .filter((edge) => edge.support > 0.45)
    .sort((left, right) => right.support - left.support)
    .slice(0, 6);

  if (focusEdge) {
    const targetList = focusEdge.support > focusEdge.pressure ? supportEdges : pressureEdges;
    if (!targetList.find((edge) => edge.source === focusEdge.source && edge.target === focusEdge.target)) {
      targetList.push(focusEdge);
    }
  }

  const edgeLayer = svgElement('g');

  supportEdges.forEach((edge) => {
    const source = layout.get(edge.source);
    const target = layout.get(edge.target);
    edgeLayer.appendChild(svgElement('path', {
      d: edgeCurve(source, target, edge.support),
      class: `edge-line support${focusEdge && focusEdge.source === edge.source && focusEdge.target === edge.target ? ' is-focus' : ''}`,
      'stroke-width': (1.2 + edge.support * 2.6).toFixed(2),
    }));
  });

  pressureEdges.forEach((edge) => {
    const source = layout.get(edge.source);
    const target = layout.get(edge.target);
    edgeLayer.appendChild(svgElement('path', {
      d: edgeCurve(source, target, edge.pressure),
      class: `edge-line pressure${focusEdge && focusEdge.source === edge.source && focusEdge.target === edge.target ? ' is-focus' : ''}`,
      'stroke-width': (1.4 + edge.pressure * 1.9).toFixed(2),
      'marker-end': 'url(#arrow-pressure)',
    }));
  });

  svg.appendChild(edgeLayer);

  const nodeLayer = svgElement('g');
  sample.nodes.forEach((node) => {
    const position = layout.get(node.id);
    const radius = 20 + node.social_pressure * 20;
    const group = svgElement('g', {
      class: `graph-node${state.selectedNodeId === node.id ? ' is-selected' : ''}${focusNodeId === node.id ? ' is-focus' : ''}`,
      transform: `translate(${position.x}, ${position.y})`,
      tabindex: '0',
      role: 'button',
      'aria-label': `${node.name} node`,
    });

    const glow = svgElement('circle', {
      r: (radius + 11).toFixed(2),
      fill: 'rgba(255,255,255,0.05)',
      stroke: 'rgba(255,255,255,0.04)',
    });
    const circle = svgElement('circle', {
      r: radius.toFixed(2),
      fill: getDisplayColor(node),
      stroke: state.selectedNodeId === node.id ? '#fff4e5' : 'rgba(255,255,255,0.42)',
      'stroke-width': state.selectedNodeId === node.id ? '4' : '2.2',
    });
    const label = svgElement('text', {
      x: '0',
      y: (radius + 22).toFixed(2),
      'text-anchor': 'middle',
      class: 'graph-label',
    });
    label.textContent = node.name;

    group.addEventListener('click', () => {
      state.selectedNodeId = node.id;
      state.activeStepId = null;
      renderScenario();
    });
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        state.selectedNodeId = node.id;
        state.activeStepId = null;
        renderScenario();
      }
    });

    group.appendChild(glow);
    group.appendChild(circle);
    group.appendChild(label);
    nodeLayer.appendChild(group);
  });

  svg.appendChild(nodeLayer);
}

function renderScenarioList() {
  refs.scenarioList.innerHTML = '';
  state.data.samples.forEach((sample, index) => {
    const gnnMisses = sample.nodes.filter((node) => node.gnn_prediction !== node.label).length;
    const mlpMisses = sample.nodes.filter((node) => node.mlp_prediction !== node.label).length;
    const overloadedCount = sample.nodes.filter((node) => node.label === 1).length;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `scenario-card${state.sampleIndex === index ? ' is-selected' : ''}`;
    button.innerHTML = `
      <h3>${sampleLabel(sample)}</h3>
      <p>${composeScenarioSummary(sample, true)}</p>
      <div class="scenario-meta">
        <span class="scenario-chip">${overloadedCount} overloaded</span>
        <span class="scenario-chip">GNN misses ${gnnMisses}</span>
        <span class="scenario-chip">Trait-only misses ${mlpMisses}</span>
      </div>
    `;
    button.addEventListener('click', () => {
      state.sampleIndex = index;
      state.selectedNodeId = chooseDefaultNode(sample);
      renderScenario();
      renderScenarioList();
    });
    refs.scenarioList.appendChild(button);
  });
}

function renderScenario() {
  const sample = getCurrentSample();
  if (!sample.nodes.some((node) => node.id === state.selectedNodeId)) {
    state.selectedNodeId = chooseDefaultNode(sample);
  }

  refs.scenarioTitle.textContent = scenarioTitle(sample);
  refs.scenarioCaption.textContent = scenarioCaption(sample);
  renderFigureNotes(sample);
  renderGraph(sample);
  updateNodePanel(sample, state.selectedNodeId ?? chooseDefaultNode(sample));

  Array.from(refs.viewToggle.querySelectorAll('button')).forEach((button) => {
    button.classList.toggle('is-active', button.dataset.view === state.view);
  });

  syncFigureSteps();
  renderScenarioList();
}

function renderHistoryChart() {
  const svg = refs.historyChart;
  svg.innerHTML = '';

  const width = 760;
  const height = 300;
  const margin = { top: 24, right: 40, bottom: 40, left: 44 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const gnnHistory = state.data.histories.gnn;
  const mlpHistory = state.data.histories.mlp;
  const allValues = [...gnnHistory, ...mlpHistory].map((item) => item.val_f1);
  const minValue = Math.min(...allValues) - 0.02;
  const maxValue = Math.max(...allValues) + 0.02;

  const xForEpoch = (epoch) => margin.left + ((epoch - 1) / Math.max(1, gnnHistory.length - 1)) * chartWidth;
  const yForValue = (value) => margin.top + chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight;

  [0, 0.5, 1].forEach(() => {});

  const axis = svgElement('g');
  axis.appendChild(svgElement('line', {
    x1: margin.left,
    y1: margin.top + chartHeight,
    x2: margin.left + chartWidth,
    y2: margin.top + chartHeight,
    class: 'chart-axis',
  }));
  axis.appendChild(svgElement('line', {
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + chartHeight,
    class: 'chart-axis',
  }));

  [minValue, (minValue + maxValue) / 2, maxValue].forEach((value) => {
    const y = yForValue(value);
    axis.appendChild(svgElement('line', {
      x1: margin.left,
      y1: y,
      x2: margin.left + chartWidth,
      y2: y,
      class: 'chart-axis',
      opacity: '0.55',
    }));
    const label = svgElement('text', {
      x: 8,
      y: y + 4,
      class: 'chart-label',
    });
    label.textContent = value.toFixed(2);
    axis.appendChild(label);
  });

  svg.appendChild(axis);

  const makePath = (history) => history.map((item, index) => {
    const x = xForEpoch(index + 1);
    const y = yForValue(item.val_f1);
    return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');

  svg.appendChild(svgElement('path', {
    d: makePath(gnnHistory),
    class: 'chart-line',
    stroke: dangerColor,
  }));
  svg.appendChild(svgElement('path', {
    d: makePath(mlpHistory),
    class: 'chart-line',
    stroke: supportColor,
  }));

  const gnnTag = svgElement('text', {
    x: width - 126,
    y: 38,
    class: 'chart-tag',
  });
  gnnTag.textContent = 'GNN';
  svg.appendChild(gnnTag);
  svg.appendChild(svgElement('line', {
    x1: width - 186,
    y1: 32,
    x2: width - 136,
    y2: 32,
    stroke: dangerColor,
    'stroke-width': '4',
    'stroke-linecap': 'round',
  }));

  const mlpTag = svgElement('text', {
    x: width - 126,
    y: 64,
    class: 'chart-tag',
  });
  mlpTag.textContent = 'Trait-only';
  svg.appendChild(mlpTag);
  svg.appendChild(svgElement('line', {
    x1: width - 186,
    y1: 58,
    x2: width - 136,
    y2: 58,
    stroke: supportColor,
    'stroke-width': '4',
    'stroke-linecap': 'round',
  }));
}

function populateMetrics() {
  const overallGnn = state.data.metrics.gnn.overall;
  const overallMlp = state.data.metrics.mlp.overall;
  const hardGnn = state.data.metrics.gnn.high_pressure_subset;
  const hardMlp = state.data.metrics.mlp.high_pressure_subset;

  refs.gnnF1.textContent = formatMetric(overallGnn.f1);
  refs.mlpF1.textContent = formatMetric(overallMlp.f1);
  refs.subsetGap.textContent = formatDelta(hardGnn.f1 - hardMlp.f1);
  refs.overallGap.textContent = formatDelta(overallGnn.f1 - overallMlp.f1);
  refs.hardGap.textContent = formatDelta(hardGnn.f1 - hardMlp.f1);
  refs.sampleCount.textContent = state.data.samples.length.toString();
}

function attachFigureObservers() {
  const stepVisibility = new Map();

  const figureObserver = new IntersectionObserver((entries) => {
    const entry = entries[0];
    refs.figureBlock.classList.toggle('is-active', entry.isIntersecting && entry.intersectionRatio > 0.14);
  }, {
    threshold: [0, 0.14, 0.3, 0.5],
  });
  figureObserver.observe(refs.figureBlock);

  const stepObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      stepVisibility.set(entry.target.dataset.step, entry.isIntersecting ? entry.intersectionRatio : 0);
    });

    const bestStep = refs.figureSteps
      .map((step) => ({ step, ratio: stepVisibility.get(step.dataset.step) ?? 0 }))
      .sort((left, right) => right.ratio - left.ratio)[0];

    if (bestStep && bestStep.ratio > 0.18 && bestStep.step.dataset.step !== state.activeStepId) {
      activateStep(bestStep.step.dataset.step);
    }
  }, {
    threshold: [0, 0.2, 0.4, 0.6, 0.8],
    rootMargin: '-18% 0px -26% 0px',
  });

  refs.figureSteps.forEach((step) => stepObserver.observe(step));
}

function attachEvents() {
  refs.viewToggle.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-view]');
    if (!button) {
      return;
    }
    state.activeStepId = null;
    state.view = button.dataset.view;
    renderScenario();
  });

  refs.prevSample.addEventListener('click', () => {
    state.sampleIndex = (state.sampleIndex - 1 + state.data.samples.length) % state.data.samples.length;
    state.selectedNodeId = state.activeStepId
      ? getFocusNodeForMode(getCurrentSample(), getStepElement(state.activeStepId)?.dataset.focusMode ?? 'contrast').id
      : chooseDefaultNode(getCurrentSample());
    renderScenario();
  });

  refs.nextSample.addEventListener('click', () => {
    state.sampleIndex = (state.sampleIndex + 1) % state.data.samples.length;
    state.selectedNodeId = state.activeStepId
      ? getFocusNodeForMode(getCurrentSample(), getStepElement(state.activeStepId)?.dataset.focusMode ?? 'contrast').id
      : chooseDefaultNode(getCurrentSample());
    renderScenario();
  });

  refs.shuffleSample.addEventListener('click', () => {
    const nextIndex = Math.floor(Math.random() * state.data.samples.length);
    state.sampleIndex = nextIndex;
    state.selectedNodeId = state.activeStepId
      ? getFocusNodeForMode(getCurrentSample(), getStepElement(state.activeStepId)?.dataset.focusMode ?? 'contrast').id
      : chooseDefaultNode(getCurrentSample());
    renderScenario();
  });

  refs.figureSteps.forEach((step) => {
    step.addEventListener('click', () => activateStep(step.dataset.step));
    step.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activateStep(step.dataset.step);
      }
    });
  });
}

async function init() {
  state.data = await loadData();
  populateMetrics();
  state.selectedNodeId = getFocusNodeForMode(getCurrentSample(), 'target').id;
  renderScenario();
  renderHistoryChart();
  attachEvents();
  attachFigureObservers();
}

init().catch((error) => {
  refs.scenarioTitle.textContent = 'Failed to load';
  refs.scenarioCaption.textContent = error.message;
  console.error(error);
});
