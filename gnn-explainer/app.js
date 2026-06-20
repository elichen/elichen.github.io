const ARTIFACT_URL = 'data/experiment-data.json?v=movielens-lightgcn2';
const SVG_NS = 'http://www.w3.org/2000/svg';

const COLORS = {
  low: '#db5b48',
  mid: '#eee7d6',
  high: '#18a872',
  ink: '#141414',
  active: '#00a676',
};

const state = {
  artifact: null,
  graph: null,
  rawLayers: [],
  finalLayers: new Map(),
  edgeWeightCache: new Map(),
  scenarioId: null,
  depth: 2,
  model: 'gnn',
  selectedNodeIdx: null,
  showMessages: true,
  hasRun: false,
  loading: true,
  error: null,
};

const refs = {
  scenarioList: document.getElementById('scenario-list'),
  scenarioKicker: document.getElementById('scenario-kicker'),
  scenarioTitle: document.getElementById('scenario-title'),
  scenarioObjective: document.getElementById('scenario-objective'),
  candidateName: document.getElementById('candidate-name'),
  candidateMeta: document.getElementById('candidate-meta'),
  modelPicker: document.getElementById('model-picker'),
  depthPicker: document.getElementById('depth-picker'),
  messageToggle: document.getElementById('message-toggle'),
  graphSvg: document.getElementById('graph'),
  scoreValue: document.getElementById('score-value'),
  scoreLabel: document.getElementById('score-label'),
  scoreFill: document.getElementById('score-fill'),
  runButton: document.getElementById('run-button'),
  resultCopy: document.getElementById('result-copy'),
  compareButton: document.getElementById('compare-button'),
  depthComparison: document.getElementById('depth-comparison'),
  nodeName: document.getElementById('node-name'),
  nodeDescription: document.getElementById('node-description'),
  nodeStats: document.getElementById('node-stats'),
  messageList: document.getElementById('message-list'),
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-clamp(value, -40, 40)));
}

function hexToRgb(hex) {
  const raw = hex.replace('#', '');
  return {
    r: parseInt(raw.slice(0, 2), 16),
    g: parseInt(raw.slice(2, 4), 16),
    b: parseInt(raw.slice(4, 6), 16),
  };
}

function rgbToHex(rgb) {
  return `#${[rgb.r, rgb.g, rgb.b].map((value) => value.toString(16).padStart(2, '0')).join('')}`;
}

function mixColor(a, b, t) {
  const left = hexToRgb(a);
  const right = hexToRgb(b);
  const amount = clamp(t, 0, 1);
  return rgbToHex({
    r: Math.round(left.r + (right.r - left.r) * amount),
    g: Math.round(left.g + (right.g - left.g) * amount),
    b: Math.round(left.b + (right.b - left.b) * amount),
  });
}

function signalColor(score) {
  if (score < 0.5) return mixColor(COLORS.low, COLORS.mid, score / 0.5);
  return mixColor(COLORS.mid, COLORS.high, (score - 0.5) / 0.5);
}

function luminance(hex) {
  const { r, g, b } = hexToRgb(hex);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function textColor(fill) {
  return luminance(fill) < 0.56 ? '#fffdf4' : COLORS.ink;
}

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  return node;
}

function getDepths() {
  return state.artifact?.meta.depths ?? [0, 1, 2, 4, 6];
}

function getThreshold() {
  return state.artifact?.meta.threshold ?? 0.5;
}

function getScenario() {
  const scenarios = state.artifact?.scenarios ?? [];
  return scenarios.find((scenario) => scenario.id === state.scenarioId) ?? scenarios[0] ?? null;
}

function getDisplayNode(idx) {
  const scenario = getScenario();
  return scenario?.display.nodes.find((node) => node.idx === idx) ?? null;
}

function embeddingOffset(idx) {
  return idx * state.graph.dim;
}

function dotAt(layer, leftIdx, rightIdx) {
  const left = embeddingOffset(leftIdx);
  const right = embeddingOffset(rightIdx);
  let total = 0;
  for (let i = 0; i < state.graph.dim; i += 1) {
    total += layer[left + i] * layer[right + i];
  }
  return total;
}

function normAt(layer, idx) {
  const start = embeddingOffset(idx);
  let total = 0;
  for (let i = 0; i < state.graph.dim; i += 1) {
    total += layer[start + i] * layer[start + i];
  }
  return Math.sqrt(total) || 1;
}

function pairScore(userIdx, movieIdx, depth = state.depth) {
  const layer = state.finalLayers.get(depth) ?? state.finalLayers.get(0);
  const logit = dotAt(layer, userIdx, movieIdx)
    + state.graph.bias[userIdx]
    + state.graph.bias[movieIdx]
    + state.graph.globalBias;
  return sigmoid(logit);
}

function nodeSignal(node, depth = state.depth) {
  const scenario = getScenario();
  if (!scenario) return 0.5;
  if (node.type === 'user') return pairScore(node.idx, scenario.movie_node, depth);
  if (node.type === 'item') return pairScore(scenario.user_node, node.idx, depth);

  const layer = state.finalLayers.get(depth) ?? state.finalLayers.get(0);
  const similarity = dotAt(layer, scenario.user_node, node.idx)
    / (normAt(layer, scenario.user_node) * normAt(layer, node.idx));
  return clamp((similarity + 1) / 2, 0, 1);
}

function predictionFor(score) {
  return score >= getThreshold() ? 'recommend' : 'hold';
}

function labelForPrediction(prediction) {
  return prediction === 'recommend' ? 'Recommend' : 'Hold';
}

function confidenceCopy(score) {
  const distance = Math.abs(score - getThreshold());
  if (distance >= 0.25) return 'high confidence';
  if (distance >= 0.12) return 'moderate confidence';
  return 'near threshold';
}

function nodePosition(node) {
  const width = 920;
  const height = 620;
  const padX = 72;
  const padY = 62;
  return {
    x: padX + node.x * (width - padX * 2),
    y: padY + node.y * (height - padY * 2),
  };
}

function edgePath(source, target, curve = 0) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const midX = (source.x + target.x) / 2;
  const midY = (source.y + target.y) / 2;
  const length = Math.hypot(dx, dy) || 1;
  const normalX = -dy / length;
  const normalY = dx / length;
  const bend = curve || (source.x < target.x ? 18 : -18);
  const cx = midX + normalX * bend;
  const cy = midY + normalY * bend;
  return {
    d: `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${target.x.toFixed(1)} ${target.y.toFixed(1)}`,
    labelX: (0.25 * source.x + 0.5 * cx + 0.25 * target.x),
    labelY: (0.25 * source.y + 0.5 * cy + 0.25 * target.y),
  };
}

function appendText(parent, tag, text, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

function prepareRuntime(artifact) {
  const graph = artifact.graph;
  state.graph = {
    dim: graph.dim,
    nodeCount: graph.nodeCount,
    src: Int32Array.from(graph.src),
    dst: Int32Array.from(graph.dst),
    weight: Float32Array.from(graph.weight),
    embedding: Float32Array.from(graph.embedding),
    bias: Float32Array.from(graph.bias),
    globalBias: graph.globalBias,
  };

  computeLayers(Math.max(...artifact.meta.depths));
}

function computeLayers(maxDepth) {
  const { dim, nodeCount, src, dst, weight, embedding } = state.graph;
  state.rawLayers = [embedding];
  state.finalLayers.clear();

  for (let depth = 1; depth <= maxDepth; depth += 1) {
    const previous = state.rawLayers[depth - 1];
    const next = new Float32Array(nodeCount * dim);
    for (let edge = 0; edge < src.length; edge += 1) {
      const sourceOffset = src[edge] * dim;
      const targetOffset = dst[edge] * dim;
      const amount = weight[edge];
      for (let i = 0; i < dim; i += 1) {
        next[targetOffset + i] += previous[sourceOffset + i] * amount;
      }
    }
    state.rawLayers.push(next);
  }

  for (const depth of getDepths()) {
    const final = new Float32Array(nodeCount * dim);
    for (let layerIndex = 0; layerIndex <= depth; layerIndex += 1) {
      const layer = state.rawLayers[layerIndex];
      for (let i = 0; i < final.length; i += 1) {
        final[i] += layer[i];
      }
    }
    const scale = 1 / (depth + 1);
    for (let i = 0; i < final.length; i += 1) {
      final[i] *= scale;
    }
    state.finalLayers.set(depth, final);
  }
}

function normalizedEdgeWeight(source, target, fallback) {
  const key = `${source}:${target}`;
  if (state.edgeWeightCache.has(key)) return state.edgeWeightCache.get(key);

  const { src, dst, weight } = state.graph;
  for (let i = 0; i < src.length; i += 1) {
    if (src[i] === source && dst[i] === target) {
      state.edgeWeightCache.set(key, weight[i]);
      return weight[i];
    }
  }
  state.edgeWeightCache.set(key, fallback);
  return fallback;
}

function directedEdgeContribution(edge, source, target) {
  if (state.depth === 0) return 0;
  const previous = state.rawLayers[Math.max(0, state.depth - 1)];
  const final = state.finalLayers.get(state.depth);
  const norm = normalizedEdgeWeight(source, target, edge.weight);
  const raw = dotAt(previous, source, target);
  const sourceNorm = normAt(previous, source);
  const targetNorm = normAt(final, target);
  return norm * raw / (sourceNorm * targetNorm || 1);
}

function localMessages(limit = 5) {
  const scenario = getScenario();
  if (!scenario || state.depth === 0) return [];

  return scenario.display.edges
    .map((edge) => {
      const forward = directedEdgeContribution(edge, edge.source, edge.target);
      const backward = directedEdgeContribution(edge, edge.target, edge.source);
      const useBackward = Math.abs(backward) > Math.abs(forward);
      const source = useBackward ? edge.target : edge.source;
      const target = useBackward ? edge.source : edge.target;
      return {
        ...edge,
        source,
        target,
        value: useBackward ? backward : forward,
        sourceNode: getDisplayNode(source),
        targetNode: getDisplayNode(target),
      };
    })
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit);
}

function scoreForScenarioDepth(scenario, depth) {
  return pairScore(scenario.user_node, scenario.movie_node, depth);
}

function depthNote(depth) {
  const trainDepth = state.artifact.meta.train_depth;
  if (depth === 0) return 'Learned user and movie embeddings only; no graph messages applied.';
  if (depth === trainDepth) return `The trained ${trainDepth}-hop LightGCN inference depth.`;
  if (depth < trainDepth) return 'A shallower graph pass than the depth used during training.';
  return 'Extra propagation with the same learned model; useful signal may smooth out.';
}

function buildScenarioList() {
  refs.scenarioList.textContent = '';
  for (const scenario of state.artifact.scenarios) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'scenario-button';
    button.dataset.scenarioId = scenario.id;
    appendText(button, 'span', scenario.level);
    appendText(button, 'strong', scenario.title);
    appendText(button, 'small', `${scenario.kicker} | best depth ${scenario.best_depth}`);
    appendText(button, 'em', `${Math.round(scoreForScenarioDepth(scenario, scenario.best_depth) * 100)}%`);
    button.addEventListener('click', () => {
      state.scenarioId = scenario.id;
      state.depth = scenario.best_depth;
      state.model = scenario.best_depth === 0 ? 'base' : 'gnn';
      state.selectedNodeIdx = scenario.user_node;
      state.hasRun = false;
      render();
    });
    refs.scenarioList.appendChild(button);
  }
}

function buildControls() {
  refs.modelPicker.textContent = '';
  [
    { id: 'base', label: 'Node-only' },
    { id: 'gnn', label: 'LightGCN' },
  ].forEach((model) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.model = model.id;
    button.textContent = model.label;
    button.addEventListener('click', () => {
      state.model = model.id;
      state.depth = model.id === 'base'
        ? 0
        : Math.max(1, state.depth || state.artifact.meta.train_depth);
      state.hasRun = false;
      render();
    });
    refs.modelPicker.appendChild(button);
  });

  refs.depthPicker.textContent = '';
  getDepths().forEach((depth) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.depth = String(depth);
    button.textContent = String(depth);
    button.addEventListener('click', () => {
      state.depth = depth;
      state.model = depth === 0 ? 'base' : 'gnn';
      state.hasRun = false;
      render();
    });
    refs.depthPicker.appendChild(button);
  });

  refs.messageToggle.addEventListener('click', () => {
    state.showMessages = !state.showMessages;
    render();
  });

  refs.runButton.addEventListener('click', () => {
    state.hasRun = true;
    render();
  });

  refs.compareButton.addEventListener('click', () => {
    const scenario = getScenario();
    state.depth = scenario.best_depth;
    state.model = scenario.best_depth === 0 ? 'base' : 'gnn';
    state.hasRun = true;
    render();
  });
}

function syncScenarioList() {
  for (const button of refs.scenarioList.children) {
    const active = button.dataset.scenarioId === state.scenarioId;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  }
}

function syncControls() {
  for (const button of refs.modelPicker.children) {
    const active = button.dataset.model === state.model;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  }

  for (const button of refs.depthPicker.children) {
    const active = Number(button.dataset.depth) === state.depth;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-pressed', String(active));
  }

  refs.messageToggle.classList.toggle('is-active', state.showMessages);
  refs.messageToggle.setAttribute('aria-pressed', String(state.showMessages));
  refs.messageToggle.textContent = state.showMessages ? 'Hide messages' : 'Show messages';
}

function renderHeader() {
  const scenario = getScenario();
  refs.scenarioKicker.textContent = `${scenario.level} / ${scenario.kicker}`;
  refs.scenarioTitle.textContent = scenario.title;
  refs.scenarioObjective.textContent = scenario.prompt;
  refs.candidateName.textContent = scenario.candidate_name;
  refs.candidateMeta.textContent = `${scenario.candidate_meta} | threshold ${Math.round(getThreshold() * 100)}%`;
}

function renderGraph() {
  const scenario = getScenario();
  const messages = localMessages();
  const activeEdges = new Set(messages.map((message) => message.id));
  const positions = new Map(scenario.display.nodes.map((node) => [node.idx, nodePosition(node)]));
  refs.graphSvg.textContent = '';

  const defs = svgEl('defs');
  const marker = svgEl('marker', {
    id: 'arrow-message',
    viewBox: '0 0 10 10',
    refX: '8',
    refY: '5',
    markerWidth: '7',
    markerHeight: '7',
    orient: 'auto',
  });
  marker.appendChild(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: COLORS.active }));
  defs.appendChild(marker);
  refs.graphSvg.appendChild(defs);

  const edgeLayer = svgEl('g', { class: 'edge-layer' });
  scenario.display.edges.forEach((edge, index) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const geometry = edgePath(source, target, index % 2 === 0 ? 16 : -16);
    const active = activeEdges.has(edge.id);
    const path = svgEl('path', {
      d: geometry.d,
      class: `graph-edge${active ? ' is-active' : ''}`,
      'stroke-width': (1.4 + edge.weight * (active ? 3.0 : 1.2)).toFixed(2),
      'marker-end': active && state.showMessages ? 'url(#arrow-message)' : null,
    });
    edgeLayer.appendChild(path);

    if (active && state.showMessages) {
      const label = svgEl('text', {
        x: geometry.labelX.toFixed(1),
        y: (geometry.labelY - 8).toFixed(1),
        class: 'edge-label',
      });
      label.textContent = edge.relation;
      edgeLayer.appendChild(label);
    }
  });
  refs.graphSvg.appendChild(edgeLayer);

  const nodeLayer = svgEl('g', { class: 'node-layer' });
  for (const node of scenario.display.nodes) {
    const pos = positions.get(node.idx);
    const score = nodeSignal(node);
    const fill = signalColor(score);
    const selected = node.idx === state.selectedNodeIdx;
    const isUser = node.idx === scenario.user_node;
    const isCandidate = node.idx === scenario.movie_node;
    const group = svgEl('g', {
      class: `graph-node node-${node.type}${selected ? ' is-selected' : ''}${isUser ? ' is-user-target' : ''}${isCandidate ? ' is-candidate' : ''}`,
      transform: `translate(${pos.x.toFixed(1)}, ${pos.y.toFixed(1)})`,
      role: 'button',
      tabindex: '0',
      'aria-label': node.name,
    });

    const haloRadius = isCandidate ? 48 : isUser ? 44 : 0;
    if (haloRadius) {
      group.appendChild(svgEl('circle', {
        r: String(haloRadius),
        class: isCandidate ? 'candidate-halo' : 'user-halo',
      }));
    }

    if (node.type === 'item') {
      group.appendChild(svgEl('rect', {
        x: '-46',
        y: '-27',
        width: '92',
        height: '54',
        rx: '15',
        class: 'node-shape',
        fill,
      }));
    } else if (node.type === 'topic') {
      group.appendChild(svgEl('path', {
        d: 'M 0 -34 L 35 -12 L 35 18 L 0 36 L -35 18 L -35 -12 Z',
        class: 'node-shape',
        fill,
      }));
    } else {
      group.appendChild(svgEl('circle', {
        r: '34',
        class: 'node-shape',
        fill,
      }));
    }

    const label = svgEl('text', {
      x: '0',
      y: node.type === 'topic' ? '3' : '-2',
      class: 'node-label',
      fill: textColor(fill),
    });
    label.textContent = node.label;
    group.appendChild(label);

    const scoreText = svgEl('text', {
      x: '0',
      y: node.type === 'item' ? '17' : '18',
      class: 'node-score',
      fill: textColor(fill),
    });
    scoreText.textContent = Math.round(score * 100);
    group.appendChild(scoreText);

    const caption = svgEl('text', {
      x: '0',
      y: node.type === 'topic' ? '54' : '52',
      class: 'node-caption',
    });
    caption.textContent = node.type === 'item' ? 'movie' : node.type === 'topic' ? 'genre' : 'user';
    group.appendChild(caption);

    group.addEventListener('click', () => {
      state.selectedNodeIdx = node.idx;
      render();
    });
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        state.selectedNodeIdx = node.idx;
        render();
      }
    });
    nodeLayer.appendChild(group);
  }
  refs.graphSvg.appendChild(nodeLayer);
}

function renderPrediction() {
  const scenario = getScenario();
  const score = scoreForScenarioDepth(scenario, state.depth);
  const prediction = predictionFor(score);
  const correct = prediction === scenario.answer;
  refs.scoreFill.style.width = `${Math.round(score * 100)}%`;
  refs.scoreFill.style.background = signalColor(score);
  refs.runButton.textContent = state.hasRun ? 'Run again' : 'Run prediction';

  if (!state.hasRun) {
    refs.scoreValue.textContent = '--';
    refs.scoreLabel.textContent = state.depth === 0 ? 'Node-only selected' : `${state.depth}-hop LightGCN selected`;
    refs.resultCopy.textContent = 'Run the selected model depth to reveal the literal MovieLens-trained score.';
    refs.resultCopy.className = 'result-copy';
    return;
  }

  refs.scoreValue.textContent = `${Math.round(score * 100)}%`;
  refs.scoreLabel.textContent = `${labelForPrediction(prediction)} | ${confidenceCopy(score)}`;

  const groundTruth = scenario.answer === 'recommend'
    ? `The held-out MovieLens rating was ${scenario.rating} stars.`
    : 'This sampled pair has no positive MovieLens rating.';
  const depthContext = state.depth === state.artifact.meta.train_depth
    ? 'This is the trained inference depth.'
    : depthNote(state.depth);
  refs.resultCopy.textContent = `${correct ? 'Correct.' : 'Wrong.'} ${groundTruth} ${depthContext}`;
  refs.resultCopy.className = `result-copy ${correct ? 'is-correct' : 'is-wrong'}`;
}

function renderComparison() {
  const scenario = getScenario();
  refs.depthComparison.textContent = '';

  getDepths().forEach((depth) => {
    const score = scoreForScenarioDepth(scenario, depth);
    const prediction = predictionFor(score);
    const correct = prediction === scenario.answer;
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `compare-row${depth === state.depth ? ' is-active' : ''}${correct ? ' is-correct' : ' is-wrong'}`;

    appendText(row, 'span', depth === 0 ? 'Base' : `${depth} hop`, 'compare-depth');
    const bar = document.createElement('span');
    bar.className = 'compare-bar';
    const fill = document.createElement('i');
    fill.style.width = `${Math.round(score * 100)}%`;
    fill.style.background = signalColor(score);
    bar.appendChild(fill);
    row.appendChild(bar);
    appendText(row, 'strong', `${Math.round(score * 100)}%`);
    appendText(row, 'small', `${labelForPrediction(prediction)}. ${depthNote(depth)}`);

    row.addEventListener('click', () => {
      state.depth = depth;
      state.model = depth === 0 ? 'base' : 'gnn';
      state.hasRun = false;
      render();
    });
    refs.depthComparison.appendChild(row);
  });
}

function renderInspector() {
  const node = getDisplayNode(state.selectedNodeIdx) ?? getDisplayNode(getScenario().user_node);
  const signal = nodeSignal(node);
  refs.nodeName.textContent = node.name;
  refs.nodeDescription.textContent = node.description;
  refs.nodeStats.textContent = '';

  [
    ['role', node.type === 'item' ? 'movie' : node.type === 'topic' ? 'genre' : 'user'],
    ['signal at depth', `${Math.round(signal * 100)}%`],
    ['weighted degree', node.degree],
    ['graph node', node.idx],
  ].forEach(([label, value]) => {
    const term = document.createElement('dt');
    const detail = document.createElement('dd');
    term.textContent = label;
    detail.textContent = value;
    refs.nodeStats.append(term, detail);
  });
}

function renderMessages() {
  const scenario = getScenario();
  const messages = localMessages(4);
  refs.messageList.textContent = '';

  const intro = document.createElement('p');
  intro.className = 'lesson-copy';
  intro.textContent = scenario.lesson;
  refs.messageList.appendChild(intro);

  const source = document.createElement('p');
  source.className = 'lesson-copy source-note';
  source.textContent = `${state.artifact.meta.model}, trained on ${state.artifact.meta.dataset}: ${state.artifact.meta.validation_accuracy} validation accuracy, ${state.artifact.meta.validation_rank_auc} rank AUC.`;
  refs.messageList.appendChild(source);

  if (state.depth === 0) {
    const item = document.createElement('article');
    item.className = 'message-item';
    const copy = document.createElement('div');
    appendText(copy, 'strong', 'No propagation at depth 0');
    appendText(copy, 'p', 'This score uses only the learned initial user and movie embeddings plus learned biases.');
    item.appendChild(copy);
    refs.messageList.appendChild(item);
    return;
  }

  messages.forEach((message) => {
    const item = document.createElement('article');
    item.className = 'message-item';
    const copy = document.createElement('div');
    appendText(copy, 'strong', `${message.sourceNode.name} -> ${message.targetNode.name}`);
    appendText(copy, 'p', `${message.relation}: one of the largest displayed local messages at depth ${state.depth}.`);
    const value = document.createElement('span');
    value.className = message.value < 0 ? 'is-negative' : 'is-positive';
    value.textContent = `${message.value >= 0 ? '+' : ''}${message.value.toFixed(3)}`;
    item.append(copy, value);
    refs.messageList.appendChild(item);
  });
}

function renderError() {
  refs.scenarioTitle.textContent = 'Could not load model artifact';
  refs.scenarioObjective.textContent = state.error?.message ?? 'Unknown error';
  refs.candidateName.textContent = '--';
  refs.candidateMeta.textContent = '--';
  refs.scoreValue.textContent = '--';
  refs.scoreLabel.textContent = 'Load failed';
  refs.resultCopy.textContent = 'Run the training script or serve this page through a local/static web server so the JSON artifact can be fetched.';
  refs.resultCopy.className = 'result-copy is-wrong';
}

function render() {
  if (state.error) {
    renderError();
    return;
  }
  if (state.loading || !state.artifact) return;

  syncScenarioList();
  syncControls();
  renderHeader();
  renderGraph();
  renderPrediction();
  renderComparison();
  renderInspector();
  renderMessages();
}

async function init() {
  try {
    refs.scenarioTitle.textContent = 'Loading MovieLens model';
    refs.scenarioObjective.textContent = 'Fetching learned graph embeddings and computing browser-side LightGCN layers.';
    const response = await fetch(ARTIFACT_URL);
    if (!response.ok) throw new Error(`Artifact request failed: ${response.status}`);
    const artifact = await response.json();
    state.artifact = artifact;
    prepareRuntime(artifact);
    const firstScenario = artifact.scenarios[0];
    state.scenarioId = firstScenario.id;
    state.depth = firstScenario.best_depth;
    state.model = state.depth === 0 ? 'base' : 'gnn';
    state.selectedNodeIdx = firstScenario.user_node;
    buildScenarioList();
    buildControls();
    state.loading = false;
    render();
  } catch (error) {
    state.loading = false;
    state.error = error;
    render();
  }
}

init();
