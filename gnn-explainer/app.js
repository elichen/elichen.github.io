const MESSAGE_STEP_ORDER = ['features', 'messages', 'aggregate', 'update', 'depth'];
const DEPTH_LEVELS = [0, 1, 2, 4, 6];
const DEPTH_SCENE_IDS = ['twohop', 'oversmooth'];

const state = {
  data: null,
  messageSceneId: null,
  messageStepId: 'features',
  messageNodeId: null,
  stepLockScrollY: null,
  depthSceneId: 'twohop',
  depthLevel: 2,
};

const svgNS = 'http://www.w3.org/2000/svg';

const refs = {
  heroEquation: document.getElementById('hero-equation'),
  heroNote: document.getElementById('hero-note'),
  messageTitle: document.getElementById('message-title'),
  messageDek: document.getElementById('message-dek'),
  messageScenePicker: document.getElementById('message-scene-picker'),
  messageGraph: document.getElementById('message-graph'),
  messageNodeName: document.getElementById('message-node-name'),
  messageNodeH0: document.getElementById('message-node-h0'),
  messageNodeAgg: document.getElementById('message-node-agg'),
  messageNodeH1: document.getElementById('message-node-h1'),
  messageBestDepth: document.getElementById('message-best-depth'),
  messageEquation: document.getElementById('message-equation'),
  messageStageNote: document.getElementById('message-stage-note'),
  messageNodeTitle: document.getElementById('message-node-title'),
  messageNodeSummary: document.getElementById('message-node-summary'),
  messageNodeValues: document.getElementById('message-node-values'),
  messageIncoming: document.getElementById('message-incoming'),
  messageSteps: Array.from(document.querySelectorAll('.figure-step')),
  depthTitle: document.getElementById('depth-title'),
  depthDek: document.getElementById('depth-dek'),
  depthScenePicker: document.getElementById('depth-scene-picker'),
  depthPicker: document.getElementById('depth-picker'),
  depthGraph: document.getElementById('depth-graph'),
  depthTrack: document.getElementById('depth-track'),
  depthLevel: document.getElementById('depth-level'),
  depthNote: document.getElementById('depth-note'),
  depthFocusState: document.getElementById('depth-focus-state'),
  depthFocusProb: document.getElementById('depth-focus-prob'),
  depthSpread: document.getElementById('depth-spread'),
  depthGap: document.getElementById('depth-gap'),
  depthSummary: document.getElementById('depth-summary'),
};

async function loadData() {
  const response = await fetch('./data/experiment-data.json');
  if (!response.ok) {
    throw new Error(`Failed to load publication data: ${response.status}`);
  }
  return response.json();
}

function formatNumber(value, digits = 3) {
  return Number(value).toFixed(digits);
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

function luminance(hex) {
  const { r, g, b } = hexToRgb(hex);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function stateColor(value) {
  if (value <= 0.5) {
    return lerpColor('#5188d8', '#ede5d8', value / 0.5);
  }
  return lerpColor('#ede5d8', '#e87430', (value - 0.5) / 0.5);
}

function textColorForFill(fill) {
  return luminance(fill) < 0.63 ? '#fffaf3' : '#1c1b17';
}

function svgElement(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function getScene(sceneId) {
  return state.data.scenes.find((scene) => scene.id === sceneId) ?? null;
}

function getMessageScene() {
  return getScene(state.messageSceneId);
}

function getDepthScene() {
  return getScene(state.depthSceneId);
}

function findNode(scene, nodeId) {
  return scene.nodes.find((node) => node.id === nodeId) ?? null;
}

function getLayer(scene, depth) {
  return scene.layers.find((layer) => layer.depth === depth) ?? scene.layers[scene.layers.length - 1];
}

function getDepthMetric(scene, depth) {
  return scene.depth_metrics.find((metric) => metric.depth === depth) ?? scene.depth_metrics[scene.depth_metrics.length - 1];
}

function getLayerNode(scene, depth, nodeId) {
  return getLayer(scene, depth).nodes.find((node) => node.id === nodeId) ?? null;
}

function getTransition(scene, fromDepth = 0) {
  return scene.transitions.find((transition) => transition.from_depth === fromDepth) ?? scene.transitions[0];
}

function getAggregate(scene, nodeId, fromDepth = 0) {
  return getTransition(scene, fromDepth).aggregates.find((entry) => entry.id === nodeId) ?? null;
}

function getUpdate(scene, nodeId, fromDepth = 0) {
  return getTransition(scene, fromDepth).updates.find((entry) => entry.id === nodeId) ?? null;
}

function getMessagesInto(scene, nodeId, fromDepth = 0) {
  return getTransition(scene, fromDepth).messages
    .filter((message) => message.target === nodeId)
    .slice()
    .sort((left, right) => right.value - left.value);
}

function edgeKey(source, target) {
  return `${source}->${target}`;
}

function getStrongestMessage(scene, nodeId, fromDepth = 0) {
  return getMessagesInto(scene, nodeId, fromDepth)[0] ?? null;
}

function sceneRoleLabel(node) {
  return node.kind.replace(/-/g, ' ');
}

function messageDisplayDepth(scene) {
  if (state.messageStepId === 'update') {
    return 1;
  }
  if (state.messageStepId === 'depth') {
    return scene.recommended_depth;
  }
  return 0;
}

function incomingHopDistances(scene, targetId, maxDepth) {
  const reverse = new Map();
  scene.edges.forEach((edge) => {
    if (!reverse.has(edge.target)) {
      reverse.set(edge.target, []);
    }
    reverse.get(edge.target).push(edge.source);
  });

  const distances = new Map([[targetId, 0]]);
  const queue = [targetId];

  while (queue.length) {
    const current = queue.shift();
    const currentDistance = distances.get(current);
    if (currentDistance >= maxDepth) {
      continue;
    }
    const incoming = reverse.get(current) ?? [];
    incoming.forEach((source) => {
      if (!distances.has(source)) {
        distances.set(source, currentDistance + 1);
        queue.push(source);
      }
    });
  }

  return distances;
}

function receptiveFieldEdges(scene, distances) {
  const edges = new Set();
  scene.edges.forEach((edge) => {
    const sourceDistance = distances.get(edge.source);
    const targetDistance = distances.get(edge.target);
    if (
      sourceDistance != null
      && targetDistance != null
      && sourceDistance === targetDistance + 1
    ) {
      edges.add(edgeKey(edge.source, edge.target));
    }
  });
  return edges;
}

function formatProbability(probability) {
  return formatNumber(probability, 3);
}

function getSceneTitleLine(scene) {
  return scene.title;
}

function sceneSubtitle(scene) {
  return `${scene.dek} ${scene.task}`;
}

function messageEquationHtml(scene, nodeId) {
  const node = findNode(scene, nodeId);
  const h0 = getLayerNode(scene, 0, nodeId);
  const aggregate = getAggregate(scene, nodeId, 0);
  const update = getUpdate(scene, nodeId, 0);
  const strongest = getStrongestMessage(scene, nodeId, 0);

  if (state.messageStepId === 'features') {
    return `h<sup>(0)</sup><sub>${node.label}</sub> = ${formatNumber(h0.state)}`;
  }

  if (state.messageStepId === 'messages') {
    if (!strongest) {
      return `m<sup>(0)</sup><sub>u→${node.label}</sub> = w<sub>u${node.label}</sub> · h<sup>(0)</sup><sub>u</sub>`;
    }
    const source = findNode(scene, strongest.source);
    const sourceState = getLayerNode(scene, 0, strongest.source);
    return `m<sup>(0)</sup><sub>${source.label}→${node.label}</sub> = ${formatNumber(strongest.weight)} × ${formatNumber(sourceState.state)} = ${formatNumber(strongest.value)}`;
  }

  if (state.messageStepId === 'aggregate') {
    const messages = getMessagesInto(scene, nodeId, 0);
    if (!messages.length) {
      return `a<sup>(0)</sup><sub>${node.label}</sub> = h<sup>(0)</sup><sub>${node.label}</sub>`;
    }
    const numerator = messages.map((message) => formatNumber(message.value)).join(' + ');
    return `a<sup>(0)</sup><sub>${node.label}</sub> = (${numerator}) / ${formatNumber(aggregate.normalizer)} = ${formatNumber(aggregate.value)}`;
  }

  if (state.messageStepId === 'update') {
    return `h<sup>(1)</sup><sub>${node.label}</sub> = 0.35 × ${formatNumber(h0.state)} + 0.65 × ${formatNumber(aggregate.value)} = ${formatNumber(update.next_state)}`;
  }

  const targetDepth = scene.recommended_depth;
  const recommended = getLayerNode(scene, targetDepth, nodeId);
  return `h<sup>(${targetDepth})</sup><sub>${node.label}</sub> = ${formatNumber(recommended.state)} · receptive field = ${targetDepth} hop${targetDepth === 1 ? '' : 's'}`;
}

function messageNodeSummary(scene, nodeId) {
  const node = findNode(scene, nodeId);
  const h0 = getLayerNode(scene, 0, nodeId);
  const h1 = getLayerNode(scene, 1, nodeId);
  const hk = getLayerNode(scene, scene.recommended_depth, nodeId);
  const isFocus = nodeId === scene.focus_node;

  if (isFocus) {
    return `${node.name} is the node we are tracking. It starts at ${formatNumber(h0.state)}, moves to ${formatNumber(h1.state)} after one update, and reaches ${formatNumber(hk.state)} by depth ${scene.recommended_depth}.`;
  }

  return `${node.name} is a ${sceneRoleLabel(node)}. Compare it against the focus node to see how the same update rule plays out in a different position.`;
}

function buildMessageValueRows(scene, nodeId) {
  const h0 = getLayerNode(scene, 0, nodeId);
  const aggregate = getAggregate(scene, nodeId, 0);
  const update = getUpdate(scene, nodeId, 0);
  const recommended = getLayerNode(scene, scene.recommended_depth, nodeId);

  return [
    ['Local state h^0', formatNumber(h0.state)],
    ['Neighborhood aggregate a^0', formatNumber(aggregate.value)],
    ['Updated state h^1', formatNumber(update.next_state)],
    [`State at depth ${scene.recommended_depth}`, formatNumber(recommended.state)],
  ];
}

function buildMessageRows(scene, nodeId) {
  const messages = getMessagesInto(scene, nodeId, 0);
  if (!messages.length) {
    refs.messageIncoming.innerHTML = '<p class="contributors-empty">This node has no incoming edges in the first update, so it keeps its own state.</p>';
    return;
  }

  refs.messageIncoming.innerHTML = '';
  messages.forEach((message) => {
    const source = findNode(scene, message.source);
    const row = document.createElement('div');
    row.className = 'message-row';
    row.innerHTML = `
      <div>
        <div><strong>${source.label}</strong> -> <strong>${findNode(scene, nodeId).label}</strong></div>
        <small>w = ${formatNumber(message.weight)} · h = ${formatNumber(getLayerNode(scene, 0, message.source).state)}</small>
      </div>
      <strong>${formatNumber(message.value)}</strong>
    `;
    refs.messageIncoming.appendChild(row);
  });
}

function lockStepSync() {
  state.stepLockScrollY = window.scrollY;
}

function releaseStepSyncIfScrolled() {
  if (state.stepLockScrollY == null) {
    return true;
  }
  if (Math.abs(window.scrollY - state.stepLockScrollY) > 96) {
    state.stepLockScrollY = null;
    return true;
  }
  return false;
}

function svgSize(svg) {
  const base = svg.viewBox.baseVal;
  return { width: base.width, height: base.height };
}

function nodePosition(node, width, height) {
  const paddingX = width * 0.12;
  const paddingY = height * 0.14;
  return {
    x: paddingX + node.x * (width - paddingX * 2),
    y: paddingY + node.y * (height - paddingY * 2),
  };
}

function edgeGeometry(source, target) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const mx = (source.x + target.x) / 2;
  const my = (source.y + target.y) / 2;
  const normalX = -dy;
  const normalY = dx;
  const normalLength = Math.hypot(normalX, normalY) || 1;
  const sign = source.x <= target.x ? 1 : -1;
  const bend = 18 * sign;
  const cx = mx + (normalX / normalLength) * bend;
  const cy = my + (normalY / normalLength) * bend;
  const labelX = 0.25 * source.x + 0.5 * cx + 0.25 * target.x;
  const labelY = 0.25 * source.y + 0.5 * cy + 0.25 * target.y;
  return {
    path: `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`,
    labelX,
    labelY,
  };
}

function addMarker(svg) {
  const defs = svgElement('defs');
  const marker = svgElement('marker', {
    id: `arrow-${svg.id}`,
    viewBox: '0 0 10 10',
    refX: '9',
    refY: '5',
    markerWidth: '7',
    markerHeight: '7',
    orient: 'auto',
  });
  marker.appendChild(svgElement('path', {
    d: 'M 0 0 L 10 5 L 0 10 z',
    fill: 'rgba(23, 23, 19, 0.32)',
  }));
  defs.appendChild(marker);
  svg.appendChild(defs);
}

function renderBackdrop(svg, scene, width, height) {
  if (scene.id === 'oversmooth') {
    svg.appendChild(svgElement('ellipse', {
      cx: width * 0.24,
      cy: height * 0.5,
      rx: width * 0.18,
      ry: height * 0.34,
      fill: 'rgba(235, 139, 65, 0.08)',
    }));
    svg.appendChild(svgElement('ellipse', {
      cx: width * 0.76,
      cy: height * 0.5,
      rx: width * 0.18,
      ry: height * 0.34,
      fill: 'rgba(106, 154, 231, 0.08)',
    }));
  }
}

function addGraphNote(svg, text, width, height) {
  const note = svgElement('text', {
    x: width * 0.06,
    y: height - 28,
    class: 'graph-note',
  });
  note.textContent = text;
  svg.appendChild(note);
}

function renderGraph(scene, svg, options) {
  const { width, height } = svgSize(svg);
  svg.innerHTML = '';
  addMarker(svg);
  renderBackdrop(svg, scene, width, height);

  const positions = new Map(
    scene.nodes.map((node) => [node.id, nodePosition(node, width, height)]),
  );
  const transition = getTransition(scene, 0);
  const messageMap = new Map(transition.messages.map((message) => [edgeKey(message.source, message.target), message]));
  const distances = options.stage === 'depth' && options.mode === 'message'
    ? incomingHopDistances(scene, options.selectedNodeId, scene.recommended_depth)
    : null;
  const receptiveEdges = distances ? receptiveFieldEdges(scene, distances) : new Set();

  const edgeLayer = svgElement('g');
  scene.edges.forEach((edge) => {
    const sourcePosition = positions.get(edge.source);
    const targetPosition = positions.get(edge.target);
    const geometry = edgeGeometry(sourcePosition, targetPosition);
    const key = edgeKey(edge.source, edge.target);
    const sourceState = getLayerNode(scene, 0, edge.source).state;
    const sourceColor = stateColor(sourceState);
    const isIncomingSelected = edge.target === options.selectedNodeId;
    const isDepthPath = receptiveEdges.has(key);

    let stroke = 'rgba(23, 23, 19, 0.16)';
    let opacity = 0.14;
    let strokeWidth = 1.4 + edge.weight * 1.8;

    if (options.mode === 'message') {
      if (options.stage === 'messages' || options.stage === 'aggregate') {
        stroke = isIncomingSelected ? sourceColor : 'rgba(23, 23, 19, 0.14)';
        opacity = isIncomingSelected ? 0.9 : 0.08;
        strokeWidth = isIncomingSelected ? 2.2 + edge.weight * 2.2 : 1.2;
      } else if (options.stage === 'update') {
        stroke = isIncomingSelected ? sourceColor : 'rgba(23, 23, 19, 0.14)';
        opacity = isIncomingSelected ? 0.54 : 0.1;
      } else if (options.stage === 'depth') {
        stroke = isDepthPath ? sourceColor : 'rgba(23, 23, 19, 0.12)';
        opacity = isDepthPath ? 0.76 : 0.08;
        strokeWidth = isDepthPath ? 2 + edge.weight * 2 : 1.2;
      }
    } else {
      stroke = stateColor(getLayerNode(scene, options.depth, edge.source).state);
      opacity = 0.28 + options.depth * 0.06;
      strokeWidth = 1.2 + edge.weight * 1.7;
    }

    edgeLayer.appendChild(svgElement('path', {
      d: geometry.path,
      class: 'graph-edge',
      stroke,
      opacity: opacity.toFixed(2),
      'stroke-width': strokeWidth.toFixed(2),
      'marker-end': `url(#arrow-${svg.id})`,
    }));

    if (options.mode === 'message' && options.stage === 'messages' && isIncomingSelected) {
      const message = messageMap.get(key);
      const label = svgElement('text', {
        x: geometry.labelX.toFixed(1),
        y: geometry.labelY.toFixed(1),
        class: 'edge-label',
      });
      label.textContent = formatNumber(message.value);
      edgeLayer.appendChild(label);
    }
  });
  svg.appendChild(edgeLayer);

  const nodeLayer = svgElement('g');
  const shownDepth = options.mode === 'message'
    ? messageDisplayDepth(scene)
    : options.depth;

  scene.nodes.forEach((node) => {
    const position = positions.get(node.id);
    const nodeState = getLayerNode(scene, shownDepth, node.id).state;
    const fill = stateColor(nodeState);
    const textFill = textColorForFill(fill);
    const isSelected = node.id === options.selectedNodeId;
    const isFocus = node.id === scene.focus_node;
    const radius = isSelected ? 33 : (node.kind === 'target' || node.kind === 'bridge' ? 30 : 28);
    const group = svgElement('g', {
      class: `graph-node${isSelected ? ' is-selected' : ''}${isFocus ? ' is-focus' : ''}`,
      transform: `translate(${position.x}, ${position.y})`,
      tabindex: options.mode === 'message' ? '0' : '-1',
      role: options.mode === 'message' ? 'button' : 'img',
      'aria-label': `${node.name}`,
    });

    if (options.mode === 'message' && options.stage === 'depth') {
      const hopDistance = distances.get(node.id);
      if (hopDistance === 1 || hopDistance === 2) {
        group.appendChild(svgElement('circle', {
          r: (radius + 10 + hopDistance * 4).toFixed(1),
          class: `hop-halo-${hopDistance}`,
        }));
      }
    }

    if (options.mode === 'depth' && isFocus) {
      group.appendChild(svgElement('circle', {
        r: (radius + 10).toFixed(1),
        class: 'hop-halo-1',
      }));
    }

    group.appendChild(svgElement('circle', {
      r: radius.toFixed(1),
      class: 'node-shell',
      fill,
      stroke: 'rgba(23, 23, 19, 0.18)',
      'stroke-width': '2.4',
    }));

    const label = svgElement('text', {
      x: '0',
      y: '1',
      class: 'node-label',
      fill: textFill,
    });
    label.textContent = node.label;
    group.appendChild(label);

    const caption = svgElement('text', {
      x: '0',
      y: (radius + 20).toFixed(1),
      class: 'node-caption',
    });
    caption.textContent = node.kind;
    group.appendChild(caption);

    if (options.mode === 'message') {
      group.addEventListener('click', () => {
        state.messageNodeId = node.id;
        renderMessageFigure();
      });
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          state.messageNodeId = node.id;
          renderMessageFigure();
        }
      });
    }

    nodeLayer.appendChild(group);
  });

  svg.appendChild(nodeLayer);

  if (options.mode === 'message') {
    const note = state.messageStepId === 'depth'
      ? `Receptive field: ${scene.recommended_depth} incoming hop${scene.recommended_depth === 1 ? '' : 's'}`
      : `Color encodes orange confidence at depth ${shownDepth}.`;
    addGraphNote(svg, note, width, height);
  } else {
    addGraphNote(svg, `Depth ${options.depth} · focus node = ${findNode(scene, scene.focus_node).label}`, width, height);
  }
}

function syncMessageSteps() {
  refs.messageSteps.forEach((step) => {
    step.classList.toggle('is-active', step.dataset.step === state.messageStepId);
  });
}

function renderMessageScenePicker() {
  const scenes = state.data.scenes.filter((scene) => scene.figure === 'message');
  refs.messageScenePicker.innerHTML = '';
  scenes.forEach((scene) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = scene.name;
    button.classList.toggle('is-active', scene.id === state.messageSceneId);
    button.addEventListener('click', () => {
      state.messageSceneId = scene.id;
      state.messageNodeId = scene.focus_node;
      renderMessageFigure();
    });
    refs.messageScenePicker.appendChild(button);
  });
}

function renderMessageFigure() {
  const scene = getMessageScene();
  if (!scene) {
    return;
  }
  if (!findNode(scene, state.messageNodeId)) {
    state.messageNodeId = scene.focus_node;
  }

  const node = findNode(scene, state.messageNodeId);
  const h0 = getLayerNode(scene, 0, node.id);
  const aggregate = getAggregate(scene, node.id, 0);
  const h1 = getLayerNode(scene, 1, node.id);
  const recommended = getLayerNode(scene, scene.recommended_depth, node.id);

  refs.messageTitle.textContent = getSceneTitleLine(scene);
  refs.messageDek.textContent = sceneSubtitle(scene);
  refs.messageNodeName.textContent = `${node.label} · ${sceneRoleLabel(node)}`;
  refs.messageNodeH0.textContent = formatNumber(h0.state);
  refs.messageNodeAgg.textContent = formatNumber(aggregate.value);
  refs.messageNodeH1.textContent = formatNumber(h1.state);
  refs.messageBestDepth.textContent = `${scene.recommended_depth} layer${scene.recommended_depth === 1 ? '' : 's'}`;
  refs.messageEquation.innerHTML = messageEquationHtml(scene, node.id);
  refs.messageStageNote.textContent = scene.stage_notes[state.messageStepId];
  refs.messageNodeTitle.textContent = node.name;
  refs.messageNodeSummary.textContent = messageNodeSummary(scene, node.id);

  refs.messageNodeValues.innerHTML = '';
  buildMessageValueRows(scene, node.id).forEach(([label, value]) => {
    const row = document.createElement('div');
    row.className = 'value-row';
    row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    refs.messageNodeValues.appendChild(row);
  });

  buildMessageRows(scene, node.id);
  renderMessageScenePicker();
  syncMessageSteps();
  renderGraph(scene, refs.messageGraph, {
    mode: 'message',
    stage: state.messageStepId,
    selectedNodeId: node.id,
  });
}

function renderDepthScenePicker() {
  refs.depthScenePicker.innerHTML = '';
  DEPTH_SCENE_IDS.forEach((sceneId) => {
    const scene = getScene(sceneId);
    if (!scene) {
      return;
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = scene.name;
    button.classList.toggle('is-active', scene.id === state.depthSceneId);
    button.addEventListener('click', () => {
      state.depthSceneId = scene.id;
      if (scene.id === 'twohop' && state.depthLevel === 6) {
        state.depthLevel = 2;
      }
      renderDepthFigure();
    });
    refs.depthScenePicker.appendChild(button);
  });
}

function renderDepthPicker() {
  refs.depthPicker.innerHTML = '';
  DEPTH_LEVELS.forEach((depth) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = depth.toString();
    button.classList.toggle('is-active', depth === state.depthLevel);
    button.addEventListener('click', () => {
      state.depthLevel = depth;
      renderDepthFigure();
    });
    refs.depthPicker.appendChild(button);
  });
}

function renderDepthTrack(scene) {
  refs.depthTrack.innerHTML = '';
  DEPTH_LEVELS.forEach((depth) => {
    const metric = getDepthMetric(scene, depth);
    const chip = document.createElement('div');
    chip.className = `depth-chip${depth === state.depthLevel ? ' is-active' : ''}`;
    chip.innerHTML = `
      <span>Depth ${depth}</span>
      <strong>${formatNumber(metric.focus_state)}</strong>
      <small>p = ${formatProbability(metric.focus_probability)} · gap = ${formatNumber(metric.community_gap)}</small>
    `;
    refs.depthTrack.appendChild(chip);
  });
}

function reachableNodeCount(scene, depth) {
  return incomingHopDistances(scene, scene.focus_node, depth).size;
}

function depthSummary(scene, depth) {
  const metric = getDepthMetric(scene, depth);
  const note = scene.depth_notes[String(depth)] ?? scene.depth_notes['6'];
  const reach = reachableNodeCount(scene, depth);
  return `${note} At this depth the focus node has ${reach} node${reach === 1 ? '' : 's'} within range. Graph-wide state spread: ${formatNumber(metric.spread)}.`;
}

function renderDepthFigure() {
  const scene = getDepthScene();
  if (!scene) {
    return;
  }
  const metric = getDepthMetric(scene, state.depthLevel);

  refs.depthTitle.textContent = scene.title;
  refs.depthDek.textContent = scene.depth_prompt;
  refs.depthLevel.textContent = `${state.depthLevel} layer${state.depthLevel === 1 ? '' : 's'}`;
  refs.depthNote.textContent = scene.depth_notes[String(state.depthLevel)] ?? '';
  refs.depthFocusState.textContent = formatNumber(metric.focus_state);
  refs.depthFocusProb.textContent = formatProbability(metric.focus_probability);
  refs.depthSpread.textContent = formatNumber(metric.spread);
  refs.depthGap.textContent = formatNumber(metric.community_gap);
  refs.depthSummary.textContent = depthSummary(scene, state.depthLevel);

  renderDepthScenePicker();
  renderDepthPicker();
  renderDepthTrack(scene);
  renderGraph(scene, refs.depthGraph, {
    mode: 'depth',
    depth: state.depthLevel,
    selectedNodeId: scene.focus_node,
  });
}

function populateHero() {
  refs.heroEquation.innerHTML = `h<sup>(k+1)</sup><sub>v</sub> = ${formatNumber(state.data.meta.self_weight, 2)} · h<sup>(k)</sup><sub>v</sub> + ${formatNumber(state.data.meta.neighbor_weight, 2)} · a<sup>(k)</sup><sub>v</sub>`;
  refs.heroNote.textContent = state.data.meta.note;
}

function attachMessageStepEvents() {
  refs.messageSteps.forEach((step) => {
    step.addEventListener('click', () => {
      lockStepSync();
      state.messageStepId = step.dataset.step;
      renderMessageFigure();
    });
    step.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        lockStepSync();
        state.messageStepId = step.dataset.step;
        renderMessageFigure();
      }
    });
  });
}

function attachMessageStepObserver() {
  const visibility = new Map();
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      visibility.set(entry.target.dataset.step, entry.isIntersecting ? entry.intersectionRatio : 0);
    });

    if (!releaseStepSyncIfScrolled()) {
      return;
    }

    const best = refs.messageSteps
      .map((step) => ({ step, ratio: visibility.get(step.dataset.step) ?? 0 }))
      .sort((left, right) => right.ratio - left.ratio)[0];

    if (best && best.ratio > 0.2 && best.step.dataset.step !== state.messageStepId) {
      state.messageStepId = best.step.dataset.step;
      renderMessageFigure();
    }
  }, {
    threshold: [0, 0.2, 0.4, 0.6, 0.8],
    rootMargin: '-18% 0px -24% 0px',
  });

  refs.messageSteps.forEach((step) => observer.observe(step));
}

async function init() {
  state.data = await loadData();
  state.messageSceneId = state.data.scenes.find((scene) => scene.figure === 'message')?.id ?? state.data.scenes[0].id;
  state.messageNodeId = getMessageScene().focus_node;

  populateHero();
  renderMessageFigure();
  renderDepthFigure();
  attachMessageStepEvents();
  attachMessageStepObserver();
}

init().catch((error) => {
  refs.messageTitle.textContent = 'Failed to load';
  refs.messageDek.textContent = error.message;
  console.error(error);
});
