// ─────────────────────────────────────────────────────────────────────────────
// Constants & theme
// ─────────────────────────────────────────────────────────────────────────────
const MESSAGE_STEP_ORDER = ['features', 'messages', 'aggregate', 'update', 'depth'];
const DEPTH_LEVELS = [0, 1, 2, 4, 6];
const DEPTH_SCENE_IDS = ['twohop', 'oversmooth'];

const THEME = {
  stateLow:    '#5188d8',
  stateMid:    '#ede5d8',
  stateHigh:   '#e87430',
  edgeIdle:    'rgba(15, 17, 21, 0.14)',
  edgeMuted:   'rgba(15, 17, 21, 0.10)',
  arrowFill:   'rgba(15, 17, 21, 0.34)',
  nodeStroke:  'rgba(15, 17, 21, 0.22)',
  labelLight:  '#fbfaf6',
  labelDark:   '#101114',
  labelPivot:  0.63,
};

const SVG_NS = 'http://www.w3.org/2000/svg';
const REDUCED_MOTION = typeof matchMedia === 'function'
  && matchMedia('(prefers-reduced-motion: reduce)').matches;
const STEP_CLICK_SUPPRESS_MS = 650;

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
const state = {
  data: null,
  messageSceneId: null,
  messageStepId: 'features',
  messageNodeId: null,
  depthSceneId: 'twohop',
  depthLevel: 2,
  // observer suppression — time-based, clears itself
  observerSuppressedUntil: 0,
};

// Per-scene memoization keyed by scene.id
const sceneCache = new Map();

// ─────────────────────────────────────────────────────────────────────────────
// DOM refs
// ─────────────────────────────────────────────────────────────────────────────
const refs = {
  heroEquation:        document.getElementById('hero-equation'),
  heroNote:            document.getElementById('hero-note'),
  messageTitle:        document.getElementById('message-title'),
  messageDek:          document.getElementById('message-dek'),
  messageScenePicker:  document.getElementById('message-scene-picker'),
  messageGraph:        document.getElementById('message-graph'),
  messageNodeName:     document.getElementById('message-node-name'),
  messageNodeH0:       document.getElementById('message-node-h0'),
  messageNodeAgg:      document.getElementById('message-node-agg'),
  messageNodeH1:       document.getElementById('message-node-h1'),
  messageBestDepth:    document.getElementById('message-best-depth'),
  messageEquation:     document.getElementById('message-equation'),
  messageStageNote:    document.getElementById('message-stage-note'),
  messageNodeTitle:    document.getElementById('message-node-title'),
  messageNodeSummary:  document.getElementById('message-node-summary'),
  messageNodeValues:   document.getElementById('message-node-values'),
  messageIncoming:     document.getElementById('message-incoming'),
  messageSteps:        Array.from(document.querySelectorAll('.figure-step')),
  depthTitle:          document.getElementById('depth-title'),
  depthDek:            document.getElementById('depth-dek'),
  depthScenePicker:    document.getElementById('depth-scene-picker'),
  depthPicker:         document.getElementById('depth-picker'),
  depthGraph:          document.getElementById('depth-graph'),
  depthTrack:          document.getElementById('depth-track'),
  depthLevel:          document.getElementById('depth-level'),
  depthNote:           document.getElementById('depth-note'),
  depthFocusState:     document.getElementById('depth-focus-state'),
  depthFocusProb:      document.getElementById('depth-focus-prob'),
  depthSpread:         document.getElementById('depth-spread'),
  depthGap:            document.getElementById('depth-gap'),
  depthSummary:        document.getElementById('depth-summary'),
};

// ─────────────────────────────────────────────────────────────────────────────
// Small utilities
// ─────────────────────────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (v, digits = 3) => Number(v).toFixed(digits);

const hexToRgb = (hex) => {
  const h = hex.replace('#', '');
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
};

const rgbToHex = ({ r, g, b }) =>
  `#${[r, g, b].map((c) => c.toString(16).padStart(2, '0')).join('')}`;

const lerpColor = (a, b, t) => {
  const l = hexToRgb(a);
  const r = hexToRgb(b);
  const k = clamp(t, 0, 1);
  return rgbToHex({
    r: Math.round(l.r + (r.r - l.r) * k),
    g: Math.round(l.g + (r.g - l.g) * k),
    b: Math.round(l.b + (r.b - l.b) * k),
  });
};

const luminance = (hex) => {
  const { r, g, b } = hexToRgb(hex);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
};

const stateColor = (value) => value <= 0.5
  ? lerpColor(THEME.stateLow,  THEME.stateMid,  value / 0.5)
  : lerpColor(THEME.stateMid,  THEME.stateHigh, (value - 0.5) / 0.5);

const textColorForFill = (fill) =>
  luminance(fill) < THEME.labelPivot ? THEME.labelLight : THEME.labelDark;

const svgEl = (name, attrs = {}) => {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v != null) node.setAttribute(k, v);
  }
  return node;
};

// ─────────────────────────────────────────────────────────────────────────────
// Scene lookups (memoized per-scene-id where it's worth it)
// ─────────────────────────────────────────────────────────────────────────────
function getScene(sceneId) {
  return state.data.scenes.find((s) => s.id === sceneId) ?? null;
}
const getMessageScene = () => getScene(state.messageSceneId);
const getDepthScene   = () => getScene(state.depthSceneId);

function getCacheFor(sceneId) {
  let c = sceneCache.get(sceneId);
  if (!c) {
    c = {
      nodesById: null,
      layersByDepth: null,
      transitionsByFrom: null,
      messagesByTarget: null,
      hopDistances: new Map(), // key: `${fromId}:${maxDepth}`
      reachCount: new Map(),   // key: depth
    };
    sceneCache.set(sceneId, c);
  }
  return c;
}

function indexOf(arr, key) {
  const m = new Map();
  for (const item of arr) m.set(item[key], item);
  return m;
}

function findNode(scene, nodeId) {
  const c = getCacheFor(scene.id);
  if (!c.nodesById) c.nodesById = indexOf(scene.nodes, 'id');
  return c.nodesById.get(nodeId) ?? null;
}

function getLayer(scene, depth) {
  const c = getCacheFor(scene.id);
  if (!c.layersByDepth) {
    c.layersByDepth = new Map(scene.layers.map((l) => [l.depth, l]));
  }
  return c.layersByDepth.get(depth) ?? scene.layers[scene.layers.length - 1];
}

function getLayerNode(scene, depth, nodeId) {
  return getLayer(scene, depth).nodes.find((n) => n.id === nodeId) ?? null;
}

function getTransition(scene, fromDepth = 0) {
  const c = getCacheFor(scene.id);
  if (!c.transitionsByFrom) {
    c.transitionsByFrom = new Map(scene.transitions.map((t) => [t.from_depth, t]));
  }
  return c.transitionsByFrom.get(fromDepth) ?? scene.transitions[0];
}

function getAggregate(scene, nodeId, fromDepth = 0) {
  return getTransition(scene, fromDepth).aggregates.find((e) => e.id === nodeId) ?? null;
}

function getUpdate(scene, nodeId, fromDepth = 0) {
  return getTransition(scene, fromDepth).updates.find((e) => e.id === nodeId) ?? null;
}

function getMessagesInto(scene, nodeId, fromDepth = 0) {
  const c = getCacheFor(scene.id);
  if (fromDepth === 0) {
    if (!c.messagesByTarget) {
      const m = new Map();
      for (const msg of getTransition(scene, 0).messages) {
        if (!m.has(msg.target)) m.set(msg.target, []);
        m.get(msg.target).push(msg);
      }
      for (const list of m.values()) list.sort((a, b) => b.value - a.value);
      c.messagesByTarget = m;
    }
    return c.messagesByTarget.get(nodeId) ?? [];
  }
  return getTransition(scene, fromDepth).messages
    .filter((msg) => msg.target === nodeId)
    .sort((a, b) => b.value - a.value);
}

function getDepthMetric(scene, depth) {
  return scene.depth_metrics.find((m) => m.depth === depth)
    ?? scene.depth_metrics[scene.depth_metrics.length - 1];
}

function getStrongestMessage(scene, nodeId, fromDepth = 0) {
  return getMessagesInto(scene, nodeId, fromDepth)[0] ?? null;
}

const edgeKey = (source, target) => `${source}->${target}`;
const sceneRoleLabel = (node) => node.kind.replace(/-/g, ' ');

// ─────────────────────────────────────────────────────────────────────────────
// Graph algorithms (cached)
// ─────────────────────────────────────────────────────────────────────────────
function incomingHopDistances(scene, targetId, maxDepth) {
  const c = getCacheFor(scene.id);
  const key = `${targetId}:${maxDepth}`;
  const hit = c.hopDistances.get(key);
  if (hit) return hit;

  const reverse = new Map();
  for (const edge of scene.edges) {
    if (!reverse.has(edge.target)) reverse.set(edge.target, []);
    reverse.get(edge.target).push(edge.source);
  }

  const distances = new Map([[targetId, 0]]);
  const queue = [targetId];
  while (queue.length) {
    const current = queue.shift();
    const d = distances.get(current);
    if (d >= maxDepth) continue;
    for (const src of reverse.get(current) ?? []) {
      if (!distances.has(src)) {
        distances.set(src, d + 1);
        queue.push(src);
      }
    }
  }

  c.hopDistances.set(key, distances);
  return distances;
}

function receptiveFieldEdges(scene, distances) {
  const edges = new Set();
  for (const edge of scene.edges) {
    const s = distances.get(edge.source);
    const t = distances.get(edge.target);
    if (s != null && t != null && s === t + 1) edges.add(edgeKey(edge.source, edge.target));
  }
  return edges;
}

function reachableNodeCount(scene, depth) {
  const c = getCacheFor(scene.id);
  if (c.reachCount.has(depth)) return c.reachCount.get(depth);
  const n = incomingHopDistances(scene, scene.focus_node, depth).size;
  c.reachCount.set(depth, n);
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Message-figure copy generation
// ─────────────────────────────────────────────────────────────────────────────
function messageDisplayDepth(scene) {
  if (state.messageStepId === 'update') return 1;
  if (state.messageStepId === 'depth')  return scene.recommended_depth;
  return 0;
}

function messageEquationHtml(scene, nodeId) {
  const node = findNode(scene, nodeId);
  const h0 = getLayerNode(scene, 0, nodeId);
  const aggregate = getAggregate(scene, nodeId, 0);
  const update = getUpdate(scene, nodeId, 0);
  const strongest = getStrongestMessage(scene, nodeId, 0);

  switch (state.messageStepId) {
    case 'features':
      return `h<sup>(0)</sup><sub>${node.label}</sub> = ${fmt(h0.state)}`;

    case 'messages': {
      if (!strongest) {
        return `m<sup>(0)</sup><sub>u→${node.label}</sub> = w<sub>u${node.label}</sub> · h<sup>(0)</sup><sub>u</sub>`;
      }
      const source = findNode(scene, strongest.source);
      const sourceState = getLayerNode(scene, 0, strongest.source);
      return `m<sup>(0)</sup><sub>${source.label}→${node.label}</sub> = ${fmt(strongest.weight)} × ${fmt(sourceState.state)} = ${fmt(strongest.value)}`;
    }

    case 'aggregate': {
      const messages = getMessagesInto(scene, nodeId, 0);
      if (!messages.length) {
        return `a<sup>(0)</sup><sub>${node.label}</sub> = h<sup>(0)</sup><sub>${node.label}</sub>`;
      }
      const numerator = messages.map((m) => fmt(m.value)).join(' + ');
      return `a<sup>(0)</sup><sub>${node.label}</sub> = (${numerator}) / ${fmt(aggregate.normalizer)} = ${fmt(aggregate.value)}`;
    }

    case 'update':
      return `h<sup>(1)</sup><sub>${node.label}</sub> = 0.35 × ${fmt(h0.state)} + 0.65 × ${fmt(aggregate.value)} = ${fmt(update.next_state)}`;

    case 'depth':
    default: {
      const k = scene.recommended_depth;
      const atK = getLayerNode(scene, k, nodeId);
      return `h<sup>(${k})</sup><sub>${node.label}</sub> = ${fmt(atK.state)} · receptive field = ${k} hop${k === 1 ? '' : 's'}`;
    }
  }
}

function messageNodeSummary(scene, nodeId) {
  const node = findNode(scene, nodeId);
  const h0 = getLayerNode(scene, 0, nodeId);
  const h1 = getLayerNode(scene, 1, nodeId);
  const hk = getLayerNode(scene, scene.recommended_depth, nodeId);
  const isFocus = nodeId === scene.focus_node;

  if (isFocus) {
    return `${node.name} is the node we are tracking. It starts at ${fmt(h0.state)}, moves to ${fmt(h1.state)} after one update, and reaches ${fmt(hk.state)} by depth ${scene.recommended_depth}.`;
  }
  return `${node.name} is a ${sceneRoleLabel(node)}. Compare it against the focus node to see how the same update rule plays out in a different position.`;
}

function buildMessageValueRows(scene, nodeId) {
  const h0 = getLayerNode(scene, 0, nodeId);
  const aggregate = getAggregate(scene, nodeId, 0);
  const update = getUpdate(scene, nodeId, 0);
  const recommended = getLayerNode(scene, scene.recommended_depth, nodeId);
  return [
    ['Local state h^0', fmt(h0.state)],
    ['Neighborhood aggregate a^0', fmt(aggregate.value)],
    ['Updated state h^1', fmt(update.next_state)],
    [`State at depth ${scene.recommended_depth}`, fmt(recommended.state)],
  ];
}

function renderMessageRows(scene, nodeId) {
  const messages = getMessagesInto(scene, nodeId, 0);
  const host = refs.messageIncoming;
  host.textContent = '';

  if (!messages.length) {
    const empty = document.createElement('p');
    empty.className = 'contributors-empty';
    empty.textContent = 'This node has no incoming edges in the first update, so it keeps its own state.';
    host.appendChild(empty);
    return;
  }

  const targetLabel = findNode(scene, nodeId).label;
  for (const message of messages) {
    const source = findNode(scene, message.source);
    const row = document.createElement('div');
    row.className = 'message-row';
    row.innerHTML = `
      <div>
        <div><strong>${source.label}</strong> → <strong>${targetLabel}</strong></div>
        <small>w = ${fmt(message.weight)} · h = ${fmt(getLayerNode(scene, 0, message.source).state)}</small>
      </div>
      <strong>${fmt(message.value)}</strong>
    `;
    host.appendChild(row);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SVG rendering
// ─────────────────────────────────────────────────────────────────────────────
function svgSize(svg) {
  const vb = svg.viewBox.baseVal;
  return { width: vb.width, height: vb.height };
}

function nodePosition(node, width, height) {
  const padX = width * 0.12;
  const padY = height * 0.14;
  return {
    x: padX + node.x * (width  - padX * 2),
    y: padY + node.y * (height - padY * 2),
  };
}

function edgeGeometry(source, target) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const mx = (source.x + target.x) / 2;
  const my = (source.y + target.y) / 2;
  const nx = -dy;
  const ny = dx;
  const nLen = Math.hypot(nx, ny) || 1;
  const sign = source.x <= target.x ? 1 : -1;
  const bend = 18 * sign;
  const cx = mx + (nx / nLen) * bend;
  const cy = my + (ny / nLen) * bend;
  return {
    path: `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`,
    labelX: 0.25 * source.x + 0.5 * cx + 0.25 * target.x,
    labelY: 0.25 * source.y + 0.5 * cy + 0.25 * target.y,
  };
}

function addMarker(svg) {
  const defs = svgEl('defs');
  const marker = svgEl('marker', {
    id: `arrow-${svg.id}`,
    viewBox: '0 0 10 10',
    refX: '9', refY: '5',
    markerWidth: '7', markerHeight: '7',
    orient: 'auto',
  });
  marker.appendChild(svgEl('path', {
    d: 'M 0 0 L 10 5 L 0 10 z',
    fill: THEME.arrowFill,
  }));
  defs.appendChild(marker);
  svg.appendChild(defs);
}

function renderBackdrop(svg, scene, width, height) {
  if (scene.id !== 'oversmooth') return;
  svg.appendChild(svgEl('ellipse', {
    cx: width * 0.24, cy: height * 0.5,
    rx: width * 0.18, ry: height * 0.34,
    fill: 'rgba(235, 139, 65, 0.08)',
  }));
  svg.appendChild(svgEl('ellipse', {
    cx: width * 0.76, cy: height * 0.5,
    rx: width * 0.18, ry: height * 0.34,
    fill: 'rgba(106, 154, 231, 0.08)',
  }));
}

function addGraphNote(svg, text, width, height) {
  const note = svgEl('text', {
    x: width * 0.06,
    y: height - 28,
    class: 'graph-note',
  });
  note.textContent = text;
  svg.appendChild(note);
}

function computeEdgeStyle(edge, { mode, stage, depth, selectedNodeId, sourceState, isIncomingSelected, isDepthPath }) {
  const sourceColor = stateColor(sourceState);

  if (mode === 'message') {
    if (stage === 'messages' || stage === 'aggregate') {
      return {
        stroke:      isIncomingSelected ? sourceColor : THEME.edgeIdle,
        opacity:     isIncomingSelected ? 0.9 : 0.08,
        strokeWidth: isIncomingSelected ? 2.2 + edge.weight * 2.2 : 1.2,
      };
    }
    if (stage === 'update') {
      return {
        stroke:      isIncomingSelected ? sourceColor : THEME.edgeIdle,
        opacity:     isIncomingSelected ? 0.54 : 0.1,
        strokeWidth: 1.4 + edge.weight * 1.8,
      };
    }
    if (stage === 'depth') {
      return {
        stroke:      isDepthPath ? sourceColor : THEME.edgeMuted,
        opacity:     isDepthPath ? 0.76 : 0.08,
        strokeWidth: isDepthPath ? 2 + edge.weight * 2 : 1.2,
      };
    }
    return { stroke: THEME.edgeIdle, opacity: 0.14, strokeWidth: 1.4 + edge.weight * 1.8 };
  }

  // depth mode
  return {
    stroke:      sourceColor,
    opacity:     0.28 + depth * 0.06,
    strokeWidth: 1.2 + edge.weight * 1.7,
  };
}

function renderGraph(scene, svg, options) {
  const { width, height } = svgSize(svg);
  svg.textContent = '';
  addMarker(svg);
  renderBackdrop(svg, scene, width, height);

  const positions = new Map(scene.nodes.map((n) => [n.id, nodePosition(n, width, height)]));
  const transition = getTransition(scene, 0);
  const messageMap = new Map(transition.messages.map((m) => [edgeKey(m.source, m.target), m]));
  const distances = (options.mode === 'message' && options.stage === 'depth')
    ? incomingHopDistances(scene, options.selectedNodeId, scene.recommended_depth)
    : null;
  const receptiveEdges = distances ? receptiveFieldEdges(scene, distances) : new Set();

  // ── edges
  const edgeLayer = svgEl('g');
  for (const edge of scene.edges) {
    const s = positions.get(edge.source);
    const t = positions.get(edge.target);
    const geo = edgeGeometry(s, t);
    const key = edgeKey(edge.source, edge.target);
    const style = computeEdgeStyle(edge, {
      mode: options.mode,
      stage: options.stage,
      depth: options.depth,
      selectedNodeId: options.selectedNodeId,
      sourceState: getLayerNode(scene, options.mode === 'message' ? 0 : options.depth, edge.source).state,
      isIncomingSelected: edge.target === options.selectedNodeId,
      isDepthPath: receptiveEdges.has(key),
    });

    edgeLayer.appendChild(svgEl('path', {
      d: geo.path,
      class: 'graph-edge',
      stroke: style.stroke,
      opacity: style.opacity.toFixed(2),
      'stroke-width': style.strokeWidth.toFixed(2),
      'marker-end': `url(#arrow-${svg.id})`,
    }));

    if (options.mode === 'message' && options.stage === 'messages' && edge.target === options.selectedNodeId) {
      const msg = messageMap.get(key);
      if (msg) {
        const label = svgEl('text', {
          x: geo.labelX.toFixed(1),
          y: geo.labelY.toFixed(1),
          class: 'edge-label',
        });
        label.textContent = fmt(msg.value);
        edgeLayer.appendChild(label);
      }
    }
  }
  svg.appendChild(edgeLayer);

  // ── nodes
  const shownDepth = options.mode === 'message' ? messageDisplayDepth(scene) : options.depth;
  const nodeLayer = svgEl('g');

  for (const node of scene.nodes) {
    const pos = positions.get(node.id);
    const nodeState = getLayerNode(scene, shownDepth, node.id).state;
    const fill = stateColor(nodeState);
    const textFill = textColorForFill(fill);
    const isSelected = node.id === options.selectedNodeId;
    const isFocus = node.id === scene.focus_node;
    const radius = isSelected ? 33
      : (node.kind === 'target' || node.kind === 'bridge' ? 30 : 28);

    const group = svgEl('g', {
      class: `graph-node${isSelected ? ' is-selected' : ''}${isFocus ? ' is-focus' : ''}`,
      transform: `translate(${pos.x}, ${pos.y})`,
      tabindex: options.mode === 'message' ? '0' : '-1',
      role: options.mode === 'message' ? 'button' : 'img',
      'aria-label': node.name,
      'aria-pressed': options.mode === 'message' ? String(isSelected) : null,
    });

    // hop halos (depth stage in message mode)
    if (options.mode === 'message' && options.stage === 'depth') {
      const d = distances.get(node.id);
      if (d === 1 || d === 2) {
        group.appendChild(svgEl('circle', {
          r: (radius + 10 + d * 4).toFixed(1),
          class: `hop-halo-${d}`,
        }));
      }
    }
    if (options.mode === 'depth' && isFocus) {
      group.appendChild(svgEl('circle', {
        r: (radius + 10).toFixed(1),
        class: 'hop-halo-1',
      }));
    }

    group.appendChild(svgEl('circle', {
      r: radius.toFixed(1),
      class: 'node-shell',
      fill,
      stroke: THEME.nodeStroke,
      'stroke-width': '2.4',
    }));

    const label = svgEl('text', { x: '0', y: '1', class: 'node-label', fill: textFill });
    label.textContent = node.label;
    group.appendChild(label);

    const caption = svgEl('text', {
      x: '0',
      y: (radius + 20).toFixed(1),
      class: 'node-caption',
    });
    caption.textContent = node.kind;
    group.appendChild(caption);

    if (options.mode === 'message') {
      const pick = () => {
        state.messageNodeId = node.id;
        renderMessageFigure();
      };
      group.addEventListener('click', pick);
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          pick();
        }
      });
    }

    nodeLayer.appendChild(group);
  }
  svg.appendChild(nodeLayer);

  const note = options.mode === 'message'
    ? (state.messageStepId === 'depth'
        ? `Receptive field: ${scene.recommended_depth} incoming hop${scene.recommended_depth === 1 ? '' : 's'}`
        : `Color encodes orange confidence at depth ${shownDepth}.`)
    : `Depth ${options.depth} · focus node = ${findNode(scene, scene.focus_node).label}`;
  addGraphNote(svg, note, width, height);
}

// ─────────────────────────────────────────────────────────────────────────────
// Segmented controls (built once, then just re-flag active)
// ─────────────────────────────────────────────────────────────────────────────
function buildSegmented(host, items, getLabel, getId, onPick) {
  host.textContent = '';
  host.setAttribute('role', 'tablist');

  items.forEach((item, index) => {
    const id = getId(item);
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('role', 'tab');
    button.dataset.value = String(id);
    button.textContent = getLabel(item);
    button.addEventListener('click', () => onPick(id));
    button.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const delta = event.key === 'ArrowRight' ? 1 : -1;
      const next = items[(index + delta + items.length) % items.length];
      onPick(getId(next));
      host.querySelector(`[data-value="${getId(next)}"]`)?.focus();
    });
    host.appendChild(button);
  });
}

function syncSegmented(host, activeValue) {
  for (const button of host.children) {
    const active = button.dataset.value === String(activeValue);
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Figures
// ─────────────────────────────────────────────────────────────────────────────
function syncMessageSteps() {
  for (const step of refs.messageSteps) {
    const active = step.dataset.step === state.messageStepId;
    step.classList.toggle('is-active', active);
    step.setAttribute('aria-selected', String(active));
  }
}

function renderMessageFigure() {
  const scene = getMessageScene();
  if (!scene) return;
  if (!findNode(scene, state.messageNodeId)) {
    state.messageNodeId = scene.focus_node;
  }

  const node = findNode(scene, state.messageNodeId);
  const h0 = getLayerNode(scene, 0, node.id);
  const aggregate = getAggregate(scene, node.id, 0);
  const h1 = getLayerNode(scene, 1, node.id);

  refs.messageTitle.textContent      = scene.title;
  refs.messageDek.textContent        = `${scene.dek} ${scene.task}`;
  refs.messageNodeName.textContent   = `${node.label} · ${sceneRoleLabel(node)}`;
  refs.messageNodeH0.textContent     = fmt(h0.state);
  refs.messageNodeAgg.textContent    = fmt(aggregate.value);
  refs.messageNodeH1.textContent     = fmt(h1.state);
  refs.messageBestDepth.textContent  = `${scene.recommended_depth} layer${scene.recommended_depth === 1 ? '' : 's'}`;
  refs.messageEquation.innerHTML     = messageEquationHtml(scene, node.id);
  refs.messageStageNote.textContent  = scene.stage_notes[state.messageStepId];
  refs.messageNodeTitle.textContent  = node.name;
  refs.messageNodeSummary.textContent = messageNodeSummary(scene, node.id);

  refs.messageNodeValues.textContent = '';
  for (const [label, value] of buildMessageValueRows(scene, node.id)) {
    const row = document.createElement('div');
    row.className = 'value-row';
    row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    refs.messageNodeValues.appendChild(row);
  }

  renderMessageRows(scene, node.id);
  syncSegmented(refs.messageScenePicker, state.messageSceneId);
  syncMessageSteps();
  renderGraph(scene, refs.messageGraph, {
    mode: 'message',
    stage: state.messageStepId,
    selectedNodeId: node.id,
  });
}

function renderDepthTrack(scene) {
  refs.depthTrack.textContent = '';
  for (const depth of DEPTH_LEVELS) {
    const metric = getDepthMetric(scene, depth);
    const chip = document.createElement('div');
    chip.className = `depth-chip${depth === state.depthLevel ? ' is-active' : ''}`;
    chip.innerHTML = `
      <span>Depth ${depth}</span>
      <strong>${fmt(metric.focus_state)}</strong>
      <small>p = ${fmt(metric.focus_probability)} · gap = ${fmt(metric.community_gap)}</small>
    `;
    refs.depthTrack.appendChild(chip);
  }
}

function depthSummary(scene, depth) {
  const metric = getDepthMetric(scene, depth);
  const note = scene.depth_notes[String(depth)] ?? scene.depth_notes['6'];
  const reach = reachableNodeCount(scene, depth);
  return `${note} At this depth the focus node has ${reach} node${reach === 1 ? '' : 's'} within range. Graph-wide state spread: ${fmt(metric.spread)}.`;
}

function renderDepthFigure() {
  const scene = getDepthScene();
  if (!scene) return;
  const metric = getDepthMetric(scene, state.depthLevel);

  refs.depthTitle.textContent       = scene.title;
  refs.depthDek.textContent         = scene.depth_prompt;
  refs.depthLevel.textContent       = `${state.depthLevel} layer${state.depthLevel === 1 ? '' : 's'}`;
  refs.depthNote.textContent        = scene.depth_notes[String(state.depthLevel)] ?? '';
  refs.depthFocusState.textContent  = fmt(metric.focus_state);
  refs.depthFocusProb.textContent   = fmt(metric.focus_probability);
  refs.depthSpread.textContent      = fmt(metric.spread);
  refs.depthGap.textContent         = fmt(metric.community_gap);
  refs.depthSummary.textContent     = depthSummary(scene, state.depthLevel);

  syncSegmented(refs.depthScenePicker, state.depthSceneId);
  syncSegmented(refs.depthPicker, state.depthLevel);
  renderDepthTrack(scene);
  renderGraph(scene, refs.depthGraph, {
    mode: 'depth',
    depth: state.depthLevel,
    selectedNodeId: scene.focus_node,
  });
}

function populateHero() {
  refs.heroEquation.innerHTML =
    `h<sup>(k+1)</sup><sub>v</sub> = ${fmt(state.data.meta.self_weight, 2)} · h<sup>(k)</sup><sub>v</sub>` +
    ` + ${fmt(state.data.meta.neighbor_weight, 2)} · a<sup>(k)</sup><sub>v</sub>`;
  refs.heroNote.textContent = state.data.meta.note;
}

// ─────────────────────────────────────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────────────────────────────────────
function suppressObserver(ms = STEP_CLICK_SUPPRESS_MS) {
  state.observerSuppressedUntil = performance.now() + ms;
}

function setMessageStep(stepId, { scrollIntoView = false, suppress = false } = {}) {
  if (!MESSAGE_STEP_ORDER.includes(stepId) || state.messageStepId === stepId) return;
  state.messageStepId = stepId;
  if (suppress) suppressObserver();
  renderMessageFigure();
  if (scrollIntoView) {
    const el = refs.messageSteps.find((s) => s.dataset.step === stepId);
    el?.scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth', block: 'center' });
  }
}

function attachMessageStepEvents() {
  refs.messageSteps.forEach((step, index) => {
    step.setAttribute('role', 'button');
    step.addEventListener('click', () => {
      setMessageStep(step.dataset.step, { scrollIntoView: true, suppress: true });
    });
    step.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        setMessageStep(step.dataset.step, { scrollIntoView: true, suppress: true });
        return;
      }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const delta = event.key === 'ArrowDown' ? 1 : -1;
        const next = refs.messageSteps[(index + delta + refs.messageSteps.length) % refs.messageSteps.length];
        next.focus();
        setMessageStep(next.dataset.step, { scrollIntoView: true, suppress: true });
      }
    });
  });
}

function attachMessageStepObserver() {
  const visibility = new Map();
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      visibility.set(entry.target.dataset.step, entry.isIntersecting ? entry.intersectionRatio : 0);
    }
    if (performance.now() < state.observerSuppressedUntil) return;

    let bestId = null;
    let bestRatio = 0;
    for (const step of refs.messageSteps) {
      const r = visibility.get(step.dataset.step) ?? 0;
      if (r > bestRatio) {
        bestRatio = r;
        bestId = step.dataset.step;
      }
    }
    if (bestId && bestRatio > 0.2 && bestId !== state.messageStepId) {
      state.messageStepId = bestId;
      renderMessageFigure();
    }
  }, {
    threshold: [0, 0.2, 0.4, 0.6, 0.8],
    rootMargin: '-18% 0px -24% 0px',
  });

  for (const step of refs.messageSteps) observer.observe(step);
}

function buildMessageScenePicker() {
  const scenes = state.data.scenes.filter((s) => s.figure === 'message');
  buildSegmented(
    refs.messageScenePicker,
    scenes,
    (s) => s.name,
    (s) => s.id,
    (id) => {
      state.messageSceneId = id;
      state.messageNodeId = getMessageScene().focus_node;
      renderMessageFigure();
    },
  );
}

function buildDepthScenePicker() {
  const scenes = DEPTH_SCENE_IDS.map(getScene).filter(Boolean);
  buildSegmented(
    refs.depthScenePicker,
    scenes,
    (s) => s.name,
    (s) => s.id,
    (id) => {
      state.depthSceneId = id;
      if (id === 'twohop' && state.depthLevel === 6) state.depthLevel = 2;
      renderDepthFigure();
    },
  );
}

function buildDepthPicker() {
  buildSegmented(
    refs.depthPicker,
    DEPTH_LEVELS,
    (d) => String(d),
    (d) => d,
    (d) => {
      state.depthLevel = d;
      renderDepthFigure();
    },
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Error UI & init
// ─────────────────────────────────────────────────────────────────────────────
function showFatalError(err) {
  console.error(err);
  const host = document.querySelector('.publication') ?? document.body;
  const banner = document.createElement('div');
  banner.setAttribute('role', 'alert');
  banner.style.cssText = 'margin:32px 0;padding:16px 20px;border:1px solid #c03030;background:#fff3f2;color:#552020;border-radius:6px;font-family:system-ui,sans-serif;';
  banner.innerHTML = `<strong>Failed to load figure data.</strong> <span style="opacity:.7">${String(err.message ?? err)}</span>`;
  host.prepend(banner);
}

async function loadData() {
  const response = await fetch('./data/experiment-data.json');
  if (!response.ok) throw new Error(`Failed to load publication data: ${response.status}`);
  return response.json();
}

async function init() {
  state.data = await loadData();
  state.messageSceneId = state.data.scenes.find((s) => s.figure === 'message')?.id ?? state.data.scenes[0].id;
  state.messageNodeId = getMessageScene().focus_node;

  populateHero();
  buildMessageScenePicker();
  buildDepthScenePicker();
  buildDepthPicker();
  renderMessageFigure();
  renderDepthFigure();
  attachMessageStepEvents();
  if (!REDUCED_MOTION) attachMessageStepObserver();
}

init().catch(showFatalError);
