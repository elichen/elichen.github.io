const DEPTHS = [0, 1, 2, 4, 6];
const RECOMMEND_THRESHOLD = 0.62;
const SVG_NS = 'http://www.w3.org/2000/svg';

const COLORS = {
  low: '#db5b48',
  mid: '#eee7d6',
  high: '#18a872',
  ink: '#141414',
  active: '#00a676',
};

const SCENARIOS = [
  {
    id: 'nearby-taste',
    level: '01',
    title: 'Nearby taste',
    kicker: 'homophily',
    prompt: 'Predict whether Maya should get the Trail Camera Guide.',
    userId: 'maya',
    candidateId: 'trail-cam',
    answer: 'recommend',
    bestDepth: 1,
    lesson: 'One-hop neighbors are already enough: Maya and nearby users save trail-running content, and that signal lands directly on the candidate.',
    nodes: [
      { id: 'maya', label: 'Maya', name: 'Maya', type: 'user', x: 0.16, y: 0.48, base: 0.52, description: 'The target user. Her profile is sparse, so a node-only model cannot see much.' },
      { id: 'ridge', label: 'Ridge', name: 'Ridge Runs', type: 'item', x: 0.36, y: 0.28, base: 0.86, description: 'A trail-running article Maya saved last week.' },
      { id: 'trail', label: 'Trail', name: 'Trail running', type: 'topic', x: 0.56, y: 0.28, base: 0.84, description: 'A shared topic that connects Maya to the candidate.' },
      { id: 'trail-cam', label: 'Guide', name: 'Trail Camera Guide', type: 'item', x: 0.76, y: 0.44, base: 0.48, candidate: true, description: 'The item we are deciding whether to recommend.' },
      { id: 'liam', label: 'Liam', name: 'Liam', type: 'user', x: 0.72, y: 0.14, base: 0.74, description: 'A nearby user with overlapping saves.' },
      { id: 'pasta', label: 'Pasta', name: 'Weeknight Pasta', type: 'item', x: 0.34, y: 0.72, base: 0.26, description: 'A weak counter-signal in Maya\'s history.' },
      { id: 'cooking', label: 'Cook', name: 'Cooking', type: 'topic', x: 0.55, y: 0.78, base: 0.25, description: 'A topic that does not support the candidate.' },
    ],
    edges: [
      { id: 'maya-ridge', source: 'maya', target: 'ridge', weight: 0.92, relation: 'saved' },
      { id: 'ridge-trail', source: 'ridge', target: 'trail', weight: 0.86, relation: 'topic' },
      { id: 'trail-cam-trail', source: 'trail-cam', target: 'trail', weight: 0.96, relation: 'tag match' },
      { id: 'liam-ridge', source: 'liam', target: 'ridge', weight: 0.72, relation: 'saved' },
      { id: 'liam-trail-cam', source: 'liam', target: 'trail-cam', weight: 0.76, relation: 'saved' },
      { id: 'maya-pasta', source: 'maya', target: 'pasta', weight: 0.26, relation: 'ignored' },
      { id: 'pasta-cooking', source: 'pasta', target: 'cooking', weight: 0.80, relation: 'topic' },
    ],
    depths: {
      0: {
        score: 0.48,
        note: 'Node-only view: title and profile are too generic.',
        result: 'The MLP holds the recommendation because the candidate alone does not reveal Maya\'s trail-running taste.',
        activeEdges: [],
        nodeScores: { maya: 0.52, ridge: 0.86, trail: 0.84, 'trail-cam': 0.48, liam: 0.74, pasta: 0.26, cooking: 0.25 },
        messages: [{ title: 'No graph context', body: 'Only Maya\'s sparse profile and the candidate item features are visible.' }],
      },
      1: {
        score: 0.78,
        note: 'Immediate neighbors make the recommendation clear.',
        result: 'A one-layer GNN recommends it: saved trail content and Liam\'s nearby save send strong support into the candidate.',
        activeEdges: ['maya-ridge', 'liam-trail-cam', 'trail-cam-trail'],
        nodeScores: { maya: 0.63, ridge: 0.88, trail: 0.86, 'trail-cam': 0.78, liam: 0.80, pasta: 0.27, cooking: 0.25 },
        messages: [
          { title: 'Maya -> Ridge Runs', body: 'The user has a direct save in the same neighborhood.', value: 0.41 },
          { title: 'Liam -> Trail Camera Guide', body: 'A nearby user supplies collaborative evidence.', value: 0.33 },
          { title: 'Trail running -> candidate', body: 'The topic edge aligns the candidate with the local taste cluster.', value: 0.39 },
        ],
      },
      2: {
        score: 0.84,
        note: 'Two hops reinforces the same local cluster.',
        result: 'The answer is still recommend, but the second layer mostly confirms what one hop already showed.',
        activeEdges: ['maya-ridge', 'ridge-trail', 'trail-cam-trail', 'liam-ridge', 'liam-trail-cam'],
        nodeScores: { maya: 0.70, ridge: 0.87, trail: 0.89, 'trail-cam': 0.84, liam: 0.82, pasta: 0.28, cooking: 0.26 },
        messages: [
          { title: 'Ridge Runs -> Trail running -> candidate', body: 'Maya\'s saved item and the candidate meet at the same topic.', value: 0.47 },
          { title: 'Liam -> Ridge Runs -> Maya', body: 'The similar-user path raises confidence without changing the decision.', value: 0.31 },
        ],
      },
      4: {
        score: 0.74,
        note: 'More depth starts blending unrelated interests.',
        result: 'Still recommend, but unrelated cooking edges begin to dilute the signal.',
        activeEdges: ['maya-ridge', 'ridge-trail', 'trail-cam-trail', 'maya-pasta', 'pasta-cooking'],
        nodeScores: { maya: 0.65, ridge: 0.78, trail: 0.80, 'trail-cam': 0.74, liam: 0.75, pasta: 0.38, cooking: 0.35 },
        messages: [{ title: 'Extra context is not free', body: 'The candidate still wins, but weak counter-signals have entered the receptive field.', value: 0.21 }],
      },
      6: {
        score: 0.63,
        note: 'Deep mixing nearly erases the local pattern.',
        result: 'Barely recommend. The graph has become smoother, so the original taste signal is less distinct.',
        activeEdges: ['maya-ridge', 'ridge-trail', 'trail-cam-trail', 'maya-pasta', 'pasta-cooking', 'liam-ridge'],
        nodeScores: { maya: 0.59, ridge: 0.66, trail: 0.67, 'trail-cam': 0.63, liam: 0.65, pasta: 0.47, cooking: 0.43 },
        messages: [{ title: 'Oversmoothing pressure', body: 'Most nearby nodes move toward the same middle value.', value: 0.11 }],
      },
    },
  },
  {
    id: 'two-hop',
    level: '02',
    title: 'Two-hop discovery',
    kicker: 'receptive field',
    prompt: 'Predict whether Maya should get the Blue Note Deep Dive.',
    userId: 'maya',
    candidateId: 'blue-note',
    answer: 'recommend',
    bestDepth: 2,
    lesson: 'The useful evidence is not adjacent to the candidate. It sits behind a similar listener, so a two-layer GNN is the first model that can reach it.',
    nodes: [
      { id: 'maya', label: 'Maya', name: 'Maya', type: 'user', x: 0.14, y: 0.48, base: 0.50, description: 'The target user. Her direct history is incomplete.' },
      { id: 'piano', label: 'Piano', name: 'Jazz Piano Loops', type: 'item', x: 0.32, y: 0.24, base: 0.78, description: 'A saved item that hints at jazz interest.' },
      { id: 'lofi', label: 'Lo-fi', name: 'Lo-fi Beats', type: 'item', x: 0.32, y: 0.72, base: 0.56, description: 'A broad music save that is not decisive by itself.' },
      { id: 'iris', label: 'Iris', name: 'Iris', type: 'user', x: 0.52, y: 0.34, base: 0.76, description: 'A similar listener who bridges Maya to the candidate.' },
      { id: 'modal', label: 'Modal', name: 'Modal jazz', type: 'topic', x: 0.70, y: 0.34, base: 0.80, description: 'The hidden shared topic.' },
      { id: 'blue-note', label: 'Deep', name: 'Blue Note Deep Dive', type: 'item', x: 0.84, y: 0.52, base: 0.44, candidate: true, description: 'The candidate recommendation. Its own features look niche and uncertain.' },
      { id: 'pop', label: 'Pop', name: 'Pop Hits', type: 'topic', x: 0.58, y: 0.76, base: 0.42, description: 'A broad music topic that does not explain the candidate.' },
    ],
    edges: [
      { id: 'maya-piano', source: 'maya', target: 'piano', weight: 0.82, relation: 'saved' },
      { id: 'maya-lofi', source: 'maya', target: 'lofi', weight: 0.58, relation: 'saved' },
      { id: 'piano-iris', source: 'piano', target: 'iris', weight: 0.74, relation: 'overlap' },
      { id: 'iris-modal', source: 'iris', target: 'modal', weight: 0.88, relation: 'likes' },
      { id: 'modal-blue-note', source: 'modal', target: 'blue-note', weight: 0.94, relation: 'topic' },
      { id: 'lofi-pop', source: 'lofi', target: 'pop', weight: 0.72, relation: 'topic' },
      { id: 'pop-blue-note', source: 'pop', target: 'blue-note', weight: 0.18, relation: 'weak' },
    ],
    depths: {
      0: {
        score: 0.44,
        note: 'The item looks too niche in isolation.',
        result: 'The MLP holds. It cannot see that Maya is close to a jazz listener who already points at the candidate.',
        activeEdges: [],
        nodeScores: { maya: 0.50, piano: 0.78, lofi: 0.56, iris: 0.76, modal: 0.80, 'blue-note': 0.44, pop: 0.42 },
        messages: [{ title: 'Local features only', body: 'The candidate has no obvious broad appeal signal on its own.' }],
      },
      1: {
        score: 0.55,
        note: 'One hop reaches weak music context, not the decisive witness.',
        result: 'Still hold. One layer sees generic music edges but misses the listener-topic bridge.',
        activeEdges: ['pop-blue-note', 'lofi-pop'],
        nodeScores: { maya: 0.52, piano: 0.77, lofi: 0.57, iris: 0.75, modal: 0.77, 'blue-note': 0.55, pop: 0.48 },
        messages: [{ title: 'Weak adjacent evidence', body: 'The candidate touches a broad music topic, but that edge is not enough.', value: 0.12 }],
      },
      2: {
        score: 0.83,
        note: 'Two hops reaches the similar-listener bridge.',
        result: 'Recommend. The GNN can now use Maya -> Jazz Piano -> Iris -> Modal jazz as evidence for the candidate.',
        activeEdges: ['maya-piano', 'piano-iris', 'iris-modal', 'modal-blue-note'],
        nodeScores: { maya: 0.66, piano: 0.82, lofi: 0.58, iris: 0.84, modal: 0.87, 'blue-note': 0.83, pop: 0.45 },
        messages: [
          { title: 'Maya -> Jazz Piano -> Iris', body: 'A similar listener becomes reachable at this depth.', value: 0.36 },
          { title: 'Iris -> Modal jazz -> candidate', body: 'The hidden jazz topic finally reaches the candidate.', value: 0.48 },
        ],
      },
      4: {
        score: 0.77,
        note: 'The recommendation remains good, with some extra dilution.',
        result: 'Still recommend. Deeper context adds broad music noise, but the jazz path remains dominant.',
        activeEdges: ['maya-piano', 'piano-iris', 'iris-modal', 'modal-blue-note', 'maya-lofi', 'lofi-pop'],
        nodeScores: { maya: 0.65, piano: 0.76, lofi: 0.60, iris: 0.78, modal: 0.80, 'blue-note': 0.77, pop: 0.55 },
        messages: [{ title: 'Useful, then noisy', body: 'Extra layers add more music context without much new signal.', value: 0.24 }],
      },
      6: {
        score: 0.59,
        note: 'Too much graph context washes out the niche signal.',
        result: 'Hold. By six layers, the local jazz witness has been averaged into a broad music neighborhood.',
        activeEdges: ['maya-piano', 'piano-iris', 'iris-modal', 'modal-blue-note', 'maya-lofi', 'lofi-pop', 'pop-blue-note'],
        nodeScores: { maya: 0.58, piano: 0.62, lofi: 0.58, iris: 0.61, modal: 0.62, 'blue-note': 0.59, pop: 0.57 },
        messages: [{ title: 'Niche signal collapsed', body: 'The candidate now looks like a generic music item instead of a specific jazz match.', value: 0.08 }],
      },
    },
  },
  {
    id: 'popularity-trap',
    level: '03',
    title: 'Popularity trap',
    kicker: 'negative context',
    prompt: 'Predict whether Maya should get the Viral Productivity Timer.',
    userId: 'maya',
    candidateId: 'timer',
    answer: 'hold',
    bestDepth: 2,
    lesson: 'Graph context can reduce a bad recommendation. The item is popular globally, but Maya lives in a quiet longform cluster that pushes against it.',
    nodes: [
      { id: 'maya', label: 'Maya', name: 'Maya', type: 'user', x: 0.15, y: 0.48, base: 0.46, description: 'The target user. Her local graph favors longform reading, not productivity hacks.' },
      { id: 'essay', label: 'Essay', name: 'Longform Essays', type: 'item', x: 0.34, y: 0.28, base: 0.78, description: 'A strong positive signal for slow reading.' },
      { id: 'quiet', label: 'Quiet', name: 'Quiet research', type: 'topic', x: 0.54, y: 0.30, base: 0.76, description: 'The local taste topic.' },
      { id: 'timer', label: 'Timer', name: 'Viral Productivity Timer', type: 'item', x: 0.78, y: 0.46, base: 0.68, candidate: true, description: 'The candidate. It has high global popularity but poor local fit.' },
      { id: 'sam', label: 'Sam', name: 'Sam', type: 'user', x: 0.60, y: 0.72, base: 0.32, description: 'A nearby user who repeatedly skips productivity content.' },
      { id: 'viral', label: 'Viral', name: 'Viral tools', type: 'topic', x: 0.84, y: 0.20, base: 0.72, description: 'Global popularity pressure.' },
      { id: 'planner', label: 'Plan', name: 'Daily Planner', type: 'item', x: 0.34, y: 0.72, base: 0.28, description: 'A related item Maya ignored.' },
    ],
    edges: [
      { id: 'maya-essay', source: 'maya', target: 'essay', weight: 0.90, relation: 'saved' },
      { id: 'essay-quiet', source: 'essay', target: 'quiet', weight: 0.84, relation: 'topic' },
      { id: 'quiet-timer', source: 'quiet', target: 'timer', weight: 0.22, relation: 'mismatch' },
      { id: 'sam-planner', source: 'sam', target: 'planner', weight: 0.76, relation: 'ignored' },
      { id: 'maya-planner', source: 'maya', target: 'planner', weight: 0.72, relation: 'ignored' },
      { id: 'planner-timer', source: 'planner', target: 'timer', weight: 0.68, relation: 'similar' },
      { id: 'viral-timer', source: 'viral', target: 'timer', weight: 0.95, relation: 'popular' },
    ],
    depths: {
      0: {
        score: 0.68,
        note: 'The candidate looks strong if popularity is all you see.',
        result: 'The MLP recommends it because the item has strong standalone popularity features.',
        activeEdges: [],
        nodeScores: { maya: 0.46, essay: 0.78, quiet: 0.76, timer: 0.68, sam: 0.32, viral: 0.72, planner: 0.28 },
        messages: [{ title: 'Popularity dominates', body: 'Node-only ranking sees a viral item and misses the local mismatch.' }],
      },
      1: {
        score: 0.57,
        note: 'Direct neighborhood starts pushing back.',
        result: 'Hold. One layer sees that the candidate sits next to ignored planner content and a weak mismatch edge.',
        activeEdges: ['planner-timer', 'quiet-timer', 'viral-timer'],
        nodeScores: { maya: 0.45, essay: 0.74, quiet: 0.69, timer: 0.57, sam: 0.34, viral: 0.70, planner: 0.32 },
        messages: [
          { title: 'Planner -> candidate', body: 'A similar item was ignored, reducing confidence.', value: -0.28 },
          { title: 'Viral tools -> candidate', body: 'Popularity still pushes upward, but not enough.', value: 0.22 },
        ],
      },
      2: {
        score: 0.34,
        note: 'Two hops exposes Maya\'s own negative evidence.',
        result: 'Correct hold. The GNN reaches Maya -> ignored planner -> timer and quiet research -> mismatch.',
        activeEdges: ['maya-planner', 'planner-timer', 'maya-essay', 'essay-quiet', 'quiet-timer'],
        nodeScores: { maya: 0.38, essay: 0.70, quiet: 0.62, timer: 0.34, sam: 0.31, viral: 0.60, planner: 0.25 },
        messages: [
          { title: 'Maya -> ignored planner -> candidate', body: 'The most relevant path is negative, not positive.', value: -0.44 },
          { title: 'Essay -> quiet research -> mismatch', body: 'Maya\'s real cluster points away from the timer.', value: -0.31 },
        ],
      },
      4: {
        score: 0.49,
        note: 'The local negative evidence remains visible.',
        result: 'Hold, but global popularity starts leaking back in.',
        activeEdges: ['maya-planner', 'planner-timer', 'maya-essay', 'essay-quiet', 'quiet-timer', 'viral-timer'],
        nodeScores: { maya: 0.43, essay: 0.61, quiet: 0.55, timer: 0.49, sam: 0.37, viral: 0.61, planner: 0.38 },
        messages: [{ title: 'Mixed evidence', body: 'The graph now contains both Maya\'s mismatch and the viral cluster.', value: -0.10 }],
      },
      6: {
        score: 0.65,
        note: 'Deep mixing lets popularity swamp local taste.',
        result: 'Wrong recommend. Too much depth turns a personalized graph into a popularity machine.',
        activeEdges: ['maya-planner', 'planner-timer', 'maya-essay', 'essay-quiet', 'quiet-timer', 'viral-timer'],
        nodeScores: { maya: 0.55, essay: 0.58, quiet: 0.58, timer: 0.65, sam: 0.50, viral: 0.67, planner: 0.54 },
        messages: [{ title: 'Popularity leaks everywhere', body: 'The candidate absorbs broad graph mass and loses personalization.', value: 0.18 }],
      },
    },
  },
  {
    id: 'oversmoothing',
    level: '04',
    title: 'Oversmoothing trap',
    kicker: 'too much depth',
    prompt: 'Predict whether Maya should get the Obscure Synth Zine.',
    userId: 'maya',
    candidateId: 'zine',
    answer: 'recommend',
    bestDepth: 2,
    lesson: 'Depth is a tradeoff. Two hops reaches the niche synth cluster; six hops blends that cluster with mainstream music and loses the recommendation.',
    nodes: [
      { id: 'maya', label: 'Maya', name: 'Maya', type: 'user', x: 0.14, y: 0.48, base: 0.51, description: 'The target user. Her taste is niche and easy to wash out.' },
      { id: 'modular', label: 'Mod', name: 'Modular Patch Notes', type: 'item', x: 0.32, y: 0.28, base: 0.82, description: 'A strong niche save.' },
      { id: 'zine', label: 'Zine', name: 'Obscure Synth Zine', type: 'item', x: 0.55, y: 0.28, base: 0.51, candidate: true, description: 'The candidate. It needs graph context to look relevant.' },
      { id: 'niche', label: 'Niche', name: 'Analog synths', type: 'topic', x: 0.44, y: 0.52, base: 0.84, description: 'The useful local topic.' },
      { id: 'vera', label: 'Vera', name: 'Vera', type: 'user', x: 0.66, y: 0.54, base: 0.78, description: 'A similar user in the niche cluster.' },
      { id: 'playlist', label: 'Pop', name: 'Pop Playlist', type: 'item', x: 0.78, y: 0.22, base: 0.42, description: 'A mainstream music item.' },
      { id: 'mainstream', label: 'Main', name: 'Mainstream music', type: 'topic', x: 0.82, y: 0.74, base: 0.48, description: 'A broad cluster that should not dominate this recommendation.' },
      { id: 'radio', label: 'Radio', name: 'Daily Radio', type: 'item', x: 0.34, y: 0.76, base: 0.40, description: 'A broad music item that enters at deeper layers.' },
    ],
    edges: [
      { id: 'maya-modular', source: 'maya', target: 'modular', weight: 0.88, relation: 'saved' },
      { id: 'modular-niche', source: 'modular', target: 'niche', weight: 0.92, relation: 'topic' },
      { id: 'niche-zine', source: 'niche', target: 'zine', weight: 0.94, relation: 'topic' },
      { id: 'vera-zine', source: 'vera', target: 'zine', weight: 0.80, relation: 'saved' },
      { id: 'vera-niche', source: 'vera', target: 'niche', weight: 0.84, relation: 'likes' },
      { id: 'zine-playlist', source: 'zine', target: 'playlist', weight: 0.24, relation: 'music' },
      { id: 'playlist-mainstream', source: 'playlist', target: 'mainstream', weight: 0.90, relation: 'topic' },
      { id: 'radio-mainstream', source: 'radio', target: 'mainstream', weight: 0.86, relation: 'topic' },
      { id: 'maya-radio', source: 'maya', target: 'radio', weight: 0.30, relation: 'skipped' },
    ],
    depths: {
      0: {
        score: 0.51,
        note: 'The candidate looks neutral alone.',
        result: 'The MLP holds. The zine does not carry enough standalone evidence.',
        activeEdges: [],
        nodeScores: { maya: 0.51, modular: 0.82, zine: 0.51, niche: 0.84, vera: 0.78, playlist: 0.42, mainstream: 0.48, radio: 0.40 },
        messages: [{ title: 'No local graph', body: 'The candidate looks obscure rather than relevant.' }],
      },
      1: {
        score: 0.70,
        note: 'Direct niche neighbors help.',
        result: 'Recommend. The candidate receives strong evidence from Vera and the analog synth topic.',
        activeEdges: ['vera-zine', 'niche-zine'],
        nodeScores: { maya: 0.58, modular: 0.82, zine: 0.70, niche: 0.84, vera: 0.80, playlist: 0.45, mainstream: 0.48, radio: 0.41 },
        messages: [
          { title: 'Vera -> candidate', body: 'A similar user saved the zine.', value: 0.34 },
          { title: 'Analog synths -> candidate', body: 'The candidate is anchored in the right niche topic.', value: 0.41 },
        ],
      },
      2: {
        score: 0.86,
        note: 'Two hops ties Maya directly to the niche cluster.',
        result: 'Best recommend. The model reaches Maya -> Modular Patch Notes -> Analog synths -> zine.',
        activeEdges: ['maya-modular', 'modular-niche', 'niche-zine', 'vera-niche', 'vera-zine'],
        nodeScores: { maya: 0.72, modular: 0.86, zine: 0.86, niche: 0.88, vera: 0.84, playlist: 0.46, mainstream: 0.49, radio: 0.42 },
        messages: [
          { title: 'Maya -> Modular Patch Notes -> Analog synths', body: 'The local taste path reaches the candidate.', value: 0.49 },
          { title: 'Vera -> Analog synths -> candidate', body: 'A similar user reinforces the same niche signal.', value: 0.36 },
        ],
      },
      4: {
        score: 0.67,
        note: 'The recommendation survives, but the signal is weaker.',
        result: 'Recommend with lower confidence. Mainstream music edges are now in range.',
        activeEdges: ['maya-modular', 'modular-niche', 'niche-zine', 'zine-playlist', 'playlist-mainstream', 'maya-radio'],
        nodeScores: { maya: 0.61, modular: 0.72, zine: 0.67, niche: 0.73, vera: 0.70, playlist: 0.56, mainstream: 0.55, radio: 0.48 },
        messages: [{ title: 'The cluster is less crisp', body: 'Mainstream music starts averaging with the niche synth neighborhood.', value: 0.16 }],
      },
      6: {
        score: 0.52,
        note: 'Oversmoothing erases the recommendation.',
        result: 'Wrong hold. The zine is averaged into the same middle state as broad music items.',
        activeEdges: ['maya-modular', 'modular-niche', 'niche-zine', 'zine-playlist', 'playlist-mainstream', 'radio-mainstream', 'maya-radio'],
        nodeScores: { maya: 0.54, modular: 0.57, zine: 0.52, niche: 0.56, vera: 0.55, playlist: 0.53, mainstream: 0.53, radio: 0.51 },
        messages: [{ title: 'Everything looks average', body: 'The model gained reach, but lost the distinction that made the zine special.', value: 0.03 }],
      },
    },
  },
];

const state = {
  scenarioId: SCENARIOS[0].id,
  depth: SCENARIOS[0].bestDepth,
  model: 'gnn',
  selectedNodeId: SCENARIOS[0].userId,
  showMessages: true,
  hasRun: false,
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
  graph: document.getElementById('graph'),
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

function getScenario() {
  return SCENARIOS.find((scenario) => scenario.id === state.scenarioId) ?? SCENARIOS[0];
}

function getLayer(depth = state.depth) {
  return getScenario().depths[depth] ?? getScenario().depths[0];
}

function getNode(id) {
  return getScenario().nodes.find((node) => node.id === id) ?? null;
}

function predictionFor(score) {
  return score >= RECOMMEND_THRESHOLD ? 'recommend' : 'hold';
}

function labelForPrediction(prediction) {
  return prediction === 'recommend' ? 'Recommend' : 'Hold';
}

function confidenceCopy(score) {
  if (score >= 0.80) return 'high confidence';
  if (score >= 0.62) return 'moderate confidence';
  if (score >= 0.46) return 'uncertain';
  return 'low match';
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
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

function buildScenarioList() {
  refs.scenarioList.textContent = '';
  for (const scenario of SCENARIOS) {
    const bestLayer = scenario.depths[scenario.bestDepth];
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'scenario-button';
    button.dataset.scenarioId = scenario.id;
    button.innerHTML = `
      <span>${scenario.level}</span>
      <strong>${scenario.title}</strong>
      <small>${scenario.kicker} · best depth ${scenario.bestDepth}</small>
      <em>${Math.round(bestLayer.score * 100)}%</em>
    `;
    button.addEventListener('click', () => {
      state.scenarioId = scenario.id;
      state.depth = scenario.bestDepth;
      state.model = scenario.bestDepth === 0 ? 'mlp' : 'gnn';
      state.selectedNodeId = scenario.userId;
      state.hasRun = false;
      render();
    });
    refs.scenarioList.appendChild(button);
  }
}

function buildControls() {
  refs.modelPicker.textContent = '';
  [
    { id: 'mlp', label: 'MLP' },
    { id: 'gnn', label: 'GNN' },
  ].forEach((model) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.model = model.id;
    button.textContent = model.label;
    button.addEventListener('click', () => {
      state.model = model.id;
      state.depth = model.id === 'mlp' ? 0 : Math.max(1, state.depth || getScenario().bestDepth);
      state.hasRun = false;
      render();
    });
    refs.modelPicker.appendChild(button);
  });

  refs.depthPicker.textContent = '';
  DEPTHS.forEach((depth) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.depth = String(depth);
    button.textContent = String(depth);
    button.addEventListener('click', () => {
      state.depth = depth;
      state.model = depth === 0 ? 'mlp' : 'gnn';
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
    state.depth = scenario.bestDepth;
    state.model = scenario.bestDepth === 0 ? 'mlp' : 'gnn';
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
  const candidate = getNode(scenario.candidateId);
  refs.scenarioKicker.textContent = `${scenario.level} / ${scenario.kicker}`;
  refs.scenarioTitle.textContent = scenario.title;
  refs.scenarioObjective.textContent = scenario.prompt;
  refs.candidateName.textContent = candidate.name;
  refs.candidateMeta.textContent = `Target user: ${getNode(scenario.userId).name}`;
}

function renderGraph() {
  const scenario = getScenario();
  const layer = getLayer();
  const activeEdges = new Set(layer.activeEdges);
  const positions = new Map(scenario.nodes.map((node) => [node.id, nodePosition(node)]));
  refs.graph.textContent = '';

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
  refs.graph.appendChild(defs);

  const edgeLayer = svgEl('g', { class: 'edge-layer' });
  scenario.edges.forEach((edge, index) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
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
  refs.graph.appendChild(edgeLayer);

  const nodeLayer = svgEl('g', { class: 'node-layer' });
  for (const node of scenario.nodes) {
    const pos = positions.get(node.id);
    const score = layer.nodeScores[node.id] ?? node.base ?? 0.5;
    const fill = signalColor(score);
    const selected = node.id === state.selectedNodeId;
    const isUser = node.id === scenario.userId;
    const isCandidate = node.id === scenario.candidateId;
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
    caption.textContent = node.type;
    group.appendChild(caption);

    group.addEventListener('click', () => {
      state.selectedNodeId = node.id;
      render();
    });
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        state.selectedNodeId = node.id;
        render();
      }
    });
    nodeLayer.appendChild(group);
  }
  refs.graph.appendChild(nodeLayer);
}

function renderPrediction() {
  const scenario = getScenario();
  const layer = getLayer();
  const prediction = predictionFor(layer.score);
  const correct = prediction === scenario.answer;
  refs.scoreFill.style.width = `${Math.round(layer.score * 100)}%`;
  refs.scoreFill.style.background = signalColor(layer.score);
  refs.runButton.textContent = state.hasRun ? 'Run again' : 'Run prediction';

  if (!state.hasRun) {
    refs.scoreValue.textContent = '--';
    refs.scoreLabel.textContent = state.depth === 0 ? 'MLP selected' : `${state.depth}-layer GNN selected`;
    refs.resultCopy.textContent = 'Run the selected model to reveal the recommendation and explanation.';
    refs.resultCopy.className = 'result-copy';
    return;
  }

  refs.scoreValue.textContent = `${Math.round(layer.score * 100)}%`;
  refs.scoreLabel.textContent = `${labelForPrediction(prediction)} · ${confidenceCopy(layer.score)}`;
  refs.resultCopy.textContent = `${correct ? 'Correct.' : 'Wrong.'} ${layer.result}`;
  refs.resultCopy.className = `result-copy ${correct ? 'is-correct' : 'is-wrong'}`;
}

function renderComparison() {
  const scenario = getScenario();
  refs.depthComparison.textContent = '';
  DEPTHS.forEach((depth) => {
    const layer = scenario.depths[depth];
    const prediction = predictionFor(layer.score);
    const correct = prediction === scenario.answer;
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `compare-row${depth === state.depth ? ' is-active' : ''}${correct ? ' is-correct' : ' is-wrong'}`;
    row.innerHTML = `
      <span class="compare-depth">${depth === 0 ? 'MLP' : `${depth} hop`}</span>
      <span class="compare-bar"><i style="width:${Math.round(layer.score * 100)}%; background:${signalColor(layer.score)}"></i></span>
      <strong>${Math.round(layer.score * 100)}%</strong>
      <small>${layer.note}</small>
    `;
    row.addEventListener('click', () => {
      state.depth = depth;
      state.model = depth === 0 ? 'mlp' : 'gnn';
      state.hasRun = false;
      render();
    });
    refs.depthComparison.appendChild(row);
  });
}

function renderInspector() {
  const layer = getLayer();
  const node = getNode(state.selectedNodeId) ?? getNode(getScenario().userId);
  const score = layer.nodeScores[node.id] ?? node.base ?? 0.5;
  refs.nodeName.textContent = node.name;
  refs.nodeDescription.textContent = node.description;
  refs.nodeStats.textContent = '';

  [
    ['role', node.type],
    ['signal at depth', `${Math.round(score * 100)}%`],
    ['local baseline', `${Math.round((node.base ?? 0.5) * 100)}%`],
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
  const layer = getLayer();
  refs.messageList.textContent = '';

  const intro = document.createElement('p');
  intro.className = 'lesson-copy';
  intro.textContent = scenario.lesson;
  refs.messageList.appendChild(intro);

  layer.messages.forEach((message) => {
    const item = document.createElement('article');
    item.className = 'message-item';
    const value = typeof message.value === 'number'
      ? `<span class="${message.value < 0 ? 'is-negative' : 'is-positive'}">${message.value > 0 ? '+' : ''}${message.value.toFixed(2)}</span>`
      : '';
    item.innerHTML = `
      <div>
        <strong>${message.title}</strong>
        <p>${message.body}</p>
      </div>
      ${value}
    `;
    refs.messageList.appendChild(item);
  });
}

function render() {
  syncScenarioList();
  syncControls();
  renderHeader();
  renderGraph();
  renderPrediction();
  renderComparison();
  renderInspector();
  renderMessages();
}

buildScenarioList();
buildControls();
render();
