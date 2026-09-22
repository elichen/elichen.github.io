(() => {
  const A = window.Autogram;

  // Ten ways to open the P.S. Some have no true version nearby, so the search rotates between them.
  const OPENINGS = [
    'P.S. From the first word to the last, this letter contains ',
    'P.S. Including this sentence, the letter contains ',
    'P.S. Count carefully and you will find that this letter has ',
    'P.S. By my count, which you are welcome to check, this letter contains ',
    'P.S. Counting every letter, this letter contains ',
    'P.S. All told, this letter is made of ',
    'P.S. For anyone keeping score, this letter has ',
    'P.S. Letter by letter, the whole thing comes to ',
    'P.S. In total, this letter uses ',
    'P.S. Every letter included, this letter holds ',
  ];
  // The first search on page load uses a fixed seed so everyone sees the same opening.
  const FIRST_SEED = 40;
  // It also starts in slow motion for a moment, so there is something to watch.
  const OVERTURE_MS = 1500;
  const SLOW_STEP_MS = 170;
  const OVERTURE_STEP_MS = 110;

  const $ = (id) => document.getElementById(id);
  const bodyEl = $('letter-body');
  const psEl = $('ps');
  const caseEl = $('case');
  const statusEl = $('status');
  const postmarkEl = $('postmark');
  const slowEl = $('slow');
  slowEl.checked = false;
  const stats = {
    guesses: $('stat-guesses'),
    near: $('stat-near'),
    phrasing: $('stat-phrasing'),
  };
  $('stat-phrasings').textContent = OPENINGS.length;

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const canHighlight = typeof Highlight === 'function' && !!(window.CSS && CSS.highlights);
  const fmt = (n) => n.toLocaleString('en-US');
  const plural = (k, n) => A.ALPHABET[k] + (n === 1 ? '' : '’s');

  const to = new URLSearchParams(location.search).get('to');
  if (to && to.trim()) $('to').textContent = to.trim().slice(0, 40);
  const originalHTML = bodyEl.innerHTML;

  // The P.S. is built once and its words are swapped in place as the search runs.
  const openingEl = document.createElement('span');
  psEl.append(openingEl);
  const items = [];
  for (let k = 0; k < 26; k++) {
    if (k === 25) psEl.append('and ');
    const el = document.createElement('span');
    el.className = 'item';
    el.dataset.k = k;
    const num = document.createElement('span');
    num.className = 'num';
    const suffix = document.createTextNode('');
    el.append(num, suffix);
    psEl.append(el, k < 25 ? ', ' : '.');
    items.push({ el, num, suffix, n: -1 });
  }

  const drawers = [];
  const tallyEl = $('tally');
  for (let k = 0; k < 26; k++) {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'drawer';
    el.dataset.k = k;
    el.innerHTML = `<span class="glyph">${A.ALPHABET[k]}</span><span class="n"></span><span class="delta"></span><span class="bar"></span>`;
    caseEl.insertBefore(el, tallyEl);
    drawers.push({ el, n: el.querySelector('.n'), delta: el.querySelector('.delta'), shown: -1, diff: 0 });
  }

  const shownX = new Int32Array(26);
  let search = null;
  let frame = 0;
  let startedAt = 0;
  let lastSlowStep = 0;
  let overtureUntil = 0;
  let fullSpeed = { iterations: 0, ms: 0 };
  let found = [];
  let debounce = 0;
  let solvedText = null;

  function setStatus(html) {
    statusEl.innerHTML = html;
  }

  function renderPS(opening, x, actual) {
    if (openingEl.textContent !== opening) openingEl.textContent = opening;
    for (let k = 0; k < 26; k++) {
      const item = items[k];
      const n = x[k];
      if (item.n !== n) {
        item.n = n;
        item.num.textContent = A.numberWords(n);
        item.suffix.data = ' ' + plural(k, n);
      }
      shownX[k] = n;
      item.el.classList.toggle('off', !!actual && actual[k] !== n);
    }
  }

  function renderCase(x, actual) {
    let max = 1;
    for (let k = 0; k < 26; k++) if (x[k] > max) max = x[k];
    for (let k = 0; k < 26; k++) {
      const d = drawers[k];
      const diff = actual ? actual[k] - x[k] : 0;
      if (d.shown !== x[k]) {
        d.shown = x[k];
        d.n.textContent = x[k];
        d.el.style.setProperty('--f', Math.sqrt(x[k] / max).toFixed(3));
      }
      if (d.diff !== diff) {
        d.diff = diff;
        d.delta.textContent = diff ? (diff > 0 ? '+' : '−') + Math.abs(diff) : '';
        d.el.classList.toggle('off', diff !== 0);
      }
      d.el.setAttribute(
        'aria-label',
        `${A.ALPHABET[k]}: the P.S. claims ${x[k]}` + (diff ? `, the letter has ${x[k] + diff}` : '')
      );
    }
  }

  function renderStats() {
    stats.guesses.textContent = fmt(search.iterations);
    stats.near.textContent = fmt(search.nearMisses);
    stats.phrasing.textContent = search.active + 1;
  }

  function pageCounts() {
    return A.countLetters(bodyEl.textContent + psEl.textContent);
  }

  // Search

  function randomSeed() {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  }

  function startSearch({ seed = randomSeed(), startAt = 0 } = {}) {
    cancelAnimationFrame(frame);
    if (seed !== FIRST_SEED) overtureUntil = 0;
    stopCounting();
    const text = bodyEl.textContent;
    if (text !== solvedText) found = [];
    search = new A.Search(
      OPENINGS.map((o) => text + o),
      { seed, startAt, exclude: found }
    );
    if (search.tooLong) {
      search = null;
      setStatus('This letter is too long to count. Trim it a little and the search will start again.');
      return;
    }
    startedAt = performance.now();
    psEl.classList.add('unsettled');
    hidePostmark();
    setStatus(slowEl.checked ? slowMessage() : 'Searching for numbers that agree with themselves.');
    draw();
    frame = requestAnimationFrame(tick);
  }

  function slowMessage() {
    return 'Slow motion: one guess at a time. Red numbers in the P.S. are the ones the letter disagrees with. Turn slow motion off to let the search finish.';
  }

  function tick(now) {
    if (!search) return;
    const overture = now < overtureUntil;
    if (slowEl.checked || overture) {
      if (now - lastSlowStep >= (overture ? OVERTURE_STEP_MS : SLOW_STEP_MS)) {
        lastSlowStep = now;
        search.run(1);
      }
    } else {
      const t0 = performance.now();
      const before = search.iterations;
      const deadline = t0 + 10;
      let done = false;
      while (!done && performance.now() < deadline) done = search.run(4000);
      fullSpeed.iterations += search.iterations - before;
      fullSpeed.ms += performance.now() - t0;
    }
    if (search.solution) {
      finish();
      return;
    }
    draw();
    frame = requestAnimationFrame(tick);
  }

  function draw() {
    const v = search.current;
    renderPS(OPENINGS[search.active], v.x, v.y);
    renderCase(v.x, v.y);
    renderStats();
  }

  function finish() {
    const s = search;
    const { variant, x } = s.solution;
    renderPS(OPENINGS[variant], x, null);
    renderCase(x, null);
    renderStats();
    stats.phrasing.textContent = variant + 1;

    // Don't trust the arithmetic: count the letters actually on the page.
    const counts = pageCounts();
    for (let k = 0; k < 26; k++) {
      if (counts[k] !== x[k]) {
        search = null;
        setStatus(
          `The search thought it was done, but a recount of the page found ${counts[k]} ${plural(k, counts[k])}, ` +
            `not ${x[k]}. Edit the letter or press Find another true version to try again.`
        );
        return;
      }
    }

    search = null;
    solvedText = bodyEl.textContent;
    found.push(variant + ':' + x.join(','));
    psEl.classList.remove('unsettled');
    const seconds = (performance.now() - startedAt) / 1000;
    const which = found.length > 1 ? ` This is true version number ${found.length} of this letter.` : '';
    setStatus(
      `<strong>True.</strong> Found after ${fmt(s.iterations)} guesses in ${seconds.toFixed(seconds < 10 ? 2 : 0)} seconds, ` +
        `then recounted from the page itself. All 26 counts match.${which}`
    );
    showPostmark();
    explainPlainRule(OPENINGS[variant]);
  }

  // Postmark

  function showPostmark() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    $('pm-date').textContent = `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}`;
    postmarkEl.classList.remove('on', 'fade');
    void postmarkEl.offsetWidth;
    postmarkEl.classList.add('on');
  }

  function hidePostmark() {
    if (postmarkEl.classList.contains('on')) {
      postmarkEl.classList.remove('on');
      postmarkEl.classList.add('fade');
    }
  }

  // Notes that describe this particular letter

  function explainPlainRule(opening) {
    const r = A.plainIteration(bodyEl.textContent + opening);
    let text;
    if (r.fixed) {
      text = `on this letter it happens to land on a true version after ${fmt(r.steps)} steps, which is unusual luck`;
    } else if (!r.loopLength) {
      text = `on this letter it wanders for ${fmt(r.stepsBeforeLoop)} steps without ever settling`;
    } else {
      text = `on this letter it falls into a loop of ${fmt(r.loopLength)} different guesses after ${fmt(r.stepsBeforeLoop)} steps and circles it forever`;
    }
    $('loop-fact').textContent = text;
    if (fullSpeed.ms > 150) {
      const rate = (fullSpeed.iterations / fullSpeed.ms) * 1000;
      $('rate').textContent =
        rate >= 1e6 ? `about ${(rate / 1e6).toFixed(1)} million` : `about ${fmt(Math.round(rate / 1000) * 1000)}`;
    }
  }

  // Finding and counting letters on the page

  function rangesFor(k) {
    const ranges = [];
    for (const root of [bodyEl, psEl]) {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        const data = node.data;
        let i = 0;
        for (const ch of data) {
          if (A.letterIndex(ch) === k) {
            const r = new Range();
            r.setStart(node, i);
            r.setEnd(node, i + ch.length);
            ranges.push(r);
          }
          i += ch.length;
        }
      }
    }
    return ranges;
  }

  let peeking = -1;
  function peek(k) {
    if (counting || peeking === k) return;
    unpeek();
    peeking = k;
    items[k].el.classList.add('peek');
    drawers[k].el.classList.add('peeking');
    if (canHighlight) CSS.highlights.set('peek', new Highlight(...rangesFor(k)));
  }

  function unpeek() {
    if (peeking < 0) return;
    items[peeking].el.classList.remove('peek');
    drawers[peeking].el.classList.remove('peeking');
    if (canHighlight) CSS.highlights.delete('peek');
    peeking = -1;
  }

  let counting = null;
  function countOut(k) {
    stopCounting();
    unpeek();
    const ranges = rangesFor(k);
    const total = ranges.length;
    const claimed = shownX[k];
    const d = drawers[k];
    const tally = canHighlight ? new Highlight() : null;
    const head = canHighlight ? new Highlight() : null;
    if (canHighlight) {
      CSS.highlights.set('tally', tally);
      CSS.highlights.set('tally-head', head);
    }
    d.el.classList.remove('counted');
    d.el.classList.add('counting');
    const duration = reducedMotion.matches ? 0 : Math.min(2600, Math.max(700, total * 26));
    const t0 = performance.now();
    let shown = 0;
    counting = { k, frame: 0, timer: 0 };
    const step = (now) => {
      const p = duration ? Math.min(1, (now - t0) / duration) : 1;
      const target = Math.round(p * total);
      while (shown < target) {
        if (tally) tally.add(ranges[shown]);
        shown++;
      }
      if (head) {
        head.clear();
        if (p < 1 && shown > 0) head.add(ranges[shown - 1]);
      }
      d.n.textContent = shown;
      if (p < 1) {
        counting.frame = requestAnimationFrame(step);
        return;
      }
      d.el.classList.remove('counting');
      d.n.textContent = claimed;
      const match = total === claimed;
      d.el.classList.toggle('counted', match);
      setStatus(
        match
          ? `Counted ${fmt(total)} ${plural(k, total)} on the page. The P.S. says ${A.numberWords(claimed)}.`
          : `Counted ${fmt(total)} ${plural(k, total)} on the page, but the P.S. says ${A.numberWords(claimed)}.`
      );
      counting.timer = setTimeout(stopCounting, 5000);
    };
    counting.frame = requestAnimationFrame(step);
  }

  function stopCounting() {
    if (!counting) return;
    cancelAnimationFrame(counting.frame);
    clearTimeout(counting.timer);
    const d = drawers[counting.k];
    d.el.classList.remove('counting', 'counted');
    d.n.textContent = shownX[counting.k];
    if (canHighlight) {
      CSS.highlights.delete('tally');
      CSS.highlights.delete('tally-head');
    }
    counting = null;
  }

  // Events

  for (const d of drawers) {
    const k = +d.el.dataset.k;
    d.el.addEventListener('pointerenter', () => peek(k));
    d.el.addEventListener('pointerleave', unpeek);
    d.el.addEventListener('focus', () => peek(k));
    d.el.addEventListener('blur', unpeek);
    d.el.addEventListener('click', () => countOut(k));
  }
  for (const item of items) {
    const k = +item.el.dataset.k;
    item.el.addEventListener('pointerenter', () => peek(k));
    item.el.addEventListener('pointerleave', unpeek);
    item.el.addEventListener('click', () => countOut(k));
  }

  bodyEl.addEventListener('input', () => {
    cancelAnimationFrame(frame);
    search = null;
    stopCounting();
    unpeek();
    hidePostmark();
    const counts = pageCounts();
    renderPS(openingEl.textContent, shownX, counts);
    renderCase(shownX, counts);
    psEl.classList.add('unsettled');
    setStatus('The letter changed, so the P.S. isn’t true anymore. The search starts again when you pause.');
    clearTimeout(debounce);
    debounce = setTimeout(() => startSearch(), 450);
  });

  bodyEl.addEventListener('paste', (e) => {
    e.preventDefault();
    const text = e.clipboardData.getData('text/plain');
    document.execCommand('insertText', false, text);
  });

  $('another').addEventListener('click', () => {
    clearTimeout(debounce);
    startSearch({ startAt: Math.floor(Math.random() * OPENINGS.length) });
  });

  $('restore').addEventListener('click', () => {
    clearTimeout(debounce);
    bodyEl.innerHTML = originalHTML;
    startSearch({ seed: FIRST_SEED });
  });

  $('copy').addEventListener('click', async () => {
    const text = bodyEl.innerText.trim() + '\n\n' + psEl.textContent;
    try {
      await navigator.clipboard.writeText(text);
      setStatus('Copied the letter, P.S. included.');
    } catch {
      setStatus('Your browser blocked copying. Select the letter and copy it by hand.');
    }
  });

  slowEl.addEventListener('change', () => {
    if (slowEl.checked && !search) {
      startSearch({ startAt: Math.floor(Math.random() * OPENINGS.length) });
    } else if (search) {
      setStatus(slowEl.checked ? slowMessage() : 'Searching for numbers that agree with themselves.');
    }
  });

  setTimeout(() => {
    overtureUntil = reducedMotion.matches ? 0 : performance.now() + OVERTURE_MS;
    startSearch({ seed: FIRST_SEED });
  }, 400);
})();
