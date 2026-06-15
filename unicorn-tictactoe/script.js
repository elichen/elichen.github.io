/*
 * Rainbow Unicorn Tic-Tac-Toe — UI layer.
 *
 * All game rules and AI live in engine.js (window.UnicornEngine). This file
 * only renders state, plays sound, handles input, and runs the atmosphere.
 * Marks: 'U' = unicorn, 'R' = rainbow.
 */
(function () {
  'use strict';

  var Engine = window.UnicornEngine;
  var STORE_KEY = 'unicorn-ttt';
  var RECORD_KEY = 'unicorn-ttt-record';
  var prefersReducedMotion =
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // =========================================================================
  // Art assets — AI-generated transparent PNG sprites + a painted backdrop.
  //
  // CODEX fills assets/ with these files (see assets/README.md for the full
  // spec: art direction, sizes, aspect ratios). This is the contract the UI
  // renders against. Every path is OPTIONAL: each asset is probed once on boot
  // and, if it loads, the renderer swaps the matching hand-coded SVG (marks /
  // confetti) or CSS painting (background) for the real art. If a file is
  // missing or fails to decode, the SVG/CSS fallback stays, so the page is
  // never broken — no missing-image icon ever shows.
  //
  // Wanted files (filename -> intended use / size / aspect):
  //   unicorn.png      512x512   1:1   Unicorn player mark (transparent)
  //   unicorn@2x.png   1024x1024 1:1   retina Unicorn mark
  //   rainbow.png      512x512   1:1   Rainbow player mark (transparent)
  //   rainbow@2x.png   1024x1024 1:1   retina Rainbow mark
  //   background.png   1600x2400 2:3   painted twilight realm backdrop (opaque)
  //   background-wide.png 2560x1440 16:9 landscape backdrop variant (opaque)
  //   sparkle.png      256x256   1:1   glowing sparkle confetti/spark sprite
  //   star.png         256x256   1:1   second celebration star sprite
  //   cloud.png        512x320   8:5   soft cloud puff scenery (transparent)
  //   win-banner.png   1200x400  3:1   win ribbon, text-free center (transparent)
  //   mote.png         64x64     1:1   soft gold-white bokeh sparkle, per-move spark
  // =========================================================================
  var ASSETS = {
    unicorn:    { src: 'assets/unicorn.png',    src2x: 'assets/unicorn@2x.png',    alt: 'Unicorn' },
    rainbow:    { src: 'assets/rainbow.png',    src2x: 'assets/rainbow@2x.png',    alt: 'Rainbow' },
    background: { src: 'assets/background.png',  src2x: 'assets/background-wide.png' },
    sparkle:    { src: 'assets/sparkle.png' },
    star:       { src: 'assets/star.png' },
    cloud:      { src: 'assets/cloud.png' },
    winBanner:  { src: 'assets/win-banner.png' },
    mote:       { src: 'assets/mote.png' },
  };

  // Probe each asset once. `ready` flips true only after a successful decode, so
  // callers can check Art.has(name) synchronously and otherwise use a fallback.
  // No asset is required; a 404 just leaves ready=false and is swallowed.
  var Art = (function () {
    var state = {};   // name -> { ready, img }
    var pending = 0;
    var onAllSettled = [];

    function settle() {
      if (pending === 0) {
        var cbs = onAllSettled.slice();
        onAllSettled.length = 0;
        for (var i = 0; i < cbs.length; i++) cbs[i]();
      }
    }

    function probe(name, spec) {
      var entry = {
        ready: false,        // base 1x source decoded
        ready2x: false,      // the @2x / wide variant decoded (separate file)
        // Whether the @2x probe has finished either way (loaded or errored). Lets
        // callers tell "still loading" from "confirmed missing" so they can wait
        // for a preferred variant instead of flashing the base art first. No @2x
        // spec means there's nothing to wait on, so it counts as settled.
        settled2x: !spec.src2x,
        img: null,
        src: spec.src,
        src2x: spec.src2x || null,
        alt: spec.alt || '',
      };
      state[name] = entry;

      pending++;
      var img = new Image();
      img.decoding = 'async';
      img.onload = function () {
        entry.ready = true;
        entry.img = img;
        applyWhenReady(name, entry);
        pending--; settle();
      };
      img.onerror = function () {
        pending--; settle();   // stay on the fallback; never show a broken image
      };
      img.src = spec.src;

      // Probe the hi-res variant independently so we only ever reference it
      // (via srcset / the wide backdrop) once it is confirmed to exist. A
      // missing @2x must never make the base mark fall back to its SVG.
      if (spec.src2x) {
        pending++;
        var img2 = new Image();
        img2.decoding = 'async';
        img2.onload = function () {
          entry.ready2x = true;
          entry.settled2x = true;
          applyWhenReady(name, entry);
          pending--; settle();
        };
        img2.onerror = function () {
          entry.settled2x = true;
          applyWhenReady(name, entry);   // let a waiting caller fall back now
          pending--; settle();
        };
        img2.src = spec.src2x;
      }
    }

    for (var key in ASSETS) {
      if (Object.prototype.hasOwnProperty.call(ASSETS, key)) probe(key, ASSETS[key]);
    }

    return {
      has: function (name) { return !!(state[name] && state[name].ready); },
      has2x: function (name) { return !!(state[name] && state[name].ready2x); },
      // True once the @2x / wide variant probe has finished, succeed or fail.
      settled: function (name) { return !!(state[name] && state[name].settled2x); },
      get: function (name) { return state[name] || null; },
      whenSettled: function (cb) { if (pending === 0) cb(); else onAllSettled.push(cb); },
    };
  })();

  // Hook called the moment an individual asset finishes loading, so art can
  // pop in live even after the board has already rendered (e.g. a slow PNG, or
  // Codex dropping files in during development). Defined as a no-op now and
  // reassigned once the renderer exists below.
  var applyWhenReady = function () {};

  // ---- DOM ----
  var cells = Array.prototype.slice.call(document.querySelectorAll('.cell'));
  var statusEl = document.getElementById('status');
  var moveLogEl = document.getElementById('move-log');
  var boardEl = document.getElementById('board');
  var controlsEl = document.getElementById('controls');
  var restartBtn = document.getElementById('restart-btn');
  var muteBtn = document.getElementById('mute-btn');
  var difficultyHint = document.getElementById('difficulty-hint');
  var streakEl = document.getElementById('streak');
  var recordEl = document.getElementById('record');
  var hintKeysEl = document.getElementById('hint-keys');
  var hintDismissBtn = document.getElementById('hint-dismiss');
  var scoreEls = {
    U: document.getElementById('score-u'),
    R: document.getElementById('score-r'),
    tie: document.getElementById('score-tie'),
  };
  var winBeam = document.createElement('span');
  winBeam.className = 'win-beam';
  winBeam.setAttribute('aria-hidden', 'true');
  boardEl.appendChild(winBeam);

  var sceneClouds = document.createElement('div');
  sceneClouds.className = 'scene-clouds';
  sceneClouds.setAttribute('aria-hidden', 'true');
  document.body.appendChild(sceneClouds);

  var winRibbon = document.createElement('div');
  winRibbon.className = 'win-ribbon';
  winRibbon.setAttribute('aria-hidden', 'true');
  // The result message painted INTO the scroll's open center, so the ribbon
  // never reads blank. Positioned over the lavender panel of win-banner.png and
  // sized to fit it; populated by showWinRibbon() and cleared on hide. The
  // status pill below still carries the full sentence for screen readers, so
  // this stays aria-hidden to avoid a double announcement.
  var winRibbonText = document.createElement('span');
  winRibbonText.className = 'win-ribbon-text';
  winRibbon.appendChild(winRibbonText);
  boardEl.parentNode.insertBefore(winRibbon, boardEl);

  // A small "Rainbow is dreaming" badge that rides at the foot of the board
  // during the AI's thinking beat — a readable, at-a-glance cue.
  var ponder = document.createElement('div');
  ponder.className = 'ponder';
  ponder.setAttribute('aria-hidden', 'true');
  ponder.innerHTML =
    '<svg class="svg-icon ponder-mark" data-mark="R" viewBox="0 0 108 100" aria-hidden="true">' +
    '<use href="#mark-rainbow"/></svg>' +
    '<span class="ponder-text">dreaming</span>' +
    '<span class="ponder-dots"><i></i><i></i><i></i></span>';
  boardEl.appendChild(ponder);

  var DIFFICULTY_HINTS = {
    easy: 'Easy plays gently — a kind first opponent.',
    medium: 'Medium pounces if you slip up.',
    hard: 'Hard plays perfectly. A tie is the best anyone can do.',
  };

  // ---- Persistent settings ----
  var defaults = {
    opponent: 'cpu',   // 'cpu' | 'human'
    difficulty: 'hard',
    first: 'U',        // who places the first mark of each game
    muted: false,
  };
  var settings = loadSettings();

  function loadSettings() {
    var s = {};
    for (var k in defaults) s[k] = defaults[k];
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        var saved = JSON.parse(raw);
        for (var key in defaults) {
          if (saved && Object.prototype.hasOwnProperty.call(saved, key)) {
            s[key] = saved[key];
          }
        }
      }
    } catch (e) { /* storage may be unavailable; fall back to defaults */ }
    return s;
  }

  function saveSettings() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(settings)); }
    catch (e) { /* ignore */ }
  }

  // ---- Scores (round-to-round session state) ----
  var scores = { U: 0, R: 0, tie: 0 };
  var round = 1;
  var streak = { mark: null, count: 0 };

  // ---- All-time record vs Hard (persisted separately from settings) ----
  // Against a perfect player you can only tie or lose, so the meaningful
  // lifetime stat is how many forced ties you've earned and your best run of
  // ties in a row. This rewards repeat visitors and frames the whole point:
  // a tie is the best anyone can do against Hard.
  var recordDefaults = { ties: 0, losses: 0, bestTieStreak: 0, tieStreak: 0 };
  var record = loadRecord();

  function loadRecord() {
    var r = {};
    for (var k in recordDefaults) r[k] = recordDefaults[k];
    try {
      var raw = localStorage.getItem(RECORD_KEY);
      if (raw) {
        var saved = JSON.parse(raw);
        for (var key in recordDefaults) {
          if (saved && typeof saved[key] === 'number' && isFinite(saved[key])) {
            r[key] = saved[key];
          }
        }
      }
    } catch (e) { /* storage may be unavailable */ }
    return r;
  }

  function saveRecord() {
    try { localStorage.setItem(RECORD_KEY, JSON.stringify(record)); }
    catch (e) { /* ignore */ }
  }

  // Fold a finished game vs Hard into the lifetime record. 'tie' or the
  // winning mark; the human is always 'U' vs the computer.
  function recordResult(result) {
    if (settings.opponent !== 'cpu' || settings.difficulty !== 'hard') return;
    if (result === 'tie') {
      record.ties += 1;
      record.tieStreak += 1;
      if (record.tieStreak > record.bestTieStreak) {
        record.bestTieStreak = record.tieStreak;
      }
    } else {
      record.tieStreak = 0;
      if (result === 'R') record.losses += 1; // human (U) lost to Hard
    }
    saveRecord();
  }

  function renderRecord() {
    var showVsHard = settings.opponent === 'cpu' && settings.difficulty === 'hard';
    if (!showVsHard || (record.ties === 0 && record.losses === 0)) {
      recordEl.hidden = true;
      recordEl.textContent = '';
      return;
    }
    recordEl.hidden = false;
    var parts = [
      'vs Hard — ',
      '<span class="rec-num">' + record.ties + '</span> ' +
        (record.ties === 1 ? 'tie earned' : 'ties earned'),
    ];
    if (record.bestTieStreak > 1) {
      parts.push(
        ' · best run <span class="rec-num">' + record.bestTieStreak + '</span>'
      );
    }
    recordEl.innerHTML = parts.join('');
  }

  // ---- Game state ----
  var game = Engine.createGame(settings.first);
  var thinking = false;          // AI's turn is in flight
  var focusIndex = 0;            // keyboard cursor on the board
  var aiTimer = null;

  // The human always plays Unicorn vs the computer. In 2-player mode both
  // marks are human-controlled.
  function isAiTurn() {
    return settings.opponent === 'cpu' && game.toMove === 'R' && !game.isOver;
  }

  // =========================================================================
  // Sound — a soft pentatonic chime palette in A major.
  // =========================================================================
  var Sound = (function () {
    var ctx = null;
    var master = null;
    var echo = null;
    var echoGain = null;
    var air = null;
    var ambience = null;

    // Comfortable default for a kids' page; not blaring on first interaction.
    var LEVEL = 0.6;

    // A-major pentatonic across two octaves (Hz). Pleasant, no harsh intervals.
    var SCALE = [220.00, 246.94, 277.18, 329.63, 369.99,
                 440.00, 493.88, 554.37, 659.25, 739.99, 880.00];

    function ensure() {
      if (ctx) return ctx;
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
      master = ctx.createGain();
      air = ctx.createBiquadFilter();
      air.type = 'lowpass';
      air.frequency.value = 6200;
      air.Q.value = 0.35;
      echo = ctx.createDelay(1.2);
      echo.delayTime.value = 0.18;
      echoGain = ctx.createGain();
      echoGain.gain.value = 0.16;
      echo.connect(echoGain).connect(air);
      air.connect(master).connect(ctx.destination);
      // Fade the master in on creation so the very first chime eases up
      // instead of snapping to full level.
      var now = ctx.currentTime;
      master.gain.setValueAtTime(0.0001, now);
      master.gain.linearRampToValueAtTime(settings.muted ? 0 : LEVEL, now + 0.25);
      return ctx;
    }

    function resume() {
      if (ctx && ctx.state === 'suspended') ctx.resume();
    }

    // A single bell-like voice: triangle + sine octave, gentle attack/release.
    function voice(freq, when, dur, level, type) {
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      var tone = ctx.createBiquadFilter();
      osc.type = type || 'triangle';
      osc.frequency.value = freq;
      tone.type = 'lowpass';
      tone.frequency.setValueAtTime(5200, when);
      tone.frequency.exponentialRampToValueAtTime(1600, when + dur);
      gain.gain.setValueAtTime(0.0001, when);
      gain.gain.exponentialRampToValueAtTime(level, when + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, when + dur);
      osc.connect(tone).connect(gain).connect(air);
      gain.connect(echo);
      osc.start(when);
      osc.stop(when + dur + 0.02);
    }

    function shimmer(when, dur, level) {
      var len = Math.max(1, Math.floor(ctx.sampleRate * dur));
      var buffer = ctx.createBuffer(1, len, ctx.sampleRate);
      var data = buffer.getChannelData(0);
      for (var i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len);
      var src = ctx.createBufferSource();
      var filter = ctx.createBiquadFilter();
      var gain = ctx.createGain();
      filter.type = 'bandpass';
      filter.frequency.value = 4200;
      filter.Q.value = 2.2;
      gain.gain.setValueAtTime(0.0001, when);
      gain.gain.exponentialRampToValueAtTime(level, when + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, when + dur);
      src.buffer = buffer;
      src.connect(filter).connect(gain).connect(air);
      src.start(when);
    }

    function chord(freqs, when, dur, level) {
      for (var i = 0; i < freqs.length; i++) {
        voice(freqs[i], when, dur, level, i === 0 ? 'triangle' : 'sine');
      }
    }

    function startAmbience() {
      if (!ctx || ambience || settings.muted) return;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      var lfo = ctx.createOscillator();
      var lfoGain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 110;
      var now = ctx.currentTime;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(0.018, now + 1.4); // ease the drone in
      lfo.frequency.value = 0.045;
      lfoGain.gain.value = 0.009;
      lfo.connect(lfoGain).connect(gain.gain);
      osc.connect(gain).connect(air);
      osc.start();
      lfo.start();
      ambience = { osc: osc, lfo: lfo, gain: gain };
    }

    // Fade the drone out and free its oscillators so nothing runs for the
    // page lifetime. Safe to call when no ambience is playing.
    function stopAmbience() {
      if (!ctx || !ambience) return;
      var a = ambience;
      ambience = null;
      var now = ctx.currentTime;
      a.gain.gain.cancelScheduledValues(now);
      a.gain.gain.setValueAtTime(a.gain.gain.value, now);
      a.gain.gain.linearRampToValueAtTime(0.0001, now + 0.4);
      a.osc.stop(now + 0.45);
      a.lfo.stop(now + 0.45);
    }

    function place(mark, index) {
      if (!ensure()) return;
      resume();
      startAmbience();
      var now = ctx.currentTime;
      // Unicorn rings a touch higher than rainbow, so moves are audibly distinct.
      var base = mark === 'U' ? 7 : 4;
      var offset = typeof index === 'number' ? (index % 3) * 0.015 : 0;
      voice(SCALE[base], now, 0.48, 0.22, 'triangle');
      voice(SCALE[base + 2], now + 0.045 + offset, 0.68, 0.12, 'sine');
      shimmer(now + 0.02, 0.24, 0.035);
    }

    function win() {
      if (!ensure()) return;
      resume();
      var now = ctx.currentTime;
      var arp = [3, 5, 7, 8, 10];
      for (var i = 0; i < arp.length; i++) {
        voice(SCALE[arp[i]], now + i * 0.1, 0.55, 0.3, 'triangle');
        voice(SCALE[arp[i]] * 2, now + i * 0.1, 0.3, 0.08, 'sine');
      }
      shimmer(now + 0.18, 0.9, 0.055);
      chord([SCALE[3], SCALE[5], SCALE[8]], now + arp.length * 0.1, 1.4, 0.22);
    }

    function tie() {
      if (!ensure()) return;
      resume();
      var now = ctx.currentTime;
      chord([SCALE[2], SCALE[4]], now, 0.7, 0.22);
      chord([SCALE[3], SCALE[5]], now + 0.22, 1.0, 0.22);
      shimmer(now + 0.1, 0.5, 0.025);
    }

    function illegal() {
      if (!ensure()) return;
      resume();
      var now = ctx.currentTime;
      voice(SCALE[1], now, 0.16, 0.12, 'sine');
      voice(SCALE[0], now + 0.06, 0.2, 0.1, 'sine');
    }

    function setMuted(m) {
      if (!master) return;
      master.gain.cancelScheduledValues(ctx.currentTime);
      master.gain.linearRampToValueAtTime(m ? 0 : LEVEL, ctx.currentTime + 0.08);
      // Don't leave the 110Hz drone running while muted; it resumes on the
      // next placed mark once sound is back on.
      if (m) stopAmbience();
    }

    return { ensure: ensure, resume: resume, place: place, win: win,
             tie: tie, illegal: illegal, setMuted: setMuted,
             stopAmbience: stopAmbience };
  })();

  // =========================================================================
  // Rendering
  // =========================================================================
  // Mark tokens: U/R are the players; S/T are decorative celebration sprites.
  // S and T share the SVG star fallback but map to different PNGs (sparkle vs
  // star) so confetti has visual variety once the real art loads.
  var MARK_HREF = { U: '#mark-unicorn', R: '#mark-rainbow', S: '#mark-star', T: '#mark-star' };
  var MARK_VIEWBOX = { U: '0 0 100 100', R: '0 0 108 100', S: '0 0 100 100', T: '0 0 100 100' };
  var MARK_NAME = { U: 'unicorn', R: 'rainbow' };
  // Which probed asset backs each mark token. Tokens with no ready asset fall
  // through to the hand-coded SVG.
  var MARK_ASSET = { U: 'unicorn', R: 'rainbow', S: 'sparkle', T: 'star' };
  var SVG_NS = 'http://www.w3.org/2000/svg';

  function makeMarkSvg(mark, className) {
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', MARK_VIEWBOX[mark]);
    svg.setAttribute('class', className);
    svg.setAttribute('aria-hidden', 'true');
    var use = document.createElementNS(SVG_NS, 'use');
    use.setAttribute('href', MARK_HREF[mark]);
    svg.appendChild(use);
    return svg;
  }

  // Build a PNG <img> for a mark, with srcset for retina and an onerror guard
  // that swaps the SVG back in if the bytes ever fail to decode after probing.
  // `decorative` keeps celebration sprites out of the accessibility tree; the
  // placed marks instead get a meaningful alt via the cell's aria-label.
  function makeMarkImg(mark, className, decorative) {
    var assetName = MARK_ASSET[mark];
    var entry = Art.get(assetName);
    var img = document.createElement('img');
    img.className = className;
    img.draggable = false;
    img.decoding = 'async';
    img.src = entry.src;
    // Only advertise the @2x candidate once it is confirmed to exist, so a
    // missing retina file never 404s and trips the SVG fallback below.
    if (entry.src2x && Art.has2x(assetName)) {
      img.srcset = entry.src + ' 1x, ' + entry.src2x + ' 2x';
    }
    img.alt = decorative ? '' : (entry.alt || MARK_NAME[mark] || '');
    if (decorative) img.setAttribute('aria-hidden', 'true');
    img.onerror = function () {
      // The probe said this loaded, but a later decode failed — degrade
      // gracefully to the SVG so nothing shows a broken-image glyph.
      var svg = makeMarkSvg(mark, className);
      if (img.parentNode) img.parentNode.replaceChild(svg, img);
    };
    return img;
  }

  // Which backend a mark token will render with right now: 'svg' (fallback),
  // 'img1x' (base PNG), or 'img2x' (base + retina srcset). Drives the render
  // cache key so the board upgrades itself as each asset settles.
  function markBackend(mark) {
    var name = MARK_ASSET[mark];
    if (!name || !Art.has(name)) return 'svg';
    return Art.has2x(name) ? 'img2x' : 'img1x';
  }

  // The renderer's single entry point for a mark element: real PNG art when the
  // asset is ready, otherwise the hand-coded SVG. Used everywhere a mark, ghost,
  // or confetti face is drawn so the upgrade is uniform.
  function makeMark(mark, className, decorative) {
    if (MARK_ASSET[mark] && Art.has(MARK_ASSET[mark])) {
      return makeMarkImg(mark, className, decorative);
    }
    var svg = makeMarkSvg(mark, className);
    if (decorative) svg.setAttribute('aria-hidden', 'true');
    return svg;
  }

  // Reveal the painted backdrop once background.png is ready. Picks the wide
  // landscape variant on short/landscape viewports if Codex shipped one. The
  // CSS sky gradient remains underneath as the permanent fallback.
  var backdropEl = document.getElementById('backdrop');
  function applyBackground() {
    if (!backdropEl || !Art.has('background')) return;
    var entry = Art.get('background');
    var landscape = window.innerWidth > window.innerHeight * 1.2;
    // On a landscape viewport prefer the wide variant up front: if it exists but
    // hasn't decoded yet, hold off rather than paint the 2:3 portrait first and
    // swap to wide a moment later (a visible flash on wide screens). Once the
    // wide image settles, applyWhenReady('background') re-runs this and shows it;
    // if the wide file is missing entirely the probe still ends pending, and
    // Art.whenSettled() runs applyBackground() again to fall back to portrait.
    if (landscape && entry.src2x && !Art.has2x('background') && !Art.settled('background')) {
      return;
    }
    // Use the wide landscape variant only when it actually exists.
    var src = (landscape && Art.has2x('background')) ? entry.src2x : entry.src;
    backdropEl.style.backgroundImage = 'url("' + src + '")';
    backdropEl.classList.add('ready');
    // Hand the scenery over to the painting; Realm may not exist yet if the PNG
    // loaded from cache before the atmosphere module ran, so guard the call.
    if (typeof Realm !== 'undefined' && Realm) Realm.useBackdrop();
  }

  function applyClouds() {
    if (!Art.has('cloud') || sceneClouds.childNodes.length) return;
    var src = Art.get('cloud').src;
    for (var i = 0; i < 2; i++) {
      var img = document.createElement('img');
      img.src = src;
      img.alt = '';
      img.decoding = 'async';
      img.draggable = false;
      sceneClouds.appendChild(img);
    }
    sceneClouds.classList.add('ready');
  }

  function applyWinRibbon() {
    if (!Art.has('winBanner')) return;
    winRibbon.style.backgroundImage = 'url("' + Art.get('winBanner').src + '")';
  }

  // Short result line painted inside the scroll, distinct from the status pill's
  // longer sentence so it fits the open ribbon area.
  function ribbonMessage(result) {
    if (result === 'U') return 'Unicorn wins!';
    if (result === 'R') return 'Rainbow wins!';
    return 'Perfect tie!';
  }

  function showWinRibbon(result) {
    if (!Art.has('winBanner')) return;
    applyWinRibbon();
    winRibbonText.textContent = ribbonMessage(result);
    winRibbon.classList.add('has-text');
    winRibbon.classList.remove('show');
    void winRibbon.offsetWidth;
    winRibbon.classList.add('show');
  }

  function hideWinRibbon() {
    winRibbon.classList.remove('show');
    winRibbon.classList.remove('has-text');
    winRibbonText.textContent = '';
  }

  window.addEventListener('resize', function () {
    if (Art.has('background')) applyBackground();
  });

  function rowCol(i) {
    return 'row ' + (Math.floor(i / 3) + 1) + ' column ' + (i % 3 + 1);
  }

  // Per-cell record of what SVG is currently in the DOM, so render() can
  // rebuild only the children that actually changed. Key is "placed:U",
  // "ghost:R", or "" for empty.
  var cellContent = ['', '', '', '', '', '', '', '', ''];

  // Render is a pure function of engine state. For 9 cells the work is tiny,
  // but only the cells whose displayed mark/ghost changed touch the DOM — no
  // per-move innerHTML churn, and nothing left dangling to leak.
  function render(opts) {
    opts = opts || {};
    var board = game.board;
    var line = game.winningLine;
    var canPlay = humanCanPlay();
    boardEl.classList.toggle('thinking', thinking);
    // Always-on faint ghost guide while a human may move (and the game is live).
    boardEl.classList.toggle('invite', canPlay && !game.isOver);

    for (var i = 0; i < 9; i++) {
      var cell = cells[i];
      var mark = board[i];

      // Decide what this cell should show. The key encodes the art backend
      // (svg / img1x / img2x), so a mark drawn as SVG is rebuilt as a PNG once
      // its asset loads, and upgraded again to srcset when the @2x arrives.
      var key = '';
      if (mark) key = 'placed:' + mark + ':' + markBackend(mark);
      else if (canPlay) key = 'ghost:' + game.toMove + ':' + markBackend(game.toMove);

      // Only rebuild children when the content key changes.
      if (key !== cellContent[i]) {
        cellContent[i] = key;
        cell.textContent = '';
        if (mark) {
          var placed = makeMark(mark, 'mark placed', true);
          // Only animate the cell that was just played.
          if (opts.justPlayed !== i) placed.style.animation = 'none';
          cell.appendChild(placed);
        } else if (canPlay) {
          cell.appendChild(makeMark(game.toMove, 'mark ghost', true));
        }
      }

      cell.classList.toggle('taken', !!mark);
      cell.classList.remove('winner', 'dim');
      cell.setAttribute(
        'aria-disabled',
        (game.isOver || !!mark || thinking || isAiTurn()) ? 'true' : 'false'
      );
      cell.setAttribute(
        'aria-label',
        mark ? rowCol(i) + ', ' + MARK_NAME[mark] : rowCol(i) + ', empty'
      );
      cell.setAttribute('tabindex', i === focusIndex ? '0' : '-1');
    }

    if (line) {
      positionWinBeam(line);
      for (var j = 0; j < 9; j++) {
        if (line.indexOf(j) === -1 && board[j]) cells[j].classList.add('dim');
      }
      for (var k = 0; k < line.length; k++) cells[line[k]].classList.add('winner');
    } else {
      winBeam.classList.remove('show');
    }
  }

  // Now that render() exists, let a late-arriving asset upgrade the live board.
  // A placed unicorn that first drew as SVG becomes the real PNG the instant
  // unicorn.png decodes — no reload, no flash for the common (already-cached or
  // missing) case because render() only rebuilds cells whose backend changed.
  applyWhenReady = function (name) {
    if (name === 'unicorn' || name === 'rainbow') { render(); upgradeIcons(); }
    if (name === 'background') applyBackground();
    if (name === 'cloud') applyClouds();
    if (name === 'winBanner') applyWinRibbon();
  };

  // Swap the small inline-SVG icons (scoreboard chips, the "dreaming" badge) for
  // the matching PNG once it loads, so every appearance of a mark shares one art
  // set. Each icon carries a data-mark; we replace it in place and tag the new
  // node so we never double-swap. SVG stays put if the asset never arrives.
  function upgradeIcons() {
    var icons = document.querySelectorAll('.svg-icon[data-mark]');
    for (var i = 0; i < icons.length; i++) {
      var node = icons[i];
      var mark = node.getAttribute('data-mark');
      var assetName = MARK_ASSET[mark];
      if (!assetName || !Art.has(assetName)) continue;
      var img = makeMarkImg(mark, node.getAttribute('class'), true);
      img.setAttribute('data-mark-upgraded', '1');
      node.parentNode.replaceChild(img, node);
    }
  }

  function positionWinBeam(line) {
    var boardRect = boardEl.getBoundingClientRect();
    var a = cells[line[0]].getBoundingClientRect();
    var c = cells[line[2]].getBoundingClientRect();
    var ax = a.left + a.width / 2 - boardRect.left;
    var ay = a.top + a.height / 2 - boardRect.top;
    var cx = c.left + c.width / 2 - boardRect.left;
    var cy = c.top + c.height / 2 - boardRect.top;
    var dx = cx - ax;
    var dy = cy - ay;
    var pad = Math.min(a.width, a.height) * 0.22;
    var len = Math.sqrt(dx * dx + dy * dy);
    var angle = Math.atan2(dy, dx);
    var x = ax - Math.cos(angle) * pad;
    var y = ay - Math.sin(angle) * pad;

    winBeam.style.setProperty('--beam-x', x + 'px');
    winBeam.style.setProperty('--beam-y', y + 'px');
    winBeam.style.setProperty('--beam-width', (len + pad * 2) + 'px');
    winBeam.style.setProperty('--beam-angle', angle + 'rad');
    if (!winBeam.classList.contains('show')) {
      winBeam.classList.add('show');
    }
  }

  // Announce a placed mark to screen readers via the polite move-log region,
  // e.g. "Unicorn, row 2, column 2". Clearing then re-setting on a short timeout
  // makes assistive tech re-read even when the same player plays twice in a row.
  // A timeout (not rAF) is used so the announcement still fires when the tab is
  // backgrounded, where rAF is suspended. Kept separate from the status pill so
  // the move and the turn prompt don't fight over one live region.
  var announceTimer = null;
  function announceMove(mark, index) {
    if (!moveLogEl) return;
    var text = MARK_NAME[mark].charAt(0).toUpperCase() +
      MARK_NAME[mark].slice(1) + ', ' + rowCol(index);
    moveLogEl.textContent = '';
    clearTimeout(announceTimer);
    announceTimer = setTimeout(function () { moveLogEl.textContent = text; }, 60);
  }

  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.classList.toggle('win', kind === 'win');
    if (kind === 'shake') {
      statusEl.classList.remove('shake');
      // reflow to restart the animation
      void statusEl.offsetWidth;
      statusEl.classList.add('shake');
    }
  }

  function turnMessage() {
    if (settings.opponent === 'cpu') {
      return game.toMove === 'U' ? 'Your turn, little unicorn!' : 'Rainbow is dreaming up a move...';
    }
    return game.toMove === 'U' ? "Unicorn's turn!" : "Rainbow's turn!";
  }

  function updateStatusForTurn() {
    setStatus(turnMessage(), null);
  }

  function bumpScore(key) {
    scores[key] += 1;
    var el = scoreEls[key];
    el.textContent = scores[key];
    el.classList.remove('bump');
    void el.offsetWidth;
    el.classList.add('bump');
  }

  function updateStreak(result) {
    if (result) {
      round += 1;
      if (result === 'tie') {
        streak.mark = null;
        streak.count = 0;
      } else if (streak.mark === result) {
        streak.count += 1;
      } else {
        streak.mark = result;
        streak.count = 1;
      }
    }
    var label = streak.mark
      ? (streak.mark === 'U' ? 'Unicorn' : 'Rainbow') + ' streak ' + streak.count
      : 'Streak 0';
    streakEl.textContent = 'Round ' + round + ' · ' + label;
  }

  // =========================================================================
  // Game flow
  // =========================================================================
  function applyMove(index) {
    var before = game;
    game = game.play(index);
    if (game === before) { // illegal — square taken or game over
      Sound.illegal();
      setStatus(turnMessage(), 'shake');
      return;
    }
    Sound.place(game.board[index], index);
    announceMove(game.board[index], index);
    render({ justPlayed: index });
    burstFromCell(index, game.board[index]);

    if (game.isOver) {
      finish();
      return;
    }
    updateStatusForTurn();

    if (isAiTurn()) scheduleAi();
  }

  function finish() {
    if (game.winner) {
      var w = game.winner;
      bumpScore(w);
      updateStreak(w);
      recordResult(w);
      renderRecord();
      var msg;
      if (settings.opponent === 'cpu') {
        msg = w === 'U' ? 'You win! The horn glows bright!' : 'Rainbow wins this one. Try again?';
      } else {
        msg = (w === 'U' ? 'Unicorn' : 'Rainbow') + ' wins! Magical!';
      }
      setStatus(msg, 'win');
      showWinRibbon(w);
      Sound.win();
      Realm.victory(game.winningLine, w);
      celebrate(game.winningLine);
    } else {
      bumpScore('tie');
      updateStreak('tie');
      recordResult('tie');
      renderRecord();
      var tieMsg = 'A friendly tie. Everyone shines!';
      if (settings.opponent === 'cpu' && settings.difficulty === 'hard') {
        tieMsg = 'A perfect tie — the best anyone can do against Hard!';
      }
      setStatus(tieMsg, null);
      showWinRibbon('tie');
      Sound.tie();
    }
    render(); // disable cells, dim losers
  }

  function scheduleAi() {
    thinking = true;
    render(); // disables cells while the AI "thinks"
    // A short, readable beat so the move feels considered, not instant.
    var delay = prefersReducedMotion ? 220 : 420 + Math.random() * 280;
    clearTimeout(aiTimer);
    aiTimer = setTimeout(function () {
      thinking = false;
      if (!isAiTurn()) { render(); return; } // state changed (e.g. restart)
      var move = Engine.chooseMove(game.board, 'R', settings.difficulty);
      if (move >= 0) {
        game = game.play(move);
        Sound.place('R', move);
        announceMove('R', move);
        render({ justPlayed: move });
        burstFromCell(move, 'R');
        if (game.isOver) { finish(); return; }
      }
      updateStatusForTurn();
    }, delay);
  }

  function newGame() {
    clearTimeout(aiTimer);
    thinking = false;
    hideWinRibbon();
    game = Engine.createGame(settings.first);
    focusIndex = firstEmpty();
    updateStreak();
    render();
    updateStatusForTurn();
    if (isAiTurn()) scheduleAi();
  }

  function firstEmpty() {
    for (var i = 0; i < 9; i++) if (!game.board[i]) return i;
    return 0;
  }

  // =========================================================================
  // Input — pointer + full keyboard navigation
  // =========================================================================
  function humanCanPlay() {
    return !game.isOver && !thinking && !isAiTurn();
  }

  cells.forEach(function (cell) {
    cell.addEventListener('click', function () {
      Sound.ensure();
      Sound.resume();
      var i = Number(cell.dataset.i);
      focusIndex = i;
      if (!humanCanPlay() || game.board[i]) {
        if (!game.isOver) { Sound.illegal(); setStatus(turnMessage(), 'shake'); }
        return;
      }
      applyMove(i);
    });
  });

  // Arrow keys move the cursor; Enter/Space place a mark.
  boardEl.addEventListener('keydown', function (e) {
    var i = focusIndex;
    var handled = true;
    switch (e.key) {
      case 'ArrowRight': i = (i % 3 === 2) ? i - 2 : i + 1; break;
      case 'ArrowLeft':  i = (i % 3 === 0) ? i + 2 : i - 1; break;
      case 'ArrowDown':  i = (i + 3) % 9; break;
      case 'ArrowUp':    i = (i + 6) % 9; break;
      case 'Home':       i = Math.floor(i / 3) * 3; break;
      case 'End':        i = Math.floor(i / 3) * 3 + 2; break;
      case 'Enter':
      case ' ':
      case 'Spacebar':
        Sound.ensure(); Sound.resume();
        if (humanCanPlay() && !game.board[focusIndex]) applyMove(focusIndex);
        else if (!game.isOver) { Sound.illegal(); setStatus(turnMessage(), 'shake'); }
        e.preventDefault();
        return;
      default: handled = false;
    }
    if (handled) {
      e.preventDefault();
      focusIndex = i;
      cells[i].focus();
      cells[i].setAttribute('tabindex', '0');
      cells.forEach(function (c, idx) { if (idx !== i) c.setAttribute('tabindex', '-1'); });
    }
  });

  cells.forEach(function (cell) {
    cell.addEventListener('focus', function () { focusIndex = Number(cell.dataset.i); });
  });

  // =========================================================================
  // Settings UI (segmented radio groups + actions)
  //
  // Each .segmented is a true ARIA radiogroup: roving tabindex keeps a single
  // tab stop, and Arrow keys move the selection like a native radio set.
  // =========================================================================

  // Reflect the chosen value as aria-checked, and give the checked button the
  // only tabindex=0 in its group so Tab lands on the active choice.
  function syncSegmented(attr, value, disabled) {
    var group = document.querySelector('.segmented[data-group="' + attr + '"]');
    var btns = document.querySelectorAll('[data-' + attr + ']');
    if (group) group.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    btns.forEach(function (b) {
      var on = b.getAttribute('data-' + attr) === value;
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.setAttribute('aria-disabled', disabled ? 'true' : 'false');
      b.setAttribute('tabindex', (!disabled && on) ? '0' : '-1');
    });
  }

  function syncControls() {
    var human = settings.opponent === 'human';
    syncSegmented('opponent', settings.opponent);
    syncSegmented('difficulty', settings.difficulty, human);
    syncSegmented('first', settings.first);
    difficultyHint.textContent = DIFFICULTY_HINTS[settings.difficulty];
    controlsEl.classList.toggle('human', human);
    muteBtn.setAttribute('aria-pressed', settings.muted ? 'true' : 'false');
    muteBtn.setAttribute('aria-label', settings.muted ? 'Unmute sound' : 'Mute sound');
    renderRecord();
  }

  // Wire one segmented group: pointer selection plus full keyboard support.
  function wireSegmented(attr, onChange) {
    var group = document.querySelector('.segmented[data-group="' + attr + '"]');
    if (!group) return;
    var options = Array.prototype.slice.call(
      group.querySelectorAll('[data-' + attr + ']')
    );

    function valueOf(btn) { return btn.getAttribute('data-' + attr); }

    function select(btn, fromKeyboard) {
      if (group.getAttribute('aria-disabled') === 'true') return;
      onChange(valueOf(btn));
      // syncControls() (called by onChange) has already refreshed tabindex.
      if (fromKeyboard) btn.focus();
    }

    options.forEach(function (btn) {
      btn.addEventListener('click', function () { select(btn, false); });
    });

    group.addEventListener('keydown', function (e) {
      var current = options.indexOf(document.activeElement);
      if (current === -1) current = 0;
      var next = current;
      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          next = (current + 1) % options.length;
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          next = (current - 1 + options.length) % options.length;
          break;
        case 'Home':
          next = 0;
          break;
        case 'End':
          next = options.length - 1;
          break;
        case ' ':
        case 'Spacebar':
        case 'Enter':
          e.preventDefault();
          select(options[current], true);
          return;
        default:
          return; // let other keys (Tab, etc.) behave normally
      }
      e.preventDefault();
      // Arrow keys in an ARIA radiogroup both move focus AND select.
      select(options[next], true);
    });
  }

  wireSegmented('opponent', function (v) {
    settings.opponent = v; saveSettings(); syncControls(); newGame();
  });
  wireSegmented('difficulty', function (v) {
    settings.difficulty = v; saveSettings(); syncControls();
  });
  wireSegmented('first', function (v) {
    settings.first = v; saveSettings(); syncControls(); newGame();
  });

  restartBtn.addEventListener('click', function () {
    Sound.ensure(); Sound.resume();
    newGame();
  });

  muteBtn.addEventListener('click', function () {
    settings.muted = !settings.muted;
    saveSettings();
    Sound.ensure();
    Sound.setMuted(settings.muted);
    syncControls();
  });

  // =========================================================================
  // Win celebration — pooled DOM sprites, rAF-driven, GPU-light.
  //
  // Both the per-move spark motes and the win confetti draw from fixed pools of
  // elements created once and parked off-screen. A burst only repositions and
  // reactivates pool members, so a win never appends ~26 fresh nodes mid-rAF
  // (no layout churn, nothing to garbage-collect, friendly to low-end mobile).
  // Sprites are SVG <use> clones of the hand-drawn marks + a custom star — no
  // OS emoji glyphs.
  // =========================================================================
  var activeSparks = [];      // currently animating pool members
  var motePool = [];          // small round spark motes (per-move bursts)
  var confettiPool = [];      // larger SVG sprites (win celebration)
  var rafId = 0;
  var lastSparkTime = 0;

  var MOTE_POOL_SIZE = 36;    // covers a couple of overlapping per-move bursts
  var CONFETTI_POOL_SIZE = 30;

  // Build a mote (a glowing dot). Class flips between u/r per use.
  function buildMote() {
    var el = document.createElement('div');
    el.className = 'spark mote';
    el.style.display = 'none';
    document.body.appendChild(el);
    return { el: el, active: false, kind: 'mote' };
  }

  // Build a confetti sprite: a stack of faces (unicorn, rainbow, sparkle star);
  // we toggle which one is visible per use so one pooled node can be any of
  // them. Each face is real PNG art when its asset is ready, else the SVG —
  // resolved at build time, and the pool is rebuilt after assets settle (below).
  function buildConfetti() {
    var el = document.createElement('div');
    el.className = 'spark sprite';
    el.style.display = 'none';
    var faces = {};
    ['U', 'R', 'S', 'T'].forEach(function (m) {
      var face = makeMark(m, 'mark face', true);
      face.style.width = '100%';
      face.style.height = '100%';
      face.style.display = 'none';
      el.appendChild(face);
      faces[m] = face;
    });
    document.body.appendChild(el);
    return { el: el, faces: faces, active: false, kind: 'confetti' };
  }

  // Find a free pool member, lazily growing the pool on first use only.
  function takeFrom(pool, builder, size) {
    for (var i = 0; i < pool.length; i++) {
      if (!pool[i].active) return pool[i];
    }
    if (pool.length < size) {
      var p = builder();
      pool.push(p);
      return p;
    }
    return null; // pool saturated — drop the extra spark rather than allocate
  }

  function launch(p, opts) {
    p.active = true;
    p.x = opts.x; p.y = opts.y;
    p.vx = opts.vx; p.vy = opts.vy;
    p.rot = Math.random() * 360;
    p.spin = opts.spin;
    p.life = 0;
    p.ttl = opts.ttl;
    p.gravity = opts.gravity;
    p.el.style.display = 'block';
    p.el.style.opacity = '1';
    p.el.style.transform = 'translate(' + opts.x + 'px,' + opts.y + 'px)';
    if (activeSparks.indexOf(p) === -1) activeSparks.push(p);
    if (!rafId) rafId = requestAnimationFrame(stepSparks);
  }

  function burstFromCell(index, mark) {
    if (prefersReducedMotion) return;
    var cell = cells[index];
    var rect = cell.getBoundingClientRect();
    var originX = rect.left + rect.width / 2;
    var originY = rect.top + rect.height / 2;
    var ripple = document.createElement('span');
    ripple.className = 'cell-ripple ' + (mark === 'U' ? 'u' : 'r');
    cell.appendChild(ripple);
    setTimeout(function () { ripple.remove(); }, 620);
    Realm.wish(index, mark);

    // Per-move motes use the painted mote.png when it has loaded, with the CSS
    // dot as the graceful fallback so the whole juice layer can be one art family
    // once Codex ships the sprite. The .png class drops the solid dot fill but
    // keeps the player-tinted glow, and the asset can pop in live: each mote
    // resolves its backend at launch time, so a cached or late mote.png upgrades
    // the next burst with no reload.
    var hasMote = Art.has('mote');
    var moteSrc = hasMote ? Art.get('mote').src : null;
    var cls = 'spark mote ' + (mark === 'U' ? 'u' : 'r') + (hasMote ? ' png' : '');
    for (var i = 0; i < 12; i++) {
      var p = takeFrom(motePool, buildMote, MOTE_POOL_SIZE);
      if (!p) break;
      var size = hasMote ? (12 + Math.random() * 14) : (7 + Math.random() * 9);
      p.el.className = cls;
      p.el.style.backgroundImage = hasMote ? 'url("' + moteSrc + '")' : '';
      p.el.style.width = size + 'px';
      p.el.style.height = size + 'px';
      var a = Math.random() * Math.PI * 2;
      var speed = 130 + Math.random() * 220;
      launch(p, {
        x: originX, y: originY,
        vx: Math.cos(a) * speed,
        vy: Math.sin(a) * speed - 70,
        spin: (Math.random() - 0.5) * 500,
        ttl: 0.62 + Math.random() * 0.3,
        gravity: 360,
      });
    }

    if (!boardEl.classList.contains('pulse')) {
      boardEl.classList.add('pulse');
      setTimeout(function () { boardEl.classList.remove('pulse'); }, 360);
    }
  }

  function celebrate(line) {
    if (prefersReducedMotion) return;
    var rect = boardEl.getBoundingClientRect();
    var originX = rect.left + rect.width / 2;
    var originY = rect.top + rect.height / 2;
    var count = 26;
    // Mostly sparkles/stars with unicorns and rainbows mixed in, so the
    // celebration leans on the celebration art, not the player marks.
    var faceCycle = ['S', 'U', 'T', 'R', 'S', 'T']; // S/T = sparkle/star sprites

    for (var i = 0; i < count; i++) {
      var p = takeFrom(confettiPool, buildConfetti, CONFETTI_POOL_SIZE);
      if (!p) break;
      var size = 16 + Math.random() * 26;
      p.el.style.width = size + 'px';
      p.el.style.height = size + 'px';
      // Show exactly one face on this pooled sprite for this run.
      var face = faceCycle[i % faceCycle.length];
      for (var m in p.faces) {
        p.faces[m].style.display = (m === face) ? 'block' : 'none';
      }
      var angle = Math.random() * Math.PI * 2;
      var speed = 280 + Math.random() * 460;
      launch(p, {
        x: originX, y: originY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 260,
        spin: (Math.random() - 0.5) * 700,
        ttl: 1.5 + Math.random() * 0.7,
        gravity: 900,
      });
    }
  }

  function retire(p) {
    p.active = false;
    p.el.style.display = 'none';
  }

  function stepSparks(now) {
    if (!lastSparkTime) lastSparkTime = now;
    var dt = Math.min((now - lastSparkTime) / 1000, 0.05);
    lastSparkTime = now;

    for (var i = activeSparks.length - 1; i >= 0; i--) {
      var s = activeSparks[i];
      s.life += dt;
      s.vy += (s.gravity || 900) * dt;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.rot += s.spin * dt;
      var fade = Math.max(0, 1 - s.life / s.ttl);
      s.el.style.transform =
        'translate(' + s.x + 'px,' + s.y + 'px) rotate(' + s.rot + 'deg)';
      s.el.style.opacity = fade;
      if (s.life >= s.ttl || s.y > window.innerHeight + 80) {
        retire(s);
        activeSparks.splice(i, 1);
      }
    }
    if (activeSparks.length) {
      rafId = requestAnimationFrame(stepSparks);
    } else {
      rafId = 0;
      lastSparkTime = 0;
    }
  }

  // =========================================================================
  // Atmosphere — drifting stars + a soft aurora band on the background canvas.
  // =========================================================================
  var Realm = (function atmosphere() {
    var canvas = document.getElementById('atmosphere');
    var ctx = canvas.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var stars = [];
    var wishes = [];
    var halos = [];
    var w = 0, h = 0;
    // When a painted background.png is showing it already supplies the moon,
    // hills and meadow, so the canvas drops its own static scenery and keeps
    // only the live, interactive layer (drifting stars, aurora, sparks, halos).
    // No image = canvas paints the whole realm, the standalone fallback look.
    var paintedBackdrop = false;

    function resize() {
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      seed();
    }

    function seed() {
      stars = [];
      // Fewer drifting stars over a painted sky (it may already show stars);
      // more when the canvas is the whole backdrop.
      var n = Math.round((w * h) / (paintedBackdrop ? 26000 : 14000));
      for (var i = 0; i < n; i++) {
        stars.push({
          x: Math.random() * w,
          y: Math.random() * h,
          r: Math.random() * 1.6 + 0.3,
          base: 0.25 + Math.random() * 0.6,
          tw: Math.random() * Math.PI * 2,
          tws: 0.6 + Math.random() * 1.4,
          drift: 4 + Math.random() * 10,
        });
      }
    }

    function aurora(time) {
      // two slow, low-opacity color bands swaying across the upper sky
      var bands = [
        { hue: 320, y: h * 0.22, amp: 26, phase: 0,   alpha: 0.10 },
        { hue: 190, y: h * 0.34, amp: 34, phase: 1.7, alpha: 0.08 },
      ];
      for (var b = 0; b < bands.length; b++) {
        var band = bands[b];
        var grad = ctx.createLinearGradient(0, band.y - 60, 0, band.y + 90);
        grad.addColorStop(0, 'hsla(' + band.hue + ',90%,72%,0)');
        grad.addColorStop(0.5, 'hsla(' + band.hue + ',90%,72%,' + band.alpha + ')');
        grad.addColorStop(1, 'hsla(' + band.hue + ',90%,72%,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (var x = 0; x <= w; x += 24) {
          var y = band.y + Math.sin(x * 0.006 + time * 0.0004 + band.phase) * band.amp;
          ctx.lineTo(x, y);
        }
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fill();
      }
    }

    function realm(time) {
      var moonX = w * 0.18, moonY = h * 0.18;
      var glow = ctx.createRadialGradient(moonX, moonY, 4, moonX, moonY, Math.min(w, h) * 0.34);
      glow.addColorStop(0, 'rgba(255,246,214,0.24)');
      glow.addColorStop(0.45, 'rgba(255,190,221,0.08)');
      glow.addColorStop(1, 'rgba(255,190,221,0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, w, h);

      ctx.fillStyle = 'rgba(255,244,216,0.82)';
      ctx.beginPath();
      ctx.arc(moonX, moonY, Math.max(28, Math.min(w, h) * 0.045), 0, Math.PI * 2);
      ctx.fill();

      var drift = prefersReducedMotion ? 0 : Math.sin(time * 0.00025) * 10;
      drawCloud(w * 0.72 + drift, h * 0.18, Math.max(34, w * 0.045), 'rgba(255,255,255,0.11)');
      drawCloud(w * 0.38 - drift * 0.8, h * 0.31, Math.max(24, w * 0.032), 'rgba(255,255,255,0.08)');

      drawHills(h * 0.73, 'rgba(23,18,58,0.58)', 0.0028, time * 0.00014);
      drawHills(h * 0.82, 'rgba(17,35,62,0.62)', 0.0042, time * 0.00018);
      drawCrystalPath();
    }

    function drawCloud(x, y, r, fill) {
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.ellipse(x - r * 0.8, y + r * 0.15, r * 0.75, r * 0.32, 0, 0, Math.PI * 2);
      ctx.ellipse(x, y, r, r * 0.42, 0, 0, Math.PI * 2);
      ctx.ellipse(x + r * 0.85, y + r * 0.12, r * 0.72, r * 0.3, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawHills(baseY, fill, freq, phase) {
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(0, h);
      ctx.lineTo(0, baseY);
      for (var x = 0; x <= w; x += 28) {
        var y = baseY + Math.sin(x * freq + phase) * 20 + Math.sin(x * freq * 2.4 + phase) * 8;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
    }

    function drawCrystalPath() {
      var grad = ctx.createLinearGradient(w * 0.5, h * 0.55, w * 0.5, h);
      grad.addColorStop(0, 'rgba(255,226,122,0)');
      grad.addColorStop(0.55, 'rgba(255,226,122,0.08)');
      grad.addColorStop(1, 'rgba(139,233,255,0.18)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(w * 0.43, h);
      ctx.quadraticCurveTo(w * 0.48, h * 0.78, w * 0.49, h * 0.58);
      ctx.quadraticCurveTo(w * 0.56, h * 0.78, w * 0.62, h);
      ctx.closePath();
      ctx.fill();
    }

    function addWish(index, mark) {
      if (prefersReducedMotion) return;
      var rect = cells[index].getBoundingClientRect();
      var color = mark === 'U' ? '255,179,230' : '139,233,255';
      wishes.push({
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
        px: rect.left + rect.width / 2,
        py: rect.top + rect.height / 2,
        vx: (Math.random() - 0.5) * 36,
        vy: -150 - Math.random() * 70,
        drift: (Math.random() - 0.5) * 20,
        color: color,
        life: 0,
        ttl: 0.85 + Math.random() * 0.25,
      });
      if (wishes.length > 24) wishes.shift();
    }

    function addVictory(line, mark) {
      if (prefersReducedMotion || !line) return;
      var color = mark === 'U' ? '255,179,230' : '139,233,255';
      for (var i = 0; i < line.length; i++) addWish(line[i], mark);
      var rect = boardEl.getBoundingClientRect();
      halos.push({
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2,
        r: rect.width * 0.18,
        max: rect.width * 0.72,
        color: color,
        life: 0,
        ttl: 1.35,
      });
      if (halos.length > 3) halos.shift();
    }

    function drawWishes(dt) {
      if (!wishes.length && !halos.length) return;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';

      for (var hIdx = halos.length - 1; hIdx >= 0; hIdx--) {
        var halo = halos[hIdx];
        halo.life += dt;
        var hp = halo.life / halo.ttl;
        var alpha = Math.max(0, 1 - hp);
        var radius = halo.r + (halo.max - halo.r) * hp;
        var grad = ctx.createRadialGradient(halo.x, halo.y, radius * 0.35, halo.x, halo.y, radius);
        grad.addColorStop(0, 'rgba(' + halo.color + ',' + (0.14 * alpha) + ')');
        grad.addColorStop(0.55, 'rgba(255,226,122,' + (0.10 * alpha) + ')');
        grad.addColorStop(1, 'rgba(' + halo.color + ',0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(halo.x, halo.y, radius, 0, Math.PI * 2);
        ctx.fill();
        if (halo.life >= halo.ttl) halos.splice(hIdx, 1);
      }

      for (var i = wishes.length - 1; i >= 0; i--) {
        var s = wishes[i];
        s.life += dt;
        s.px = s.x;
        s.py = s.y;
        s.vx += Math.sin(s.life * 8) * s.drift * dt;
        s.vy -= 12 * dt;
        s.x += s.vx * dt;
        s.y += s.vy * dt;
        var p = s.life / s.ttl;
        var fade = Math.max(0, 1 - p);
        var trail = ctx.createLinearGradient(s.px, s.py, s.x, s.y);
        trail.addColorStop(0, 'rgba(' + s.color + ',0)');
        trail.addColorStop(1, 'rgba(' + s.color + ',' + (0.72 * fade) + ')');
        ctx.strokeStyle = trail;
        ctx.lineWidth = 2.4 + 2.8 * fade;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(s.px, s.py);
        ctx.lineTo(s.x, s.y);
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,' + (0.85 * fade) + ')';
        ctx.beginPath();
        ctx.arc(s.x, s.y, 2.2 + 2.2 * fade, 0, Math.PI * 2);
        ctx.fill();
        if (s.life >= s.ttl || s.y < -30) wishes.splice(i, 1);
      }
      ctx.restore();
    }

    var running = true;
    var frameQueued = false;   // a frame is already scheduled via rAF
    var lastRealmTime = 0;

    // Schedule the next frame at most once, so refocusing the tab can never
    // stack a second concurrent loop on top of the running one (no double-speed
    // canvas). Every requestAnimationFrame for `frame` goes through here.
    function queueFrame() {
      if (frameQueued) return;
      frameQueued = true;
      requestAnimationFrame(frame);
    }

    function frame(now) {
      frameQueued = false;
      if (!running) return;
      if (!lastRealmTime) lastRealmTime = now;
      var dt = Math.min((now - lastRealmTime) / 1000, 0.05);
      lastRealmTime = now;
      ctx.clearRect(0, 0, w, h);
      if (!paintedBackdrop) realm(now);
      if (!prefersReducedMotion) aurora(now);
      drawWishes(dt);
      for (var i = 0; i < stars.length; i++) {
        var s = stars[i];
        var tw = prefersReducedMotion ? 1 : (0.6 + 0.4 * Math.sin(s.tw + now * 0.001 * s.tws));
        if (!prefersReducedMotion) {
          s.y -= s.drift * 0.0016;
          if (s.y < -2) { s.y = h + 2; s.x = Math.random() * w; }
        }
        ctx.globalAlpha = s.base * tw;
        ctx.fillStyle = '#fff8ff';
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      queueFrame();
    }

    // Pause the loop when the tab is hidden to save battery and avoid drift.
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) { running = false; lastRealmTime = 0; }
      else if (!running) { running = true; queueFrame(); }
    });

    window.addEventListener('resize', resize);
    resize();
    queueFrame();
    return {
      wish: addWish,
      victory: addVictory,
      // Called when background.png loads: drop the canvas's own scenery and
      // re-seed a sparser starfield so the painted art leads.
      useBackdrop: function () {
        if (paintedBackdrop) return;
        paintedBackdrop = true;
        seed();
      },
    };
  })();

  // =========================================================================
  // Onboarding — a one-time, dismissible keyboard nudge.
  //
  // Shown until the visitor either dismisses it or plays a move; the choice is
  // remembered in localStorage so it never nags a returning player.
  // =========================================================================
  var HINT_KEY = 'unicorn-ttt-seen-hint';
  var hintDismissed = false;

  function hintAlreadySeen() {
    try { return localStorage.getItem(HINT_KEY) === '1'; }
    catch (e) { return false; }
  }

  function dismissHint(persist) {
    if (hintDismissed) return;
    hintDismissed = true;
    hintKeysEl.hidden = true;
    if (persist) {
      try { localStorage.setItem(HINT_KEY, '1'); } catch (e) { /* ignore */ }
    }
  }

  function maybeShowHint() {
    if (hintAlreadySeen()) { hintKeysEl.hidden = true; return; }
    hintKeysEl.hidden = false;
    // Auto-tuck after a calm beat so it never overstays; still counts as seen.
    if (!prefersReducedMotion) {
      setTimeout(function () { dismissHint(true); }, 11000);
    }
  }

  hintDismissBtn.addEventListener('click', function () { dismissHint(true); });
  // The first real interaction with the board also retires the hint.
  boardEl.addEventListener('pointerdown', function () { dismissHint(true); });
  boardEl.addEventListener('keydown', function () { dismissHint(true); }, true);

  // =========================================================================
  // Boot
  // =========================================================================
  syncControls();
  newGame();
  maybeShowHint();
  // Catch any asset that finished loading before applyWhenReady was wired up
  // (e.g. a cached background.png), then keep the board in sync once every
  // probe settles so late marks/scenery pop in without a manual refresh.
  applyBackground();
  applyClouds();
  applyWinRibbon();
  Art.whenSettled(function () {
    applyBackground();
    applyClouds();
    applyWinRibbon();
    render();
    upgradeIcons();
  });
})();
