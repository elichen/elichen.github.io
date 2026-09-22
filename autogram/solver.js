// Autogram solver: finds counts that make a sentence's letter inventory describe itself.
// The inventory reads "N a's, N b's, ..., and N z's", with "one b" for a count of one.
(function (root) {
  const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';
  const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
  const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
  const MAX_COUNT = 9999;

  function numberWords(n) {
    if (n < 20) return ONES[n];
    if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? '-' + ONES[n % 10] : '');
    if (n < 1000) {
      const rest = n % 100;
      return ONES[Math.floor(n / 100)] + ' hundred' + (rest ? ' ' + numberWords(rest) : '');
    }
    const rest = n % 1000;
    return ONES[Math.floor(n / 1000)] + ' thousand' + (rest ? ' ' + numberWords(rest) : '');
  }

  // Map a character to 0..25, folding case and accents (é counts as e).
  function letterIndex(ch) {
    let code = ch.charCodeAt(0);
    if (code >= 65 && code <= 90) return code - 65;
    if (code >= 97 && code <= 122) return code - 97;
    if (code < 192) return -1;
    code = ch.normalize('NFD').toLowerCase().charCodeAt(0);
    return code >= 97 && code <= 122 ? code - 97 : -1;
  }

  function countLetters(text, into) {
    const counts = into || new Int32Array(26);
    for (const ch of text) {
      const i = letterIndex(ch);
      if (i >= 0) counts[i]++;
    }
    return counts;
  }

  function itemText(letter, n) {
    return numberWords(n) + ' ' + ALPHABET[letter] + (n === 1 ? '' : "'s");
  }

  function inventory(x) {
    return Array.from(x, (n, i) => (i === 25 ? 'and ' : '') + itemText(i, n)).join(', ') + '.';
  }

  // Sparse table of the letters each count adds to the sentence: its number word plus the plural s.
  let table = null;
  function buildTable() {
    const offsets = new Int32Array(MAX_COUNT + 2);
    const letters = [];
    const amounts = [];
    const scratch = new Int32Array(26);
    for (let n = 0; n <= MAX_COUNT; n++) {
      offsets[n] = letters.length;
      scratch.fill(0);
      countLetters(numberWords(n) + (n === 1 ? '' : 's'), scratch);
      for (let k = 0; k < 26; k++) {
        if (scratch[k]) { letters.push(k); amounts.push(scratch[k]); }
      }
    }
    offsets[MAX_COUNT + 1] = letters.length;
    table = { offsets, letters: Int32Array.from(letters), amounts: Int32Array.from(amounts) };
  }

  // Letters that are fixed no matter what the counts are: the prefix, the "and",
  // and each letter's own name in "N x's".
  function constantsFor(prefix) {
    const c = countLetters(prefix + 'and');
    for (let k = 0; k < 26; k++) c[k] += 1;
    return c;
  }

  // What the page would actually contain if the inventory claimed x.
  function actualCounts(c, x, out) {
    if (!table) buildTable();
    const { offsets, letters, amounts } = table;
    const y = out || new Int32Array(26);
    for (let k = 0; k < 26; k++) y[k] = c[k];
    for (let L = 0; L < 26; L++) {
      const n = x[L];
      for (let p = offsets[n], e = offsets[n + 1]; p < e; p++) y[letters[p]] += amounts[p];
    }
    return y;
  }

  function tooLong(c) {
    for (let k = 0; k < 26; k++) if (c[k] > MAX_COUNT - 2000) return true;
    return false;
  }

  // What happens if you just keep replacing every guess with its count.
  function plainIteration(prefix, maxSteps = 5000) {
    const c = constantsFor(prefix);
    let x = Int32Array.from(c);
    const seen = new Map();
    for (let step = 0; step < maxSteps; step++) {
      const key = x.join(',');
      if (seen.has(key)) {
        const first = seen.get(key);
        return { fixed: false, stepsBeforeLoop: first, loopLength: step - first };
      }
      seen.set(key, step);
      const y = actualCounts(c, x);
      let same = true;
      for (let k = 0; k < 26; k++) if (y[k] !== x[k]) { same = false; break; }
      if (same) return { fixed: true, steps: step, x: Array.from(x) };
      x = y;
    }
    return { fixed: false, stepsBeforeLoop: maxSteps, loopLength: 0 };
  }

  // Randomized relaxation across several phrasings of the prefix. Each phrasing keeps its
  // own state; the search rotates between them so one hard phrasing can't stall it.
  class Search {
    constructor(prefixes, { seed = Date.now(), slice = 150000, startAt = 0, exclude = [] } = {}) {
      if (!table) buildTable();
      this.rng = (seed >>> 0) || 0x9e3779b9;
      this.slice = slice;
      this.exclude = new Set(exclude);
      this.variants = prefixes.map((prefix) => {
        const c = constantsFor(prefix);
        return { prefix, c, x: new Int32Array(26), y: new Int32Array(26), next: new Int32Array(26), tried: false };
      });
      this.tooLong = this.variants.some((v) => tooLong(v.c));
      this.active = startAt % prefixes.length;
      this.sliceLeft = slice;
      this.iterations = 0;
      this.nearMisses = 0;
      this.variantsTried = 0;
      this.solution = null;
      if (this.tooLong) return;
      for (const v of this.variants) this.reseed(v);
      this.enter(this.variants[this.active]);
    }

    random() {
      let s = this.rng;
      s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
      this.rng = s;
      return (s >>> 0) * 2.3283064365386963e-10;
    }

    enter(v) {
      if (!v.tried) { v.tried = true; this.variantsTried++; }
    }

    reseed(v) {
      const { c, x, y } = v;
      x.set(c);
      actualCounts(c, x, y);
      for (let k = 0; k < 26; k++) x[k] = Math.max(1, y[k] + Math.floor(this.random() * 7) - 3);
      actualCounts(c, x, y);
    }

    get current() { return this.variants[this.active]; }

    // Run up to `budget` iterations. Returns true once a self-describing inventory is found.
    // Hot loop: state lives in locals and is written back at the end.
    run(budget) {
      if (this.solution || this.tooLong) return !!this.solution;
      const { offsets, letters, amounts } = table;
      let rng = this.rng;
      let iterations = this.iterations;
      let nearMisses = this.nearMisses;
      let sliceLeft = this.sliceLeft;
      let active = this.active;
      let v = this.variants[active];
      let c = v.c, x = v.x, y = v.y, next = v.next;
      for (let i = 0; i < budget; i++) {
        for (let k = 0; k < 26; k++) y[k] = c[k];
        for (let L = 0; L < 26; L++) {
          const n = x[L];
          for (let p = offsets[n], e = offsets[n + 1]; p < e; p++) y[letters[p]] += amounts[p];
        }
        let disagreements = 0;
        for (let k = 0; k < 26; k++) {
          const a = x[k], b = y[k];
          if (a === b) { next[k] = a; continue; }
          disagreements++;
          rng ^= rng << 13; rng ^= rng >>> 17; rng ^= rng << 5;
          const r = (rng >>> 0) * 2.3283064365386963e-10;
          next[k] = a < b ? a + ((r * (b - a + 1)) | 0) : b + ((r * (a - b + 1)) | 0);
        }
        iterations++;
        if (disagreements === 0) {
          if (!this.exclude.has(active + ':' + x.join(','))) {
            this.solution = { variant: active, prefix: v.prefix, x: Array.from(x) };
            break;
          }
          this.rng = rng;
          this.reseed(v);
          rng = this.rng;
          continue;
        }
        if (disagreements === 1) nearMisses++;
        x.set(next);
        if (--sliceLeft <= 0) {
          sliceLeft = this.slice;
          active = (active + 1) % this.variants.length;
          v = this.variants[active];
          c = v.c; x = v.x; y = v.y; next = v.next;
          this.enter(v);
        }
      }
      this.rng = rng;
      this.iterations = iterations;
      this.nearMisses = nearMisses;
      this.sliceLeft = sliceLeft;
      this.active = active;
      if (this.solution) return true;
      // Leave y consistent with x for display.
      actualCounts(c, x, y);
      return false;
    }
  }

  const Autogram = {
    ALPHABET, MAX_COUNT, numberWords, letterIndex, countLetters, itemText, inventory,
    constantsFor, actualCounts, plainIteration, Search,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = Autogram;
  else root.Autogram = Autogram;
})(typeof self !== 'undefined' ? self : this);
