// Run with: node autogram/solver.test.js
const assert = require('assert');
const A = require('./solver.js');

// Number words.
assert.strictEqual(A.numberWords(0), 'zero');
assert.strictEqual(A.numberWords(28), 'twenty-eight');
assert.strictEqual(A.numberWords(100), 'one hundred');
assert.strictEqual(A.numberWords(164), 'one hundred sixty-four');
assert.strictEqual(A.numberWords(2019), 'two thousand nineteen');

// Letter folding.
assert.strictEqual(A.letterIndex('E'), 4);
assert.strictEqual(A.letterIndex('é'), 4);
assert.strictEqual(A.letterIndex('’'), -1);
assert.strictEqual(A.letterIndex('7'), -1);

// Lee Sallows's pangram is a true autogram under these rules.
const sallows =
  "This pangram tallies five a's, one b, one c, two d's, twenty-eight e's, eight f's, six g's, eight h's, " +
  "thirteen i's, one j, one k, three l's, two m's, eighteen n's, fifteen o's, two p's, one q, seven r's, " +
  "twenty-five s's, twenty-two t's, four u's, four v's, nine w's, two x's, four y's and one z.";
const claimed = [5, 1, 1, 2, 28, 8, 6, 8, 13, 1, 1, 3, 2, 18, 15, 2, 1, 7, 25, 22, 4, 4, 9, 2, 4, 1];
assert.deepStrictEqual(Array.from(A.countLetters(sallows)), claimed);
assert.deepStrictEqual(Array.from(A.actualCounts(A.constantsFor('This pangram tallies '), claimed)), claimed);

// The search finds inventories that survive an independent recount.
const prefixes = [
  'Hi there. This sentence contains ',
  'Dear reader, I am a small letter and I hold ',
  'Ten thousand monkeys at ten thousand typewriters would still need to write ',
];
for (const [i, prefix] of prefixes.entries()) {
  const search = new A.Search([prefix, prefix + 'exactly '], { seed: 1234 + i });
  search.run(5e7);
  assert.ok(search.solution, `no solution for ${JSON.stringify(prefix)}`);
  const { x, prefix: used } = search.solution;
  const sentence = used + A.inventory(x);
  assert.deepStrictEqual(Array.from(A.countLetters(sentence)), x, sentence);
}

// Excluding a solution makes the search find a different one.
{
  const first = new A.Search(['This sentence contains '], { seed: 7 });
  first.run(5e7);
  const key = '0:' + first.solution.x.join(',');
  const second = new A.Search(['This sentence contains ', 'This sentence has '], { seed: 7, exclude: [key] });
  second.run(5e7);
  assert.notStrictEqual(second.solution.variant + ':' + second.solution.x.join(','), key);
}

// The plain rule reports either a fixed point or a loop.
const plain = A.plainIteration('This sentence contains ');
assert.ok(plain.fixed || plain.loopLength > 0);

console.log('autogram solver: all tests passed');
