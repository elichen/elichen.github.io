// JS tokenizer vs the Python tokenizer on the export's test strings (tools/test/ref.json)
import { readFileSync } from 'fs';
import { Tokenizer } from '../tokenizer.js';
const dir = new URL('..', import.meta.url).pathname;
const t0 = Date.now();
const tok = new Tokenizer(JSON.parse(readFileSync(dir + 'data/vocab.json')), readFileSync(dir + 'data/merges.bin').buffer.slice(0));
console.log(`loaded in ${Date.now() - t0} ms`);
const ref = JSON.parse(readFileSync(dir + 'tools/test/ref.json'));
let bad = 0;
for (const { text, ids } of ref.tokenize) {
  const mine = tok.encode(text);
  const ok = JSON.stringify(mine) === JSON.stringify(ids);
  const round = tok.decode(ids) === text;
  if (!ok || !round) { bad++; console.log('MISMATCH', JSON.stringify(text), ok ? '' : `\n  py ${ids}\n  js ${mine}`, round ? '' : `\n  decode ${JSON.stringify(tok.decode(ids))}`); }
}
console.log(bad ? `${bad} FAIL` : `PASS ${ref.tokenize.length} strings`);
