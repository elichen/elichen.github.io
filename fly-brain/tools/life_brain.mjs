// What the brain + nerve cord (BANC) offers a fly's life, each 3 s, in engine.wasm: walking drive, sugar's stop, feeding, the giant fiber under looms.
//   node tools/life_brain.mjs [seeds]
import fs from 'node:fs'; import zlib from 'node:zlib';
const seeds=+(process.argv[2]||3), n=JSON.parse(fs.readFileSync('model/neurons.json')), L=n.life, bytes=zlib.gunzipSync(fs.readFileSync('model/banc.gz'));
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const looms=s=>[...Object.values(n.looms).flatMap(v=>v[s])];
const cases={ 'nothing':[], 'P9 100':[[L.P9,100]],'P9 150':[[L.P9,150]],'P9 150 + sugar 100':[[L.P9,150],[L.sugar,100]],'P9 150 + sugar 150':[[L.P9,150],[L.sugar,150]],'sugar 100':[[L.sugar,100]],'sugar 150':[[L.sugar,150]],
 'loom both 100':[[looms('left'),100],[looms('right'),100]],'loom left 100':[[looms('left'),100]],'loom both + LPLC2':[[looms('left'),100],[looms('right'),100],[L.LPLC2.left,100],[L.LPLC2.right,100]],'LPLC2 both':[[L.LPLC2.left,100],[L.LPLC2.right,100]],'loom both + P9':[[looms('left'),100],[looms('right'),100],[L.P9,150]]};
const read={MN9:L.MN9,oDN1:L.oDN1,GF:L.GF,DNp03L:n.named.DNp03.left,DNp03R:n.named.DNp03.right};
for(const [k,inp] of Object.entries(cases)){const tot={};let all=0,t0=performance.now();for(let s=1;s<=seeds;s++){const w=(await WebAssembly.instantiate(wasm)).exports;const p=w.alloc(bytes.length);new Uint8Array(w.memory.buffer,p,bytes.length).set(bytes);w.init(p,bytes.length,s*1000);
  for(const [c,hz] of inp)for(const i of c)w.set_rate(i,hz);const c=new Map();for(let i=0;i<300;i++){const m=w.run(100);const a=new Uint32Array(w.memory.buffer,w.spikes(),2*m);all+=m;for(let j=1;j<a.length;j+=2)c.set(a[j],(c.get(a[j])||0)+1);}
  for(const r in read){tot[r]=(tot[r]||0)+read[r].reduce((x,i)=>x+(c.get(i)||0),0)/read[r].length/3/seeds;}}
  console.log(k.padEnd(22),Object.entries(tot).map(([a,b])=>`${a} ${b.toFixed(1)}`).join('  '),'brain',(all/3/seeds|0),'spk/s',((performance.now()-t0)/1000/(3*seeds)).toFixed(2),'s/s');}
