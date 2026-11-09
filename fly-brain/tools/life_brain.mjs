// What the brain + nerve cord (BANC) offers a fly's life, each 3 s, in engine.wasm: walking drive, sugar's stop, feeding, the two takeoff pathways under looms, and the body's own sensory neurons.
//   node tools/life_brain.mjs [seeds]
import fs from 'node:fs'; import zlib from 'node:zlib';
const seeds=+(process.argv[2]||3), n=JSON.parse(fs.readFileSync('model/neurons.json')), L=n.life, bytes=zlib.gunzipSync(fs.readFileSync('model/banc.gz'));
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const looms=s=>[...Object.values(n.looms).flatMap(v=>v[s])];   // LC4, LPLC1, LPLC4: they follow how fast an object grows. LPLC2 follows its size
const legs=Object.values(n.feel.legs).flatMap(l=>Object.values(l).flat()),beat=[...n.feel.haltere.left,...n.feel.haltere.right,...n.feel.wing.left,...n.feel.wing.right];
const cases={ 'nothing':[], 'P9 100':[[L.P9,100]],'P9 150':[[L.P9,150]],'P9 150 + sugar 100':[[L.P9,150],[L.sugar,100]],'P9 150 + sugar 150':[[L.P9,150],[L.sugar,150]],'sugar 100':[[L.sugar,100]],'sugar 150':[[L.sugar,150]],
 'grows fast, left 100':[[looms('left'),100]],'grows fast, right 100':[[looms('right'),100]],'grows fast, left 40':[[looms('left'),40]],'grows fast, right 40':[[looms('right'),40]],
 'large, left 100':[[L.LPLC2.left,100]],'large, right 100':[[L.LPLC2.right,100]],'both, left 100':[[looms('left'),100],[L.LPLC2.left,100]],'both, right 100':[[looms('right'),100],[L.LPLC2.right,100]],
 'both, both eyes 100':[[looms('left'),100],[looms('right'),100],[L.LPLC2.left,100],[L.LPLC2.right,100]],
 'legs feel 50':[[legs,50]],'P9 + legs feel 50':[[L.P9,150],[legs,50]],'P9 + sugar + legs feel 50':[[L.P9,150],[L.sugar,150],[legs,50]],'wings feel 220':[[beat,220]],'both, left 100 + wings feel 220':[[looms('left'),100],[L.LPLC2.left,100],[beat,220]]};
const N=n.named,read={MN9:L.MN9,oDN1:L.oDN1,GFL:N.DNp01.left,GFR:N.DNp01.right,slowL:[...N.DNp04.left,...N.DNp11.left],slowR:[...N.DNp04.right,...N.DNp11.right],DNp03L:N.DNp03.left,DNp03R:N.DNp03.right};
for(const [k,inp] of Object.entries(cases)){const tot={};let all=0,t0=performance.now();for(let s=1;s<=seeds;s++){const w=(await WebAssembly.instantiate(wasm)).exports;const p=w.alloc(bytes.length);new Uint8Array(w.memory.buffer,p,bytes.length).set(bytes);w.init(p,bytes.length,s*1000);w.set_adaptation(0.05,200);
  for(const [c,hz] of inp)for(const i of c)w.set_rate(i,hz);const c=new Map();for(let i=0;i<300;i++){const m=w.run(100);const a=new Uint32Array(w.memory.buffer,w.spikes(),2*m);all+=m;for(let j=1;j<a.length;j+=2)c.set(a[j],(c.get(a[j])||0)+1);}
  for(const r in read){tot[r]=(tot[r]||0)+read[r].reduce((x,i)=>x+(c.get(i)||0),0)/read[r].length/3/seeds;}}
  console.log(k.padEnd(34),Object.entries(tot).map(([a,b])=>`${a} ${b.toFixed(1)}`).join('  '),'brain',(all/3/seeds|0),'spk/s',((performance.now()-t0)/1000/(3*seeds)).toFixed(2),'s/s');}
