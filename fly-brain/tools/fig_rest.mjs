// The article's figure: does the whole model go back to rest when input stops? Spikes per second in 100 ms bins, mean of 6 seeds, for the
// three versions of the model. Writes data/rest.json. The original rule's wiring is not shipped: make it with export_banc.py before its serotonin lines.
//   node tools/fig_rest.mjs <original banc.gz>
import fs from 'node:fs'; import zlib from 'node:zlib';
import { ADAPT } from '../day.js';
const n=JSON.parse(fs.readFileSync('model/neurons.json')),L=n.life,wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const looms=Object.values(n.looms).flatMap(v=>[...v.left,...v.right]);
const models={original:[process.argv[2],false],noSerotonin:['model/banc.gz',false],adapting:['model/banc.gz',true]};
const inputs={walk:{cells:L.P9,hz:150,on:3},loom:{cells:looms,hz:100,on:1}};
const SEEDS=6,T=6,BIN=100,out={binMs:BIN,seeds:SEEDS,seconds:T,inputs:{walk:{name:'P9, 150 Hz',on:3},loom:{name:'Looming neurons, both eyes, 100 Hz',on:1}},series:{}};
for(const [m,[file,adapt]] of Object.entries(models)){const bytes=zlib.gunzipSync(fs.readFileSync(file));out.series[m]={};
 for(const [k,inp] of Object.entries(inputs)){const sum=new Float64Array(T*1000/BIN);
  for(let s=1;s<=SEEDS;s++){const w=(await WebAssembly.instantiate(wasm)).exports,p=w.alloc(bytes.length);new Uint8Array(w.memory.buffer,p,bytes.length).set(bytes);w.init(p,bytes.length,s);if(adapt)w.set_adaptation(ADAPT.step,ADAPT.tau);
   for(const i of inp.cells)w.set_rate(i,inp.hz);for(let b=0;b<sum.length;b++){if(b===inp.on*1000/BIN)for(const i of inp.cells)w.set_rate(i,0);sum[b]+=w.run(BIN*10)/(BIN/1000)/SEEDS;}}
  out.series[m][k]=Array.from(sum,x=>Math.round(x));console.log(m,k,'last bin',out.series[m][k].at(-1));}}
fs.writeFileSync('data/rest.json',JSON.stringify(out));
