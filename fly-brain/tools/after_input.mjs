// Does the brain + nerve cord go back to rest when its input stops? Spikes per second by 1 s bins, for the walking and feeding inputs; then, from rest and from
// the state that outlasts P9, the page's outputs under looms and sugar (Hz, 1 s, 3 seeds). tools/life_after.mjs is the same question for looms.
//   node tools/after_input.mjs
import fs from 'node:fs'; import zlib from 'node:zlib';
const n=JSON.parse(fs.readFileSync('model/neurons.json')),L=n.life,bytes=zlib.gunzipSync(fs.readFileSync('model/banc.gz')),wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const fresh=async seed=>{const w=(await WebAssembly.instantiate(wasm)).exports,p=w.alloc(bytes.length);new Uint8Array(w.memory.buffer,p,bytes.length).set(bytes);w.init(p,bytes.length,seed);return w;};
const set=(w,cells,hz)=>{for(const i of cells)w.set_rate(i,hz);};
const looms=s=>Object.values(n.looms).flatMap(v=>v[s]);
const read={oDN1:L.oDN1,MN9:L.MN9,GF:L.GF,dnL:n.named.DNp03.left,dnR:n.named.DNp03.right};
const measure=(w,ms)=>{const c=new Map();let all=0;for(let i=0;i<ms;i++){const m=w.run(10);all+=m;const a=new Uint32Array(w.memory.buffer,w.spikes(),2*m);for(let j=1;j<a.length;j+=2)c.set(a[j],(c.get(a[j])||0)+1);}const o={spikes:all/ms*1000};for(const r in read)o[r]=read[r].reduce((x,i)=>x+(c.get(i)||0),0)/read[r].length/ms*1000;return o;};
for(const [name,cells,secs] of [['P9 150 Hz',L.P9,3],['sugar 150 Hz',L.sugar,3]])for(const seed of [1,2,3]){const w=await fresh(seed);set(w,cells,150);const on=[];for(let i=0;i<secs;i++)on.push(measure(w,1000).spikes|0);set(w,cells,0);const off=[];for(let i=0;i<4;i++)off.push(measure(w,1000).spikes|0);console.log(name.padEnd(13),'seed',seed,'on:',on.join(' '),'off:',off.join(' '));}
// how long P9 takes to tip the model over (spikes per 100 ms past 1,500)
for(const hz of [150,100,50])for(const seed of [1,2,3,4,5,6]){const w=await fresh(seed);set(w,L.P9,hz);let ms=0,c=0,at=null;while(ms<20000&&at===null){c+=w.run(10);ms++;if(ms%100===0){if(c>1500)at=ms;c=0;}}console.log(`P9 ${hz} Hz, seed ${seed}: tips over by ${at} ms`);}
const cases={'looms, both eyes 100 Hz':[[looms('left'),100],[looms('right'),100]],'looms, left eye 100 Hz':[[looms('left'),100]],'looms, right eye 100 Hz':[[looms('right'),100]],'sugar 150 Hz':[[L.sugar,150]]};
for(const [k,inp] of Object.entries(cases))for(const busy of [false,true]){const tot={};for(const seed of [11,12,13]){const w=await fresh(seed);if(busy){set(w,L.P9,150);let ms=0,c=0;while(ms<20000){c+=w.run(10);ms++;if(ms%100===0){if(c>1500)break;c=0;}}measure(w,500);set(w,L.P9,0);measure(w,1000);}for(const [cells,hz] of inp)set(w,cells,hz);const o=measure(w,1000);for(const a in o)tot[a]=(tot[a]||0)+o[a]/3;}console.log(k.padEnd(24),busy?'busy':'rest',Object.entries(tot).map(([a,b])=>`${a} ${b.toFixed(a==='spikes'?0:1)}`).join('  '));}
