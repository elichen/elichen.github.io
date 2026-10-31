// Does the brain + nerve cord calm down after looms? Firing by 0.25 s bins, input off at 1 s.
import fs from 'node:fs'; import zlib from 'node:zlib';
const n=JSON.parse(fs.readFileSync('model/neurons.json')),bytes=zlib.gunzipSync(fs.readFileSync('model/banc.gz'));
const w=(await WebAssembly.instantiate(await WebAssembly.compile(fs.readFileSync('engine.wasm')))).exports;const p=w.alloc(bytes.length);new Uint8Array(w.memory.buffer,p,bytes.length).set(bytes);w.init(p,bytes.length,5);
const looms=Object.values(n.looms).flatMap(v=>[...v.left,...v.right]),G=new Set(n.life.GF);
for(const hz of [100,50,30]){for(const i of looms)w.set_rate(i,hz);const out=[];for(let b=0;b<16;b++){if(b===4)for(const i of looms)w.set_rate(i,0);let tot=0,gf=0;for(let k=0;k<25;k++){const m=w.run(100);tot+=m;const a=new Uint32Array(w.memory.buffer,w.spikes(),2*m);for(let j=1;j<a.length;j+=2)if(G.has(a[j]))gf++;}out.push(`${(tot/250).toFixed(0)}k/${(gf/2/.25).toFixed(0)}`);}console.log(hz,'Hz, kspikes per s / GF Hz by 0.25 s:',out.join(' '));}
