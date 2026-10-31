// A fly's day, no page: what it does and how often the handovers fail.
//   node tools/day.mjs <full body dir> [sim seconds] [seed] [swat every s]
import fs from 'node:fs'; import zlib from 'node:zlib'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js';
import { Flyer } from '../flyer.js';
import { Day, FOODS } from '../day.js';
const dir=process.argv[2],T=+(process.argv[3]||120),seed=+(process.argv[4]||1),swatEvery=+(process.argv[5]||0);
const mujoco=await loadMujoco(); const vfs=new mujoco.MjVFS();
for(const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f,new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m=mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`,'utf8'),vfs); const d=new mujoco.MjData(m);
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm')),bytes=zlib.gunzipSync(fs.readFileSync('model/banc.gz')),n=JSON.parse(fs.readFileSync('model/neurons.json'));
const load=(p,shape)=>{const b=fs.readFileSync(p);const e=new WebAssembly.Instance(wasm).exports;return new WasmPolicy(e,new Float32Array(b.buffer,b.byteOffset,b.length/4),shape);};
const W=new Walker(mujoco,m,d,JSON.parse(fs.readFileSync('policy/walk.json')),load('policy/walk.bin')); W.meta=JSON.parse(fs.readFileSync('policy/walk.json'));
const F=new Flyer(mujoco,m,d,JSON.parse(fs.readFileSync('policy/fly.json')),load('policy/fly.bin',[104,256,2,12]));
const b=new WebAssembly.Instance(wasm).exports;const p=b.alloc(bytes.length);new Uint8Array(b.memory.buffer,p,bytes.length).set(bytes);b.init(p,bytes.length,seed);
const D=new Day(mujoco,m,d,W,F,b,n,seed*7919);
if(process.env.SWATTEST){ // the swat table: per side and speed, standing fly; NOJUMP=1 is the control with the giant fiber's jump off
  const trials=+process.env.SWATTEST; D.noJump=!!process.env.NOJUMP;
  for(const [name,side] of [['above',0],['left',1.2],['right',-1.2]])for(const sp of [30,40,60,90]){let hits=0,jumps=0,lat=[],gf=[];
    for(let i=0;i<trials;i++){D.hits=0;D.hunger=0;D.bored=0;D.reset(-12,-16,D.rnd()*6.28);D.calm();const t0=D.t;while(D.t<t0+.8){D.goal=null;D.bored=0;D.step();}D.goal=null;D.bored=0;D.jumps=0;D.swat(side,30,sp,3);const t1=D.t;let fired=null,mg=0;while(D.object){D.step();mg=Math.max(mg,D.GF.rate);if(fired===null&&D.jumps)fired=D.t-t1;}hits+=D.hits;if(D.jumps)jumps++;if(fired!==null)lat.push(fired/(30/sp));gf.push(mg);}
    console.log(`${name} ${sp} cm/s: jumped ${jumps}/${trials}, hit ${hits}/${trials}, jump at ${lat.length?(lat.reduce((a,b)=>a+b)/lat.length).toFixed(2):'-'} of the way, peak GF ${(gf.reduce((a,b)=>a+b)/trials).toFixed(0)} Hz`);}
  process.exit(0);
}
for(const [x,y] of FOODS)D.addFood(x,y);
let last='',next=swatEvery||1e9,t0=performance.now(),meals=0,was=null,flights=0,time={};
const log=s=>console.log(D.t.toFixed(2).padStart(7),s);
let pb=0,trace=[];while(D.t<T){const before=D.life.mode;const u0=D.life.up(),z0=F.position()[2],m0=D.life.t;if(before==='walk'&&Math.round(D.t*1000)%100===0){trace.push(`${D.t.toFixed(1)}:v${D.life.speed.toFixed(1)} y${D.life.yaw.toFixed(1)} u${u0.toFixed(2)} z${z0.toFixed(2)} m${D.mouth.MN9.rate.toFixed(0)}${D.eating?'E':''}`);if(trace.length>12)trace.shift();}D.step();const l=D.life;if((D.putBacks||0)>pb){pb=D.putBacks;log(`PUTBACK from ${before} t ${m0.toFixed(2)} up ${u0.toFixed(2)} z ${z0.toFixed(2)} | ${trace.join(' ')}`);trace=[];}time[l.mode]=(time[l.mode]||0)+1e-3;
  if(l.mode!==before){log(`${before} -> ${l.mode} at ${F.position().map(x=>x.toFixed(1))} up ${l.up().toFixed(2)} goal ${D.goal?D.goal.why:'-'} hunger ${D.hunger.toFixed(2)}`);if(l.mode==='launch')flights++;}
  const g=(D.goal?D.goal.why:'-')+(D.eating?' eating':'');if(g!==last){log(`wish ${g} oDN1 ${D.oDN1.rate.toFixed(1)} MN9 ${D.mouth.MN9.rate.toFixed(1)} hunger ${D.hunger.toFixed(2)}`);last=g;}
  if(D.eating&&!was)meals++;was=D.eating;
  if(D.t>next){next+=swatEvery;D.swat([0,1.2,-1.2][Math.floor(D.rnd()*3)]);log(`SWAT in ${l.mode}`);}
  if(D.lastSwat){log(`swat ${D.lastSwat}, GF ${D.GF.rate.toFixed(0)}`);D.lastSwat=null;}
}
console.log(JSON.stringify({T,flights,meals,food:D.foods.length,hunger:+D.hunger.toFixed(2),putBacks:D.putBacks||0,jumps:D.jumps||0,hits:D.hits||0,aborts:D.life.aborts,calms:D.calms||0,time:Object.fromEntries(Object.entries(time).map(([k,v])=>[k,+v.toFixed(1)])),speed:+(T/((performance.now()-t0)/1000)).toFixed(2)}));
