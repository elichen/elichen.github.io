// The whole cycle on one body through life.js, as the page will run it: walk -> take off -> fly -> land -> walk -> take off again.
//   node tools/life.mjs <full body dir> [trials] [LANDING overrides as JSON]
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js';
import { Flyer } from '../flyer.js';
import { Life, TAKEOFF, LANDING } from '../life.js';
const dir=process.argv[2], N=+(process.argv[3]||6), over=JSON.parse(process.argv[4]||'{}'); let seed=7; const rnd=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/2**32;};
const mujoco=await loadMujoco(); const vfs=new mujoco.MjVFS();
for(const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f,new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m=mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`,'utf8'),vfs); const d=new mujoco.MjData(m);
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const load=(p,shape)=>{const b=fs.readFileSync(p);const e=new WebAssembly.Instance(wasm).exports;return new WasmPolicy(e,new Float32Array(b.buffer,b.byteOffset,b.length/4),shape);};
const wm=JSON.parse(fs.readFileSync('policy/walk.json')), fm=JSON.parse(fs.readFileSync('policy/fly.json'));
const W=new Walker(mujoco,m,d,wm,load('policy/walk.bin')); const F=new Flyer(mujoco,m,d,fm,load('policy/fly.bin',[104,256,2,12]));
const L=new Life(mujoco,m,d,W,F,TAKEOFF,{...LANDING,...over});
const f=F.freeq, wait=(T,fn)=>{let t=0;while(t<T){t+=L.step();fn&&fn();if(!Number.isFinite(d.qpos[f]))return false;}return true;};
const reset=(h)=>{mujoco.mj_resetData(m,d);m.opt.timestep=2e-4;W.setState(wm.qpos0);for(const q of F.wq)d.qpos[q]=m.qpos_spring[q];d.qpos[f+3]=Math.cos(h/2);d.qpos[f+4]=0;d.qpos[f+5]=0;d.qpos[f+6]=Math.sin(h/2);mujoco.mj_forward(m,d);L.startWalking();};
const until=(mode,T)=>{let t=0;while(L.mode!==mode&&t<T){t+=L.step();}return L.mode===mode;};
const tally={launch:0,flew:0,touched:0,upright:0,walked:0,again:0};
for(let i=0;i<N;i++){reset((rnd()-.5)*6.2);L.speed=2;L.yaw=0;wait(.4+rnd()*.5);L.speed=0;L.takeOff();
  if(!until('launch',.3)){console.log(i,'aborted');continue;}tally.launch++;until('fly',1);L.speed=14;wait(.8);
  const z=d.qpos[f+2],up=L.up();if(z<.3||up<.2){console.log(i,'fell in flight z',z.toFixed(2));continue;}tally.flew++;
  L.land();const p=[];let vz=0;let t=0;while(L.mode==='descend'&&t<3){t+=L.step();vz=d.qvel[F.freev+2];}
  if(L.mode==='descend'){console.log(i,'never touched');continue;}tally.touched++;const atTouch={z:d.qpos[f+2].toFixed(2),pitch:(L.pitch()*57.3).toFixed(0),up:L.up().toFixed(2),vz:vz.toFixed(1),t:t.toFixed(2),hv:Math.hypot(d.qvel[F.freev],d.qvel[F.freev+1]).toFixed(1)};
  until('walk',.2);L.speed=0;wait(.5);const a={z:d.qpos[f+2].toFixed(3),up:L.up().toFixed(2)};const ok=L.up()>.8&&d.qpos[f+2]<.3;if(ok)tally.upright++;
  let walked=false,dist=0;if(ok){const x=d.qpos[f],y=d.qpos[f+1];L.speed=2;wait(1);dist=Math.hypot(d.qpos[f]-x,d.qpos[f+1]-y);walked=L.up()>.8&&dist>1;if(walked)tally.walked++;}
  let again='-';if(walked){L.speed=0;L.takeOff();until('launch',.3);if(L.mode==='launch'){until('fly',1);wait(.6);again=d.qpos[f+2]>.3&&L.up()>.2;if(again)tally.again++;}else again='aborted';}
  console.log(i,'touch',JSON.stringify(atTouch),'after',JSON.stringify(a),'walked',dist.toFixed(2),'again',again);}
console.log(JSON.stringify(tally),'of',N,'aborts',L.aborts);
