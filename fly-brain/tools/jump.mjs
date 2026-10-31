// Takeoff with no new policy: on the floor, the legs push by a stated rule while flybody's flight policy runs the wings from the first
// moment, on a path that starts at the fly's own pose and blends into cruise. Searches the push.
//   node tools/jump.mjs <full body dir> [samples] [seed]
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js';
import { Flyer, CONTROL_DT as DT } from '../flyer.js';
const dir=process.argv[2], N=+(process.argv[3]||40); let seed=+(process.argv[4]||1); const rnd=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/2**32;};
const mujoco=await loadMujoco(); const vfs=new mujoco.MjVFS();
for(const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f,new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m=mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`,'utf8'),vfs); const d=new mujoco.MjData(m);
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const load=(p,shape)=>{const b=fs.readFileSync(p);const e=new WebAssembly.Instance(wasm).exports;return new WasmPolicy(e,new Float32Array(b.buffer,b.byteOffset,b.length/4),shape);};
const wm=JSON.parse(fs.readFileSync('policy/walk.json')), fm=JSON.parse(fs.readFileSync('policy/fly.json'));
const W=new Walker(mujoco,m,d,wm,load('policy/walk.bin')); const F=new Flyer(mujoco,m,d,fm,load('policy/fly.bin',[104,256,2,12]));
const smooth=x=>{const u=Math.min(1,Math.max(0,x));return u*u*(3-2*u);};
const CRUISE=-47.5*Math.PI/180, f=F.freeq;
const nm=j=>mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT.value,j), act=n=>mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR.value,n);
const legs=[...Array(m.njnt).keys()].filter(j=>/coxa|femur|tibia|tarsus/.test(nm(j)||'')).map(j=>{const n=nm(j),q=m.jnt_qposadr[j],a=act(n);return{n,q,a,pair:/T1/.test(n)?0:/T2/.test(n)?1:2,joint:n.match(/coxa|femur|tibia|tarsus/)[0],tuck:m.qpos_spring[q]*0.6};}).filter(x=>x.a>=0);
const adh=[...Array(m.nu).keys()].filter(a=>/adhere_claw/.test(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR.value,a)));
console.log('legs',legs.length,'adh',adh.length,'mass',m.body_subtreemass[1]);
function stand(){mujoco.mj_resetData(m,d);W.setState(wm.qpos0);for(const q of F.wq)d.qpos[q]=m.qpos_spring[q];d.qpos[f]=0;d.qpos[f+1]=0;for(const l of legs)d.ctrl[l.a]=d.qpos[l.q];mujoco.mj_forward(m,d);}
stand(); const foot=['T1','T2','T3'].flatMap(t=>['left','right'].map(s=>m.body(`walker/claw_${t}_${s}`)?.id));
for(const l of legs){ // +1 if a bigger angle moves this leg's foot further from the thorax
  const s=/left/.test(l.n)?'left':'right',t=['T1','T2','T3'][l.pair],c=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY.value,`walker/claw_${t}_${s}`),th=F.root;
  const r=v=>{const q0=d.qpos[l.q];d.qpos[l.q]=q0+v;mujoco.mj_forward(m,d);const z=d.xpos[3*th+2]-d.xpos[3*c+2];d.qpos[l.q]=q0;return z;};
  l.sign=r(.1)>r(-.1)?1:-1;l.q0=d.qpos[l.q];}
mujoco.mj_forward(m,d);
console.log('stance z',d.qpos[f+2].toFixed(3),'pitch deg',(2*Math.atan2(d.qpos[f+5],d.qpos[f+3])*57.3).toFixed(1),'nleg sample',legs.slice(0,4).map(l=>l.n+':'+l.q0.toFixed(2)+':'+l.tuck.toFixed(2)).join(' '));
m.opt.timestep=5e-5;
function run(P,dur=0.7){
  stand();
  // the rear-up, wings shut: front legs reach, hind legs crouch (Card & Dickinson 2008: a fly sets its posture before it jumps)
  const pre=Math.round(P.pre/m.opt.timestep); for(const a of adh)d.ctrl[a]=0;
  for(let k=0;k<pre;k++){const u=smooth(k/pre/0.8);for(const l of legs){const j=l.joint==='coxa'?0:P.pamp[l.pair];d.ctrl[l.a]=l.q0+l.sign*j*u;}mujoco.mj_step(m,d);}
  const p0=2*Math.atan2(d.qpos[f+5],d.qpos[f+3]), set=legs.map(l=>d.ctrl[l.a]);
  F.phase=0;F.freq=fm.base_freq;F.buf.fill(0);F.nbuf=0;F.yaw=0;F.speed=0;F.pitch=p0;F.height=1;
  const X0=d.qpos[f],Y0=d.qpos[f+1],{p}=F.pose(0,0,0,0),off=[p[0],p[1],p[2]-d.qpos[f+2]];
  F.ref={x:X0-off[0],y:Y0-off[1],heading:0};F.height=1-off[2];const h0=F.height;
  let late=0,minUp=1,zmax=0,liftoff=-1;
  for(let k=0;k<dur/DT;k++){const t=k*DT,s=smooth(t/P.T);
    F.pitch=p0+(CRUISE-p0)*s;F.height=h0+P.rise*s;F.speed=14*smooth((t-P.T*.5)/P.T);
    const u=smooth((t-P.t0)/P.tp),v=smooth((t-P.t0-P.tp)/0.06);
    legs.forEach((l,i)=>{const j=l.joint==='coxa'?0:P.amp[l.pair];const c=set[i]+u*l.sign*j;d.ctrl[l.a]=c*(1-v)+l.tuck*v;});
    F.step(); if(!Number.isFinite(d.qpos[f]))return{nan:1,p0};
    const P_=F.position(),w=F.pose(F.ref.x,F.ref.y,0,0).p,e=Math.hypot(P_[0]-w[0],P_[1]-w[1],P_[2]-w[2]);
    if(t>P.T+0.15)late=Math.max(late,e); minUp=Math.min(minUp,1-2*(d.qpos[f+4]**2+d.qpos[f+5]**2)); zmax=Math.max(zmax,P_[2]);
    if(liftoff<0&&d.ncon===0)liftoff=t;
  }
  return{pitch0:+(p0*57.3).toFixed(0),late:+late.toFixed(2),minUp:+minUp.toFixed(2),zmax:+zmax.toFixed(2),liftoff:+liftoff.toFixed(3)};
}
// from real walking: walk, then the same takeoff from wherever the gait left the legs, at any heading and wing phase
function walkTo(speed,time,heading){m.opt.timestep=2e-4;stand();const h=heading/2;d.qpos[f+3]=Math.cos(h);d.qpos[f+4]=0;d.qpos[f+5]=0;d.qpos[f+6]=Math.sin(h);mujoco.mj_forward(m,d);Object.values(W.buf).forEach(b=>b.fill(0));W.nbuf=0;W.setRef(0,0,heading);W.yaw=0;W.speed=speed;for(let k=0;k<time/0.002;k++)W.step();endState={z:d.qpos[f+2],up:1-2*(d.qpos[f+4]**2+d.qpos[f+5]**2)};W.speed=0;for(let k=0;k<(settle/0.002);k++)W.step();settleState={z:d.qpos[f+2],up:1-2*(d.qpos[f+4]**2+d.qpos[f+5]**2)};m.opt.timestep=5e-5;}
let endState,settleState,settle=0;
function trial(P,o,dur=0.7){
  walkTo(o.speed,o.time,o.heading); for(const a of adh)d.ctrl[a]=0;
  const start=legs.map(l=>d.qpos[l.q]),pre=Math.round(P.pre/5e-5);
  for(let k=0;k<pre;k++){const u=smooth(k/pre/0.8);legs.forEach((l,i)=>{const j=l.joint==='coxa'?0:P.pamp[l.pair];d.ctrl[l.a]=start[i]*(1-u)+(l.q0+l.sign*j)*u;});mujoco.mj_step(m,d);}
  const set=legs.map(l=>d.ctrl[l.a]);
  const qw=d.qpos[f+3],qx=d.qpos[f+4],qy=d.qpos[f+5],qz=d.qpos[f+6];
  const th=Math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)),p0=Math.asin(Math.max(-1,Math.min(1,2*(qw*qy-qz*qx))));
  F.phase=o.phase;F.freq=fm.base_freq;F.buf.fill(0);F.nbuf=0;F.yaw=0;F.speed=0;F.pitch=p0;F.height=1;
  const X=d.qpos[f],Y=d.qpos[f+1],{p}=F.pose(0,0,th,0);F.ref={x:X-p[0],y:Y-p[1],heading:th};F.height=1-(p[2]-d.qpos[f+2]);const h0=F.height;
  let late=0,minUp=1,zmax=0;
  for(let k=0;k<dur/DT;k++){const t=k*DT,s=smooth(t/P.T);F.pitch=p0+(CRUISE-p0)*s;F.height=h0+P.rise*s;F.speed=14*smooth((t-P.T*.5)/P.T);
    const u=smooth((t-P.t0)/P.tp),v=smooth((t-P.t0-P.tp)/0.06);legs.forEach((l,i)=>{const j=l.joint==='coxa'?0:P.amp[l.pair];const c=set[i]+u*l.sign*j;d.ctrl[l.a]=c*(1-v)+l.tuck*v;});
    F.step();if(!Number.isFinite(d.qpos[f]))return{late:99,minUp:-1,p0:0};
    const Q=F.position(),w=F.pose(F.ref.x,F.ref.y,th,0).p,e=Math.hypot(Q[0]-w[0],Q[1]-w[1],Q[2]-w[2]);if(t>P.T+.15)late=Math.max(late,e);minUp=Math.min(minUp,1-2*(d.qpos[f+4]**2+d.qpos[f+5]**2));zmax=Math.max(zmax,Q[2]);}
  return{late,minUp,p0:p0*57.3,zmax};
}
const best={pre:0.0977,pamp:[0.8887,0.1825,-1.2234],T:0.0587,rise:0.3675,t0:0.0201,tp:0.00535,amp:[0,0.5462,0.6799]};
for(const S of [0,0.1,0.25,0.5]){settle=S;for(const sp of [0,0.5,1,1.5,2]){let ok=0,walkbad=0,bad=0,n=N,ps=[];for(let i=0;i<n;i++){const r=trial(best,{speed:sp,time:.3+rnd()*.8,heading:(rnd()-.5)*6.2,phase:rnd()});if(settleState.up<.8)walkbad++;if(r.late<.5&&r.minUp>-.2)ok++;ps.push(r.p0.toFixed(0));}console.log('settle',S,'speed',sp,`fly ${ok}/${n}`,'fell before takeoff',walkbad,'pitch',ps.join(','));}}
