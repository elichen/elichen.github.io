// One body: do flybody's walking and flight policies both work on the full fly (legs, wings, mouth all free)?
//   node tools/one_body.mjs <full body dir> 
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy, CONTROL_DT as WALK_DT } from '../walker.js';
import { Flyer, CONTROL_DT as FLY_DT } from '../flyer.js';
const dir=process.argv[2]; const mujoco=await loadMujoco(); const vfs=new mujoco.MjVFS();
for(const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f,new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m=mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`,'utf8'),vfs); const d=new mujoco.MjData(m);
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const load=(p,shape)=>{const b=fs.readFileSync(p);const e=new WebAssembly.Instance(wasm).exports;return new WasmPolicy(e,new Float32Array(b.buffer,b.byteOffset,b.length/4),shape);};
const wm=JSON.parse(fs.readFileSync('policy/walk.json')), fm=JSON.parse(fs.readFileSync('policy/fly.json'));
const W=new Walker(mujoco,m,d,wm,load('policy/walk.bin')); const F=new Flyer(mujoco,m,d,fm,load('policy/fly.bin',[104,256,2,12]));
const name=(t,i)=>mujoco.mj_id2name(m,mujoco.mjtObj[t].value,i);
const joints=[...Array(m.njnt).keys()].map(j=>({j,n:name('mjOBJ_JOINT',j),q:m.jnt_qposadr[j]}));
const wings=joints.filter(x=>/wing_/.test(x.n)), legs=joints.filter(x=>/coxa|femur|tibia|tarsus/.test(x.n));
console.log('nq',m.nq,'nv',m.nv,'nu',m.nu,'na',m.na,'legs',legs.length,'wings',wings.length);
const act=n=>mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR.value,n);
const LEG=0.6;
const holdLegs=()=>{for(const x of legs){const a=act(x.n); const t=m.qpos_spring[x.q]*LEG; d.qpos[x.q]=t; if(a>=0)d.ctrl[a]=t; const ad=m.actuator_actadr[a]; if(a>=0&&ad>=0)d.act[ad]=t;}};
// flight
m.opt.timestep=5e-5; m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT.value;
for(const [sp,y,T] of [[14,0,1],[14,'sac',1.5],[0,0,1]]){F.speed=sp;F.yaw=0;F.start(); holdLegs(); mujoco.mj_forward(m,d); let e=0,z=9; const sd=.04/2.355,t0=performance.now(); for(let k=0;k<T/FLY_DT;k++){const t=k*FLY_DT; F.yaw=y==='sac'?43*Math.exp(-((t-.7)**2)/(2*sd*sd)):0; F.step(); const p=F.position(),{p:w}=F.pose(F.ref.x,F.ref.y,F.ref.heading,F.yaw); e=Math.max(e,Math.hypot(p[0]-w[0],p[1]-w[1],p[2]-w[2])); z=Math.min(z,p[2]);} console.log(`fly speed ${sp} ${y}: err max ${e.toFixed(3)} cm, lowest ${z.toFixed(2)}, ${(T/((performance.now()-t0)/1000)).toFixed(2)}x`);}
// walk
m.opt.disableflags&=~mujoco.mjtDisableBit.mjDSBL_CONTACT.value;
for(const dt of [2e-4,5e-5]){ m.opt.timestep=dt; for(const [sp,y] of [[2,0],[2,4],[0,0]]){ mujoco.mj_resetData(m,d); W.setState(wm.qpos0); for(const x of wings)d.qpos[x.q]=m.qpos_spring[x.q]; mujoco.mj_forward(m,d); W.buf&&Object.values(W.buf).forEach(b=>b.fill(0)); W.nbuf=0; W.setRef(0,0,0); W.speed=sp;W.yaw=y; let e=0,z=9,zx=0,wmax=0; const sub=Math.round(2e-4/dt); const t0=performance.now();
  for(let k=0;k<1/WALK_DT;k++){ if(sub===1) W.step(); else { /* same control, finer physics */ W.writeObs(); W.setAction(W.policy.run(W.obs)); Object.values(W.buf).forEach(b=>b.fill(0)); W.nbuf=0; for(let i=0;i<10*sub;i++){mujoco.mj_step(m,d); if(i%sub===sub-1)W.readSensors();} const r=W.ref; r.x+=W.speed*WALK_DT*Math.cos(r.heading); r.y+=W.speed*WALK_DT*Math.sin(r.heading); r.heading+=W.yaw*WALK_DT; }
   const f=W.freeq; e=Math.max(e,Math.hypot(d.qpos[f]-W.ref.x,d.qpos[f+1]-W.ref.y)); z=Math.min(z,d.qpos[f+2]); zx=Math.max(zx,d.qpos[f+2]); for(const x of wings)wmax=Math.max(wmax,Math.abs(d.qpos[x.q]-m.qpos_spring[x.q])); if(!Number.isFinite(d.qpos[f])){console.log('NaN');break;}}
  console.log(`walk dt ${dt} speed ${sp} yaw ${y}: err max ${e.toFixed(3)} cm, z ${z.toFixed(3)}-${zx.toFixed(3)}, wing from fold ${(wmax*57.3).toFixed(1)} deg, ${(1/((performance.now()-t0)/1000)).toFixed(2)}x`);}}
