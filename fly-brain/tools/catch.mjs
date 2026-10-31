// Can flybody's flight policy, unchanged, catch a fly that is in the air in its walking pose? Contacts off; the path asks for the
// pose to blend into cruise over T seconds. Decides whether a takeoff needs a new policy at all.
//   node tools/catch.mjs <full body dir>
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy } from '../walker.js';
import { Flyer, CONTROL_DT as DT } from '../flyer.js';
const dir=process.argv[2]; const mujoco=await loadMujoco(); const vfs=new mujoco.MjVFS();
for(const f of fs.readdirSync(`${dir}/meshes`)) vfs.addBuffer(f,new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m=mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`,'utf8'),vfs); const d=new mujoco.MjData(m);
const wasm=await WebAssembly.compile(fs.readFileSync('engine.wasm'));
const load=(p,shape)=>{const b=fs.readFileSync(p);const e=new WebAssembly.Instance(wasm).exports;return new WasmPolicy(e,new Float32Array(b.buffer,b.byteOffset,b.length/4),shape);};
const wm=JSON.parse(fs.readFileSync('policy/walk.json')), fm=JSON.parse(fs.readFileSync('policy/fly.json'));
const W=new Walker(mujoco,m,d,wm,load('policy/walk.bin')); const F=new Flyer(mujoco,m,d,fm,load('policy/fly.bin',[104,256,2,12]));
const smooth=x=>{const u=Math.min(1,Math.max(0,x));return u*u*(3-2*u);};
const CRUISE=-47.5*Math.PI/180;
const nm=j=>mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT.value,j);
const legs=[...Array(m.njnt).keys()].filter(j=>/coxa|femur|tibia|tarsus/.test(nm(j)||'')).map(j=>{const a=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR.value,nm(j));return{q:m.jnt_qposadr[j],a,t:m.qpos_spring[m.jnt_qposadr[j]]*0.6};}).filter(x=>x.a>=0);
m.opt.timestep=5e-5; m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT.value;
function run({T,wings,vz=0,phase=0,dur=0.8,rise=0.5,startPitch=null,startLegs=false}){
  mujoco.mj_resetData(m,d); W.setState(wm.qpos0);
  const f=F.freeq; d.qpos[f]=0;d.qpos[f+1]=0;d.qpos[f+2]=1; for(let i=0;i<F.wq.length;i++)d.qpos[F.wq[i]]=m.qpos_spring[F.wq[i]];
  d.qvel[F.freev+2]=vz; if(wings==='open'){const a=F.wing(phase);F.wq.forEach((q,i)=>d.qpos[q]=a[i]);}
  if(startPitch!==null){d.qpos[f+3]=Math.cos(startPitch/2);d.qpos[f+4]=0;d.qpos[f+5]=Math.sin(startPitch/2);d.qpos[f+6]=0;}
  if(startLegs){F.setState(fm.qpos0);d.qpos[f]=0;d.qpos[f+1]=0;d.qpos[f+2]=1;d.qpos[f+3]=Math.cos(startPitch/2);d.qpos[f+4]=0;d.qpos[f+5]=Math.sin(startPitch/2);d.qpos[f+6]=0;const a=F.wing(phase);F.wq.forEach((q,i)=>d.qpos[q]=a[i]);}
  mujoco.mj_forward(m,d);
  const p0=2*Math.atan2(d.qpos[f+5],d.qpos[f+3]);
  F.phase=phase;F.freq=fm.base_freq;F.buf.fill(0);F.nbuf=0;F.yaw=0;F.speed=0;F.pitch=p0;F.height=1;
  const {p}=F.pose(0,0,0,0); const off=[p[0]-d.qpos[f],p[1]-d.qpos[f+1],p[2]-d.qpos[f+2]];
  F.ref={x:-off[0],y:-off[1],heading:0}; F.height=1-off[2]; const h0=F.height, x0=F.ref.x;
  const leg0=legs.map(l=>d.qpos[l.q]);
  let late=0,up=1,zmin=9;
  for(let k=0;k<dur/DT;k++){const t=k*DT,s=smooth(t/T);
    F.pitch=p0+(CRUISE-p0)*s;F.height=h0+rise*s;F.speed=14*s;
    const ls=smooth(t/0.1);legs.forEach((l,i)=>{d.ctrl[l.a]=leg0[i]+(l.t-leg0[i])*ls;});
    F.step();
    if(!Number.isFinite(d.qpos[f]))return{nan:true};
    const P=F.position(),w=F.pose(F.ref.x,F.ref.y,0,0).p;const e=Math.hypot(P[0]-w[0],P[1]-w[1],P[2]-w[2]);
    if(t>T+0.1)late=Math.max(late,e); up=Math.min(up,1-2*(d.qpos[f+4]**2+d.qpos[f+5]**2)); zmin=Math.min(zmin,P[2]);
  }
  return{late:+late.toFixed(2),minUp:+up.toFixed(2),zmin:+zmin.toFixed(2)};
}
const deg=Math.PI/180;
console.log('control: flight qpos0, cruise pitch, rise 0, speed ramp',JSON.stringify(run({T:0.2,wings:'open',startPitch:CRUISE,startLegs:true,rise:0})));
for(const sl of [true,false])for(const a of [47.5,40,30,20,10,0])for(const T of [0.1,0.3])console.log(sl?'flight body':'walk body','pitch',a,'T',T,JSON.stringify(run({T,wings:'open',startPitch:-a*deg,startLegs:sl,rise:0})));
