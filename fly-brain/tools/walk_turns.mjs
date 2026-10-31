// Which turns can the walking policy take on this body, starting from a stand? Falls in 5 s by top speed, top turn rate and speed ramp.
import fs from 'node:fs'; import loadMujoco from '@mujoco/mujoco';
import { Walker, WasmPolicy, CONTROL_DT as DT } from '../walker.js';
const dir=process.argv[2];const mujoco=await loadMujoco();const vfs=new mujoco.MjVFS();for(const f of fs.readdirSync(`${dir}/meshes`))vfs.addBuffer(f,new Uint8Array(fs.readFileSync(`${dir}/meshes/${f}`)));
const m=mujoco.MjModel.from_xml_string(fs.readFileSync(`${dir}/fly.xml`,'utf8'),vfs),d=new mujoco.MjData(m);const wm=JSON.parse(fs.readFileSync('policy/walk.json'));
const b=fs.readFileSync('policy/walk.bin'),e=new WebAssembly.Instance(await WebAssembly.compile(fs.readFileSync('engine.wasm'))).exports,W=new Walker(mujoco,m,d,wm,new WasmPolicy(e,new Float32Array(b.buffer,b.byteOffset,b.length/4)));
const f=W.freeq,N=+(process.argv[3]||6);m.opt.timestep=2e-4;
const wq=[...Array(m.njnt).keys()].filter(j=>/wing_/.test(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT.value,j)||'')).map(j=>m.jnt_qposadr[j]);
function run(v,y,ramp,flip,tau){mujoco.mj_resetData(m,d);W.setState(wm.qpos0);for(const q of wq)d.qpos[q]=m.qpos_spring[q];mujoco.mj_forward(m,d);for(const k in W.buf)W.buf[k].fill(0);W.nbuf=0;W.setRef(0,0,0);let s=0,yy=0;
  for(let k=0;k<5/DT;k++){const t=k*DT;s=Math.min(v,s+ramp*DT);W.speed=s;const want=s>=.8?y*(flip&&Math.sin(t*flip)<0?-1:1):0;yy+=(want-yy)*(1-Math.exp(-DT/tau));W.yaw=yy;W.step();if(1-2*(d.qpos[f+4]**2+d.qpos[f+5]**2)<.3)return t;}return null;}
for(const [v,y,ramp,flip,tau] of [[2,0,2,0,.05],[2,0,99,0,.05],[1,0,2,0,.05],[1,4,2,0,.05],[1,2,2,0,.05],[1,1,2,0,.05],[2,4,2,0,.05],[2,2,2,0,.05],[1.5,3,2,0,.05],[2,4,2,6,.05],[2,2,2,6,.05],[2,2,2,6,.3],[1,2,2,6,.3],[1,4,2,6,.3]]){const r=[];for(let i=0;i<N;i++)r.push(run(v,y,ramp,flip,tau));const F=r.filter(x=>x!==null);console.log(`speed ${v} yaw ${y} ramp ${ramp} flip ${flip} tau ${tau}: fell ${F.length}/${N}`,F.map(x=>x.toFixed(1)).join(' '));}
