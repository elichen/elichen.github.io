import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const canvas = document.querySelector('#garden');
const button = document.querySelector('#harvest');
const message = document.querySelector('#message');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
try {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.setClearColor(0xf5efe3, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(33, 1, .1, 80);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, .8, 0);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 9; controls.maxDistance = 19;
  controls.minPolarAngle = .65; controls.maxPolarAngle = 1.35;
  controls.minAzimuthAngle = -.7; controls.maxAzimuthAngle = .7;
  scene.add(new THREE.HemisphereLight(0xfff7e8, 0x9ca88c, 1.9));
  const sun = new THREE.DirectionalLight(0xffedda, 2.8);
  sun.position.set(-3, 7, 5); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: .1, far: 20 });
  sun.shadow.normalBias = .025; sun.shadow.bias = -.0002;
  sun.shadow.radius = 3.5; scene.add(sun);
  const mat = (color, extra={}) => new THREE.MeshStandardMaterial({color, roughness:1,...extra});
  const cream=mat('#fffaf0'), orange=mat('#ed772e'), black=mat('#38352e'), leaf=mat('#6a893f'), leafLight=mat('#94a952'), soil=mat('#927052'), wood=mat('#e4c591'), pink=mat('#e7a49d'), yellow=mat('#e8bd4c');
  const sphere = new THREE.SphereGeometry(1, 32, 24);
  function mesh(geometry, material, parent, x=0,y=0,z=0) {
    const m=new THREE.Mesh(geometry,material);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;
  }
  function ell(parent,material,x,y,z,sx,sy,sz){const m=mesh(sphere,material,parent,x,y,z);m.scale.set(sx,sy,sz);return m;}
  function rod(parent,material,a,b,r=.025){const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b);const m=mesh(new THREE.CylinderGeometry(r,r,av.distanceTo(bv),12),material,parent);m.position.copy(av).add(bv).multiplyScalar(.5);m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),bv.sub(av).normalize());return m;}
  function softBox(w,h,d){
    const b=Math.min(.025,w/5,h/5,d/5);
    const shape=new THREE.Shape();
    shape.moveTo(-w/2+b,-h/2+b);shape.lineTo(w/2-b,-h/2+b);
    shape.lineTo(w/2-b,h/2-b);shape.lineTo(-w/2+b,h/2-b);shape.closePath();
    const geometry=new THREE.ExtrudeGeometry(shape,{depth:d-2*b,bevelEnabled:true,bevelThickness:b,bevelSize:b,bevelSegments:3,steps:1});
    geometry.translate(0,0,-d/2+b);return geometry;
  }
  const ground=mesh(new THREE.PlaneGeometry(200,200),mat('#eee5d5'),scene,0,-.42,0);ground.rotation.x=-Math.PI/2;
  // A soft, round patch of earth, like a small handmade model.
  mesh(new THREE.CylinderGeometry(3.25,3.05,.36,96),mat('#ccae83'),scene,0,-.2,0);
  mesh(new THREE.CylinderGeometry(3.25,3.25,.12,96),mat('#a5b779'),scene,0,.035,0);
  mesh(new THREE.BoxGeometry(2.55,.13,1.6),soil,scene,-.28,.13,1.08);
  for(const z of [.23,1.93])mesh(softBox(2.76,.24,.13),wood,scene,-.28,.19,z);
  for(const x of [-1.59,1.03])mesh(softBox(.13,.24,1.7),wood,scene,x,.19,1.08);
  for(let i=0;i<5;i++) { const x=-2.05+i*.98;mesh(softBox(.12,.8,.12),cream,scene,x,.46,-1.55);ell(scene,cream,x,.87,-1.55,.06,.06,.06); }
  for(const y of [.35,.69])mesh(softBox(4.1,.10,.10),cream,scene,-.09,y,-1.56);
  // Seeded placement keeps the garden the same on every visit.
  let seed=23;const rand=()=>{seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};
  const flowers=[];
  function flower(x,z,h,color){const g=new THREE.Group();g.position.set(x,.1,z);scene.add(g);rod(g,leaf,[0,0,0],[0,h,0],.014);const l=ell(g,leaf, .065,h*.43,0,.1,.035,.04);l.rotation.z=.45;const head=new THREE.Group();head.position.y=h;head.rotation.x=-.35;g.add(head);for(let j=0;j<5;j++){const a=j*Math.PI*2/5;ell(head,color,Math.sin(a)*.071,Math.cos(a)*.071,0,.06,.065,.028);}ell(head,yellow,0,0,.027,.04,.04,.022);flowers.push({g,phase:rand()*6});}
  for(let i=0;i<65;i++){const a=rand()*Math.PI*2,r=1.85+rand()*1.18;const x=Math.cos(a)*r,z=Math.sin(a)*r;if(z>.1&&Math.abs(x)<1.75)continue;flower(x,z,.19+rand()*.28,i%4===0?pink:cream);}
  for(let i=0;i<55;i++){const a=rand()*6.28,r=1.65+rand()*1.4;const x=Math.cos(a)*r,z=Math.sin(a)*r; if(z>.0&&Math.abs(x)<1.8)continue;const blade=ell(scene,leafLight,x,.16,z,.018,.10+rand()*.07,.025);blade.rotation.z=(rand()-.5)*.7;}
  // Quiet surface details, kept low to preserve the simple silhouette.
  const soilDetail=mat('#795a40');
  for(const z of [.63,1.43])rod(scene,soilDetail,[-1.35,.18,z],[.84,.18,z],.025);
  const crumbs=new THREE.InstancedMesh(new THREE.SphereGeometry(1,8,6),soilDetail,70);
  const crumbPose=new THREE.Object3D();
  for(let i=0;i<70;i++){
    crumbPose.position.set(-1.45+rand()*2.3,.199,.36+rand()*1.40);
    const r=.008+rand()*.013;crumbPose.scale.set(r,.005,r*.7);
    crumbPose.updateMatrix();crumbs.setMatrixAt(i,crumbPose.matrix);
  }
  crumbs.receiveShadow=true;scene.add(crumbs);
  ell(scene,soilDetail,-.4,.197,.65,.115,.007,.095);
  function carrot(parent,x,y,z,scale=1){const g=new THREE.Group();g.position.set(x,y,z);g.scale.setScalar(scale);parent.add(g);const root=mesh(new THREE.ConeGeometry(.095,.46,20),orange,g,0,-.17,0);root.rotation.z=Math.PI;for(let i=0;i<5;i++){const a=i*2.4;const l=ell(g,i%2?leaf:leafLight,Math.sin(a)*.075,.14,Math.cos(a)*.065,.033,.18,.025);l.rotation.z=-Math.sin(a)*.48;l.rotation.x=Math.cos(a)*.4;}return g;}
  for(const x of [-1.12,-.48,.19,.72])for(const z of [.63,1.43]){if(x===-.48&&z===.63)continue;carrot(scene,x,.19,z,.75);}
  const pulled=carrot(scene,-.4,.19,.65,1.1);
  // Proportions checked against the official Medicom UDF and classic figure.
  const miffyWhite=mat('#ffffff');
  const dress=mat('#f45b00');
  const miffy=new THREE.Group();miffy.position.set(-.4,.12,-.17);scene.add(miffy);
  ell(miffy,miffyWhite,-.22,.145,.15,.215,.145,.285);
  ell(miffy,miffyWhite,.22,.145,.15,.215,.145,.285);
  const upper=new THREE.Group();upper.position.y=.22;miffy.add(upper);
  mesh(new THREE.CylinderGeometry(.28,.45,.49,48),dress,upper,0,.30,0);
  ell(upper,dress,0,.52,0,.285,.12,.26);
  ell(upper,miffyWhite,0,.43,.326,.033,.033,.015);
  const head=new THREE.Group();head.position.set(0,.87,0);upper.add(head);
  // A round forehead, broad cheeks and a gently flattened chin.
  const outline=new THREE.CatmullRomCurve3([
    [0,-.35,0],[.27,-.34,0],[.43,-.29,0],[.53,-.17,0],
    [.555,0,0],[.51,.17,0],[.38,.32,0],[.20,.395,0],[0,.42,0]
  ].map(p=>new THREE.Vector3(...p)));
  const profile=outline.getPoints(64).map(p=>new THREE.Vector2(Math.max(0,p.x),p.y));
  const face=mesh(new THREE.LatheGeometry(profile,64),miffyWhite,head);
  face.scale.z=.72;
  for(const sign of [-1,1]){
    const ear=ell(head,miffyWhite,sign*.245,.60,-.035,.18,.43,.145);
    ear.rotation.z=-sign*.07;
  }
  // Place the graphic features on the surface, including when viewed obliquely.
  function faceZ(x,y){
    let r=.55;
    for(let i=1;i<profile.length;i++)if(profile[i].y>=y){
      const a=profile[i-1],b=profile[i];r=THREE.MathUtils.lerp(a.x,b.x,(y-a.y)/(b.y-a.y));break;
    }
    return .72*Math.sqrt(Math.max(0,r*r-x*x))+.006;
  }
  for(const x of [-.235,.235])ell(head,black,x,-.025,faceZ(x,-.025),.023,.032,.012);
  for(const sign of [-1,1]){
    const points=[];
    for(let i=0;i<=12;i++){
      const x=-.067+i*.134/12,y=-.225+sign*x*.40;
      points.push(new THREE.Vector3(x,y,faceZ(x,y)));
    }
    mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),16,.010,8,false),black,head);
  }
  const arms=[];
  for(const sign of [-1,1]){const pivot=new THREE.Group();upper.add(pivot);const sleeve=mesh(new THREE.CylinderGeometry(.12,.135,1,24),dress,pivot);const hand=ell(upper,miffyWhite,0,0,0,.115,.115,.115);arms.push({sign,pivot,sleeve,hand});}
  function poseArms(tug,lift){for(const a of arms){const shoulder=new THREE.Vector3(a.sign*.285,.46,.015);const hand=new THREE.Vector3(a.sign*(.40-.29*tug),.17-.24*tug+lift*.67,.08+.57*tug);a.hand.position.copy(hand);a.pivot.position.copy(shoulder).add(hand).multiplyScalar(.5);a.pivot.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),hand.clone().sub(shoulder).normalize());a.sleeve.scale.y=shoulder.distanceTo(hand);}}
  // Woven basket, with a little handle and a visible harvest.
  const basket=new THREE.Group();basket.position.set(1.65,.12,.9);scene.add(basket);
  const wicker=mat('#bb8b52');mesh(new THREE.CylinderGeometry(.43,.31,.44,40,1,true),wicker,basket,0,.22,0);
  mesh(new THREE.CylinderGeometry(.32,.32,.035,40),wicker,basket,0,.035,0);
  for(let i=0;i<6;i++){const r=.32+i*.021;const ring=mesh(new THREE.TorusGeometry(r,.023,8,48),wood,basket,0,.035+i*.079,0);ring.rotation.x=Math.PI/2;}
  for(let i=0;i<24;i++){const a=i*Math.PI/12;rod(basket,wood,[Math.sin(a)*.32,.02,Math.cos(a)*.32],[Math.sin(a)*.425,.44,Math.cos(a)*.425],.014);}
  const handle=mesh(new THREE.TorusGeometry(.39,.028,10,40,Math.PI),wicker,basket,0,.42,0);handle.rotation.y=Math.PI/2;
  const stored=[];for(let i=0;i<8;i++){const c=carrot(basket,(i%3-1)*.14,.42+Math.floor(i/3)*.07,(Math.floor(i/3)-1)*.12,.8);c.rotation.z=(i%2?1:-1)*.5;c.visible=false;stored.push(c);}
  // One gentle interaction. No score, timer, or complicated controls.
  let elapsed=0,start=null,growthStart=null,count=0;const duration=reducedMotion?1.8:4.8;
  const growthDuration=reducedMotion?.45:1.8;
  const smooth=(a,b,t)=>{const x=THREE.MathUtils.clamp((t-a)/(b-a),0,1);return x*x*(3-2*x);};
  function harvest(){if(start!==null||growthStart!==null)return;start=elapsed;button.disabled=true;button.querySelector('span').textContent='A little tug…';message.textContent='Steady… steady…';}
  button.addEventListener('click',harvest);
  function resize(){const w=innerWidth,h=innerHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();const d=Math.max(12.3,6.9/(2*Math.tan(THREE.MathUtils.degToRad(16.5))*camera.aspect)/1.1);controls.minDistance=d*.8;controls.maxDistance=d*1.55;camera.position.set(2.0,d*.48,d);controls.update();}
  window.addEventListener('resize',resize);resize();poseArms(0,0);button.disabled=false;
  const clock=new THREE.Clock();
  renderer.setAnimationLoop(()=>{
    const dt=clock.getDelta();elapsed+=dt;
    let bend=0,lift=0,tug=0;
    if(start!==null){const t=(elapsed-start)/duration;
      tug=smooth(0,.22,t)*(1-smooth(.66,.85,t));
      lift=smooth(.35,.48,t)*(1-smooth(.66,.85,t));
      bend=.32*smooth(0,.22,t)*(1-smooth(.34,.50,t));
      if(t>.22&&t<.35&&!reducedMotion)bend+=Math.sin(t*120)*.025;
      upper.rotation.x=bend;
      miffy.position.y=.12+Math.sin(smooth(.35,.53,t)*Math.PI)*.1;
      poseArms(tug,lift);
      pulled.position.set(-.4,.19+smooth(.35,.49,t)*1.00,.65);
      const fly=smooth(.57,.80,t);
      if(fly>0){pulled.position.x=THREE.MathUtils.lerp(-.4,1.65,fly);pulled.position.y=1.19+(Math.sin(fly*Math.PI)*.62)-fly*.59;pulled.position.z=THREE.MathUtils.lerp(.65,.9,fly);pulled.rotation.z=-fly*.7;}
      pulled.visible=t<.80;
      if(t>=.80){stored[count%8].visible=true;message.textContent='A lovely little carrot.';}
      if(t>=1){
        count++;start=null;growthStart=elapsed;
        pulled.visible=true;pulled.position.set(-.4,.19,.65);pulled.rotation.z=0;pulled.scale.setScalar(.001);
        upper.rotation.x=0;miffy.position.y=.12;poseArms(0,0);
        button.querySelector('span').textContent='A little patience…';message.textContent='And a new little carrot grows.';
      }
    }else if(!reducedMotion){upper.position.y=.22+Math.sin(elapsed*1.7)*.012;head.rotation.z=Math.sin(elapsed*.65)*.025;}
    if(growthStart!==null){
      const progress=(elapsed-growthStart)/growthDuration;
      const growth=smooth(0,1,progress);
      pulled.scale.set(1.1*(.2+.8*growth),Math.max(.001,1.1*growth),1.1*(.2+.8*growth));
      if(progress>=1){
        growthStart=null;pulled.scale.setScalar(1.1);button.disabled=false;
        button.querySelector('span').textContent='Pull another carrot';
        message.textContent=count===1?'One carrot. A small, happy moment.':`${count} carrots picked. A little more sunshine?`;
      }
    }
    if(!reducedMotion)for(const {g,phase}of flowers)g.rotation.z=Math.sin(elapsed*1.15+phase)*.045;
    controls.update();renderer.render(scene,camera);
  });
  // Let the scene settle before Miffy shows the first little harvest.
  if(!reducedMotion)setTimeout(()=>{if(count===0&&start===null)harvest();},1400);
  document.addEventListener('visibilitychange',()=>{clock.getDelta();});
} catch(error) {console.error(error);document.querySelector('#error').hidden=false;button.disabled=true;message.textContent='The garden couldn’t open.';}
