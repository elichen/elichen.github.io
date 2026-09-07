import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { makeDoraemon, material } from "./model.js";
import { makeWorld } from "./world.js";

const $ = (id) => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = THREE.MathUtils.clamp;
const damp = (a, b, rate, dt) =>
  THREE.MathUtils.lerp(a, b, 1 - Math.exp(-rate * dt));
let renderer, scene, camera, controls, doraemon, world, lights;
let autopilot = false;
let cameraStyle = "cinematic",
  cinemaTime = 0,
  cameraArc = 0,
  cameraHeading = Math.PI,
  lastFrontCycle = -1;
const mouseFlight = {
  active: false,
  id: null,
  originX: 0,
  originY: 0,
  x: 0,
  y: 0,
};
let mode = "orbit",
  mood = "day",
  time = 0,
  last = performance.now(),
  interacted = false;
let speed = 0,
  targetSpeed = 5,
  heading = Math.PI,
  turnVelocity = 0,
  verticalVelocity = 0,
  flightTime = 0;
let ringIndex = 0,
  flightStarted = false,
  toastTimeout,
  photoBusy = false;
let frameCount = 0,
  frameSeconds = 0,
  qualityAdjusted = false;
const gaze = { x: 0, y: 0 };
const keys = new Set();
const touch = { x: 0, y: 0, rise: 0 };
const position = new THREE.Vector3(2, 6, 0),
  cameraOffset = new THREE.Vector3(),
  target = new THREE.Vector3(),
  desiredCamera = new THREE.Vector3(),
  temp = new THREE.Vector3(),
  projected = new THREE.Vector3();
const ringPositions = [
  [0, 6.8, 10],
  [-7, 8, -7],
  [-21, 10, -20],
  [-37, 13, -9],
  [-34, 16, 13],
  [-18, 12, 31],
  [5, 10, 33],
  [17, 8, 13],
].map((p) => new THREE.Vector3(...p));
const rings = [],
  bursts = [];
let trail;

function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => $("toast").classList.remove("visible"), 3600);
}
function setPressed(el, pressed) {
  el.classList.toggle("active", pressed);
  el.setAttribute("aria-pressed", String(pressed));
}
function frameOrbit() {
  const narrow = innerWidth < 701;
  camera.setViewOffset(
    innerWidth,
    innerHeight,
    narrow ? 0 : -innerWidth * 0.18,
    narrow
      ? -innerHeight * (innerHeight < 700 ? 0.12 : 0.03)
      : innerHeight * 0.07,
    innerWidth,
    innerHeight,
  );
}
function frameFlight() {
  camera.clearViewOffset();
  if (innerWidth < 701)
    camera.setViewOffset(
      innerWidth,
      innerHeight,
      0,
      -innerHeight * (innerHeight < 700 ? 0.16 : 0.1),
      innerWidth,
      innerHeight,
    );
}
function homeCamera() {
  const narrow = innerWidth < 701;
  target.set(2, 7.3, 0);
  desiredCamera.set(10, narrow ? 12.9 : 11.2, narrow ? 23.6 : 19.8);
  if (narrow) {
    camera.fov = 43;
    camera.updateProjectionMatrix();
  } else {
    camera.fov = 38;
    camera.updateProjectionMatrix();
  }
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  controls.update();
  controls.target.copy(target);
  camera.position.copy(desiredCamera);
  frameOrbit();
  controls.update();
  controls.enableDamping = damping;
}
function setMode(next) {
  if (mode === next) return;
  keys.clear();
  touch.x = touch.y = touch.rise = 0;
  releaseJoystick();
  releaseMouse();
  mode = next;
  document.body.classList.toggle("flying", mode === "fly");
  $("flight-hud").hidden = mode !== "fly";
  $("gadget-label").hidden = mode !== "orbit";
  $("touch-flight").hidden = mode !== "fly";
  $("ring-guide").hidden = mode !== "fly" || ringIndex >= 8;
  setPressed($("orbit-mode"), mode === "orbit");
  setPressed($("fly-mode"), mode === "fly");
  controls.enabled = mode === "orbit";
  interacted = false;
  if (mode === "fly") {
    if (!flightStarted) resetFlight();
    $("world").focus({ preventScroll: true });
    camera.fov = 49;
    camera.updateProjectionMatrix();
    frameFlight();
    $("status-label").innerHTML = "<i></i> EXPLORING THE BLUE";
    toast(
      matchMedia("(pointer: coarse)").matches
        ? "Off we go! Joystick to steer. Arrows to rise & descend."
        : "Hold & drag to steer and climb. Scroll to change speed.",
    );
  } else {
    doraemon.root.position.set(2, 6, 0);
    doraemon.root.rotation.set(0, 0, 0);
    homeCamera();
    $("status-label").innerHTML = "<i></i> FREE TO WANDER";
  }
  rings.forEach((r, i) => (r.visible = mode === "fly" && i >= ringIndex));
  if (mode === "orbit") {
    clearTimeout(toastTimeout);
    $("toast").classList.remove("visible");
  }
}
function resetFlight() {
  flightStarted = true;
  position.set(0, 5.5, 23);
  heading = Math.PI;
  speed = 0;
  targetSpeed = 5;
  turnVelocity = verticalVelocity = 0;
  ringIndex = 0;
  flightTime = 0;
  cinemaTime = 0;
  cameraArc = 0;
  cameraHeading = Math.PI;
  lastFrontCycle = -1;
  releaseMouse();
  releaseJoystick();
  keys.clear();
  touch.rise = 0;
  rings.forEach((r) => {
    r.visible = mode === "fly";
    r.scale.setScalar(1);
    r.userData.passed = false;
  });
  $("rings-value").innerHTML = "00<span>/08</span>";
  $("course-progress").style.width = "0%";
  $("ring-guide").hidden = mode !== "fly";
  doraemon.root.position.copy(position);
  doraemon.root.rotation.set(0, heading, 0);
  camera.position.set(0, 10.5, 37);
  controls.target.copy(position).add(new THREE.Vector3(0, 2, 0));
  resetTrail();
}
function setMood(next) {
  mood = next;
  world.setMood(next);
  document.body.classList.toggle("night", next === "night");
  document
    .querySelectorAll("[data-sky]")
    .forEach((b) => setPressed(b, b.dataset.sky === next));
  try {
    localStorage.setItem("pocket-skies-mood", next);
  } catch {}
}

// Synthesized rotor, wind, and soft pentatonic chimes. No audio downloads.
const sound = {
  context: null,
  enabled: false,
  master: null,
  rotor: null,
  wind: null,
  pulse: null,
  nextNote: 0,
  async toggle() {
    try {
      if (!this.context) this.init();
      await this.context.resume();
      this.enabled = !this.enabled;
      this.master.gain.setTargetAtTime(
        this.enabled ? 0.12 : 0,
        this.context.currentTime,
        0.35,
      );
      $("sound").setAttribute("aria-pressed", String(this.enabled));
      $("sound").setAttribute(
        "aria-label",
        this.enabled ? "Turn sound off" : "Turn sound on",
      );
      $("sound").title = this.enabled ? "Turn sound off" : "Turn sound on";
      $("sound-waves").setAttribute(
        "d",
        this.enabled
          ? "M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"
          : "m16 9 5 6m0-6-5 6",
      );
    } catch {
      toast("Sound isn’t available in this browser.");
    }
  },
  init() {
    const C = window.AudioContext || window.webkitAudioContext;
    if (!C) throw new Error("No audio");
    this.context = new C();
    const c = this.context;
    this.master = c.createGain();
    this.master.gain.value = 0;
    this.master.connect(c.destination);
    const buffer = c.createBuffer(1, c.sampleRate * 2, c.sampleRate);
    const data = buffer.getChannelData(0);
    let smooth = 0;
    for (let i = 0; i < data.length; i++) {
      smooth = (smooth + Math.random() * 0.12 - 0.06) * 0.98;
      data[i] = smooth;
    }
    const noise = c.createBufferSource();
    noise.buffer = buffer;
    noise.loop = true;
    const filter = c.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 520;
    noise.connect(filter);
    this.wind = c.createGain();
    this.wind.gain.value = 0.55;
    filter.connect(this.wind).connect(this.master);
    noise.start();
    this.rotor = c.createOscillator();
    this.rotor.type = "sine";
    this.rotor.frequency.value = 83;
    const rotorGain = c.createGain();
    rotorGain.gain.value = 0.09;
    this.rotor.connect(rotorGain).connect(this.master);
    this.pulse = c.createOscillator();
    this.pulse.frequency.value = 18;
    const pulseGain = c.createGain();
    pulseGain.gain.value = 0.055;
    this.pulse.connect(pulseGain).connect(rotorGain.gain);
    this.rotor.start();
    this.pulse.start();
  },
  chime(index = 0) {
    if (!this.enabled) return;
    const c = this.context;
    const notes = [
      523.25, 587.33, 659.25, 783.99, 880, 1046.5, 1174.66, 1318.51,
    ];
    [0, 7].forEach((interval, i) => {
      const o = c.createOscillator(),
        g = c.createGain();
      o.type = "sine";
      o.frequency.value = notes[index % 8] * (i ? 1.5 : 1);
      g.gain.setValueAtTime(0, c.currentTime);
      g.gain.linearRampToValueAtTime(i ? 0.15 : 0.3, c.currentTime + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + 1.9);
      o.connect(g).connect(this.master);
      o.start(c.currentTime + i * 0.08);
      o.stop(c.currentTime + 2.1);
    });
  },
  update() {
    if (!this.context || !this.enabled) return;
    const t = this.context.currentTime;
    this.rotor.frequency.setTargetAtTime(78 + speed * 2, t, 0.2);
    this.pulse.frequency.setTargetAtTime(17 + speed * 0.7, t, 0.2);
    this.wind.gain.setTargetAtTime(
      mode === "fly" ? 0.5 + speed * 0.045 : 0.35,
      t,
      0.3,
    );
    if (t > this.nextNote && mode === "orbit") {
      this.nextNote = t + 7 + Math.random() * 6;
      this.chime(Math.floor(Math.random() * 5));
    }
  },
};

function updateAutopilot() {
  $("autopilot").setAttribute("aria-pressed", String(autopilot));
  $("auto-camera-controls").hidden = !autopilot;
  document.body.classList.toggle("autopiloting", autopilot);
  $("autopilot").classList.toggle("enabled", autopilot);
  $("autopilot").innerHTML =
    "<span>" +
    (autopilot ? "✧" : "⌁") +
    "</span> " +
    (autopilot ? "Guided flight on" : "Let Doraemon guide") +
    "<small>" +
    (autopilot ? "Take the controls anytime" : "Sit back & follow the breeze") +
    "</small>";
}

function createRings() {
  const ringGeometry = new THREE.TorusGeometry(2.3, 0.078, 12, 80);
  const ringMat = material("#ffcb60", {
    metalness: 0.58,
    roughness: 0.25,
    emissive: "#e6a831",
    emissiveIntensity: 0.23,
  });
  const innerMat = new THREE.MeshBasicMaterial({
    color: "#ffedb4",
    transparent: true,
    opacity: 0.25,
    depthWrite: false,
  });
  ringPositions.forEach((p, i) => {
    const g = new THREE.Group();
    g.position.copy(p);
    const previous = i ? ringPositions[i - 1] : new THREE.Vector3(0, 6.8, 23);
    temp.copy(p).sub(previous);
    g.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      temp.normalize(),
    );
    g.add(new THREE.Mesh(ringGeometry, ringMat));
    const halo = new THREE.Mesh(
      new THREE.TorusGeometry(2.3, 0.19, 8, 80),
      innerMat,
    );
    g.add(halo);
    for (let j = 0; j < 4; j++) {
      const a = (j * Math.PI) / 2;
      const bead = new THREE.Mesh(new THREE.OctahedronGeometry(0.13), ringMat);
      bead.position.set(Math.cos(a) * 2.3, Math.sin(a) * 2.3, 0);
      g.add(bead);
    }
    g.visible = false;
    scene.add(g);
    rings.push(g);
  });
  const trailGeo = new THREE.BufferGeometry();
  const trailPositions = new Float32Array(90 * 3);
  trailGeo.setAttribute(
    "position",
    new THREE.BufferAttribute(trailPositions, 3),
  );
  trail = new THREE.Points(
    trailGeo,
    new THREE.PointsMaterial({
      color: "#fff4d4",
      size: 0.075,
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
    }),
  );
  trail.frustumCulled = false;
  scene.add(trail);
  resetTrail();
}
function resetTrail() {
  if (!trail) return;
  const p = trail.geometry.attributes.position;
  for (let i = 0; i < p.count; i++)
    p.setXYZ(i, position.x, position.y, position.z);
  p.needsUpdate = true;
}
function celebrate(p) {
  const count = 60,
    coords = new Float32Array(count * 3),
    velocities = [];
  for (let i = 0; i < count; i++) {
    coords.set([p.x, p.y, p.z], i * 3);
    velocities.push(
      new THREE.Vector3(
        (Math.random() - 0.5) * 6,
        (Math.random() - 0.3) * 5,
        (Math.random() - 0.5) * 6,
      ),
    );
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(coords, 3));
  const particles = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      color: "#ffe3a0",
      size: 0.13,
      transparent: true,
      opacity: 1,
      depthWrite: false,
    }),
  );
  scene.add(particles);
  bursts.push({ mesh: particles, velocities, life: 1.8 });
}
function collectRing() {
  const index = ringIndex;
  const ring = rings[index];
  ring.userData.passed = true;
  celebrate(ring.position);
  sound.chime(index);
  doraemon.wink();
  ringIndex++;
  $("rings-value").innerHTML =
    String(ringIndex).padStart(2, "0") + "<span>/08</span>";
  $("course-progress").style.width = `${(ringIndex / 8) * 100}%`;
  const lines = [
    "A little lift. A lot of possibility.",
    "Looking good, sky explorer.",
    "The world is smaller from up here.",
    "Halfway to a little magic.",
    "Cloud nine looks good on you.",
    "Just you, a friend, and the sky.",
    "One more little adventure…",
  ];
  if (ringIndex < 8) toast(lines[index]);
  else {
    $("ring-guide").hidden = true;
    toast("All eight! You’re officially a sky explorer.");
    setTimeout(() => {
      if (mode === "fly" && ringIndex === 8) {
        keys.clear();
        releaseMouse();
        releaseJoystick();
        touch.rise = 0;
        $("complete").showModal();
      }
    }, 1000);
  }
}
function updateFlight(dt) {
  flightTime += dt;
  let steering =
    (keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0) -
    (keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0) -
    touch.x -
    mouseFlight.x;
  const throttle =
    (keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0) -
    (keys.has("KeyS") || keys.has("ArrowDown") ? 1 : 0) -
    touch.y;
  let lift =
    (keys.has("Space") ? 1 : 0) -
    (keys.has("ShiftLeft") || keys.has("ShiftRight") ? 1 : 0) +
    touch.rise -
    mouseFlight.y;
  if (autopilot && ringIndex < 8) {
    const goal = ringPositions[ringIndex],
      dx = goal.x - position.x,
      dz = goal.z - position.z;
    const desiredHeading = Math.atan2(dx, dz),
      error = Math.atan2(
        Math.sin(desiredHeading - heading),
        Math.cos(desiredHeading - heading),
      );
    steering = clamp(error * 2.3, -1, 1);
    lift = clamp((goal.y - 1.3 - position.y) * 1.1, -1, 1);
    targetSpeed = clamp(8 - Math.abs(error) * 5, 2.5, 8);
  } else targetSpeed = clamp(targetSpeed + throttle * dt * 6, 0, 13);
  steering = clamp(steering, -1, 1);
  lift = clamp(lift, -1, 1);
  speed = damp(speed, targetSpeed, 2, dt);
  turnVelocity = damp(turnVelocity, steering * 1.05, 4, dt);
  heading += turnVelocity * dt;
  verticalVelocity = damp(verticalVelocity, lift * 5, 3, dt);
  position.x += Math.sin(heading) * speed * dt;
  position.z += Math.cos(heading) * speed * dt;
  position.y += verticalVelocity * dt;
  // Soft floor keeps flight above the buildings and trees; a gentle boundary turns us home.
  const minimum = world.flightFloor(position.x, position.z);
  position.y = clamp(position.y, minimum, 45);
  if (Math.hypot(position.x, position.z) > 150) {
    const home = Math.atan2(-position.x, -position.z);
    heading +=
      Math.atan2(Math.sin(home - heading), Math.cos(home - heading)) * dt;
    if (flightTime % 4 < dt)
      toast("The breeze is bringing us back toward the island.");
  }
  doraemon.root.position.copy(position);
  doraemon.root.position.y += Math.sin(time * 2) * 0.055;
  doraemon.root.rotation.set(0, heading, -turnVelocity * 0.23, "YXZ");
  updateFlightCamera(dt);
  const center = temp.copy(position);
  center.y += 1.3;
  if (ringIndex < 8 && center.distanceTo(ringPositions[ringIndex]) < 2.55)
    collectRing();
  $("altitude-value").innerHTML =
    Math.round((position.y + 5.3) * 3) + "<span>m</span>";
  $("speed-value").innerHTML = Math.round(speed * 10.8) + "<span>km/h</span>";
  $("flight-speed").value = targetSpeed;
  $("throttle-output").value = Math.round(targetSpeed * 10.8) + " km/h";
  $("flight-speed").setAttribute(
    "aria-valuetext",
    Math.round(targetSpeed * 10.8) + " kilometers per hour",
  );
  if (ringIndex < 8) {
    projected.copy(ringPositions[ringIndex]);
    projected.project(camera);
    const inFront = projected.z < 1 && projected.z > -1;
    let x = projected.x,
      y = projected.y;
    if (!inFront) {
      x = -x;
      y = 0;
    }
    const offscreen = !inFront || Math.abs(x) > 0.84 || Math.abs(y) > 0.7;
    const narrow = innerWidth < 701;
    const minY = narrow ? Math.min(innerHeight * 0.48, 320) : 110;
    const maxY = narrow ? Math.max(minY, innerHeight - 300) : innerHeight - 160;
    const px = clamp((x * 0.5 + 0.5) * innerWidth, 85, innerWidth - 85);
    const py = clamp((-y * 0.5 + 0.5) * innerHeight, minY, maxY);
    $("ring-guide").style.left = `${px}px`;
    $("ring-guide").style.top = `${py}px`;
    let arrow = "◇",
      hint = "";
    if (offscreen) {
      if (inFront && Math.abs(y) > 0.7 && Math.abs(x) < 0.84) {
        arrow = y > 0 ? "↑" : "↓";
        hint = y > 0 ? "RISE" : "DESCEND";
      } else {
        arrow = x < 0 ? "‹" : "›";
        hint = x < 0 ? "TURN LEFT" : "TURN RIGHT";
      }
    }
    $("ring-guide").firstElementChild.textContent = arrow;
    $("ring-distance").textContent =
      `${Math.round(center.distanceTo(ringPositions[ringIndex]) * 3)} m${hint ? " · " + hint : ""}`;
  }
  const p = trail.geometry.attributes.position;
  p.array.copyWithin(3, 0, p.array.length - 3);
  doraemon.tailAnchor.getWorldPosition(projected);
  p.setXYZ(0, projected.x, projected.y, projected.z);
  p.needsUpdate = true;
}
// Spherical camera passes keep their distance from Doraemon. Interpolating
// the orbit angle avoids a straight-line transition through his head.
function updateFlightCamera(dt) {
  const scenic = autopilot && cameraStyle === "cinematic";
  let arcTarget = 0;
  if (scenic && reduced) arcTarget = Math.PI * 0.82;
  else if (scenic) {
    cinemaTime += dt;
    const cycle = Math.floor(cinemaTime / 16),
      phase = cinemaTime % 16;
    const ease = (t) => t * t * (3 - 2 * t);
    let amount = 0;
    if (phase >= 2 && phase < 5) amount = ease((phase - 2) / 3);
    else if (phase >= 5 && phase < 9) amount = 1;
    else if (phase >= 9 && phase < 12) amount = 1 - ease((phase - 9) / 3);
    arcTarget = (cycle % 2 === 0 ? 1 : -1) * Math.PI * 0.82 * amount;
    if (phase >= 5 && phase < 9 && lastFrontCycle !== cycle) {
      lastFrontCycle = cycle;
      doraemon.greet();
    }
  }
  cameraArc = damp(cameraArc, arcTarget, 3.5, dt);
  $("ring-guide").hidden =
    ringIndex >= 8 || (autopilot && Math.abs(cameraArc) > 0.8);
  cameraHeading +=
    Math.atan2(
      Math.sin(heading - cameraHeading),
      Math.cos(heading - cameraHeading),
    ) *
    (1 - Math.exp(-dt * 4));
  const faceAmount = THREE.MathUtils.smoothstep(Math.abs(cameraArc), 0.6, 2.3);
  const radius =
    THREE.MathUtils.lerp(15.5, 12.8, faceAmount) *
    (innerWidth < 701 ? (innerHeight < 700 ? 1.6 : 1.2) : 1);
  const elevation = THREE.MathUtils.lerp(6.3, 3.3, faceAmount);
  const angle = cameraHeading + Math.PI + cameraArc;
  cameraOffset.set(
    Math.sin(angle) * radius,
    elevation,
    Math.cos(angle) * radius,
  );
  desiredCamera.copy(position).add(cameraOffset);
  camera.position.lerp(desiredCamera, 1 - Math.exp(-dt * 5));
  const lookAhead = THREE.MathUtils.lerp(5, 0.35, faceAmount);
  target.set(
    position.x + Math.sin(heading) * lookAhead,
    position.y + THREE.MathUtils.lerp(2.5, 1.5, faceAmount),
    position.z + Math.cos(heading) * lookAhead,
  );
  controls.target.lerp(target, 1 - Math.exp(-dt * 5));
  camera.lookAt(controls.target);
}
function takeManualControl() {
  if (autopilot) {
    autopilot = false;
    updateAutopilot();
  }
}
function changeSpeed(value) {
  takeManualControl();
  targetSpeed = clamp(value, 0, 13);
  $("flight-speed").value = targetSpeed;
  $("throttle-output").value = Math.round(targetSpeed * 10.8) + " km/h";
  $("flight-speed").setAttribute(
    "aria-valuetext",
    Math.round(targetSpeed * 10.8) + " kilometers per hour",
  );
}
function releaseMouse() {
  const id = mouseFlight.id;
  mouseFlight.active = false;
  mouseFlight.id = null;
  mouseFlight.x = mouseFlight.y = 0;
  $("mouse-stick").hidden = true;
  if (id !== null && $("world").hasPointerCapture(id))
    $("world").releasePointerCapture(id);
}

async function takePhoto() {
  if (photoBusy) return;
  photoBusy = true;
  $("photo").disabled = true;
  try {
    renderer.render(scene, camera);
    const out = document.createElement("canvas");
    out.width = renderer.domElement.width;
    out.height = renderer.domElement.height;
    const ctx = out.getContext("2d");
    ctx.drawImage(renderer.domElement, 0, 0);
    const scale = out.width / 1440;
    ctx.fillStyle = mood === "night" ? "#edf2e7" : "#173f4d";
    ctx.font = `600 ${Math.max(12, 17 * scale)}px sans-serif`;
    ctx.fillText("POCKET SKIES", out.width * 0.04, out.height * 0.91);
    ctx.font = `${Math.max(9, 11 * scale)}px sans-serif`;
    ctx.fillText(
      "A little friend. A big blue sky.",
      out.width * 0.04,
      out.height * 0.945,
    );
    const blob = await new Promise((resolve) =>
      out.toBlob(resolve, "image/png"),
    );
    if (!blob) throw new Error("Image encoding failed");
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = `pocket-skies-${mood}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    toast("A little piece of sky, saved as a postcard.");
  } catch {
    toast("Couldn’t save this postcard. Try again in a moment.");
  } finally {
    photoBusy = false;
    $("photo").disabled = false;
  }
}
let joystickId = null;
function releaseJoystick() {
  touch.x = touch.y = 0;
  joystickId = null;
  $("joystick-knob").style.transform = "";
}
function bindUI() {
  $("start-flight").addEventListener("click", () => setMode("fly"));
  $("fly-mode").addEventListener("click", () => setMode("fly"));
  $("orbit-mode").addEventListener("click", () => setMode("orbit"));
  $("autopilot").addEventListener("click", () => {
    autopilot = !autopilot;
    cinemaTime = 0;
    lastFrontCycle = -1;
    releaseMouse();
    updateAutopilot();
    if (autopilot) toast("Follow the breeze. Doraemon knows the way.");
    $("world").focus({ preventScroll: true });
  });
  document.querySelectorAll("[data-camera]").forEach((button) =>
    button.addEventListener("click", () => {
      cameraStyle = button.dataset.camera;
      cinemaTime = 0;
      lastFrontCycle = -1;
      document
        .querySelectorAll("[data-camera]")
        .forEach((b) => setPressed(b, b.dataset.camera === cameraStyle));
    }),
  );
  $("flight-speed").addEventListener("input", (event) =>
    changeSpeed(Number(event.target.value)),
  );
  $("slower").addEventListener("click", () => changeSpeed(targetSpeed - 1));
  $("faster").addEventListener("click", () => changeSpeed(targetSpeed + 1));
  $("hover").addEventListener("click", () => {
    changeSpeed(0);
    toast("Hovering. Drag up or down to change altitude.");
  });
  $("reset-flight").addEventListener("click", () => {
    resetFlight();
    $("world").focus({ preventScroll: true });
    toast("A fresh sky. A fresh start.");
  });
  $("reset-view").addEventListener("click", () => {
    if (mode === "fly") resetFlight();
    else {
      homeCamera();
      interacted = false;
      doraemon.wink();
    }
    toast("Back to our favorite view.");
  });
  document
    .querySelectorAll("[data-sky]")
    .forEach((b) => b.addEventListener("click", () => setMood(b.dataset.sky)));
  $("sound").addEventListener("click", () => sound.toggle());
  $("photo").addEventListener("click", takePhoto);
  $("help-button").addEventListener("click", () => {
    keys.clear();
    releaseMouse();
    releaseJoystick();
    touch.rise = 0;
    $("help").showModal();
  });
  document
    .querySelectorAll(".close-dialog")
    .forEach((b) =>
      b.addEventListener("click", () => b.closest("dialog").close()),
    );
  document
    .querySelector(".close-guide")
    .addEventListener("click", () => $("help").close());
  document.querySelectorAll("dialog").forEach((d) => {
    d.addEventListener("click", (e) => {
      const r = d.getBoundingClientRect();
      if (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      )
        d.close();
    });
    d.addEventListener("close", () => {
      keys.clear();
      last = performance.now();
    });
  });
  $("play-again").addEventListener("click", () => {
    $("complete").close();
    resetFlight();
  });
  $("keep-flying").addEventListener("click", () => $("complete").close());
  addEventListener("keydown", (e) => {
    if (
      document.querySelector("dialog[open]") ||
      e.ctrlKey ||
      e.metaKey ||
      e.altKey
    )
      return;
    if (e.code === "Escape") {
      setMode("orbit");
      return;
    }
    if (e.target instanceof HTMLInputElement) return;
    const isButton =
      e.target instanceof HTMLElement &&
      ["BUTTON", "A", "INPUT"].includes(e.target.tagName);
    if (isButton && (e.code === "Space" || e.code === "Enter")) return;
    if (e.code === "KeyP" && !e.repeat) {
      takePhoto();
      return;
    }
    if (
      mode === "fly" &&
      [
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "Space",
        "ShiftLeft",
        "ShiftRight",
      ].includes(e.code)
    ) {
      e.preventDefault();
      keys.add(e.code);
      takeManualControl();
    }
  });
  addEventListener("keyup", (e) => keys.delete(e.code));
  addEventListener("blur", () => {
    keys.clear();
    touch.rise = 0;
    releaseMouse();
    releaseJoystick();
  });
  document.addEventListener("visibilitychange", () => {
    keys.clear();
    releaseMouse();
    releaseJoystick();
    touch.rise = 0;
    last = performance.now();
    if (sound.context) {
      if (document.hidden) sound.context.suspend();
      else if (sound.enabled) sound.context.resume().catch(() => {});
    }
  });
  const stick = $("joystick");
  function moveStick(e) {
    if (e.pointerId !== joystickId) return;
    if (autopilot) {
      autopilot = false;
      updateAutopilot();
    }
    const r = stick.getBoundingClientRect();
    let x = (e.clientX - r.left - r.width / 2) / 38,
      y = (e.clientY - r.top - r.height / 2) / 38;
    const len = Math.hypot(x, y);
    if (len > 1) {
      x /= len;
      y /= len;
    }
    touch.x = Math.abs(x) < 0.08 ? 0 : x;
    touch.y = Math.abs(y) < 0.08 ? 0 : y;
    $("joystick-knob").style.transform = `translate(${x * 32}px,${y * 32}px)`;
  }
  stick.addEventListener("pointerdown", (e) => {
    joystickId = e.pointerId;
    stick.setPointerCapture(e.pointerId);
    moveStick(e);
    e.preventDefault();
  });
  stick.addEventListener("pointermove", moveStick);
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
    stick.addEventListener(name, releaseJoystick);
  for (const [id, value] of [
    ["rise", 1],
    ["descend", -1],
  ]) {
    const b = $(id);
    b.addEventListener("pointerdown", (e) => {
      b.setPointerCapture(e.pointerId);
      touch.rise = value;
      autopilot = false;
      updateAutopilot();
      e.preventDefault();
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
      b.addEventListener(name, () => (touch.rise = 0));
  }
  let wasNarrow = innerWidth < 701;
  addEventListener("resize", () => {
    const narrow = innerWidth < 701;
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    if (mode === "orbit") {
      frameOrbit();
      if (narrow !== wasNarrow) homeCamera();
    } else frameFlight();
    wasNarrow = narrow;
  });
  // A floating mouse joystick: capture the pointer so release always cancels
  // steering, even if the pointer leaves the canvas or passes over a panel.
  const canvas = $("world");
  canvas.addEventListener("pointerdown", (event) => {
    if (mode !== "fly" || event.pointerType === "touch" || event.button !== 0)
      return;
    takeManualControl();
    mouseFlight.active = true;
    mouseFlight.id = event.pointerId;
    mouseFlight.originX = event.clientX;
    mouseFlight.originY = event.clientY;
    mouseFlight.x = mouseFlight.y = 0;
    canvas.setPointerCapture(event.pointerId);
    $("mouse-stick").hidden = false;
    $("mouse-stick").style.left = event.clientX + "px";
    $("mouse-stick").style.top = event.clientY + "px";
    $("mouse-stick").querySelector("b").style.transform = "";
    canvas.focus({ preventScroll: true });
    event.preventDefault();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!mouseFlight.active || event.pointerId !== mouseFlight.id) return;
    const dx = clamp((event.clientX - mouseFlight.originX) / 130, -1, 1),
      dy = clamp((event.clientY - mouseFlight.originY) / 110, -1, 1);
    mouseFlight.x = Math.abs(dx) < 0.055 ? 0 : dx;
    mouseFlight.y = Math.abs(dy) < 0.055 ? 0 : dy;
    $("mouse-stick").querySelector("b").style.transform =
      `translate(${dx * 42}px,${dy * 42}px)`;
  });
  for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"])
    canvas.addEventListener(eventName, (event) => {
      if (event.pointerId === mouseFlight.id) releaseMouse();
    });
  canvas.addEventListener(
    "wheel",
    (event) => {
      if (mode !== "fly" || document.querySelector("dialog[open]")) return;
      event.preventDefault();
      const pixels =
        event.deltaY *
        (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
      changeSpeed(targetSpeed - clamp(pixels, -160, 160) * 0.01);
    },
    { passive: false },
  );
  const raycaster = new THREE.Raycaster(),
    pointer = new THREE.Vector2();
  let tapStart = null;
  $("world").addEventListener(
    "pointerdown",
    (e) => (tapStart = { x: e.clientX, y: e.clientY, time: performance.now() }),
  );
  $("world").addEventListener("pointermove", (e) => {
    gaze.x = (e.clientX / innerWidth) * 2 - 1;
    gaze.y = 1 - (e.clientY / innerHeight) * 2;
  });
  $("world").addEventListener("pointerup", (e) => {
    if (
      mode !== "orbit" ||
      !tapStart ||
      performance.now() - tapStart.time > 400 ||
      Math.hypot(e.clientX - tapStart.x, e.clientY - tapStart.y) > 8
    )
      return;
    pointer.set(
      (e.clientX / innerWidth) * 2 - 1,
      1 - (e.clientY / innerHeight) * 2,
    );
    raycaster.setFromCamera(pointer, camera);
    if (raycaster.intersectObject(doraemon.root, true).length) {
      doraemon.greet();
      sound.chime(2);
      toast("Hello, friend. The sky is better with you in it.");
    }
    tapStart = null;
  });
  $("world").addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    $("error-message").textContent =
      "The graphics connection was interrupted. Reload to return to your sky.";
    $("error").hidden = false;
  });
}
function animate(now) {
  requestAnimationFrame(animate);
  if (document.hidden) return;
  const rawDt = (now - last) / 1000,
    dt = Math.min(rawDt, 0.045);
  last = now;
  if (document.querySelector("dialog[open]")) {
    renderer.render(scene, camera);
    return;
  }
  time += dt;
  if (mode === "fly") updateFlight(dt);
  else {
    doraemon.root.position.set(
      2,
      6 + (reduced ? 0 : Math.sin(time * 1.3) * 0.16),
      0,
    );
    doraemon.root.rotation.set(
      reduced ? 0 : Math.sin(time * 0.8) * 0.025,
      reduced ? 0 : Math.sin(time * 0.3) * 0.055,
      reduced ? 0 : Math.sin(time * 0.9) * 0.028,
    );
    controls.update();
    if (innerWidth > 1050) {
      doraemon.rotor.getWorldPosition(projected);
      projected.project(camera);
      $("gadget-label").style.top =
        `${clamp((-projected.y * 0.5 + 0.5) * innerHeight, 110, innerHeight - 200)}px`;
      $("gadget-label").style.opacity = interacted ? "0" : "1";
    }
  }
  doraemon.update(time, dt, {
    speed: mode === "fly" ? speed : 0,
    flying: mode === "fly",
    reduced,
    gaze,
  });
  world.update(time, dt, camera, lights, doraemon.root.position);
  sound.update();
  trail.visible = mode === "fly";
  rings.forEach((r) => {
    if (r.userData.passed && r.visible) {
      r.scale.multiplyScalar(1 + dt * 2.7);
      if (r.scale.x > 1.6) r.visible = false;
    } else if (r.visible) {
      r.children[1].material.opacity = 0.19 + Math.sin(time * 2) * 0.08;
      r.children.slice(2).forEach((bead, j) => (bead.rotation.z = time + j));
    }
  });
  for (let i = bursts.length - 1; i >= 0; i--) {
    const b = bursts[i];
    b.life -= dt;
    if (b.life <= 0) {
      scene.remove(b.mesh);
      b.mesh.geometry.dispose();
      b.mesh.material.dispose();
      bursts.splice(i, 1);
      continue;
    }
    const p = b.mesh.geometry.attributes.position;
    for (let j = 0; j < p.count; j++) {
      b.velocities[j].y -= dt * 0.7;
      p.setXYZ(
        j,
        p.getX(j) + b.velocities[j].x * dt,
        p.getY(j) + b.velocities[j].y * dt,
        p.getZ(j) + b.velocities[j].z * dt,
      );
    }
    p.needsUpdate = true;
    b.mesh.material.opacity = b.life / 1.8;
  }
  // Keep the detailed model's shadow centered even when flying out to sea.
  lights.sun.position
    .copy(doraemon.root.position)
    .add(new THREE.Vector3(-14, 24, 12));
  lights.sun.target.position.copy(doraemon.root.position);
  renderer.render(scene, camera);
  if (!qualityAdjusted && time > 3) {
    frameCount++;
    frameSeconds += rawDt;
    if (frameCount === 100) {
      if (frameSeconds > 3.6) {
        renderer.setPixelRatio(Math.min(devicePixelRatio, 1.25));
        renderer.setSize(innerWidth, innerHeight);
      }
      qualityAdjusted = true;
    }
  }
}
async function init() {
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: $("world"),
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(innerWidth, innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.13;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2("#d6e9e3", 0.0065);
    camera = new THREE.PerspectiveCamera(
      38,
      innerWidth / innerHeight,
      0.1,
      600,
    );
    const hemi = new THREE.HemisphereLight("#b6e0ef", "#8a9a7a", 2.2);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight("#fff3d5", 2.5);
    sun.position.set(-14, 24, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, {
      left: -23,
      right: 23,
      top: 23,
      bottom: -23,
      near: 1,
      far: 80,
    });
    sun.shadow.bias = -0.00025;
    sun.shadow.normalBias = 0.035;
    sun.shadow.radius = 3;
    scene.add(sun, sun.target);
    const fill = new THREE.DirectionalLight("#e1f5ff", 0.8);
    fill.position.set(8, 5, -10);
    scene.add(fill);
    lights = { sun, hemi, fill };
    world = makeWorld(scene);
    doraemon = makeDoraemon();
    scene.add(doraemon.root);
    doraemon.root.position.copy(position);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.065;
    controls.enablePan = false;
    controls.minDistance = 7;
    controls.maxDistance = 48;
    controls.minPolarAngle = 0.2;
    controls.maxPolarAngle = Math.PI * 0.53;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.8;
    controls.addEventListener("start", () => (interacted = true));
    homeCamera();
    createRings();
    bindUI();
    try {
      const saved = localStorage.getItem("pocket-skies-mood");
      if (["day", "sunset", "night"].includes(saved)) setMood(saved);
    } catch {}
    renderer.compile(scene, camera);
    world.update(0, 1, camera, lights, doraemon.root.position);
    renderer.render(scene, camera);
    $("loading").classList.add("done");
    last = performance.now();
    requestAnimationFrame(animate);
    // Read-only scene diagnostics for smoke tests and performance inspection.
    window.__pocketSkies = {
      get state() {
        const copterProjection = doraemon.rotor
          .getWorldPosition(new THREE.Vector3())
          .project(camera);
        return {
          copterScreen: [
            (copterProjection.x * 0.5 + 0.5) * innerWidth,
            (-copterProjection.y * 0.5 + 0.5) * innerHeight,
          ],
          mode,
          mood,
          autopilot,
          cameraStyle,
          cameraArc,
          cinemaTime,
          mouseActive: mouseFlight.active,
          pose: doraemon.pose,
          fadedClouds: world.fadedClouds,
          ringIndex,
          speed,
          targetSpeed,
          heading,
          position: position.toArray(),
          camera: camera.position.toArray(),
          flightTime,
          drawCalls: renderer.info.render.calls,
          triangles: renderer.info.render.triangles,
        };
      },
    };
  } catch (error) {
    console.error(error);
    $("loading").classList.add("done");
    $("error").hidden = false;
    $("error-message").textContent =
      `This adventure needs WebGL 2. Try a current browser with hardware acceleration enabled. (${error.message})`;
  }
}
init();
