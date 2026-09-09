import { atelier, makeSam, V } from "./objects.js";
import { makeWorld, stops } from "./world.js";
const B = window.BABYLON,
  $ = (id) => document.getElementById(id),
  TAU = Math.PI * 2;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
let engine,
  scene,
  camera,
  world,
  sam,
  playing = !reduced,
  speed = 1,
  time = 0,
  gait = 0,
  travel = 0,
  wave = 0,
  mood = "day",
  currentStop = -1,
  toastTimer,
  audio,
  muted = true;
function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("show"), 3300);
}
function updatePlay() {
  $("play").textContent = playing ? "Ⅱ" : "▶";
  $("play").setAttribute(
    "aria-label",
    playing ? "Pause walking" : "Resume walking",
  );
  $("play").title = playing ? "Pause" : "Play";
}
function setStop(i) {
  if (i === currentStop) return;
  if (currentStop !== -1) wave = 2.4;
  currentStop = i;
}
function resetCamera() {
  const mobile = innerWidth <= 800;
  camera.alpha = 1.2;
  camera.beta = 1.2;
  camera.radius = mobile ? 18 : 12.8;
  camera.target.copyFrom(V(0, mobile ? 0.9 : 1.6, 0));
  camera.inertialAlphaOffset = 0;
  camera.inertialBetaOffset = 0;
  camera.inertialRadiusOffset = 0;
}
const moods = {
  day: {
    bg: "#102c40",
    sun: "#ffe2b1",
    fill: "#b9e0e9",
    ground: "#547f89",
    power: 2.05,
    ambient: 1.05,
    exposure: 1.1,
  },
  sunset: {
    bg: "#382c43",
    sun: "#ffb27e",
    fill: "#c4b0cc",
    ground: "#77627a",
    power: 2.4,
    ambient: 0.8,
    exposure: 1.04,
  },
  night: {
    bg: "#09172d",
    sun: "#b9d8ff",
    fill: "#8eafdc",
    ground: "#425573",
    power: 1.4,
    ambient: 1.0,
    exposure: 1.02,
  },
};
function makeAudio() {
  const C = window.AudioContext || window.webkitAudioContext;
  if (!C) throw Error("Audio unavailable");
  const ctx = new C(),
    gain = ctx.createGain();
  gain.gain.value = 0;
  gain.connect(ctx.destination);
  const melody = [
    72, 76, 79, 83, 81, 79, 76, 74, 72, 67, 71, 74, 76, 74, 71, 67,
  ];
  let beat = 0,
    next = ctx.currentTime;
  const interval = setInterval(() => {
    if (ctx.state !== "running") return;
    while (next < ctx.currentTime + 0.3) {
      if (!muted && !document.hidden) {
        const note = melody[beat % melody.length];
        for (const [n, amp] of [
          [note, 0.14],
          [note - 24, 0.045],
        ]) {
          const osc = ctx.createOscillator(),
            env = ctx.createGain();
          osc.type = "sine";
          osc.frequency.value = 440 * Math.pow(2, (n - 69) / 12);
          env.gain.setValueAtTime(0, next);
          env.gain.linearRampToValueAtTime(amp, next + 0.015);
          env.gain.exponentialRampToValueAtTime(0.0001, next + 1.5);
          osc.connect(env);
          env.connect(gain);
          osc.start(next);
          osc.stop(next + 1.6);
        }
        beat++;
      }
      next += 0.43;
    }
  }, 120);
  return { ctx, gain, interval };
}
async function toggleSound() {
  try {
    if (!audio) audio = makeAudio();
    await audio.ctx.resume();
    muted = !muted;
    audio.gain.gain.setTargetAtTime(
      muted ? 0 : 0.45,
      audio.ctx.currentTime,
      0.2,
    );
    $("sound").setAttribute("aria-pressed", String(!muted));
    $("sound").setAttribute(
      "aria-label",
      muted ? "Enable music" : "Mute music",
    );
    $("sound").style.color = muted ? "" : "#f2cd87";
  } catch {
    toast("Music is unavailable in this browser.");
  }
}
async function postcard() {
  try {
    scene.render();
    const exportCamera = new B.ArcRotateCamera(
      "postcard view",
      camera.alpha,
      camera.beta,
      16.8,
      V(0, 1, 0),
      scene,
    );
    exportCamera.fov = 0.7;
    let data;
    try {
      data = await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(Error("capture timeout")),
          12000,
        );
        B.Tools.CreateScreenshotUsingRenderTarget(
          engine,
          exportCamera,
          { width: 1600, height: 1100 },
          (url) => {
            clearTimeout(timeout);
            resolve(url);
          },
          "image/png",
          1,
          true,
        );
      });
    } finally {
      exportCamera.dispose();
    }
    const img = new Image();
    img.src = data;
    await img.decode();
    const out = document.createElement("canvas");
    out.width = 1720;
    out.height = 1320;
    const c = out.getContext("2d");
    c.fillStyle = "#f6efd9";
    c.fillRect(0, 0, out.width, out.height);
    c.drawImage(img, 60, 60, 1600, 1100);
    c.fillStyle = "#244b58";
    c.font = "44px Georgia";
    c.fillText("A world of small wonders.", 75, 1233);
    c.font = "15px sans-serif";
    c.fillStyle = "#627d7c";
    c.fillText(
      `TUXEDO SAM’S LITTLE ATLAS  /  ${stops[currentStop].name.toUpperCase()}  /  ${mood.toUpperCase()}`,
      78,
      1276,
    );
    c.strokeStyle = "#aa8a63";
    c.lineWidth = 2;
    c.beginPath();
    c.arc(1570, 1235, 49, 0, TAU);
    c.stroke();
    c.font = "11px sans-serif";
    c.textAlign = "center";
    c.fillText("WISH YOU", 1570, 1228);
    c.fillText("WERE HERE", 1570, 1246);
    const blob = await new Promise((resolve) =>
      out.toBlob(resolve, "image/png"),
    );
    if (!blob) throw Error("capture failed");
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = `sam-${stops[currentStop].type}-postcard.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    toast("Postcard saved");
  } catch (e) {
    console.error(e);
    toast("The postcard couldn’t be saved. Please try again.");
  }
}
async function init() {
  try {
    if (!B || !B.Engine.isSupported())
      throw Error(
        "This little world needs WebGL. Try a browser with hardware acceleration enabled.",
      );
    engine = new B.Engine(
      $("scene"),
      true,
      { preserveDrawingBuffer: false, stencil: true, antialias: true },
      true,
    );
    engine.setHardwareScalingLevel(1 / Math.min(devicePixelRatio, 1.65));
    scene = new B.Scene(engine);
    scene.clearColor = B.Color4.FromHexString("#102c40ff");
    scene.ambientColor = B.Color3.FromHexString("#698e9b").scale(0.2);
    camera = new B.ArcRotateCamera(
      "the scenic view",
      1.2,
      1.15,
      15.5,
      V(0, 0.8, 0),
      scene,
    );
    camera.attachControl($("scene"), true);
    camera.lowerRadiusLimit = 8;
    camera.upperRadiusLimit = 38;
    camera.lowerBetaLimit = 0.35;
    camera.upperBetaLimit = 1.9;
    camera.wheelDeltaPercentage = 0.012;
    camera.pinchDeltaPercentage = 0.006;
    camera.panningSensibility = 0;
    camera.minZ = 0.1;
    camera.maxZ = 100;
    camera.fov = 0.7;
    resetCamera();
    const hemi = new B.HemisphericLight("soft sky", V(0, 1, 0), scene);
    hemi.intensity = 1.05;
    hemi.diffuse = B.Color3.FromHexString("#b9e0e9");
    hemi.groundColor = B.Color3.FromHexString("#547f89");
    const sun = new B.DirectionalLight(
      "warm afternoon",
      V(-0.65, -1, -0.7),
      scene,
    );
    sun.position = V(8, 13, 8);
    sun.intensity = 2.05;
    sun.diffuse = B.Color3.FromHexString("#ffe2b1");
    const rim = new B.DirectionalLight("blue rim", V(0.4, -0.3, 0.8), scene);
    rim.intensity = 0.65;
    rim.diffuse = B.Color3.FromHexString("#a5d6eb");
    scene.imageProcessingConfiguration.toneMappingEnabled = true;
    scene.imageProcessingConfiguration.toneMappingType =
      B.ImageProcessingConfiguration.TONEMAPPING_ACES;
    scene.imageProcessingConfiguration.exposure = 1.1;
    scene.imageProcessingConfiguration.contrast = 1.12;
    const a = atelier(scene);
    sam = makeSam(a);
    world = await makeWorld(scene, a);
    sam.root.position.y = world.R - 0.055;
    sam.root.rotation.y = 0.3;
    const shadow = new B.ShadowGenerator(innerWidth <= 800 ? 1024 : 2048, sun);
    shadow.useBlurExponentialShadowMap = true;
    shadow.blurKernel = 24;
    shadow.darkness = 0.23;
    shadow.bias = 0.0002;
    shadow.normalBias = 0.015;
    sam.root.getChildMeshes().forEach((m) => shadow.addShadowCaster(m));
    world.root
      .getChildMeshes()
      .filter((m) => m !== world.globe)
      .forEach((m) => shadow.addShadowCaster(m));
    // A transparent contact shadow prevents little feet from floating above the atlas.
    const contactTex = new B.DynamicTexture(
        "soft footfall",
        { width: 128, height: 128 },
        scene,
        false,
      ),
      ct = contactTex.getContext();
    const g = ct.createRadialGradient(64, 64, 5, 64, 64, 62);
    g.addColorStop(0, "rgba(17,47,57,.35)");
    g.addColorStop(1, "rgba(17,47,57,0)");
    ct.fillStyle = g;
    ct.fillRect(0, 0, 128, 128);
    contactTex.hasAlpha = true;
    contactTex.update();
    const contact = B.MeshBuilder.CreateGround(
      "footfall",
      { width: 1.55, height: 1.1 },
      scene,
    );
    contact.position.y = world.R + 0.015;
    const cm = new B.StandardMaterial("soft shade", scene);
    cm.diffuseTexture = contactTex;
    cm.opacityTexture = contactTex;
    cm.disableLighting = true;
    cm.emissiveColor = B.Color3.White();
    contact.material = cm;
    contact.isPickable = false;
    scene.onPointerObservable.add((info) => {
      if (
        info.type === B.PointerEventTypes.POINTERTAP &&
        info.pickInfo?.pickedMesh?.isDescendantOf(sam.root)
      ) {
        wave = 3;
      }
    });
    $("play").onclick = () => {
      playing = !playing;
      updatePlay();
    };
    $("mood").onclick = () => {
      const names = ["day", "sunset", "night"];
      const icons = ["☀", "◒", "☾"];
      const index = (names.indexOf(mood) + 1) % names.length;
      mood = names[index];
      $("mood").textContent = icons[index];
      $("mood").setAttribute(
        "aria-label",
        `Lighting: ${mood}. Switch to ${names[(index + 1) % names.length]}`,
      );
    };
    $("sound").onclick = toggleSound;
    $("postcard").onclick = async () => {
      $("postcard").disabled = true;
      try {
        await postcard();
      } finally {
        $("postcard").disabled = false;
      }
    };
    $("scene").addEventListener("dblclick", resetCamera);
    window.addEventListener("keydown", (e) => {
      if (e.target.matches("input,button,a")) return;
      if (e.code === "Space") {
        e.preventDefault();
        playing = !playing;
        updatePlay();
      }
      if (e.key.toLowerCase() === "r") resetCamera();
    });
    document.addEventListener("visibilitychange", () => {
      if (audio) {
        if (document.hidden) audio.ctx.suspend();
        else if (!muted) audio.ctx.resume();
      }
    });
    let layoutMode =
      (innerWidth <= 800 ? "mobile" : "desktop") +
      (innerHeight < 740 ? "short" : "tall");
    window.addEventListener("resize", () => {
      engine.resize();
      const next =
        (innerWidth <= 800 ? "mobile" : "desktop") +
        (innerHeight < 740 ? "short" : "tall");
      if (layoutMode !== next) {
        layoutMode = next;
        resetCamera();
      }
    });
    engine.onContextLostObservable.add(() => {
      $("error-message").textContent =
        "The graphics connection took a little break. Reload to continue the journey.";
      $("error").hidden = false;
    });
    setStop(0);
    updatePlay();
    world.update(0, 0);
    sam.update(0, false, 0);
    await scene.whenReadyAsync();
    $("loading").hidden = true;
    if (matchMedia("(pointer: coarse)").matches)
      $("hint").textContent = "Drag to look around · Pinch to zoom";
    setTimeout(() => $("hint").classList.add("dismissed"), 6500);
    let smoothFps = 60,
      frames = 0;
    engine.runRenderLoop(() => {
      if (document.hidden) return;
      const dt = Math.min(engine.getDeltaTime() / 1000, 0.05);
      if (!reduced || playing) {
        time += dt;
        if (playing) {
          gait += dt * speed * 0.69;
          travel += (dt * speed * 0.32) / (world.totalLength * world.R);
          wave = Math.max(0, wave - dt);
        } else wave = Math.max(0, wave - dt);
      }
      world.update(travel, time);
      sam.update(gait, playing, wave, dt);
      const progress = ((travel % 1) + 1) % 1;
      let stop = 0;
      for (let i = 0; i < world.stopProgress.length; i++)
        if (progress + 1e-7 >= world.stopProgress[i]) stop = i;
      setStop(stop);
      const palette = moods[mood],
        blend = 1 - Math.exp(-dt * 2.5);
      scene.clearColor = B.Color4.Lerp(
        scene.clearColor,
        B.Color4.FromHexString(palette.bg + "ff"),
        blend,
      );
      sun.diffuse = B.Color3.Lerp(
        sun.diffuse,
        B.Color3.FromHexString(palette.sun),
        blend,
      );
      hemi.diffuse = B.Color3.Lerp(
        hemi.diffuse,
        B.Color3.FromHexString(palette.fill),
        blend,
      );
      hemi.groundColor = B.Color3.Lerp(
        hemi.groundColor,
        B.Color3.FromHexString(palette.ground),
        blend,
      );
      sun.intensity += (palette.power - sun.intensity) * blend;
      hemi.intensity += (palette.ambient - hemi.intensity) * blend;
      scene.imageProcessingConfiguration.exposure +=
        (palette.exposure - scene.imageProcessingConfiguration.exposure) *
        blend;
      scene.render();
      if (++frames % 120 === 0) {
        smoothFps = smoothFps * 0.6 + engine.getFps() * 0.4;
        if (smoothFps < 35 && engine.getHardwareScalingLevel() < 1.3) {
          engine.setHardwareScalingLevel(
            Math.min(1.3, engine.getHardwareScalingLevel() + 0.15),
          );
        }
      }
    });
    window.__sam = {
      scene,
      engine,
      camera,
      sam,
      world,
      get state() {
        return {
          playing,
          speed,
          travel,
          mood,
          currentStop,
        };
      },
    };
  } catch (error) {
    console.error(error);
    $("loading").hidden = true;
    $("error-message").textContent =
      error.message || "The little world could not be loaded.";
    $("error").hidden = false;
  }
}
init();
