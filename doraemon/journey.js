import * as THREE from "three";

// The flight chart is drawn at a fixed print resolution. All positions come from
// the same world coordinates used by the flight simulation and its landmarks.
export function makeJourney(scene, landmarks, route, reduced) {
  const chart = document.getElementById("chart-canvas");
  const ctx = chart.getContext("2d");
  const discovered = new Set();
  const breadcrumbs = [];
  let distance = 0, elapsed = 0, chartClock = 0, sampleClock = 0;
  let region = "Home waters";
  const previous = new THREE.Vector3();
  const colors = { ink: "#285967", water: "#e8f0ec", land: "#c0d1a8", sand: "#ece2c9", gold: "#b68b37" };
  const width = chart.width, height = chart.height;
  const project = (x, z) => [22 + (x + 56) / 100 * (width - 44), 20 + (z + 43) / 94 * (height - 40)];
  const strokes = [];
  const pointCount = 66;
  const sideVector = new THREE.Vector3();
  const handPosition = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (let side = 0; side < 2; side++) {
    const positions = new Float32Array(pointCount * 2 * 3);
    const uv = new Float32Array(pointCount * 2 * 2);
    const indices = [];
    for (let i = 0; i < pointCount; i++) {
      uv.set([i / (pointCount - 1), 0, i / (pointCount - 1), 1], i * 4);
      if (i < pointCount - 1) {
        const a = i * 2;
        indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geometry.setIndex(indices);
    const mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      uniforms: { opacity: { value: 0 }, tint: { value: new THREE.Color("#fff9de") } },
      vertexShader: `varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
      fragmentShader: `varying vec2 vUv;uniform float opacity;uniform vec3 tint;void main(){float edge=pow(max(0.,1.-abs(vUv.y*2.-1.)),1.8);float tail=pow(1.-vUv.x,1.6)*smoothstep(0.,.08,vUv.x);gl_FragColor=vec4(tint,edge*tail*opacity);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>}`,
    });
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.frustumCulled = false;
    mesh.visible = false;
    scene.add(mesh);
    strokes.push({mesh, centers: new Float32Array(pointCount * 3)});
  }
  let wakeClock = 0, wakeReady = false;

  function island(x, z, rx, rz, phase) {
    const [cx, cy] = project(x, z);
    const sx = (width - 44) / 100, sy = (height - 40) / 94;
    for (let layer = 2; layer >= 0; layer--) {
      ctx.beginPath();
      for (let i = 0; i <= 48; i++) {
        const a = i / 48 * Math.PI * 2;
        const wobble = 1 + Math.sin(a * 3 + phase) * 0.1 + Math.cos(a * 5 + phase) * 0.05;
        const px = cx + Math.cos(a) * (rx + layer * 1.35) * sx * wobble;
        const py = cy + Math.sin(a) * (rz + layer * 1.35) * sy * wobble;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = layer === 2 ? "#d3e4df" : layer === 1 ? colors.sand : colors.land;
      ctx.fill();
    }
  }
  function draw(position, heading, ringIndex) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = colors.water;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#aacacb55";
    ctx.lineWidth = 0.6;
    for (let x = -40; x < 45; x += 20) {
      const [px] = project(x, 0);
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, height); ctx.stroke();
    }
    for (let z = -40; z < 54; z += 20) {
      const [, py] = project(0, z);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(width, py); ctx.stroke();
    }
    landmarks.slice(0, 3).forEach((landmark, i) => {
      const radii = [[8, 8.5], [6, 5.1], [7, 4.8]][i];
      island(landmark.position.x, landmark.position.z, ...radii, i * 1.2 + 0.3);
    });
    // Route and breadcrumb trail use actual ring and flight coordinates.
    ctx.beginPath();
    route.forEach((p, i) => { const [x, y] = project(p.x, p.z); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.setLineDash([3, 5]); ctx.lineWidth = 1.3; ctx.strokeStyle = "#b08b4b80"; ctx.stroke(); ctx.setLineDash([]);
    if (breadcrumbs.length > 1) {
      ctx.beginPath();
      breadcrumbs.forEach((p, i) => { const [x, y] = project(...p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.strokeStyle = "#438a9c70"; ctx.lineWidth = 1.2; ctx.stroke();
    }
    route.forEach((p, i) => {
      const [x, y] = project(p.x, p.z);
      ctx.beginPath(); ctx.arc(x, y, i === ringIndex ? 5.6 : 3, 0, Math.PI * 2);
      ctx.fillStyle = i < ringIndex ? colors.gold : "#fff9e8";
      ctx.strokeStyle = colors.gold; ctx.lineWidth = i === ringIndex ? 2 : 1;
      ctx.fill(); ctx.stroke();
    });
    ctx.font = "600 13px 'DM Sans', sans-serif";
    ctx.textAlign = "center";
    landmarks.forEach((landmark, i) => {
      const [x, y] = project(landmark.position.x, landmark.position.z);
      const seen = discovered.has(landmark.id);
      ctx.fillStyle = seen ? colors.ink : "#819999";
      ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill();
      if (seen) { ctx.strokeStyle = colors.gold; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 6.5, 0, Math.PI * 2); ctx.stroke(); }
      const labels = ["HOME", "WINDMILL", "GARDEN", "BALLOONS"];
      ctx.fillText(labels[i] || landmark.name.toUpperCase(), x, y + 18);
    });
    let [px, py] = project(position.x, position.z);
    px = THREE.MathUtils.clamp(px, 10, width - 10); py = THREE.MathUtils.clamp(py, 10, height - 10);
    ctx.save(); ctx.translate(px, py); ctx.rotate(Math.PI - heading);
    ctx.shadowColor = "#25546740"; ctx.shadowBlur = 7;
    ctx.beginPath(); ctx.moveTo(0, -8); ctx.lineTo(5.5, 6); ctx.lineTo(0, 3.5); ctx.lineTo(-5.5, 6); ctx.closePath();
    ctx.fillStyle = "#147f9b"; ctx.strokeStyle = "#fffdf3"; ctx.lineWidth = 1.8; ctx.fill(); ctx.stroke(); ctx.restore();
    // A restrained compass rose, like a chart tucked into a travel journal.
    ctx.save(); ctx.translate(width - 21, 22);
    ctx.strokeStyle = "#567d86"; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(0, 4); ctx.lineTo(0, 21); ctx.moveTo(-4, 10); ctx.lineTo(0, 4); ctx.lineTo(4, 10); ctx.stroke();
    ctx.fillStyle = colors.ink; ctx.font = "12px 'DM Sans', sans-serif"; ctx.fillText("N", 0, 0); ctx.restore();
  }

  return {
    drawChart: draw,
    get stats() { return { distance, elapsed, discovered: discovered.size, total: landmarks.length, region }; },
    reset(position) {
      distance = elapsed = chartClock = sampleClock = wakeClock = 0;
      discovered.clear(); breadcrumbs.length = 0; previous.copy(position); wakeReady = false;
      document.getElementById("discovery-count").textContent = `0 / ${landmarks.length}`;
    },
    update(position, heading, dt, ringIndex) {
      elapsed += dt; distance += previous.distanceTo(position) * 3; previous.copy(position);
      sampleClock += dt; chartClock += dt;
      if (sampleClock > 0.45) {
        sampleClock = 0; breadcrumbs.push([position.x, position.z]);
        if (breadcrumbs.length > 260) breadcrumbs.shift();
      }
      let found = null, closest = Infinity;
      region = "Open blue";
      for (const landmark of landmarks) {
        const d = Math.hypot(position.x - landmark.position.x, position.z - landmark.position.z);
        if (d < closest && d < landmark.radius * 1.5) { region = landmark.name; closest = d; }
        if (elapsed > 2.5 && !found && d < landmark.radius * 0.7 && !discovered.has(landmark.id)) {
          discovered.add(landmark.id); found = landmark;
          document.getElementById("discovery-count").textContent = `${discovered.size} / ${landmarks.length}`;
        }
      }
      if (chartClock > 0.12) {
        chartClock = 0;
        document.getElementById("chart-region").textContent = region;
        if (!document.getElementById("sky-chart").classList.contains("is-collapsed") && !document.body.classList.contains("focus-mode")) draw(position, heading, ringIndex);
      }
      return found;
    },
    updateWake(dt, model, speed, flying, camera) {
      const visible = flying && !reduced && speed > 1.2;
      strokes.forEach(({mesh}) => { mesh.visible = visible; });
      if (!visible) { wakeReady = false; return; }
      wakeClock += dt;
      const sample = wakeClock > 1 / 50;
      if (sample) wakeClock %= 1 / 50;
      model.root.updateWorldMatrix(true, true);
      for (let j = 0; j < 2; j++) {
        const {mesh, centers} = strokes[j];
        // The wake starts at the hands; this follows the animated pose exactly.
        const arm = model.arms[j];
        arm.localToWorld(handPosition.set(j === 0 ? -0.66 : 0.66, -0.03, 0.055));
        if (!wakeReady) for (let i = 0; i < pointCount; i++) centers.set(handPosition.toArray(), i * 3);
        if (sample) centers.copyWithin(3, 0, centers.length - 3);
        centers[0] = handPosition.x; centers[1] = handPosition.y; centers[2] = handPosition.z;
        sideVector.copy(camera.position).sub(handPosition).cross(up).normalize();
        const p = mesh.geometry.attributes.position;
        for (let i = 0; i < pointCount; i++) {
          const halfWidth = 0.035 + i / pointCount * 0.16;
          for (let edge = 0; edge < 2; edge++) {
            const sign = edge ? 1 : -1;
            p.setXYZ(i * 2 + edge, centers[i * 3] + sideVector.x * halfWidth * sign, centers[i * 3 + 1], centers[i * 3 + 2] + sideVector.z * halfWidth * sign);
          }
        }
        p.needsUpdate = true;
        mesh.material.uniforms.opacity.value = THREE.MathUtils.smoothstep(speed, 1.2, 9) * 0.4;
      }
      wakeReady = true;
    },
  };
}

export function formatFlightTime(seconds) {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

// Draw a keepsake around the rendered scene; the PNG never includes the HUD.
export function composePostcard(source, {mood, region, distance}) {
  const out = document.createElement("canvas");
  const imageWidth = Math.min(2400, source.width);
  const imageHeight = Math.round(source.height / source.width * imageWidth);
  const border = Math.round(imageWidth * 0.028);
  const footer = Math.max(110, Math.round(imageWidth * 0.095));
  out.width = imageWidth + border * 2; out.height = imageHeight + border * 2 + footer;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#fffaf0"; ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(source, border, border, imageWidth, imageHeight);
  const y = border + imageHeight;
  const scale = imageWidth / 1440;
  ctx.fillStyle = "#1b4d5b";
  ctx.font = `500 ${Math.max(25, 38 * scale)}px 'Fraunces', Georgia, serif`;
  ctx.fillText("A little piece of sky.", border * 1.7, y + footer * 0.46);
  ctx.fillStyle = "#52727a";
  ctx.font = `500 ${Math.max(9, 12 * scale)}px 'DM Sans', sans-serif`;
  const skyName = {day: "DAYDREAM", sunset: "GOLDEN HOUR", night: "MOONLIGHT"}[mood];
  ctx.fillText(`POCKET SKIES   /   ${skyName}   /   ${region.toUpperCase()}`, border * 1.7, y + footer * 0.72);
  const stampX = out.width - border - footer * 0.61, stampY = y + footer * 0.5, r = footer * 0.34;
  ctx.save(); ctx.translate(stampX, stampY); ctx.rotate(-0.12);
  ctx.strokeStyle = "#b2975f"; ctx.lineWidth = Math.max(1, 1.4 * scale);
  for (const radius of [r, r * 0.87]) { ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2); ctx.stroke(); }
  ctx.textAlign = "center"; ctx.fillStyle = "#8e743f";
  ctx.font = `600 ${Math.max(7, 9 * scale)}px 'DM Sans', sans-serif`;
  ctx.fillText("SENT FROM", 0, -r * 0.25); ctx.fillText("THE SKY", 0, r * 0.06);
  ctx.font = `${Math.max(6, 8 * scale)}px 'DM Sans', sans-serif`;
  ctx.fillText(distance > 50 ? `${(distance / 1000).toFixed(1)} KM OF WONDER` : "WITH A LITTLE MAGIC", 0, r * 0.4);
  ctx.restore();
  return out;
}
