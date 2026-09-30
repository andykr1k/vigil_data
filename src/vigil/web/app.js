import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// Binary messages: u8 kind, u8 camera, 2 pad, u32 seq (must match pipeline.py).
const MSG_MESH = 1, MSG_JPEG = 2, MSG_CLOUD = 3, MSG_ULTRASOUND = 4;
const FUSED = 255; // camera index of the fused world-frame cloud

const COLORS = {
  left: new THREE.Color("#22e4ff"),
  right: new THREE.Color("#ff3df2"),
  center: new THREE.Color("#8fa8ff"),
  upper: new THREE.Color("#4d6a9a"),
  mesh: new THREE.Color("#3fd8ff"),
  grid: new THREE.Color("#1fb6ff"),
  tip: new THREE.Color("#ff2a2a"),
  gold: new THREE.Color("#ffd23d"),
};
const SEGMENT_LABELS = { thigh: "THIGH", shin: "SHIN", foot: "FOOT" };

// ───────────────────────────── settings ─────────────────────────────
const store = {
  get(k, d) { try { const v = localStorage.getItem("vigil." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("vigil." + k, JSON.stringify(v)); } catch { /* ignore */ } },
};
const opts = {};
document.querySelectorAll("[data-toggle]").forEach((el) => {
  const key = el.dataset.toggle;
  el.checked = store.get("t." + key, el.checked);
  opts[key] = el.checked;
  el.addEventListener("change", () => { opts[key] = el.checked; store.set("t." + key, el.checked); applyVisibility(); });
});
document.querySelectorAll(".panel.collapsible").forEach((p) => {
  if (store.get("c." + p.id, false)) p.classList.add("collapsed");
  p.querySelector("header").addEventListener("click", () => {
    p.classList.toggle("collapsed");
    store.set("c." + p.id, p.classList.contains("collapsed"));
  });
});

// ───────────────────────────── renderer ─────────────────────────────
const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
stage.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
labelRenderer.domElement.className = "labels";
stage.appendChild(labelRenderer.domElement);

const BACKGROUND = new THREE.Color("#02050a");
const scene = new THREE.Scene();
scene.background = BACKGROUND;
scene.fog = new THREE.FogExp2("#02050a", 0.075);

// Only the probe model needs lighting; everything else is emissive.
scene.add(new THREE.HemisphereLight("#bfe9ff", "#0a1420", 1.4));
const keyLight = new THREE.DirectionalLight("#ffffff", 1.6);
keyLight.position.set(2, 4, 3);
scene.add(keyLight);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.02, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotateSpeed = 0.6;
controls.maxPolarAngle = Math.PI * 0.495;

// Selective bloom: only objects on BLOOM_LAYER glow, so tags, the probe model and the
// point cloud stay crisp. Non-glowing objects are rendered black into the bloom buffer.
const BLOOM_LAYER = 1;
const bloomLayer = new THREE.Layers();
bloomLayer.set(BLOOM_LAYER);
const glow = (obj) => { obj.traverse((o) => o.layers.enable(BLOOM_LAYER)); return obj; };

const bloomComposer = new EffectComposer(renderer);
bloomComposer.renderToScreen = false;
bloomComposer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.9, 0.5, 0.05);
bloomComposer.addPass(bloom);

const mixPass = new ShaderPass(new THREE.ShaderMaterial({
  uniforms: { baseTexture: { value: null }, bloomTexture: { value: bloomComposer.renderTarget2.texture }, uStrength: { value: 1 } },
  vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D baseTexture; uniform sampler2D bloomTexture; uniform float uStrength; varying vec2 vUv;
    void main() { gl_FragColor = texture2D(baseTexture, vUv) + uStrength * texture2D(bloomTexture, vUv); }`,
}), "baseTexture");
mixPass.needsSwap = true;
const finalComposer = new EffectComposer(renderer);
finalComposer.addPass(new RenderPass(scene, camera));
finalComposer.addPass(mixPass);
finalComposer.addPass(new OutputPass());

const darkMaterials = {
  mesh: new THREE.MeshBasicMaterial({ color: 0x000000 }),
  points: new THREE.PointsMaterial({ color: 0x000000, size: 0.014 }),
  line: new THREE.LineBasicMaterial({ color: 0x000000 }),
};
const savedMaterials = new Map();
function darkenNonBloomed(obj) {
  if (!obj.material || bloomLayer.test(obj.layers)) return;
  savedMaterials.set(obj, obj.material);
  obj.material = obj.isPoints ? darkMaterials.points : obj.isLine ? darkMaterials.line : darkMaterials.mesh;
}
function restoreMaterial(obj) {
  const m = savedMaterials.get(obj);
  if (m) { obj.material = m; savedMaterials.delete(obj); }
}
function render() {
  if (opts.bloom) {
    scene.background = null;
    scene.traverse(darkenNonBloomed);
    bloomComposer.render();
    scene.traverse(restoreMaterial);
    scene.background = BACKGROUND;
  }
  mixPass.uniforms.uStrength.value = opts.bloom ? 1 : 0;
  finalComposer.render();
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  bloomComposer.setSize(innerWidth, innerHeight);
  finalComposer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
});

// ───────────────────────────── world frames ─────────────────────────────
// worldRoot: levels the scene using the floor plane (y = 0 is the floor).
// sensorRoot: world = first camera's optical frame (x right, y down, z forward)
//             → three.js (x right, y up, z back). Per-camera groups hang off it.
const worldRoot = new THREE.Group();
scene.add(worldRoot);
const sensorRoot = new THREE.Group();
sensorRoot.rotation.x = Math.PI;
worldRoot.add(sensorRoot);
const levelTarget = { quat: new THREE.Quaternion(), height: 1.2, known: false };
worldRoot.position.y = levelTarget.height;

// ───────────────────────────── environment ─────────────────────────────
const floorMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: { uColor: { value: COLORS.grid }, uFocus: { value: new THREE.Vector2(0, -2.5) } },
  vertexShader: /* glsl */ `
    varying vec3 vWorld;
    void main() {
      vec4 w = modelMatrix * vec4(position, 1.0);
      vWorld = w.xyz;
      gl_Position = projectionMatrix * viewMatrix * w;
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor; uniform vec2 uFocus;
    varying vec3 vWorld;
    float grid(vec2 p, float s, float w) {
      vec2 q = p / s;
      vec2 g = abs(fract(q - 0.5) - 0.5) / (fwidth(q) * w);
      return 1.0 - min(min(g.x, g.y), 1.0);
    }
    void main() {
      vec2 p = vWorld.xz;
      float d = length(p - uFocus);
      float fade = exp(-d * 0.22);
      float g = grid(p, 0.25, 1.0) * 0.22 + grid(p, 1.0, 1.4) * 0.75;
      float a = g * fade;
      gl_FragColor = vec4(uColor * a, a);
    }`,
});
const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), floorMat); // additive already; no bloom
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

const dust = (() => {
  const n = 1800, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 40;
    pos[i * 3 + 1] = Math.random() * 12;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 40;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({
    size: 0.03, color: "#3a6d9a", transparent: true, opacity: 0.6, depthWrite: false,
  }));
})();
scene.add(dust);

// ───────────────────────────── cameras (rig) ─────────────────────────────
const CLOUD_MAX = 400_000;
let rigCams = []; // per camera: { info, group, cloud, feed, image }

function makeSensorGlyph(cam, highlight, calibrated) {
  const color = !calibrated ? "#ffc53d" : highlight ? "#22e4ff" : "#7d8cff";
  const g = new THREE.Group();
  const d = 0.35, w = cam.width, h = cam.height;
  const corners = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
    new THREE.Vector3(((u - cam.cx) / cam.fx) * d, ((v - cam.cy) / cam.fy) * d, d));
  const pts = [];
  corners.forEach((c, i) => { pts.push(new THREE.Vector3(), c, c, corners[(i + 1) % 4]); });
  g.add(glow(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }))));
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.025, 0.025), new THREE.MeshBasicMaterial({ color: "#0f3550" }));
  body.position.z = -0.013;
  g.add(body, glow(new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 12), new THREE.MeshBasicMaterial({ color: "#8ff4ff" }))));
  const el = document.createElement("div");
  el.className = "tag";
  el.style.color = color;
  el.textContent = calibrated ? `CAM ${cam.index + 1}` : `CAM ${cam.index + 1} · UNCALIBRATED`;
  const label = new CSS2DObject(el);
  label.position.set(0, -0.05, 0);
  g.add(label);
  return g;
}

// CSS2D labels are DOM elements the label renderer positions each frame; removing their 3D
// object doesn't remove the element, so without this every rebuild leaves stale labels behind.
function removeWithLabels(obj) {
  obj.traverse((o) => { if (o.isCSS2DObject) o.element.remove(); });
  obj.removeFromParent();
}

function buildRig(cameras) {
  for (const rc of rigCams) removeWithLabels(rc.group);
  const feeds = $("feeds");
  feeds.innerHTML = "";
  rigCams = cameras.map((cam) => {
    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    const calibrated = !!cam.T_world_camera;
    // Until CALIBRATE RIG places it, show the camera beside CAM 1 so its viewpoint and
    // cloud are still visible (amber = placeholder pose, not its real position).
    if (calibrated) group.matrix.set(...cam.T_world_camera);
    else group.matrix.makeTranslation(0.35 * cam.index, 0, 0);
    group.add(makeSensorGlyph(cam, cam.index === 0, calibrated));

    // Calibrated cameras feed the single fused cloud; only an uncalibrated camera draws its
    // own (amber-tinted) cloud at its placeholder pose.
    const cloud = makeCloud(new THREE.Color(0.55, 0.42, 0.18));
    cloud.visible = opts.cloud && !calibrated;
    group.add(cloud);
    sensorRoot.add(group);

    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="feed-head"><span><b>CAM ${cam.index + 1}</b> …${cam.serial.slice(-4)}</span>` +
      `<span>${cam.width}×${cam.height}@${cam.fps}</span></div><canvas></canvas>`;
    feeds.appendChild(wrap);
    const feed = wrap.querySelector("canvas");
    feed.style.aspectRatio = `${cam.width} / ${cam.height}`;
    return { info: cam, group, cloud, feed, ctx: feed.getContext("2d"), image: null };
  });
  renderRigList();
  const views = $("view-cams");
  views.innerHTML = "";
  for (const rc of rigCams) {
    const b = document.createElement("button");
    b.textContent = `CAM ${rc.info.index + 1}`;
    b.title = "View the scene from this camera";
    b.onclick = () => viewFromCamera(rc);
    views.appendChild(b);
  }
}

function makeCloud(color) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(CLOUD_MAX * 3), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("color", new THREE.BufferAttribute(new Uint8Array(CLOUD_MAX * 3), 3, true).setUsage(THREE.DynamicDrawUsage));
  geo.setDrawRange(0, 0);
  const cloud = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.008, vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false, color }));
  cloud.frustumCulled = false;
  return cloud;
}
// Every calibrated camera merged into one cloud, already in the world frame.
const fusedCloud = makeCloud(new THREE.Color(0.32, 0.4, 0.48));
sensorRoot.add(fusedCloud);

function onCloud(cam, buf) {
  const cloud = cam === FUSED ? fusedCloud : rigCams[cam]?.cloud;
  if (!cloud) return;
  const geo = cloud.geometry;
  const total = new DataView(buf).getUint32(8, true), n = Math.min(total, CLOUD_MAX);
  const xyz = new Int16Array(buf, 12, n * 3);
  const rgb = new Uint8Array(buf, 12 + total * 6, n * 3);
  const pos = geo.attributes.position.array, col = geo.attributes.color.array;
  for (let i = 0; i < n * 3; i++) pos[i] = xyz[i] * 0.001;
  col.set(rgb);
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.setDrawRange(0, n);
  if (cam === FUSED || cam === 0) cloudCentre = medianPoint(pos, n);
}
let cloudCentre = null; // sensor-frame centre of the scene, for centring the view

function medianPoint(pos, n) {
  if (n < 50) return null;
  const step = Math.max(1, Math.floor(n / 2000));
  const axes = [[], [], []];
  for (let i = 0; i < n; i += step) for (let k = 0; k < 3; k++) axes[k].push(pos[i * 3 + k]);
  const med = (a) => a.sort((x, y) => x - y)[a.length >> 1];
  return new THREE.Vector3(med(axes[0]), med(axes[1]), med(axes[2]));
}

// ───────────────────────────── hologram mesh ─────────────────────────────
const holoMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  side: THREE.DoubleSide,
  blending: THREE.AdditiveBlending,
  uniforms: { uColor: { value: COLORS.mesh }, uTime: { value: 0 }, uOpacity: { value: 0 } },
  vertexShader: /* glsl */ `
    varying vec3 vN; varying vec3 vV; varying float vY;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vN = normalize(normalMatrix * normal);
      vV = normalize(-mv.xyz);
      vY = (modelMatrix * vec4(position, 1.0)).y;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor; uniform float uTime; uniform float uOpacity;
    varying vec3 vN; varying vec3 vV; varying float vY;
    void main() {
      float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4);
      float scan = smoothstep(0.44, 0.5, abs(fract(vY * 22.0 - uTime * 0.9) - 0.5));
      float sweep = exp(-pow((fract(uTime * 0.25) * 2.4 - vY) * 8.0, 2.0));
      float legs = smoothstep(1.1, 0.7, vY);   // brighter below the hips
      float a = (0.035 + f * 0.75 + scan * 0.10 + sweep * 0.25) * (0.55 + 0.45 * legs) * uOpacity;
      gl_FragColor = vec4(uColor * (0.5 + f * 1.5 + sweep), a);
    }`,
});
const meshGeo = new THREE.BufferGeometry();
const holo = glow(new THREE.Mesh(meshGeo, holoMat));
holo.frustumCulled = false;
holo.visible = false;
sensorRoot.add(holo);
let meshReady = false;
let lastMeshAt = 0;

async function loadFaces() {
  const r = await fetch("/api/mesh/faces");
  if (!r.ok) return;
  meshGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(await r.arrayBuffer()), 1));
  meshReady = true;
}

function onMesh(buf) {
  if (!meshReady) return;
  const n = new DataView(buf).getUint32(8, true);
  const src = new Int16Array(buf, 12, n * 3);
  let attr = meshGeo.attributes.position;
  if (!attr || attr.count !== n) {
    attr = new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage);
    meshGeo.setAttribute("position", attr);
  }
  for (let i = 0; i < n * 3; i++) attr.array[i] = src[i] * 0.001;
  attr.needsUpdate = true;
  meshGeo.computeVertexNormals();
  lastMeshAt = performance.now();
}

// ───────────────────────────── skeleton ─────────────────────────────
let skel = null; // built on hello

function sideOf(name) {
  if (name.startsWith("left_")) return "left";
  if (name.startsWith("right_")) return "right";
  return "center";
}

function buildSkeleton(hello) {
  if (skel) removeWithLabels(skel.group);
  const group = new THREE.Group();
  const legSet = new Set(hello.focus_joints); // the procedure's target: leg or chest
  const joints = {};
  const sphere = new THREE.SphereGeometry(1, 20, 14);
  for (const name of hello.joints) {
    const leg = legSet.has(name);
    const color = leg ? COLORS[sideOf(name)] : COLORS.upper;
    const m = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ color, transparent: true }));
    m.scale.setScalar(leg ? (name.includes("toe") || name.includes("heel") ? 0.014 : 0.024) : 0.016);
    m.userData = { leg, name };
    m.visible = false;
    const halo = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.12, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.setScalar(2.2);
    m.add(halo);
    group.add(m);
    joints[name] = m;
  }
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true);
  const bones = hello.bones.map(([a, b]) => {
    const leg = legSet.has(a) && legSet.has(b) && !(hello.region === "leg" && a === "pelvis" && b === "neck");
    const s = sideOf(b) === "center" ? sideOf(a) : sideOf(b);
    const color = leg ? COLORS[s] : COLORS.upper;
    const m = new THREE.Mesh(cyl, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: leg ? 0.95 : 0.55 }));
    m.userData = { a, b, leg, radius: leg ? 0.009 : 0.005 };
    m.visible = false;
    group.add(m);
    return m;
  });

  const tags = {};
  for (const side of ["left", "right"]) {
    for (const j of ["hip", "knee", "ankle"]) {
      const el = document.createElement("div");
      el.className = "tag";
      el.style.color = side === "left" ? "#22e4ff" : "#ff3df2";
      const obj = new CSS2DObject(el);
      // CSS2DRenderer owns the element's transform; offset the tag via its anchor instead.
      obj.center.set(side === "left" ? -0.2 : 1.2, 0.5);
      joints[`${side}_${j}`].add(obj);
      tags[`${side}_${j}`] = { el, obj };
    }
  }
  glow(group);
  sensorRoot.add(group);
  skel = { group, joints, bones, tags };
}

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _up = new THREE.Vector3(0, 1, 0);
function updateSkeleton(person) {
  if (!skel) return;
  const J = person ? person.joints : {};
  for (const [name, m] of Object.entries(skel.joints)) {
    const j = J[name];
    m.visible = !!j && opts.skeleton && (m.userData.leg || opts.upper);
    if (j) {
      m.position.set(j[0], j[1], j[2]);
      m.material.opacity = j[3] < 0.5 ? 0.35 : 1.0;
    }
  }
  for (const m of skel.bones) {
    const ja = J[m.userData.a], jb = J[m.userData.b];
    const show = !!(ja && jb) && opts.skeleton && (m.userData.leg || opts.upper);
    m.visible = show;
    if (!show) continue;
    _a.set(ja[0], ja[1], ja[2]);
    _b.set(jb[0], jb[1], jb[2]);
    const len = _a.distanceTo(_b);
    m.position.copy(_a).add(_b).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(_up, _b.sub(_a).normalize());
    m.scale.set(m.userData.radius, len, m.userData.radius);
  }
  const angles = person ? person.angles : {};
  for (const [key, t] of Object.entries(skel.tags)) {
    const v = angles[key];
    t.obj.visible = opts.labels && v != null && skel.joints[key].visible;
    if (v != null) t.el.innerHTML = `<small>${key.split("_")[1].toUpperCase()}</small>${v.toFixed(0)}°`;
  }
}

// ───────────────────────────── trails ─────────────────────────────
function makeTrail(color, n, width = 1) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage));
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const k = Math.pow(i / (n - 1), 1.6);
    c[i * 3] = color.r * k; c[i * 3 + 1] = color.g * k; c[i * 3 + 2] = color.b * k;
  }
  g.setAttribute("color", new THREE.BufferAttribute(c, 3));
  g.setDrawRange(0, 0);
  const line = glow(new THREE.Line(g, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, linewidth: width })));
  line.frustumCulled = false;
  sensorRoot.add(line);
  const trail = {
    line, n, count: 0,
    push(p) {
      const a = g.attributes.position.array;
      a.copyWithin(0, 3);
      a.set(p, (n - 1) * 3);
      trail.count = Math.min(trail.count + 1, n);
      g.setDrawRange(n - trail.count, trail.count); // only the filled tail of the buffer
      g.attributes.position.needsUpdate = true;
    },
    clear() { trail.count = 0; g.setDrawRange(0, 0); },
  };
  return trail;
}
const footTrails = { left: makeTrail(COLORS.left, 90), right: makeTrail(COLORS.right, 90) };
function updateFootTrails(person) {
  for (const side of ["left", "right"]) {
    const J = person?.joints ?? {};
    const j = J[`${side}_heel`] ?? J[`${side}_ankle`];
    if (j) footTrails[side].push([j[0], j[1], j[2]]);
  }
}

// ───────────────────────────── probe ─────────────────────────────
// probeGroup's matrix is the cube pose (object → world); children live in the cube frame.
const probeGroup = new THREE.Group();
probeGroup.matrixAutoUpdate = false;
// Until the tags are first seen, park the probe where DataCollection's viewer starts:
// 0.6 m in front of CAM 1, tag 0 (+Z face) towards the camera, probe pointing up.
probeGroup.matrix.set(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0.6, 0, 0, 0, 1);
probeGroup.visible = false;
sensorRoot.add(probeGroup);
const tipTrail = makeTrail(COLORS.tip, 240);
const probeState = { info: null, lastSeen: 0, tipLabel: null, method: null, materials: [], live: null };

function probeMaterial(mat) {
  mat.transparent = true;
  mat.userData.baseOpacity = mat.opacity;
  probeState.materials.push(mat);
  return mat;
}
// Tracked: solid. Not tracked (never seen, or tags lost): ghosted at the last known pose.
function setProbeLive(live) {
  if (live === probeState.live) return;
  probeState.live = live;
  for (const m of probeState.materials) m.opacity = m.userData.baseOpacity * (live ? 1 : 0.3);
}

const nearestLine = glow(new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: COLORS.gold, dashSize: 0.01, gapSize: 0.006, transparent: true, opacity: 0.9 })));
nearestLine.frustumCulled = false;
nearestLine.visible = false;
sensorRoot.add(nearestLine);

function buildProbe(info) {
  for (const child of [...probeGroup.children]) removeWithLabels(child);
  probeState.materials = [];
  probeState.live = null;
  probeState.info = info;
  if (!info) return;

  new GLTFLoader().load(info.model_url, (gltf) => {
    const model = gltf.scene;
    model.traverse((o) => {
      if (!o.isMesh) return;
      o.geometry.computeVertexNormals(); // exported without normals
      o.material = probeMaterial(new THREE.MeshStandardMaterial({
        color: "#9fb0c2", metalness: 0.35, roughness: 0.42, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      }));
      // Faint cyan edges read as "tracked object" and survive the dark scene.
      const edges = glow(new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 35),
        probeMaterial(new THREE.LineBasicMaterial({ color: "#22e4ff", opacity: 0.35 }))));
      o.add(edges);
    });
    probeGroup.add(model);
    const live = probeState.live;
    probeState.live = null;
    setProbeLive(!!live); // apply the current state to the freshly loaded materials
  }, undefined, (err) => console.warn("probe model failed to load", err));

  // ArUco tags on their cube faces (texture includes the printed white margin).
  const loader = new THREE.TextureLoader();
  for (const m of info.mounts) {
    const tex = loader.load(info.tag_url.replace("{id}", m.id));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.NearestFilter; // keep the code cells sharp
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(info.tag_size, info.tag_size),
      probeMaterial(new THREE.MeshBasicMaterial({ map: tex, color: new THREE.Color(0.78, 0.78, 0.78), polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 })));
    const x = new THREE.Vector3(...m.x_axis), y = new THREE.Vector3(...m.y_axis);
    const z = new THREE.Vector3().crossVectors(x, y);
    plane.matrix.makeBasis(x, y, z).setPosition(new THREE.Vector3(...m.center).addScaledVector(z, 0.0004));
    plane.matrixAutoUpdate = false;
    plane.userData.markerId = m.id;
    probeGroup.add(plane);
  }

  // The tip: a red dot with a halo and a label.
  const tip = new THREE.Group();
  tip.position.set(...info.tip);
  tip.add(glow(new THREE.Mesh(new THREE.SphereGeometry(0.0045, 20, 14), probeMaterial(new THREE.MeshBasicMaterial({ color: COLORS.tip })))));
  tip.add(glow(new THREE.Mesh(new THREE.SphereGeometry(0.011, 20, 14), probeMaterial(new THREE.MeshBasicMaterial({
    color: COLORS.tip, opacity: 0.18, depthWrite: false, blending: THREE.AdditiveBlending })))));
  const el = document.createElement("div");
  el.className = "tag";
  el.style.color = "#ff5a5a";
  const label = new CSS2DObject(el);
  label.center.set(-0.15, 0.5);
  tip.add(label);
  probeState.tipLabel = { el, obj: label };
  probeGroup.add(tip);

  // Cube axes, as in DataCollection's viewer (X red, Y green, Z blue).
  const axes = new THREE.AxesHelper(0.06);
  probeMaterial(axes.material);
  probeGroup.add(axes);
  setProbeMethod(info.method);
  setProbeLive(false);
  probeState.tipLabel.el.innerHTML = "<small>PROBE</small>NOT TRACKED";
}

const _m4 = new THREE.Matrix4();
function updateProbe(probe) {
  const panel = $("probe-state");
  if (!probe) {
    panel.textContent = probeState.info ? "NO DATA" : "DISABLED";
    return;
  }
  setProbeMethod(probe.method);
  const now = performance.now();
  const chips = new Set(probe.tracked ? probe.marker_ids : []);
  // Tags a camera saw but dropped as inconsistent with the others (bad detection / loose tag).
  const bad = new Set((lastFrame?.views ?? []).flatMap((v) => v.rejected_tags ?? []));
  document.querySelectorAll("#tag-chips span").forEach((s) => {
    s.classList.toggle("on", chips.has(+s.dataset.tag));
    s.classList.toggle("bad", bad.has(+s.dataset.tag) && !chips.has(+s.dataset.tag));
  });
  const fit = probe.fit;
  $("probe-fit").innerHTML = fit
    ? `FIT <b>${fit.rms_px.toFixed(2)} px</b> · ${fit.corners} corners` +
      (fit.depth_samples ? ` · depth <b>±${fit.depth_rms_mm?.toFixed(0) ?? "—"} mm</b>` : "") +
      (fit.rig_error_px != null
        ? `<br><span class="warn">CAMERAS DISAGREE ${fit.rig_error_px.toFixed(0)} px · ONE CAMERA IN USE — CALIBRATE RIG</span>`
        : "")
    : "";
  $("probe-cams").textContent = probe.tracked ? `${probe.cameras} CAM${probe.cameras === 1 ? "" : "S"}` : "";
  panel.textContent = probe.tracked ? "TRACKED" : "SEARCHING";
  panel.style.color = probe.tracked ? "var(--ok)" : "var(--warn)";

  probeState.tracked = probe.tracked;
  if (!probe.tracked) {
    nearestLine.visible = false;
    return;
  }
  probeState.lastSeen = now;
  const [r0, r1, r2, r3, r4, r5, r6, r7, r8] = probe.rotation;
  const [px, py, pz] = probe.position;
  _m4.set(r0, r1, r2, px, r3, r4, r5, py, r6, r7, r8, pz, 0, 0, 0, 1);
  probeGroup.matrix.copy(_m4);
  probeGroup.matrixWorldNeedsUpdate = true;
  tipTrail.push(probe.tip);

  // Tip in the levelled scene frame: y is height above the floor.
  const tipScene = sensorRoot.localToWorld(new THREE.Vector3(...probe.tip));
  $("tip-x").textContent = (tipScene.x * 1000).toFixed(0);
  $("tip-y").textContent = (tipScene.y * 1000).toFixed(0);
  $("tip-z").textContent = (tipScene.z * 1000).toFixed(0);

  const near = probe.nearest;
  if (near) {
    const [side, seg] = near.segment.split("_");
    $("nearest-seg").textContent = `${side.toUpperCase()} ${SEGMENT_LABELS[seg]}`;
    $("nearest-seg").style.color = side === "left" ? "#22e4ff" : "#ff3df2";
    $("nearest-dist").textContent = `${near.distance_mm.toFixed(0)} mm`;
    const pos = nearestLine.geometry.attributes.position;
    pos.setXYZ(0, ...probe.tip);
    pos.setXYZ(1, ...near.point);
    pos.needsUpdate = true;
    nearestLine.computeLineDistances();
    nearestLine.visible = opts.probe && opts.skeleton;
  } else {
    $("nearest-seg").textContent = "—";
    $("nearest-dist").textContent = "—";
    nearestLine.visible = false;
  }
  if (probeState.tipLabel) {
    probeState.tipLabel.el.innerHTML = near
      ? `<small>TIP</small>${near.distance_mm.toFixed(0)} mm · ${near.segment.replace("_", " ").toUpperCase()}`
      : "<small>TIP</small>";
  }
}

function setProbeMethod(method) {
  if (method === probeState.method) return;
  probeState.method = method;
  document.querySelectorAll("#probe-filter button").forEach((b) => b.classList.toggle("on", b.dataset.method === method));
}
document.querySelectorAll("#probe-filter button").forEach((b) => {
  b.onclick = () => send({ cmd: "probe_filter", method: b.dataset.method });
});
$("probe-reset").onclick = () => { send({ cmd: "probe_reset" }); tipTrail.clear(); };
// ───────────────────────────── scene levelling ─────────────────────────────
const _n = new THREE.Vector3();
function updateLevel(frame) {
  if (frame.floor) {
    const [nx, ny, nz] = frame.floor.normal;
    _n.set(nx, -ny, -nz).normalize(); // optical frame → sensorRoot parent frame
    levelTarget.quat.setFromUnitVectors(_n, _up);
    levelTarget.height = frame.floor.height;
    levelTarget.known = true;
  } else if (frame.person && !levelTarget.known) {
    // No plane from depth: put the floor under the lowest foot point.
    const J = frame.person.joints;
    let lowest = -Infinity;
    for (const k of ["left_heel", "right_heel", "left_big_toe", "right_big_toe"]) if (J[k]) lowest = Math.max(lowest, J[k][1]);
    if (lowest === -Infinity) for (const k of ["left_ankle", "right_ankle"]) if (J[k]) lowest = Math.max(lowest, J[k][1] + 0.08);
    if (lowest > -Infinity) levelTarget.height += (lowest - levelTarget.height) * 0.05;
  }
}

// ───────────────────────────── HUD ─────────────────────────────
function $(id) { return document.getElementById(id); }
const angleCells = [...document.querySelectorAll("[data-angle]")];
const spark = $("spark"), sparkCtx = spark.getContext("2d");
const history = [];
let lastFrame = null, hello = null;

function setStatus(state, message) {
  const dot = $("status-dot");
  dot.className = "dot " + ({ running: "ok", loading: "warn", starting: "warn", error: "err", offline: "err" }[state] ?? "");
  $("status-state").textContent = { running: "ONLINE", loading: "BOOTING", starting: "BOOTING", error: "FAULT", offline: "NO LINK" }[state] ?? state.toUpperCase();
  const msg = $("status-msg");
  msg.textContent = message ?? "";
  msg.classList.toggle("err", state === "error" || state === "offline");
}

function updateHud(frame) {
  const fpsCell = (id, v) => {
    $(id).textContent = v.toFixed(1);
    $(id).style.color = v >= 29 ? "" : "var(--warn)"; // target is ≥30 fps
  };
  fpsCell("stat-fps", frame.fps);
  fpsCell("stat-body-fps", frame.body_fps);
  $("stat-latency").textContent = `${frame.latency_ms.toFixed(0)} ms`;
  $("stat-pose").textContent = `${frame.timings.pose_ms.toFixed(0)} ms`;
  $("stat-probe").textContent = `${frame.timings.probe_ms.toFixed(0)} ms`;
  const age = frame.timings.detection_age_ms;
  $("stat-detect").textContent = age == null ? "—" : `${age} ms`;
  $("stat-people").textContent = frame.people;
  const subj = $("subject-state");
  const target = (hello?.region ?? "subject").toUpperCase();
  subj.textContent = frame.person ? `${target} LOCK` : `NO ${target}`;
  subj.classList.toggle("on", !!frame.person);

  for (const v of frame.views) {
    const stateEl = document.querySelector(`#rig-list li[data-cam="${v.cam}"] .state`);
    if (stateEl) {
      const reconnecting = v.state === "reconnecting";
      stateEl.classList.toggle("err", reconnecting);
      if (reconnecting) stateEl.textContent = "RECONNECTING";
      else if (stateEl.textContent === "RECONNECTING") renderRigList();
    }
  }

  const angles = frame.person?.angles ?? {};
  for (const cell of angleCells) {
    const v = angles[cell.dataset.angle];
    cell.classList.toggle("none", v == null);
    cell.querySelector("b").textContent = v == null ? "—" : v.toFixed(0);
    cell.querySelector("i").style.width = v == null ? "0" : `${Math.min(Math.abs(v) / 150, 1) * 100}%`;
  }
  const now = performance.now() / 1000;
  history.push([now, angles.left_knee ?? null, angles.right_knee ?? null]);
  while (history.length && now - history[0][0] > 10) history.shift();
  drawSpark(now);

  if (frame.calibration) {
    const { progress, target } = frame.calibration;
    const others = rigCams.slice(1).map((rc) => progress[rc.info.serial] ?? 0);
    const pct = others.length ? Math.min(...others) / target : 0;
    $("cal-progress").hidden = false;
    $("cal-progress").firstElementChild.style.width = `${Math.round(pct * 100)}%`;
    rigMsg(`Collecting still views… ${rigCams.slice(1).map((rc) => `CAM ${rc.info.index + 1}: ${progress[rc.info.serial] ?? 0}/${target}`).join(" · ")}`);
  }
}

function drawSpark(now) {
  const w = spark.clientWidth, h = spark.clientHeight, dpr = devicePixelRatio;
  if (spark.width !== w * dpr) { spark.width = w * dpr; spark.height = h * dpr; }
  const c = sparkCtx;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.strokeStyle = "rgba(160,210,255,0.08)";
  c.lineWidth = 1;
  for (const deg of [0, 45, 90, 135]) {
    const y = h - (deg / 140) * h;
    c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke();
  }
  c.fillStyle = "rgba(111,139,168,0.7)";
  c.font = "9px JetBrains Mono, monospace";
  c.fillText("90°", 2, h - (90 / 140) * h - 3);
  for (const [idx, color] of [[1, "#22e4ff"], [2, "#ff3df2"]]) {
    c.strokeStyle = color;
    c.shadowColor = color;
    c.shadowBlur = 6;
    c.lineWidth = 1.5;
    c.beginPath();
    let pen = false;
    for (const s of history) {
      const v = s[idx];
      if (v == null) { pen = false; continue; }
      const x = w - ((now - s[0]) / 10) * w;
      const y = h - (Math.max(0, Math.min(v, 140)) / 140) * h;
      pen ? c.lineTo(x, y) : c.moveTo(x, y);
      pen = true;
    }
    c.stroke();
  }
  c.shadowBlur = 0;
}

function drawFeed(rc) {
  if (!rc.image) return;
  const w = rc.feed.clientWidth, h = rc.feed.clientHeight, dpr = devicePixelRatio;
  if (!w || !h) return; // panel collapsed
  if (rc.feed.width !== Math.round(w * dpr)) { rc.feed.width = Math.round(w * dpr); rc.feed.height = Math.round(h * dpr); }
  const c = rc.ctx;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.drawImage(rc.image, 0, 0, w, h);
  c.fillStyle = "rgba(2,6,12,0.25)";
  c.fillRect(0, 0, w, h);
  const view = lastFrame?.views?.[rc.info.index];
  if (!view) return;

  if (view.bbox) {
    const [x1, y1, x2, y2] = view.bbox;
    c.strokeStyle = "rgba(34,228,255,0.7)";
    c.lineWidth = 1;
    const bw = (x2 - x1) * w, bh = (y2 - y1) * h, L = Math.min(14, bw / 4, bh / 4);
    for (const [cx, cy, sx, sy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]]) {
      c.beginPath();
      c.moveTo(cx * w + sx * L, cy * h); c.lineTo(cx * w, cy * h); c.lineTo(cx * w, cy * h + sy * L);
      c.stroke();
    }
  }
  const K = view.kp2d;
  if (K) {
    const legSet = new Set(hello?.focus_joints ?? []);
    for (const [a, b] of hello?.bones ?? []) {
      if (!K[a] || !K[b]) continue;
      const leg = legSet.has(a) && legSet.has(b);
      if (!leg && !opts.upper) continue;
      const side = sideOf(b) === "center" ? sideOf(a) : sideOf(b);
      c.strokeStyle = leg ? (side === "left" ? "#22e4ff" : side === "right" ? "#ff3df2" : "#8fa8ff") : "rgba(143,168,255,0.5)";
      c.lineWidth = leg ? 2 : 1;
      c.beginPath(); c.moveTo(K[a][0] * w, K[a][1] * h); c.lineTo(K[b][0] * w, K[b][1] * h); c.stroke();
    }
    for (const [name, [u, v]] of Object.entries(K)) {
      if (!legSet.has(name) && !opts.upper) continue;
      c.fillStyle = "#fff";
      c.beginPath(); c.arc(u * w, v * h, 2, 0, Math.PI * 2); c.fill();
    }
  }
  // ArUco outlines (first corner marked, like cv2.aruco.drawDetectedMarkers) and the tip dot.
  for (const m of view.markers ?? []) {
    c.strokeStyle = "#3dffa8";
    c.lineWidth = 1.5;
    c.beginPath();
    m.corners.forEach(([u, v], i) => (i ? c.lineTo(u * w, v * h) : c.moveTo(u * w, v * h)));
    c.closePath();
    c.stroke();
    c.fillStyle = "#ff3d6e";
    c.fillRect(m.corners[0][0] * w - 2, m.corners[0][1] * h - 2, 4, 4);
    const cx = m.corners.reduce((s, p) => s + p[0], 0) / 4 * w, cy = m.corners.reduce((s, p) => s + p[1], 0) / 4 * h;
    c.fillStyle = "#3dffa8";
    c.font = "bold 10px JetBrains Mono, monospace";
    c.fillText(m.id, cx - 3, cy + 4);
  }
  if (view.tip && lastFrame?.probe?.tracked) {
    const [u, v] = view.tip;
    c.fillStyle = "#ff2a2a";
    c.shadowColor = "#ff2a2a";
    c.shadowBlur = 10;
    c.beginPath(); c.arc(u * w, v * h, 4.5, 0, Math.PI * 2); c.fill();
    c.shadowBlur = 0;
  }
}

// ───────────────────────────── rig panel ─────────────────────────────
function renderRigList() {
  const list = $("rig-list");
  list.innerHTML = "";
  for (const rc of rigCams) {
    const cam = rc.info;
    const li = document.createElement("li");
    li.dataset.cam = cam.index;
    const world = cam.index === 0;
    const ok = world || cam.T_world_camera;
    li.innerHTML = `<span class="dot ${ok ? "ok" : "idle"}"></span>` +
      `<b>CAM ${cam.index + 1} · …${cam.serial.slice(-4)}</b>` +
      `<span class="state ${ok ? "" : "warn"}">${world ? "WORLD" : ok ? "CALIBRATED" : "UNCALIBRATED"}</span>` +
      `<small>${cam.name} · ${cam.width}×${cam.height}@${cam.fps} · USB ${cam.usb}</small>`;
    list.appendChild(li);
  }
  $("rig-meta").textContent = `${rigCams.length} CAM${rigCams.length === 1 ? "" : "S"}`;
  const calBtn = $("rig-calibrate");
  calBtn.disabled = rigCams.length < 2 || !probeState.info;
  calBtn.title = rigCams.length < 2 ? "Connect a second camera to calibrate the rig" : "";
  if (rigCams.length < 2) rigMsg("Single camera — connect another RealSense to fuse views.");
  else if (rigCams.slice(1).some((rc) => !rc.info.T_world_camera))
    rigMsg("Hold the probe where every camera sees its tags, then press CALIBRATE RIG.");
}

function updateRigHealth(frame) {
  const health = frame.rig_health ?? {};
  for (const rc of rigCams) {
    const li = document.querySelector(`#rig-list li[data-cam="${rc.info.index}"]`);
    if (!li || rc.info.index === 0) continue;
    let el = li.querySelector(".health");
    if (!el) { el = document.createElement("small"); el.className = "health"; li.appendChild(el); }
    const h = health[rc.info.serial];
    el.className = "health " + (h?.state ?? "");
    el.textContent = h
      ? `agrees with CAM 1 to ${h.mm.toFixed(0)} mm / ${h.deg.toFixed(1)}° (${h.state.toUpperCase()})`
      : rc.info.T_world_camera ? "agreement: show the probe to both cameras" : "";
  }
  if (rigCams.length > 1) $("rig-meta").textContent = `${rigCams.length} CAMS · SYNC ${frame.sync_ms?.toFixed(0) ?? "—"} MS`;
}

let calibrating = false;
function rigMsg(text, kind = "") {
  const el = $("rig-msg");
  el.textContent = text;
  el.className = "rig-msg " + kind;
}
$("rig-calibrate").onclick = () => {
  if (calibrating) { send({ cmd: "cancel_calibration" }); return; }
  send({ cmd: "calibrate_rig" });
};
function onCalibration(msg) {
  calibrating = msg.state === "collecting";
  $("rig-calibrate").textContent = calibrating ? "CANCEL" : "CALIBRATE RIG";
  $("cal-progress").hidden = !calibrating;
  if (!calibrating) $("cal-progress").firstElementChild.style.width = "0";
  rigMsg(msg.message, { done: "ok", failed: "err" }[msg.state] ?? "");
}

// ───────────────────────────── depth window ─────────────────────────────
// Only pixels whose depth (from their own camera) is inside [min, max] reach the cloud,
// and optionally the feeds. Applied on the server so every camera and view agrees.
const depthUI = {
  min: $("depth-min"), max: $("depth-max"), minNum: $("depth-min-num"), maxNum: $("depth-max-num"),
  mask: $("depth-mask"), limits: [0.2, 6.0], dragging: false, sendTimer: null, localUntil: 0,
};
function depthValues() { return [parseFloat(depthUI.min.value), parseFloat(depthUI.max.value)]; }
function renderDepthUI() {
  const [lo, hi] = depthValues();
  const [a, b] = depthUI.limits;
  const fill = $("depth-fill");
  fill.style.left = `${((lo - a) / (b - a)) * 100}%`;
  fill.style.right = `${(1 - (hi - a) / (b - a)) * 100}%`;
  if (document.activeElement !== depthUI.minNum) depthUI.minNum.value = lo.toFixed(2);
  if (document.activeElement !== depthUI.maxNum) depthUI.maxNum.value = hi.toFixed(2);
  const full = lo <= a + 1e-3 && hi >= b - 1e-3;
  $("depth-readout").textContent = full ? "ALL" : `${lo.toFixed(2)}–${hi.toFixed(2)} M`;
  $("depth-readout").style.color = full ? "" : "var(--cyan)";
}
function setDepth(lo, hi, { send: doSend = true } = {}) {
  const [a, b] = depthUI.limits;
  lo = Math.min(Math.max(lo, a), b);
  hi = Math.min(Math.max(hi, a), b);
  if (hi - lo < 0.02) hi = Math.min(lo + 0.02, b); // never collapse to an empty window
  depthUI.min.value = lo;
  depthUI.max.value = hi;
  renderDepthUI();
  if (!doSend) return;
  store.set("depth", { min: lo, max: hi, mask: depthUI.mask.checked });
  depthUI.localUntil = performance.now() + 500; // let the server catch up before syncing back
  // Throttle while dragging: at most one command every 50 ms, last value always sent.
  clearTimeout(depthUI.sendTimer);
  depthUI.sendTimer = setTimeout(() => send({ cmd: "depth_range", min: lo, max: hi, mask_feeds: depthUI.mask.checked }), 50);
}
depthUI.min.addEventListener("input", () => {
  const [lo, hi] = depthValues();
  setDepth(Math.min(lo, hi - 0.02), hi);
});
depthUI.max.addEventListener("input", () => {
  const [lo, hi] = depthValues();
  setDepth(lo, Math.max(hi, lo + 0.02));
});
for (const el of [depthUI.min, depthUI.max]) {
  el.addEventListener("pointerdown", () => (depthUI.dragging = true));
  el.addEventListener("pointerup", () => (depthUI.dragging = false));
}
depthUI.minNum.addEventListener("change", () => setDepth(parseFloat(depthUI.minNum.value) || 0, depthValues()[1]));
depthUI.maxNum.addEventListener("change", () => setDepth(depthValues()[0], parseFloat(depthUI.maxNum.value) || 0));
depthUI.mask.addEventListener("change", () => setDepth(...depthValues()));
$("depth-reset").onclick = () => setDepth(...depthUI.limits);

function initDepthUI(limits) {
  depthUI.limits = limits;
  for (const el of [depthUI.min, depthUI.max, depthUI.minNum, depthUI.maxNum]) {
    el.min = limits[0];
    el.max = limits[1];
  }
  // Restore this browser's last window (re-sent so a restarted server picks it up).
  const saved = store.get("depth", null);
  depthUI.mask.checked = !!saved?.mask;
  if (saved) setDepth(saved.min, saved.max);
  else setDepth(limits[0], limits[1], { send: false });
}
function syncDepthUI(range) {
  // Another dashboard (or a server restart) changed the window: follow it unless dragging.
  if (!range || depthUI.dragging || performance.now() < depthUI.localUntil) return;
  const [lo, hi] = depthValues();
  if (Math.abs(range.min - lo) > 0.005 || Math.abs(range.max - hi) > 0.005) setDepth(range.min, range.max, { send: false });
  if (depthUI.mask.checked !== range.mask_feeds) depthUI.mask.checked = range.mask_feeds;
}

// ───────────────────────────── clarius ultrasound ─────────────────────────────
const us = { canvas: $("us-canvas"), image: null, localUntil: 0, sendTimer: null, imaging: false };
us.ctx = us.canvas.getContext("2d");

function onUltrasound(buf) {
  createImageBitmap(new Blob([new Uint8Array(buf, 8)], { type: "image/jpeg" }))
    .then((bmp) => {
      us.image?.close?.();
      us.image = bmp;
      const c = us.canvas, w = c.clientWidth;
      if (!w) return; // panel collapsed
      c.style.aspectRatio = `${bmp.width} / ${bmp.height}`;
      if (c.width !== bmp.width) { c.width = bmp.width; c.height = bmp.height; }
      us.ctx.drawImage(bmp, 0, 0);
    })
    .catch(() => {});
}

function updateClarius(c) {
  if (!c) return;
  const dot = $("cl-dot");
  dot.className = "dot " + (c.imaging ? "ok" : c.connected ? "warn" : "err");
  $("cl-state").textContent = c.state ?? "—";
  const meter = (id, v, bad) => {
    $(id).textContent = v == null ? "—" : `${v}%`;
    const bar = $(id + "-bar");
    bar.style.width = v == null ? "0" : `${Math.max(0, Math.min(100, v))}%`;
    bar.style.background = v == null ? "" : bad(v) ? "var(--err)" : bad(v + 15) ? "var(--warn)" : "var(--ok)";
  };
  meter("cl-batt", c.battery, (v) => v < 15);
  meter("cl-temp", c.temperature, (v) => v > 85);
  if (c.charging) $("cl-batt").textContent += " ⚡";
  $("cl-fps").textContent = c.fps ? c.fps.toFixed(0) : "—";
  $("cl-err").textContent = c.error ?? "";
  $("us-meta").textContent = c.imaging ? `${c.image_size?.[0] ?? ""}×${c.image_size?.[1] ?? ""}` : (c.connected ? "FROZEN" : "OFFLINE");
  us.imaging = !!c.imaging;
  $("us-run").textContent = c.imaging ? "FREEZE" : "RUN";
  for (const id of ["us-depth", "us-gain", "us-run"]) $(id).disabled = !c.connected; // nothing to control yet
  // Sliders follow the probe (its ranges and current values) unless you're adjusting them.
  if (performance.now() > us.localUntil) {
    for (const [name, value, range] of [["depth", c.depth_cm, c.depth_range], ["gain", c.gain, c.gain_range]]) {
      const el = $(`us-${name}`);
      if (range) { el.min = range[0]; el.max = range[1]; }
      if (value != null) { el.value = value; $(`us-${name}-v`).textContent = (+value).toFixed(name === "depth" ? 1 : 0); }
    }
  }
}

for (const name of ["depth", "gain"]) {
  const el = $(`us-${name}`);
  el.addEventListener("input", () => {
    us.localUntil = performance.now() + 1500;
    $(`us-${name}-v`).textContent = (+el.value).toFixed(name === "depth" ? 1 : 0);
    clearTimeout(us.sendTimer);
    us.sendTimer = setTimeout(() => send({ cmd: "clarius_param", name, value: +el.value }), 120);
  });
}
$("us-run").onclick = () => send({ cmd: "clarius_run", run: !us.imaging });

// ───────────────────────────── views ─────────────────────────────
const followTarget = new THREE.Vector3(0, 0.9, -2.5);
function setView(kind) {
  camera.fov = 50;
  camera.updateProjectionMatrix();
  const t = followTarget;
  const offsets = {
    reset: new THREE.Vector3(1.9, 0.9, 2.6),
    side: new THREE.Vector3(3.2, 0.1, 0.001),
    top: new THREE.Vector3(0.001, 4.5, 0.4),
  };
  camera.position.copy(t).add(offsets[kind]);
  controls.target.copy(t);
}
function viewFromCamera(rc) {
  // Stand where the camera is, look down its optical axis, with its vertical field of view.
  setFollow(false);
  scene.updateMatrixWorld(true);
  const m = rc.group.matrixWorld;
  const pos = new THREE.Vector3().setFromMatrixPosition(m);
  const fwd = new THREE.Vector3(0, 0, 1).transformDirection(m);
  camera.position.copy(pos);
  controls.target.copy(pos).addScaledVector(fwd, 1.5);
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(rc.info.height / 2 / rc.info.fy));
  camera.updateProjectionMatrix();
}
function setFollow(on) {
  const el = document.querySelector('[data-toggle="follow"]');
  el.checked = opts.follow = on;
  store.set("t.follow", on);
}
$("view-reset").onclick = () => setView("reset");
$("view-side").onclick = () => setView("side");
$("view-top").onclick = () => setView("top");
setView("reset");

function applyVisibility() {
  fusedCloud.visible = opts.cloud;
  for (const rc of rigCams) rc.cloud.visible = opts.cloud && !rc.info.T_world_camera;
  floor.visible = opts.grid;
  footTrails.left.line.visible = footTrails.right.line.visible = opts.trails;
  tipTrail.line.visible = opts.tiptrail && opts.probe;
  controls.autoRotate = opts.orbit;
  if (lastFrame) updateSkeleton(lastFrame.person);
}
applyVisibility();

// ───────────────────────────── websocket ─────────────────────────────
let socket = null;
let retry = 0;
function send(msg) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
}
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket = ws;
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { retry = 0; };
  ws.onclose = () => {
    setStatus("offline", "Pipeline unreachable — retrying…");
    setTimeout(connect, Math.min(500 * 2 ** retry++, 5000));
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return onBinary(ev.data);
    const msg = JSON.parse(ev.data);
    if (msg.type === "hello") onHello(msg);
    else if (msg.type === "status") { setStatus(msg.state, msg.message); onBoot(msg); }
    else if (msg.type === "calibration") onCalibration(msg);
    else if (msg.type === "frame") onFrame(msg);
  };
}

// ───────────────────────────── startup overlay ─────────────────────────────
const PROC_ICONS = {
  cardiac: '<path d="M16 27 C6 20 3 14 3 10 A6.5 6.5 0 0 1 16 8 A6.5 6.5 0 0 1 29 10 C29 14 26 20 16 27 Z"/><path d="M6 15 H11 L13 11 L16 19 L18 14 H26"/>',
  lower_limb: '<path d="M12 3 L13 13 L11 22 L11 27 L19 29"/><path d="M19 3 L18 13 L17 22 L18 25"/><circle cx="12.5" cy="13" r="1.6"/>',
};
const CHECK_ICONS = { pending: "○", active: "", done: "✓", warn: "!", error: "✕" };
let bootState = null;

function onBoot(msg) {
  const boot = $("boot"), prev = bootState;
  bootState = msg.state;
  const show = msg.state === "select" || msg.state === "loading" || (msg.state === "error" && msg.checks);
  if (!show) {
    if (!boot.hidden && !boot.classList.contains("leaving")) {
      boot.classList.add("leaving"); // fade out once everything is up
      setTimeout(() => { boot.hidden = true; boot.classList.remove("leaving"); }, 600);
    }
    return;
  }
  boot.hidden = false;
  const selecting = msg.state === "select";
  $("boot-choices").hidden = !selecting;
  $("boot-progress").hidden = selecting;
  if (selecting) {
    $("boot-sub").textContent = "SELECT PROCEDURE";
    const box = $("boot-choices");
    if (prev !== "select") {
      box.replaceChildren(); // fresh buttons each time the server asks
      for (const p of msg.procedures) {
        const b = document.createElement("button");
        b.className = "boot-choice";
        b.innerHTML = `<svg viewBox="0 0 32 32" aria-hidden="true">${PROC_ICONS[p.id] ?? ""}</svg>` +
          `<span class="name">${p.label}</span><span class="preset">PROBE PRESET · ${p.application.toUpperCase()}</span>`;
        b.onclick = () => {
          box.querySelectorAll("button").forEach((x) => (x.disabled = true));
          b.classList.add("picked");
          send({ cmd: "start", procedure: p.id });
        };
        box.append(b);
      }
    }
    return;
  }
  $("boot-sub").textContent = `${(msg.procedure ?? "").toUpperCase()} · ${msg.state === "error" ? "STARTUP FAILED" : "INITIALISING"}`;
  if (msg.progress != null) {
    const pct = Math.round(msg.progress * 100);
    $("boot-bar").style.width = `${pct}%`;
    $("boot-pct").textContent = `${pct}%`;
    const s = Math.ceil(msg.eta_s);
    $("boot-eta").textContent = s <= 0 ? "finishing…" : `~${s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s} s`} left`;
  }
  $("boot").classList.toggle("failed", msg.state === "error");
  $("boot-err").hidden = msg.state !== "error";
  $("boot-err").textContent = msg.state === "error" ? msg.message : "";
  $("boot-checks").replaceChildren(...(msg.checks ?? []).map((c) => {
    const li = document.createElement("li");
    li.className = c.state;
    li.innerHTML = `<span class="ico">${CHECK_ICONS[c.state] ?? ""}</span><span class="lbl"></span><span class="det"></span>`;
    li.querySelector(".lbl").textContent = c.label;
    li.querySelector(".det").textContent = c.detail;
    return li;
  }));
}

function onHello(msg) {
  hello = msg;
  $("stat-backend").textContent = msg.backend === "sam3d_body" ? "SAM3D-B" : "VITPOSE+D";
  $("stat-backend").title = msg.backend;
  if (!probeState.info || JSON.stringify(probeState.info) !== JSON.stringify(msg.probe)) buildProbe(msg.probe);
  buildRig(msg.cameras);
  buildSkeleton(msg);
  initDepthUI(msg.depth_limits ?? [0.2, 6.0]);
  $("panel-angles").hidden = msg.region !== "leg"; // joint angles are a lower-limb measure
  $("clarius").hidden = $("panel-us").hidden = !msg.clarius;
  if (msg.clarius) $("cl-model").textContent = `CLARIUS ${msg.clarius.model} · ${msg.clarius.application.toUpperCase()}`;
  meshReady = false;
  if (msg.has_mesh) loadFaces();
  applyVisibility();
}

function onBinary(buf) {
  const head = new Uint8Array(buf, 0, 2);
  const kind = head[0], cam = head[1];
  if (kind === MSG_MESH) onMesh(buf);
  else if (kind === MSG_ULTRASOUND) onUltrasound(buf);
  else if (kind === MSG_CLOUD) onCloud(cam, buf);
  else if (kind === MSG_JPEG) {
    const rc = rigCams[cam];
    if (!rc) return;
    createImageBitmap(new Blob([new Uint8Array(buf, 8)], { type: "image/jpeg" }))
      .then((bmp) => { rc.image?.close?.(); rc.image = bmp; drawFeed(rc); })
      .catch(() => {});
  }
}

let personSeenAt = 0;
let focusSeenAt = 0;
function onFrame(frame) {
  lastFrame = frame;
  if (frame.person) personSeenAt = performance.now();
  updateLevel(frame);
  updateSkeleton(frame.person);
  updateFootTrails(frame.person);
  updateProbe(frame.probe);
  updateHud(frame);
  syncDepthUI(frame.depth_range);
  updateClarius(frame.clarius);
  updateRigHealth(frame);
  // Keep the view centred on what matters: the subject, else the probe, else the scene.
  const focus = frame.person?.joints?.pelvis ?? (frame.probe?.tracked ? frame.probe.position : null);
  if (focus) {
    focusSeenAt = performance.now();
    const target = sensorRoot.localToWorld(new THREE.Vector3(focus[0], focus[1], focus[2]));
    followTarget.lerp(target, frame.person ? 0.15 : 0.05);
  } else if (cloudCentre) {
    focusSeenAt = performance.now();
    followTarget.lerp(sensorRoot.localToWorld(cloudCentre.clone()), 0.05);
  }
}

// ───────────────────────────── render loop ─────────────────────────────
const clock = new THREE.Clock();
const _focus = new THREE.Vector3();
function tick() {
  requestAnimationFrame(tick);
  const t = clock.getElapsedTime();
  holoMat.uniforms.uTime.value = t;

  worldRoot.quaternion.slerp(levelTarget.quat, 0.05);
  worldRoot.position.y += (levelTarget.height - worldRoot.position.y) * 0.05;

  // mesh fades in while fresh, out when the subject is lost
  const meshLive = meshReady && opts.mesh && performance.now() - lastMeshAt < 600;
  const u = holoMat.uniforms.uOpacity;
  u.value += ((meshLive ? 1 : 0) - u.value) * 0.12;
  holo.visible = u.value > 0.01;

  // The probe is always shown: solid while tracked, ghosted at its last pose otherwise.
  // Live = the newest frame says tracked (and frames are still arriving).
  const probeFresh = !!probeState.tracked && performance.now() - probeState.lastSeen < 1500;
  probeGroup.visible = opts.probe && !!probeState.info;
  setProbeLive(probeFresh);
  if (probeState.tipLabel) {
    probeState.tipLabel.obj.visible = probeGroup.visible && opts.labels;
    if (!probeFresh) probeState.tipLabel.el.innerHTML = "<small>PROBE</small>NOT TRACKED";
  }

  if (opts.follow && performance.now() - focusSeenAt < 1500) {
    const delta = _focus.copy(followTarget).sub(controls.target).multiplyScalar(0.08);
    controls.target.add(delta);
    camera.position.add(delta);
  }
  floorMat.uniforms.uFocus.value.set(controls.target.x, controls.target.z);
  dust.rotation.y = t * 0.004;

  controls.update();
  render();
  labelRenderer.render(scene, camera);
}
tick();
connect();

// Debug handle for the browser console.
window.vigil = { scene, camera, controls, probeGroup, get skel() { return skel; }, get frame() { return lastFrame; }, get rig() { return rigCams; } };
