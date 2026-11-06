// Pipeline page for one saved mesh: pipeline.html?recording=<name>&stem=<stem>
// Every stage of build_tsdf (meshing/tsdf.py) as a card: what it does, its numbers from the
// mesh's sidecar, and, for meshes saved with stage maps (.npz), what it removed on each
// camera's frame nearest the mesh's time.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const q = new URLSearchParams(location.search);
const REC = q.get("recording"), STEM = q.get("stem");
const enc = encodeURIComponent;
const base = `/api/surface/${enc(REC ?? "")}/${enc(STEM ?? "")}`;
const $ = (id) => document.getElementById(id);

// Stage-map labels (tsdf.STAGE_LABELS) and their colours.
const L = { REGION: 0, NO_DEPTH: 1, BOX: 2, SUPPORT: 3, SECTION: 4, TUBE: 5, KEPT: 6 };
const COLOR = { 1: [255, 77, 106], 2: [255, 197, 61], 3: [255, 140, 40], 4: [255, 61, 242], 5: [179, 136, 255], 6: [61, 255, 168] };
const css = (c) => `rgb(${c.join(",")})`;

const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : typeof v === "number" ? v.toFixed(d) : String(v));
const mmss = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let report, entry, stats, params, cams, labels = {};

async function main() {
  if (!REC || !STEM) { $("sub").innerHTML = `<span class="err">Open this page from SAVED MESHES (STEPS).</span>`; return; }
  const r = await fetch(`${base}/report`);
  if (!r.ok) { $("sub").innerHTML = `<span class="err">No saved mesh ${esc(REC)}/${esc(STEM)}.</span>`; return; }
  report = await r.json();
  entry = report.entry; stats = entry.stats ?? {}; params = stats.params ?? {};
  cams = Object.keys(entry.stages?.cameras ?? stats.views ?? {}).map(Number).sort();
  if (!cams.length) cams = report.cameras.map((c) => c.index);
  document.title = `VIGIL · Pipeline · frame ${entry.frame}`;
  $("title").textContent = `MESH PIPELINE · FRAME ${entry.frame} · ${mmss(entry.time_s)}`;
  $("sub").textContent = `${entry.recording} · ${entry.method} · built ${entry.created.replace("T", " ")}` +
    (stats.build_s ? ` in ${fmt(stats.build_s)} s` : "");
  $("ply").href = `${base}.ply`;
  $("json").href = `${base}.json`;
  if (entry.method === "flat") {
    $("cards").innerHTML = `<div class="card">This is a flattened mesh. Its pipeline is on its source: ` +
      `<a href="?recording=${enc(entry.source.recording)}&stem=${enc(entry.source.stem)}">open the source mesh</a>.</div>`;
    return;
  }
  if (entry.stages) await Promise.all(cams.map(loadLabel));
  render();
}

async function loadLabel(cam) {
  const img = await loadImg(`${base}/image/${cam}/label`).catch(() => null);
  if (!img) return;
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  const rgba = g.getImageData(0, 0, c.width, c.height).data;
  const v = new Uint8Array(c.width * c.height);
  for (let i = 0; i < v.length; i++) v[i] = rgba[i * 4];
  labels[cam] = { w: c.width, h: c.height, v };
}

function loadImg(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });
}

// ───────────────────────────── stage cards ─────────────────────────────
function stages() {
  const seg = stats.segmentation, auto = stats.auto, sec = stats.section, al = stats.align, holes = stats.holes;
  const px = (key) => cams.reduce((s, c) => s + (entry.stages?.cameras?.[c]?.pixels?.[key] ?? 0), 0);
  const region = { pose: "Pose capsule around the recorded leg joints", sam: "SAM 2, from clicks / auto prompts",
                   window: "Depth window only (no leg detection)" }[params.region] ?? params.region;
  const out = [];

  out.push({
    id: "input", title: "Input frames", status: "ok",
    what: "Depth and colour from every camera within ± the time window of this frame. Each depth frame is fused separately.",
    kv: [["Frame", `${entry.frame} (${mmss(entry.time_s)})`],
         ["Window", entry.window ? `±${entry.window.seconds} s · frames ${entry.window.from_frame}–${entry.window.to_frame}` : "—"],
         ...Object.entries(stats.views ?? {}).map(([c, n]) => [`CAM ${+c + 1} frames`, n]),
         ["Depth window", `${fmt(params.depth_min, 2)} – ${fmt(params.depth_max, 2)} m`],
         ["Max frames", params.max_frames]],
    images: cams.flatMap((c) => [
      { src: `${base}/image/${c}/color`, cap: `CAM ${c + 1} colour` },
      { src: `${base}/image/${c}/depth`, cap: `CAM ${c + 1} depth (near = red)` }]),
  });

  const regionKv = [["Mode", region]];
  if (params.region === "pose") regionKv.push(["Leg", params.leg], ["Radius", `${params.radius_cm} cm`]);
  if (seg) {
    const kinds = {};
    for (const c of seg.clicks ?? []) kinds[c.kind] = (kinds[c.kind] ?? 0) + 1;
    regionKv.push(["Model", seg.model?.split("/").pop()],
      ["Clicks", Object.entries(kinds).map(([k, n]) => `${n} ${k}`).join(" · ") || "none"]);
    for (const [c, r] of Object.entries(seg.cameras ?? {})) {
      const cv = r.mask_px_cv;
      regionKv.push([`CAM ${+c + 1}`, `${r.source ?? "none"} · ${(r.mask_px ?? 0).toLocaleString()} px` +
        (cv != null ? ` · area jitter ${(cv * 100).toFixed(0)}%` : ""), cv != null && cv > 0.15]);
    }
  }
  if (auto) {
    regionKv.push(["Auto side", `${auto.side} (${auto.source}, ${auto.model})`],
      ["Knee votes", auto.knee ? `${auto.knee.votes} · cameras disagree ${fmt(auto.knee.disagree_cm)} cm` : "—", auto.knee?.disagree_cm > 3],
      ["Hip votes", auto.hip ? `${auto.hip.votes} · thigh ${fmt(auto.hip.length_cm)} cm` : "—"],
      ["Auto score", fmt(auto.score, 2)]);
  }
  const autoWarn = [...(auto?.warnings ?? []), ...(auto?.precheck?.warnings ?? [])];
  out.push({
    id: "region", title: "Leg region", status: autoWarn.length ? "warn" : "ok", warnings: autoWarn,
    what: "Which pixels belong to the leg. Everything else is ignored from here on.",
    kv: regionKv,
    overlay: { region: true, prompts: true },
    legend: [["leg region", [34, 228, 255]], ["+ leg click", [61, 255, 168]], ["− not leg", [255, 77, 106]],
             ["knee", [255, 61, 242]], ["hip", [255, 210, 61]], ["carried over from the other camera", [200, 200, 200]]],
  });

  out.push({
    id: "depth", title: "Depth cleanup", status: "ok",
    what: "Pixels of the region without usable depth: outside the depth window, flying pixels at depth edges, and the mask's eroded rim.",
    kv: [["Edge filter", params.edge_filter ? "on" : "off"], ["Removed (stage frames)", px("no_depth").toLocaleString() + " px"]],
    overlay: { removed: [L.NO_DEPTH] },
  });

  const alignBad = al && (al.failed > al.aligned / 2 || al.rms_after_mm > 3);
  out.push({
    id: "align", title: "Align views (ICP)", status: !params.align ? "off" : !al ? "off" : alignBad ? "warn" : "ok",
    off: !params.align || !al,
    what: "Each camera's frame nearest this time is snapped onto camera 1's (calibration error), and every other frame onto its own camera's (leg motion).",
    kv: al ? [["Aligned", al.aligned], ["Failed", al.failed, al.failed > al.aligned / 2],
              ["RMS before", `${fmt(al.rms_before_mm, 2)} mm`], ["RMS after", `${fmt(al.rms_after_mm, 2)} mm`, al.rms_after_mm > 3]]
            : [["Align", params.align ? "only one frame: nothing to align" : "off"]],
  });

  if (seg?.box) out.push({
    id: "box", title: "Box crop", status: "ok",
    what: "Keeps what lies inside every drawn BOX.",
    kv: [["Box cameras", (seg.box.cameras ?? []).map((c) => `CAM ${c + 1}`).join(", ")], ["Removed", `${fmt(seg.box.removed * 100)}% of points`]],
    overlay: { removed: [L.BOX] },
  });

  const sup = stats.support;
  out.push({
    id: "support", title: "Support surface", status: !params.support ? "off" : sup ? "ok" : "off", off: !params.support || !sup,
    what: "The plane the leg rests on (stool, bed) is found by RANSAC and everything within the margin of it is cut.",
    kv: sup ? [["Plane normal", sup.normal.map((v) => v.toFixed(2)).join(", ")], ["Margin", `${params.support_margin_cm} cm`],
               ["Removed", seg?.support_removed != null ? `${fmt(seg.support_removed * 100)}% of points` : px("support").toLocaleString() + " px"]]
            : [["Support", params.support ? "no supporting plane found" : "off"]],
    overlay: { removed: [L.SUPPORT] },
  });

  out.push({
    id: "section", title: "Knee / hip cut", status: sec ? "ok" : "off", off: !sec,
    what: "Keeps one side of the knee (and below the hip when there's a HIP click), then the thigh tube drops what is far from the thigh's axis (garments).",
    kv: sec ? [["Mode", sec.mode], ["Knee shift", `${sec.shift_cm} cm`],
               ["Knee", sec.knee.map((v) => v.toFixed(3)).join(", ")], ["Hip", sec.hip ? sec.hip.map((v) => v.toFixed(3)).join(", ") : "—"],
               ["Knee detections", `${sec.knees_used} of ${sec.knees_seen}`],
               ["Thigh tube", sec.tube ? `r ${sec.tube.radius_cm.join("→")} cm · removed ${fmt(sec.tube.removed * 100)}%` : params.tube ? "—" : "off"]]
            : [["Section", params.section ?? "whole leg"]],
    overlay: { removed: [L.SECTION, L.TUBE], section: true },
    legend: [["past the knee / hip", COLOR[4]], ["outside the thigh tube", COLOR[5]], ["knee", [255, 61, 242]], ["hip", [255, 210, 61]]],
  });

  out.push({
    id: "fusion", title: "Fusion (TSDF)", status: "ok",
    what: "What's left of every frame is integrated into a signed-distance volume; green is what this frame contributed.",
    kv: [["Voxel", `${fmt(stats.voxel_mm)} mm`], ["Grid", (stats.grid ?? []).join(" × ")], ["Truncation", `${params.trunc_mm} mm`],
         ["View blending", params.blend ? "on" : "off"], ["Fused (stage frames)", px("kept").toLocaleString() + " px"]],
    overlay: { removed: [], kept: true },
  });

  out.push({
    id: "surface", title: "Surface", status: (stats.warnings ?? []).length ? "warn" : "ok",
    what: "Marching cubes, largest piece, small holes filled, Taubin smoothing. Drag to rotate; magenta = knee, gold = hip.",
    kv: [["Vertices", (stats.vertices ?? 0).toLocaleString()], ["Triangles", (stats.faces ?? 0).toLocaleString()],
         ["Area", `${fmt(stats.area_cm2)} cm²`], ["Largest piece only", params.largest ? "yes" : "no"],
         ["Holes", holes ? `${holes.holes_filled} of ${holes.holes_found} filled (largest ${fmt(holes.largest_filled_cm)} cm)` : "off"],
         ["Smoothing", params.smooth ? `${params.smooth} iterations` : "off"], ["Build time", `${fmt(stats.build_s)} s`]],
    viewer: true,
  });

  out.push({
    id: "flat", title: "Flattened", status: report.flats.length ? "ok" : "off", off: !report.flats.length,
    what: report.flats.length ? "Unrolled versions of this surface (FLATTEN MESH), coloured by stretch: green = no stretch, red = 50%+."
                              : "Not flattened yet: FLATTEN MESH in the dashboard unrolls it.",
    flats: report.flats,
  });
  return out;
}

function render() {
  const all = stages();
  const warnings = [...(stats.warnings ?? [])];
  const cards = $("cards");
  if (warnings.length) {
    const w = document.createElement("div");
    w.className = "card";
    w.innerHTML = `<h2><span class="chip warn">WARNINGS</span></h2><ul class="warn">${warnings.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
    cards.append(w);
  }
  if (!entry.stages) {
    const n = document.createElement("p");
    n.className = "note";
    n.textContent = "This mesh was saved before stage maps existed, so the per-stage overlays are missing. Meshes built from now on have them.";
    cards.append(n);
  }
  $("flow").innerHTML = all.map((s, i) =>
    `<a href="#${s.id}" class="${s.off ? "off" : s.status === "warn" ? "warn" : ""}">${i + 1}. ${esc(s.title)}</a>`).join("");
  all.forEach((s, i) => cards.append(card(s, i + 1)));
}

function card(s, n) {
  const el = document.createElement("section");
  el.className = "card" + (s.off ? " off" : "");
  el.id = s.id;
  const chip = { ok: "OK", warn: "CHECK", off: "SKIPPED" }[s.status] ?? s.status;
  el.innerHTML = `<h2><span class="n">${n}</span>${esc(s.title.toUpperCase())}<span class="chip ${s.status}">${chip}</span></h2>` +
    `<p class="what">${esc(s.what)}</p>` +
    (s.kv ? `<div class="kv">${s.kv.map(([k, v, bad]) => `<div><b>${esc(k)}</b><span class="${bad ? "bad" : ""}">${esc(v ?? "—")}</span></div>`).join("")}</div>` : "") +
    (s.warnings?.length ? `<ul class="warn">${s.warnings.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "");
  if (s.images) {
    const box = document.createElement("div");
    box.className = "imgs four";
    for (const im of s.images) {
      const f = document.createElement("figure");
      f.innerHTML = `<img loading="lazy" alt="" src="${im.src}"><figcaption>${esc(im.cap)}</figcaption>`;
      f.querySelector("img").onerror = () => (f.querySelector("figcaption").textContent += " · not in the recording");
      box.append(f);
    }
    el.append(box);
  }
  if (s.overlay && (entry.stages || s.overlay.prompts)) {
    const box = document.createElement("div");
    box.className = "imgs";
    for (const c of cams) box.append(overlayFigure(c, s.overlay));
    el.append(box);
    const legend = s.legend ?? [...s.overlay.removed.map((k) => [entry.stages?.labels?.[k] ?? k, COLOR[k]]),
      ...(s.overlay.kept ? [["fused into the mesh", COLOR[6]]] : [["still kept", COLOR[6]], ["removed earlier", [70, 70, 70]]])];
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = legend.map(([t, c]) => `<span><i style="background:${css(c)}"></i>${esc(t)}</span>`).join("");
    el.append(lg);
  }
  if (s.viewer) {
    const v = document.createElement("div");
    v.id = "viewer";
    el.append(v);
    requestAnimationFrame(() => viewer(v).catch((e) => (v.innerHTML = `<p class="err">${esc(e.message)}</p>`)));
  }
  if (s.flats?.length) {
    const box = document.createElement("div");
    box.className = "imgs flats";
    for (const f of s.flats) box.append(flatFigure(f));
    el.append(box);
  }
  return el;
}

// ───────────────────────────── image overlays ─────────────────────────────
function overlayFigure(cam, o) {
  const f = document.createElement("figure");
  const canvas = document.createElement("canvas");
  f.append(canvas);
  const cap = document.createElement("figcaption");
  cap.textContent = `CAM ${cam + 1}`;
  f.append(cap);
  loadImg(`${base}/image/${cam}/color`).then((img) => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const g = canvas.getContext("2d");
    g.drawImage(img, 0, 0);
    const lab = labels[cam];
    if (lab && lab.w === canvas.width && lab.h === canvas.height) paint(g, lab, o);
    if (o.prompts) drawPrompts(g, cam);
    if (o.section) drawSection(g, cam);
    if (lab && !o.region) cap.textContent += ` · ${counts(lab, o)}`;
  }).catch(() => (cap.textContent += " · image not in the recording"));
  return f;
}

function paint(g, lab, o) {
  const im = g.getImageData(0, 0, lab.w, lab.h), d = im.data, v = lab.v;
  const removed = new Set(o.removed ?? []);
  const first = o.removed?.length ? Math.min(...o.removed) : 7;
  const last = o.removed?.length ? Math.max(...o.removed) : 0;
  for (let i = 0; i < v.length; i++) {
    const k = v[i];
    let c = null, a = 0;
    if (o.region) { if (k > 0) { c = [34, 228, 255]; a = 0.45; } }
    else if (removed.has(k)) { c = COLOR[k]; a = 0.8; }
    else if (o.kept ? k === L.KEPT : k > last && k > 0) { c = COLOR[6]; a = o.kept ? 0.55 : 0.3; }
    else if (k > 0 && k < first) { c = [70, 70, 70]; a = 0.6; }
    if (c) {
      const j = i * 4;
      d[j] += (c[0] - d[j]) * a; d[j + 1] += (c[1] - d[j + 1]) * a; d[j + 2] += (c[2] - d[j + 2]) * a;
    }
  }
  g.putImageData(im, 0, 0);
}

function counts(lab, o) {
  const n = new Array(7).fill(0);
  for (const k of lab.v) n[k]++;
  const region = n.slice(1).reduce((a, b) => a + b, 0) || 1;
  if (o.kept) return `${n[6].toLocaleString()} px fused (${(100 * n[6] / region).toFixed(0)}% of the region)`;
  const r = (o.removed ?? []).reduce((s, k) => s + n[k], 0);
  return `${r.toLocaleString()} px removed (${(100 * r / region).toFixed(1)}% of the region)`;
}

function drawPrompts(g, cam) {
  const seg = stats.segmentation;
  if (!seg) return;
  const lw = Math.max(2, g.canvas.width / 400);
  g.lineWidth = lw;
  for (const [x, y, lbl] of seg.cameras?.[cam]?.source && seg.cameras[cam].source !== "clicks" ? seg.cameras[cam].points ?? [] : []) {
    g.strokeStyle = lbl ? "#e0e0e0" : "#ff4d6a";
    g.beginPath(); g.arc(x, y, 7, 0, Math.PI * 2); g.stroke();
  }
  const style = { leg: "#3dffa8", not: "#ff4d6a", knee: "#ff3df2", hip: "#ffd23d" };
  for (const c of seg.clicks ?? []) {
    if (c.cam !== cam) continue;
    g.strokeStyle = g.fillStyle = style[c.kind] ?? "#fff";
    g.beginPath();
    if (c.kind === "leg" || c.kind === "not") {
      g.moveTo(c.x - 8, c.y); g.lineTo(c.x + 8, c.y);
      if (c.kind === "leg") { g.moveTo(c.x, c.y - 8); g.lineTo(c.x, c.y + 8); }
    } else g.arc(c.x, c.y, 8, 0, Math.PI * 2);
    g.stroke();
  }
  const box = seg.cameras?.[cam]?.box;
  if (box) { g.strokeStyle = "#ffc53d"; g.setLineDash([8, 6]); g.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]); g.setLineDash([]); }
}

function project(cam, X) {
  const c = report.cameras.find((k) => k.index === cam);
  if (!c?.T_world_camera) return null;
  const T = c.T_world_camera; // row-major 4x4, camera → world
  const d = [X[0] - T[3], X[1] - T[7], X[2] - T[11]];
  const x = T[0] * d[0] + T[4] * d[1] + T[8] * d[2];
  const y = T[1] * d[0] + T[5] * d[1] + T[9] * d[2];
  const z = T[2] * d[0] + T[6] * d[1] + T[10] * d[2];
  return z > 0.05 ? [c.fx * x / z + c.cx, c.fy * y / z + c.cy] : null;
}

function drawSection(g, cam) {
  const sec = stats.section;
  if (!sec) return;
  const knee = project(cam, sec.knee);
  const along = project(cam, sec.knee.map((v, i) => v + 0.15 * sec.axis[i]));
  g.lineWidth = 3;
  if (knee && along) {
    g.strokeStyle = "#ffffff"; g.setLineDash([6, 5]);
    g.beginPath(); g.moveTo(...knee); g.lineTo(...along); g.stroke(); g.setLineDash([]);
  }
  for (const [X, col, name] of [[sec.knee, "#ff3df2", "KNEE"], [sec.hip, "#ffd23d", "HIP"]]) {
    const p = X && project(cam, X);
    if (!p) continue;
    g.fillStyle = col; g.strokeStyle = "#000";
    g.beginPath(); g.arc(p[0], p[1], 7, 0, Math.PI * 2); g.fill(); g.stroke();
    g.font = "bold 14px JetBrains Mono, monospace";
    g.lineWidth = 4; g.strokeText(name, p[0] + 10, p[1] - 8); g.fillText(name, p[0] + 10, p[1] - 8); g.lineWidth = 3;
  }
}

// ───────────────────────────── 3D surface ─────────────────────────────
async function viewer(el) {
  const r = await fetch(`${base}.bin`);
  if (!r.ok) throw new Error("mesh file missing");
  const buf = await r.arrayBuffer();
  const [V, F] = new Uint32Array(buf, 0, 2);
  let off = 8;
  const pos = new Float32Array(buf, off, V * 3); off += V * 12;
  const nrm = new Float32Array(buf, off, V * 3); off += V * 12;
  const rgb = new Uint8Array(buf, off, V * 3); off += V * 3 + ((4 - (V * 3) % 4) % 4);
  const idx = new Uint32Array(buf, off, F * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  const lin = new Float32Array(V * 3); // camera colours are sRGB; three.js lights in linear
  for (let i = 0; i < lin.length; i++) lin[i] = Math.pow(rgb[i] / 255, 2.2);
  geo.setAttribute("color", new THREE.BufferAttribute(lin, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeBoundingSphere();
  const { center, radius } = geo.boundingSphere;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.0);
  scene.add(sun);
  scene.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })));
  const sec = stats.section;
  if (sec) {
    for (const [X, col] of [[sec.knee, 0xff3df2], [sec.hip, 0xffd23d]]) {
      if (!X) continue;
      const s = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.035), new THREE.MeshBasicMaterial({ color: col, depthTest: false }));
      s.position.set(...X); s.renderOrder = 2; scene.add(s);
    }
    const a = new THREE.Vector3(...sec.knee), b = a.clone().add(new THREE.Vector3(...sec.axis).multiplyScalar(radius * 0.8));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
    line.renderOrder = 2; scene.add(line);
  }
  const camera = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, radius / 50, radius * 50);
  camera.up.set(0, -1, 0); // world = camera 1's optical frame (y down)
  const toCam1 = new THREE.Vector3().sub(center).normalize(); // look from where camera 1 stands
  camera.position.copy(center).addScaledVector(toCam1, radius * 2.8);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.update();
  new ResizeObserver(() => {
    camera.aspect = el.clientWidth / el.clientHeight; camera.updateProjectionMatrix();
    renderer.setSize(el.clientWidth, el.clientHeight);
  }).observe(el);
  renderer.setAnimationLoop(() => { sun.position.copy(camera.position); renderer.render(scene, camera); });
}

// ───────────────────────────── flattened sheets ─────────────────────────────
function flatFigure(f) {
  const fig = document.createElement("figure");
  const canvas = document.createElement("canvas");
  canvas.width = 640; canvas.height = 420;
  fig.append(canvas);
  const cap = document.createElement("figcaption");
  cap.textContent = `${(f.flatten_method ?? "").toUpperCase()} · ${Math.round(f.area_cm2 ?? 0)} cm² · stretch p95 ${fmt(f.stretch_p95, 2)} · ${f.created.replace("T", " ")}`;
  fig.append(cap);
  fetch(`/api/surface/${enc(f.recording)}/${enc(f.stem)}.flat`).then((r) => r.arrayBuffer()).then((buf) => {
    const [V, F] = new Uint32Array(buf, 0, 2);
    let off = 8;
    const uv = new Float32Array(buf, off, V * 2); off += V * 8;
    off += V * 12; // xyz
    off += V * 3 + ((4 - (V * 3) % 4) % 4); // rgb
    const idx = new Uint32Array(buf, off, F * 3); off += F * 12;
    const stretch = new Float32Array(buf, off, F);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < V; i++) {
      x0 = Math.min(x0, uv[2 * i]); x1 = Math.max(x1, uv[2 * i]);
      y0 = Math.min(y0, uv[2 * i + 1]); y1 = Math.max(y1, uv[2 * i + 1]);
    }
    const s = Math.min((canvas.width - 20) / (x1 - x0), (canvas.height - 20) / (y1 - y0));
    const ox = (canvas.width - s * (x1 - x0)) / 2, oy = (canvas.height - s * (y1 - y0)) / 2;
    const g = canvas.getContext("2d");
    g.fillStyle = "#000"; g.fillRect(0, 0, canvas.width, canvas.height);
    const P = (k) => [ox + s * (uv[2 * k] - x0), oy + s * (y1 - uv[2 * k + 1])];
    for (let t = 0; t < F; t++) {
      const a = Math.min(Math.max((stretch[t] - 1) / 0.5, 0), 1);
      g.fillStyle = g.strokeStyle = `rgb(${Math.round(61 + 194 * Math.min(1, 2 * a))},${Math.round(255 - 178 * Math.max(0, 2 * a - 1))},${Math.round(168 * (1 - a))})`;
      const [p, q, r] = [P(idx[3 * t]), P(idx[3 * t + 1]), P(idx[3 * t + 2])];
      g.beginPath(); g.moveTo(...p); g.lineTo(...q); g.lineTo(...r); g.closePath(); g.fill(); g.stroke();
    }
    const cm = 10 / 100 * s; // 10 cm scale bar
    g.fillStyle = "#fff"; g.fillRect(12, canvas.height - 16, cm, 3);
    g.font = "11px JetBrains Mono, monospace"; g.fillText("10 cm", 12, canvas.height - 22);
  }).catch(() => (cap.textContent += " · could not load"));
  return fig;
}

main().catch((e) => { $("sub").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
