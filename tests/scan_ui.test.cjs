// Exercise the actual live-scan controller without starting cameras or a WebGL renderer.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const app = fs.readFileSync(path.join(__dirname, '../src/vigil/web/app.js'), 'utf8');
const controller = app.slice(app.indexOf('// Live capture →'), app.indexOf('const surfaceMats ='));

function setup() {
  const elements = new Map();
  const calls = { sent: [], shown: [], flat: [] };
  const cameras = [{ serial: 'A', T_world_camera: [1, 0, 0, 0] }];
  const $ = (id) => {
    if (!elements.has(id)) elements.set(id, {
      value: id === 'flat-iters' ? '23' : id === 'sam-auto-leg' ? 'left' : '2',
      checked: true, disabled: false, textContent: '', className: '',
      classList: { toggle() {} }, parentElement: { querySelector: () => ({ textContent: '' }) },
    });
    return elements.get(id);
  };
  const ctx = {
    $, store: { get: (_key, fallback) => fallback, set() {} },
    hello: { cameras }, flat: { method: 'slim', running: false }, surface: { building: false },
    socket: { readyState: 1 }, WebSocket: { OPEN: 1 },
    renderFlatSources() {}, meshParams: () => ({ voxel_mm: 4, max_frames: 12 }),
    send: (msg) => calls.sent.push(msg),
    meshUrl: (key, ext) => `${key.recording}/${key.stem}.${ext}`,
    sameKey: (a, b) => a && b && a.recording === b.recording && a.stem === b.stem,
    fetch: async () => ({ ok: true, json: async () => ({ stats: {} }), arrayBuffer: async () => new ArrayBuffer(8) }),
    closeFlat() { ctx.flat.entry = null; },
    showSurface(key, entry, data, options) { calls.shown.push({ key, options }); ctx.surface.id = key; },
    async loadFlat(key, current) { if (current()) calls.flat.push(key); },
  };
  vm.createContext(ctx);
  vm.runInContext(controller, ctx);
  const result = (extra = {}) => ({ type: 'scan', state: 'unfolding', message: 'Unfolding',
    recording: 'capture', stem: 'mesh', cameras, ...extra });
  const drain = () => vm.runInContext('scanUI.display', ctx);
  return { ctx, calls, $, result, drain };
}

test('scan sends selected mesh and unfolding settings and prevents duplicate clicks', () => {
  const { $, calls } = setup();
  $('scan-run').onclick();
  $('scan-run').onclick();
  assert.equal(calls.sent.length, 1);
  assert.equal(calls.sent[0].cmd, 'scan');
  assert.equal(calls.sent[0].params.voxel_mm, 4);
  assert.equal(calls.sent[0].unfold.method, 'slim');
  assert.equal(calls.sent[0].unfold.iterations, 23);
});

test('mesh appears before unfolding finishes, without replay or viewpoint changes', async () => {
  const { ctx, calls, $, result, drain } = setup();
  ctx.onScan(result());
  await drain();
  assert.equal(calls.shown.length, 1);
  assert.equal(calls.shown[0].options.focus, false);
  assert.equal($('mesh-hide-cloud').checked, false);
  assert.equal(calls.flat.length, 0);
  assert.equal($('scan-run').disabled, true);
  ctx.onScan(result({ state: 'done', flat_stem: 'flat' }));
  await drain();
  assert.equal(calls.shown.length, 1);
  assert.equal(calls.flat[0].stem, 'flat');
  assert.equal($('scan-run').disabled, false);
  $('scan-view').onclick();
  await drain();
  assert.deepEqual(calls.sent, []); // neither replay nor replay_ctl
});

test('partial success keeps a live mesh without attempting to open a nonexistent unfolding', async () => {
  const { ctx, calls, $, result, drain } = setup();
  ctx.onScan(result({ state: 'partial', message: '3D mesh saved; unfolding failed' }));
  await drain();
  assert.equal(calls.shown.length, 1);
  assert.equal(calls.flat.length, 0);
  assert.equal($('scan-view').disabled, false);
  assert.equal($('scan-run').disabled, false);
});

test('scan completion does not replace a replay scene', async () => {
  const { ctx, calls, result, drain } = setup();
  ctx.hello.replay = true;
  ctx.onScan(result({ state: 'done', flat_stem: 'flat' }));
  await drain();
  assert.equal(calls.shown.length, 0);
  assert.equal(calls.flat.length, 0);
  assert.deepEqual(calls.sent, []);
});

test('a scene change during result download discards the live overlay', async () => {
  const { ctx, calls, result, drain } = setup();
  let release;
  const ready = new Promise((r) => { release = r; });
  ctx.fetch = async () => { await ready; return { ok: true, json: async () => ({}), arrayBuffer: async () => new ArrayBuffer(8) }; };
  ctx.onScan(result({ state: 'done', flat_stem: 'flat' }));
  await new Promise(setImmediate);
  vm.runInContext('scanUI.epoch += 1', ctx);
  ctx.hello.replay = true;
  release();
  await drain();
  assert.equal(calls.shown.length, 0);
  assert.equal(calls.flat.length, 0);
});

test('changed camera calibration refuses to place a stale mesh in the live scene', async () => {
  const { ctx, calls, $, result, drain } = setup();
  ctx.onScan(result({ cameras: [{ serial: 'B', T_world_camera: [1, 0, 0, 0] }] }));
  await drain();
  assert.equal(calls.shown.length, 0);
  assert.match($('scan-msg').textContent, /calibration changed/);
});
