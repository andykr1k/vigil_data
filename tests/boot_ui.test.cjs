const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const app = fs.readFileSync(path.join(__dirname, '../src/vigil/web/app.js'), 'utf8');
const source = app.slice(app.indexOf('const PROC_ICONS ='), app.indexOf('// ───────────────────────────── record / replay'));

function setup() {
  function element() {
    const classes = new Set();
    return {
      hidden: true, children: [], style: {},
      classList: {
        add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c),
        toggle: (c, on) => on ? classes.add(c) : classes.delete(c),
      },
      replaceChildren(...children) { this.children = children; },
      append(child) { this.children.push(child); },
      querySelectorAll() { return this.children; },
    };
  }
  const nodes = new Map(), timers = new Map(), sent = [];
  let timerId = 0;
  const $ = (id) => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  };
  const ctx = { $, document: { createElement: element }, send: (msg) => sent.push(msg),
    window: { dispatchEvent() {} }, Event,
    setTimeout: (fn) => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: (id) => timers.delete(id),
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  const select = () => ctx.onBoot({state: 'select', procedures: [
    {id: 'cardiac', label: 'Cardiac', application: 'cardiac'},
    {id: 'lower_limb', label: 'Lower Limb', application: 'dvt'},
  ]});
  const advance = () => { const pending = [...timers.values()]; timers.clear(); pending.forEach((fn) => fn()); };
  return { ctx, $, sent, select, advance };
}

test('procedure choices are visible and send the selected procedure', () => {
  const { $, sent, select } = setup();
  select();
  assert.equal($('boot').hidden, false);
  assert.equal($('boot-choices').children.length, 2);
  $('boot-choices').children[1].onclick();
  assert.equal(sent[0].cmd, 'start');
  assert.equal(sent[0].procedure, 'lower_limb');
});

test('a pending fade-out cannot hide a new procedure selection', () => {
  const { ctx, $, select, advance } = setup();
  select();
  ctx.onBoot({state: 'running'});
  assert.equal($('boot').classList.contains('leaving'), true);
  select();
  advance();
  assert.equal($('boot').hidden, false);
  assert.equal($('boot').classList.contains('leaving'), false);
  assert.equal($('boot-choices').hidden, false);
});

test('new loading screen survives old dismissal; completed startup still dismisses', () => {
  const { ctx, $, select, advance } = setup();
  select();
  ctx.onBoot({state: 'running'});
  ctx.onBoot({state: 'loading', checks: [], progress: 0.2, eta_s: 5});
  advance();
  assert.equal($('boot').hidden, false);
  assert.equal($('boot-progress').hidden, false);
  ctx.onBoot({state: 'running'});
  advance();
  assert.equal($('boot').hidden, true);
});
