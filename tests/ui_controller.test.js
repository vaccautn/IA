const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const htmlPath = path.join(__dirname, '..', 'src', 'vacca_api', 'static', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const firstScript = html.match(/<script>\s*(\(function \(root\)[\s\S]*?\n\}\)\([^<]+<\/script>)/);
assert(firstScript, 'controller script must remain embedded before the DOM adapter');
const adapterMatch = html.match(/<script>\s*(const dropzone =[\s\S]*?)\n<\/script>/);
assert(adapterMatch, 'DOM adapter script must remain embedded after the controller');
const adapterScript = adapterMatch[1];
const sandbox = {
  module: { exports: {} },
  exports: {},
  URL: {
    createObjectURL(file) {
      const url = `blob:${file.name}`;
      return url;
    },
    revokeObjectURL() {},
  },
  AbortController: class {
    constructor() { this.signal = {}; this.aborted = false; }
    abort() { this.aborted = true; }
  },
  FormData: class {
    constructor() { this.entries = []; }
    append(name, value) { this.entries.push([name, value]); }
  },
};
vm.runInNewContext(firstScript[1].replace('</script>', ''), sandbox, { filename: htmlPath });
const { createUiController } = sandbox.module.exports;

function response(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function deferred() {
  let resolve;
  const promise = new Promise(value => { resolve = value; });
  return { promise, resolve };
}

function harness(fetchImpl) {
  const events = [];
  const states = [];
  const forms = [];
  const controllers = [];
  const urls = { created: [], revoked: [] };
  const controller = createUiController({
    fetchImpl,
    formDataFactory: () => {
      const form = { entries: [], append(name, value) { this.entries.push([name, value]); } };
      forms.push(form);
      return form;
    },
    abortControllerFactory: () => {
      const item = { signal: {}, aborted: false, abort() { this.aborted = true; } };
      controllers.push(item);
      return item;
    },
    urlApi: {
      createObjectURL(file) { const url = `blob:${file.name}`; urls.created.push(url); return url; },
      revokeObjectURL(url) { urls.revoked.push(url); },
    },
    onState: state => states.push(state),
    onEvent: event => events.push(event),
  });
  return { controller, events, states, forms, controllers, urls };
}

function adapterHarness(fetchImpl) {
  const elements = new Map();
  const windowHandlers = {};
  const context = {
    strokeRects: [],
    clearRect() {},
    strokeRect(...args) { this.strokeRects.push(args); },
    measureText(text) { return { width: text.length }; },
    fillRect() {},
    fillText() {},
  };
  function element(id = null) {
    const handlers = {};
    const item = {
      id,
      style: {},
      className: '',
      classList: { add() {}, remove() {}, toggle() {} },
      children: [],
      dataset: {},
      files: [],
      value: '',
      hidden: false,
      tabIndex: 0,
      clientWidth: 80,
      clientHeight: 60,
      addEventListener(type, listener) { handlers[type] = listener; },
      dispatch(type, event = {}) { return handlers[type]?.(event); },
      setAttribute(name, value) { this[name] = value; },
      removeAttribute(name) { delete this[name]; },
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; },
      focus() {},
      getContext() { return context; },
    };
    if (id) elements.set(id, item);
    return item;
  }
  [
    'dropzone', 'fileInput', 'preview', 'bboxCanvas', 'uploadHint', 'spinner',
    'summary', 'detectTab', 'bcsTab', 'detectPanel', 'bcsPanel', 'bcsReadiness',
    'refreshReadiness', 'calculateBcs', 'bcsProgress', 'bcsResults', 'results',
  ].forEach(id => element(id));
  const document = {
    getElementById(id) { return elements.get(id); },
    createElement() { return element(); },
  };
  const window = {
    VaccaUiController: { createUiController },
    addEventListener(type, listener) {
      (windowHandlers[type] ||= []).push(listener);
    },
  };
  const urls = { created: [], revoked: [] };
  const urlApi = {
    createObjectURL(file) {
      const url = `blob:${file.name}`;
      urls.created.push(url);
      return url;
    },
    revokeObjectURL(url) { urls.revoked.push(url); },
  };
  const adapterSandbox = { document, window, fetch: fetchImpl, URL: urlApi };
  vm.runInNewContext(adapterScript, adapterSandbox, { filename: htmlPath });
  return { elements, windowHandlers, context, urls };
}

const image = name => ({ name, type: 'image/jpeg' });
const flush = () => new Promise(resolve => setImmediate(resolve));

test('valid selection detects automatically and BCS submits the same file as multipart', async () => {
  let readinessCalls = 0;
  const requests = [];
  const h = harness(async (url, options) => {
    requests.push({ url, options });
    if (url === '/detect') return response(200, { cow_detected: false, detections: [] });
    if (url === '/ready/bcs') return response(503, readinessCalls++ === 0
      ? { status: 'not_loaded', message: 'configured' } : { status: 'ready', message: 'ready' });
    return response(200, { status: 'ok', message: 'computed', bcs_category: 4, cow_detected: null });
  });
  const file = image('new-cow.jpg');
  assert.equal(h.controller.selectFile(file), true);
  await flush();
  assert.deepEqual(requests.map(request => request.url), ['/detect']);
  await h.controller.refreshReadiness();
  assert.equal(h.controller.getState().canCalculate, true);
  await h.controller.calculateBcs();
  await flush();
  assert.equal(requests.filter(request => request.url === '/bcs').length, 1);
  assert.deepEqual(h.forms.at(-1).entries, [['file', file]]);
  const result = h.events.find(event => event.type === 'bcs-result');
  assert.equal(result.data.bcs_category, 4);
  assert.equal(result.data.cow_detected, null);
  assert.equal(h.controller.getState().bcsStatus, 'ready');
});

test('invalid reselection invalidates and aborts all old work before showing an error', async () => {
  const oldDetection = deferred();
  const oldReadiness = deferred();
  const oldBcs = deferred();
  const h = harness((url) => {
    if (url === '/detect') return oldDetection.promise;
    if (url === '/ready/bcs') return oldReadiness.promise;
    return oldBcs.promise;
  });
  const oldFile = image('old-cow.jpg');
  h.controller.selectFile(oldFile);
  await flush();
  h.controller.refreshReadiness();
  await flush();
  const generation = h.controller.getState().generation;
  assert.equal(h.controller.selectFile({ name: 'notes.txt', type: 'text/plain' }), false);
  const state = h.controller.getState();
  assert.equal(state.selectedFile, null);
  assert.equal(state.objectUrl, null);
  assert.equal(state.generation, generation + 1);
  assert.deepEqual(h.urls.revoked, ['blob:old-cow.jpg']);
  assert(h.controllers.every(item => item.aborted));
  assert.equal(h.events.at(-1).type, 'error');
  assert.equal(await h.controller.calculateBcs(), false);
  oldDetection.resolve(response(200, { cow_detected: false, detections: [] }));
  oldReadiness.resolve(response(200, { status: 'ready', message: 'stale' }));
  oldBcs.resolve(response(200, { status: 'ok', bcs_category: 5, cow_detected: null }));
  await flush();
  assert.equal(h.events.some(event => event.type === 'bcs-result'), false);
});

test('a BCS 503 disables calculation and rechecks readiness', async () => {
  let readinessCalls = 0;
  const h = harness(async url => {
    if (url === '/detect') return response(200, { cow_detected: false, detections: [] });
    if (url === '/ready/bcs') return response(503, readinessCalls++ === 0
      ? { status: 'ready', message: 'ready' } : { status: 'unconfigured', message: 'not configured' });
    return response(503, { detail: 'BCS capability is unavailable' });
  });
  h.controller.selectFile(image('cow.jpg'));
  await flush();
  await h.controller.refreshReadiness();
  assert.equal(h.controller.getState().canCalculate, true);
  await h.controller.calculateBcs();
  await flush();
  assert.equal(h.controller.getState().bcsStatus, 'unconfigured');
  assert.equal(h.controller.getState().canCalculate, false);
  assert.deepEqual(h.events.filter(event => event.type === 'error').map(event => event.status), [503]);
});

test('stale readiness response cannot replace a newer request', async () => {
  const first = deferred();
  const h = harness(url => url === '/ready/bcs' ? (h.calls++ ? Promise.resolve(response(200, { status: 'ready' })) : first.promise) : Promise.resolve(response(200, {})));
  h.calls = 0;
  const firstRequest = h.controller.refreshReadiness();
  await flush();
  const secondRequest = h.controller.refreshReadiness();
  await secondRequest;
  first.resolve(response(503, { status: 'unavailable' }));
  await firstRequest;
  assert.equal(h.controller.getState().bcsStatus, 'ready');
});

test('DOM adapter redraws the latest detection on resize', async () => {
  const h = adapterHarness(async url => {
    assert.equal(url, '/detect');
    return response(200, {
      cow_detected: true,
      detection_count: 1,
      detections: [{
        class_name: 'cow', confidence: 0.9,
        bbox: { x_center: 0.5, y_center: 0.5, width: 0.5, height: 0.5 },
        x1: 2, y1: 1, x2: 6, y2: 5,
      }],
      image_width: 8, image_height: 6, inference_time_ms: 1,
    });
  });

  h.elements.get('fileInput').files = [image('resize.jpg')];
  h.elements.get('fileInput').dispatch('change');
  await flush();
  assert.equal(h.context.strokeRects.length, 1);

  h.windowHandlers.resize[0]();
  assert.equal(h.context.strokeRects.length, 2);
});

test('DOM adapter unload delegates object URL cleanup to the controller owner', () => {
  const h = adapterHarness(() => new Promise(() => {}));
  h.elements.get('fileInput').files = [image('cleanup.jpg')];
  h.elements.get('fileInput').dispatch('change');

  assert.doesNotThrow(() => h.windowHandlers.beforeunload[0]());
  assert.deepEqual(h.urls.revoked, ['blob:cleanup.jpg']);
});
