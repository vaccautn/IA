const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const htmlPath = path.join(__dirname, '..', 'src', 'vacca_api', 'static', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const firstScript = html.match(/<script>\s*(\(function \(root\)[\s\S]*?\n\}\)\([^<]+<\/script>)/);
assert(firstScript, 'controller script must remain embedded before the DOM adapter');
const sandbox = { module: { exports: {} }, exports: {} };
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
    return response(200, { status: 'ok', message: 'computed', bcs_score: 4, cow_detected: null });
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
  assert.equal(result.data.bcs_score, 4);
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
  oldBcs.resolve(response(200, { status: 'ok', bcs_score: 5, cow_detected: null }));
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
