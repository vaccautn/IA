const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(path.join(__dirname, '../src/vacca_video/report.html'), 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];

function render(overrides = {}, frames = [
  { image: 'frames/0.jpg', frame_index: 0, timestamp_ms: 0, detection_count: 1, inference_time_ms: 8 },
  { image: 'frames/1.jpg', frame_index: 9, timestamp_ms: 350, detection_count: 2, inference_time_ms: 9 },
]) {
  const elements = new Map();
  const callbacks = new Map();
  const delays = [];
  let nextTimer = 0;
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        hidden: true,
        disabled: true,
        textContent: '',
        classList: { values: new Set(), add(value) { this.values.add(value); } },
      });
      return elements.get(id);
    },
    addEventListener() {},
  };
  const summary = {
    status: 'completed', stop_reason: 'end_of_source', frames_analyzed: 2,
    max_cows_in_frame: 2, processing_fps: 5, mean_inference_ms: 8.5,
    source: { name: 'test.mp4' }, model: { name: 'local.pt', confidence: 0.25 },
    annotated_video: null, ...overrides,
  };
  document.getElementById('video-link');
  document.getElementById('play');
  vm.runInNewContext(script, {
    window: { VACCA_SUMMARY: summary, VACCA_FRAMES: frames }, document,
    setTimeout(callback, delay) { const id = ++nextTimer; callbacks.set(id, callback); delays.push(delay); return id; },
    clearTimeout(id) { callbacks.delete(id); },
  });
  return { get: id => elements.get(id), callbacks, delays };
}

test('seeking selects the image and count from the same frame', () => {
  const { get } = render();
  assert.equal(get('frame').src, 'frames/0.jpg');
  assert.equal(get('current-count').textContent, 1);
  get('seek').oninput({ target: { value: '1' } });
  assert.equal(get('frame').src, 'frames/1.jpg');
  assert.equal(get('index').textContent, 9);
  assert.equal(get('current-count').textContent, 2);
  assert.equal(get('time').textContent, '00:00.350');
  assert.equal(get('next').disabled, true);
});

test('playback uses source timestamps and stops at the last preview frame', () => {
  const { get, callbacks, delays } = render();
  get('play').onclick();
  assert.deepEqual(delays, [350]);
  [...callbacks.values()][0]();
  assert.equal(get('frame').src, 'frames/1.jpg');
  assert.equal(get('play').textContent, 'Reproducir');
});

test('partial and failed processing cannot appear as a complete source', () => {
  const partial = render({ stop_reason: 'frame_limit' });
  assert.match(partial.get('status').textContent, /parte del video/);
  assert.equal(partial.get('status').classList.values.has('warn'), true);
  const failed = render({ status: 'failed', error: '<img src=x onerror=alert(1)>' });
  assert.match(failed.get('status').textContent, /falló/);
  assert.match(failed.get('status').textContent, /<img/);
  assert.equal(failed.get('status').innerHTML, undefined);
});

test('no-video mode hides video download and no frames disables playback', () => {
  const { get } = render({ frames_analyzed: 0 }, []);
  assert.equal(get('video-link').hidden, true);
  // The script deliberately leaves the disabled markup unchanged.
  assert.equal(get('play').disabled, true);
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

test('web frame owner aborts reselection and ignores stale results and errors', async () => {
  const { createRequestOwner } = await import('../web/src/requests.js');
  const owner = createRequestOwner(), first = deferred(), second = deferred();
  const values = [], errors = [];
  let signal;
  const old = owner.run((value) => { signal = value; return first.promise; },
    (value) => values.push(value), (error) => errors.push(error));
  const latest = owner.run(() => second.promise,
    (value) => values.push(value), (error) => errors.push(error));
  assert.equal(signal.aborted, true);
  second.resolve({ image_url: 'new.jpg', detection_count: 2 });
  await latest;
  first.reject(new Error('late failure'));
  await old;
  assert.deepEqual(values, [{ image_url: 'new.jpg', detection_count: 2 }]);
  assert.deepEqual(errors, []);
});

test('web request cleanup suppresses late upload and list results', async () => {
  const { createRequestOwner } = await import('../web/src/requests.js');
  for (const result of [{ id: 'uploaded' }, [{ id: 'old-list' }]]) {
    const owner = createRequestOwner(), request = deferred(), values = [];
    const pending = owner.run(() => request.promise,
      (value) => values.push(value), (error) => values.push(error));
    owner.cancel();
    request.resolve(result);
    await pending;
    assert.deepEqual(values, []);
  }
});

test('web polling waits for settlement and cleanup cannot schedule another poll', async () => {
  const { startPolling } = await import('../web/src/requests.js');
  const request = deferred(), timers = new Map(), delays = [];
  let calls = 0;
  const clock = {
    setTimeout(callback, delay) { timers.set(1, callback); delays.push(delay); return 1; },
    clearTimeout(id) { timers.delete(id); }
  };
  const stop = startPolling(() => { calls++; return request.promise; }, 2000, clock);
  assert.equal(calls, 1);
  assert.equal(timers.size, 0);
  request.resolve();
  await Promise.resolve();
  assert.deepEqual(delays, [2000]);
  assert.equal(calls, 1);
  stop();
  assert.equal(timers.size, 0);

  const late = deferred();
  const stopBeforeSettlement = startPolling(() => late.promise, 2000, clock);
  stopBeforeSettlement();
  late.resolve();
  await Promise.resolve();
  assert.equal(timers.size, 0);
  assert.deepEqual(delays, [2000]);
});

test('web current request failure is reported without stopping the next poll', async () => {
  const { createRequestOwner } = await import('../web/src/requests.js');
  const owner = createRequestOwner(), errors = [], values = [];
  await owner.run(() => Promise.reject(new Error('offline')),
    (value) => values.push(value), (error) => errors.push(error.message));
  await owner.run(() => Promise.resolve([]),
    (value) => values.push(value), (error) => errors.push(error.message));
  assert.deepEqual(errors, ['offline']);
  assert.deepEqual(values, [[]]);
});
