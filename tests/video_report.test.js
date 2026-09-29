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
