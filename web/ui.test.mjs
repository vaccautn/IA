import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { build } from "esbuild";
import { chromium } from "playwright-core";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const reply = (body, status = 200) => ({ body, status });
const job = (id, frames = 3) => ({
  id, name: `${id}.mp4`, status: frames ? "completed" : "pending",
  created_at: "2026-10-12T12:00:00Z", error: null, frames_analyzed: frames,
  summary: null, schema_version: "vacca-video-job-v1"
});
const frame = (id, index) => ({ items: [{
  image_url: `/api/v1/analyses/${id}/frames/${index}/image`, timestamp_ms: (id === "b" ? 2000 : 1000) + index * 200,
  detection_count: index + 1
}], total: 3 });

function deferred() {
  let resolve;
  const promise = new Promise((yes) => { resolve = yes; });
  return { promise, resolve };
}

test("real video hook and rendered interactions with simulated API", { timeout: 60000 }, async (t) => {
  // Bundle the production App and hook, not a reimplementation. Only root
  // ownership is supplied by the harness so React unmount can be exercised.
  const bundle = await build({
    stdin: {
      contents: `import React from "react";
        import { createRoot } from "react-dom/client";
        import { App } from "./src/App.jsx";
        import "./src/brand.css";
        import "./src/style.css";
        const root = createRoot(document.getElementById("root"));
        window.unmountVideo = () => root.unmount();
        root.render(<App />);`,
      resolveDir: webRoot, loader: "jsx"
    },
    bundle: true, write: false, external: ["/videos/fonts/*"],
    outdir: path.join(tmpdir(), "opencode", "vacca-video-ui"),
    define: { "process.env.NODE_ENV": '"production"' }
  });
  const assets = new Map(bundle.outputFiles.map((file) => [path.extname(file.path), file.contents]));
  const publicAssets = new Map(await Promise.all([
    ["vacca_logo.png", "image/png"], ["favicon.svg", "image/svg+xml"],
    ["fonts/Manrope-VariableFont_wght.woff2", "font/woff2"]
  ].map(async ([name, type]) => [`/videos/${name}`, { type, body: await readFile(path.join(webRoot, "public", name)) }])));
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48"><rect width="64" height="48" fill="green"/></svg>';
  const server = createServer((request, response) => {
    if (request.url === "/videos/") {
      response.setHeader("Content-Type", "text/html; charset=utf-8");
      response.end('<html lang="es"><head><link rel="icon" href="/videos/favicon.svg"><link rel="stylesheet" href="/videos/test.css"></head><body><div id="root"></div><script src="/videos/test.js"></script></body></html>');
    } else if (request.url === "/videos/test.js" || request.url === "/videos/test.css") {
      const extension = path.extname(request.url);
      response.setHeader("Content-Type", extension === ".js" ? "text/javascript" : "text/css");
      response.end(assets.get(extension));
    } else if (publicAssets.has(request.url)) {
      const asset = publicAssets.get(request.url);
      response.setHeader("Content-Type", asset.type);
      response.end(asset.body);
    } else {
      response.writeHead(204).end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  // Missing Edge or npm dependencies is a failure, never a silent skip.
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  t.after(() => browser.close());
  const url = `http://127.0.0.1:${server.address().port}/videos/`;

  async function fixture(t) {
    const page = await browser.newPage();
    page.setDefaultTimeout(5000);
    const errors = [], calls = { history: 0, upload: 0, frames: 0 };
    page.on("pageerror", (error) => errors.push(error.message));
    t.after(async () => {
      await page.close();
      assert.deepEqual(errors, [], "No browser JavaScript exceptions");
    });
    // Control only application polling/playback timers. Let React and browser
    // rendering run normally. Ignoring abort deliberately delivers late responses
    // so the real hook must suppress them, rather than relying on the transport.
    await page.addInitScript(() => {
      const timers = new Map();
      let next = -1;
      const timeout = window.setTimeout, interval = window.setInterval;
      const clearTimeout = window.clearTimeout, clearInterval = window.clearInterval;
      window.setTimeout = (callback, delay, ...args) => {
        if (delay !== 2000) return timeout(callback, delay, ...args);
        const id = next--;
        timers.set(id, { callback: () => callback(...args), repeat: false });
        return id;
      };
      window.setInterval = (callback, delay, ...args) => {
        if (delay !== 400) return interval(callback, delay, ...args);
        const id = next--;
        timers.set(id, { callback: () => callback(...args), repeat: true });
        return id;
      };
      window.clearTimeout = (id) => { timers.delete(id); clearTimeout(id); };
      window.clearInterval = (id) => { timers.delete(id); clearInterval(id); };
      window.timerCounts = () => ({
        poll: [...timers.values()].filter((timer) => !timer.repeat).length,
        playback: [...timers.values()].filter((timer) => timer.repeat).length
      });
      window.advanceVideoTimers = (repeat) => {
        for (const [id, timer] of [...timers]) {
          if (timer.repeat !== repeat) continue;
          if (!repeat) timers.delete(id);
          timer.callback();
        }
      };
      const fetch = window.fetch;
      window.apiSettled = 0;
      window.fetch = async (input, options) => {
        const response = await fetch(input, { ...options, signal: undefined });
        await response.clone().text();
        window.apiSettled++;
        return response;
      };
    });
    const mock = {
      history: () => reply([job("a"), job("b")]),
      upload: () => reply(job("created", 0), 202),
      frames: (id, index) => reply(frame(id, index))
    };
    await page.route("**/api/v1/**", async (route) => {
      const request = route.request(), endpoint = new URL(request.url());
      if (/\/frames\/\d+\/image$/.test(endpoint.pathname) || endpoint.pathname.endsWith("/candidate/image")) {
        return route.fulfill({ contentType: "image/svg+xml", body: svg });
      }
      let result;
      if (endpoint.pathname.endsWith("/frames")) {
        calls.frames++;
        result = await mock.frames(endpoint.pathname.split("/")[4], Number(endpoint.searchParams.get("offset")));
      } else if (request.method() === "POST") {
        calls.upload++;
        result = await mock.upload();
      } else {
        calls.history++;
        result = await mock.history();
      }
      await route.fulfill({ status: result.status, contentType: "application/json", body: JSON.stringify(result.body) });
    });
    await page.goto(url);
    assert.equal(new URL(page.url()).pathname, "/videos/");
    assert.equal(await page.getByAltText("VACCA", { exact: true }).getAttribute("src"), "/videos/vacca_logo.png");
    assert.equal(await page.locator('link[rel="icon"]').getAttribute("href"), "/videos/favicon.svg");
    await page.locator('.viewer img[src="/api/v1/analyses/a/frames/0/image"]').waitFor();
    assert.equal(await page.evaluate(async () => {
      await document.fonts.ready;
      return document.fonts.check("16px Manrope");
    }), true, "Prefixed font asset must load");
    await page.getByAltText("VACCA", { exact: true }).evaluate((image) => image.decode());
    await page.waitForFunction(() => window.timerCounts().poll === 1);
    const flush = async (settled) => {
      await page.waitForFunction((count) => window.apiSettled >= count, settled);
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    const selectFile = () => page.locator("input[type=file]").setInputFiles({
      name: "upload.mp4", mimeType: "video/mp4", buffer: Buffer.from("simulated upload")
    });
    return { page, mock, calls, flush, selectFile };
  }

  await t.test("upload selection survives late history and history polling resumes", async (t) => {
    const { page, mock, calls, flush, selectFile } = await fixture(t);
    const late = deferred(), started = deferred();
    mock.history = () => { started.resolve(); return late.promise; };
    await page.evaluate(() => window.advanceVideoTimers(false));
    await started.promise;
    await selectFile();
    await page.getByRole("button", { name: "Procesar video" }).click();
    await page.getByRole("button", { name: /created\.mp4/ }).waitFor();
    const settled = await page.evaluate(() => window.apiSettled);
    late.resolve(reply([job("a"), job("b")]));
    await flush(settled + 1);
    const history = page.getByRole("region", { name: "Mis análisis" });
    assert.equal(await history.getByRole("button", { name: /created\.mp4/ }).getAttribute("aria-current"), "true");
    assert.equal(await page.getByRole("region", { name: "Resultados del análisis" }).getByText("created.mp4", { exact: true }).isVisible(), true);
    await page.waitForFunction(() => window.timerCounts().poll === 1);
    mock.history = () => reply([job("created", 0), job("a"), job("b")]);
    await page.evaluate(() => window.advanceVideoTimers(false));
    await flush(settled + 2);
    assert.equal(calls.history, 3);
    assert.equal(await history.getByRole("button", { name: /created\.mp4/ }).getAttribute("aria-current"), "true");
    assert.equal(await history.getByRole("button").count(), 3);
  });

  await t.test("late frames cannot replace a newly selected analysis or index", async (t) => {
    const { page, mock, flush } = await fixture(t);
    const late = deferred(), started = deferred();
    mock.frames = (id, index) => {
      if (id === "a" && index === 1) { started.resolve(); return late.promise; }
      return reply(frame(id, index));
    };
    await page.getByRole("slider").fill("1");
    await started.promise;
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.locator('.viewer img[src="/api/v1/analyses/b/frames/0/image"]').waitFor();
    let settled = await page.evaluate(() => window.apiSettled);
    late.resolve(reply(frame("a", 1)));
    await flush(settled + 1);
    assert.equal(await page.locator(".viewer img").getAttribute("src"), "/api/v1/analyses/b/frames/0/image");
    assert.equal(await page.getByRole("region", { name: "Mis análisis" }).getByRole("button", { name: /b\.mp4/ }).getAttribute("aria-current"), "true");

    const oldIndex = deferred(), indexStarted = deferred();
    mock.frames = (id, index) => {
      if (index === 1) { indexStarted.resolve(); return oldIndex.promise; }
      return reply(frame(id, index));
    };
    await page.getByRole("slider").fill("1");
    await indexStarted.promise;
    await page.getByRole("slider").fill("2");
    await page.locator('.viewer img[src="/api/v1/analyses/b/frames/2/image"]').waitFor();
    settled = await page.evaluate(() => window.apiSettled);
    oldIndex.resolve(reply(frame("b", 1)));
    await flush(settled + 1);
    assert.equal(await page.getByRole("slider").inputValue(), "2");
    assert.equal(await page.locator(".viewer img").getAttribute("src"), "/api/v1/analyses/b/frames/2/image");
    assert.match(await page.locator(".viewer img").getAttribute("alt"), /2\.40 segundos con 3 detecciones/);
  });

  await t.test("BCS belongs to its selected original frame, not the current preview", async (t) => {
    const { page, mock } = await fixture(t);
    const selected = { frame_index: 97, sample_ordinal: 2, timestamp_ms: 19400,
      timestamp_source: "decoder", detector_confidence: 0.9123, target_index: 0,
      original_bbox: { x1: 10, y1: 8, x2: 30, y2: 24 },
      artifact: "candidate.png", encoding: "PNG", image_width: 1920, image_height: 1080,
      detector_model: { name: "fake", sha256: "fake-hash", confidence: 0.25 } };
    const update = async (bcs, candidate = selected) => {
      mock.history = () => reply([{ ...job("a"), summary: {
        candidate, bcs, selection_policy: "exactly_one_detected_cow_highest_confidence_first_tie"
      } }, job("b")]);
      await page.waitForFunction(() => window.timerCounts().poll === 1);
      await page.evaluate(() => window.advanceVideoTimers(false));
    };
    await update({ status: "completed", bcs_category: 4, model_status: "experimental_not_approved", package_id: "fake-package", reason: null });
    const result = page.getByRole("region", { name: "BCS del frame seleccionado" });
    await result.getByText("Categoría experimental BCS: 4 / 5", { exact: true }).waitFor();
    assert.match(await result.textContent(), /19\.40 s.*97/);
    assert.match(await result.textContent(), /91\.23 %/);
    assert.match(await result.textContent(), /fake-package/);
    assert.match(await result.textContent(), /No aprobado para producción/);
    assert.equal(await result.locator("img").getAttribute("src"), "/api/v1/analyses/a/candidate/image");
    assert.match(await result.locator("img").getAttribute("alt"), /original.*19\.40/);
    assert.equal(await page.locator(".viewer img").getAttribute("src"), "/api/v1/analyses/a/frames/0/image");
    await page.getByRole("slider").fill("1");
    await page.locator('.viewer img[src="/api/v1/analyses/a/frames/1/image"]').waitFor();
    assert.match(await result.textContent(), /19\.40 s/);
    for (const [status, reason, text] of [
      ["busy", "capacity_busy", "capacidad BCS está ocupada"],
      ["unavailable", "capability_unavailable", "no está disponible"],
      ["failed", "inference_failed", "Falló la evaluación"],
      ["invalid_input", "image_rejected", "imagen original no cumple"],
      ["not_evaluated", "no_eligible_frame", "No se puede evaluar"]
    ]) {
      await update({ status, reason }, reason === "no_eligible_frame" ? null : selected);
      await result.getByText(new RegExp(text)).waitFor();
      assert.equal(await result.getByText(/Categoría experimental BCS:/).count(), 0);
    }
    assert.equal(await result.locator("img").count(), 0);
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.locator('.viewer img[src="/api/v1/analyses/b/frames/0/image"]').waitFor();
    assert.equal(await result.count(), 0, "Old jobs without BCS still work");
  });

  await t.test("corrupt summaries display an unavailable diagnostic, not a BCS fallback", async (t) => {
    const { page, mock } = await fixture(t);
    mock.history = () => reply([
      { ...job("a"), summary: null,
        summary_error: "No se pudo leer el resumen guardado; las detecciones y el historial siguen disponibles." },
      job("b")
    ]);
    await page.evaluate(() => window.advanceVideoTimers(false));
    const alert = page.getByRole("alert");
    await alert.getByText(/No se pudo leer el resumen guardado/).waitFor();
    assert.equal(await page.getByRole("region", { name: "BCS del frame seleccionado" }).count(), 0);
    assert.equal(await page.getByText(/No se puede evaluar: ningún frame muestreado/).count(), 0);
    assert.equal(await page.getByRole("button", { name: /b\.mp4/ }).count(), 1);
  });

  await t.test("failed upload clears busy and renders the API error", async (t) => {
    const { page, mock, selectFile } = await fixture(t);
    const failed = deferred(), started = deferred();
    mock.upload = () => { started.resolve(); return failed.promise; };
    await selectFile();
    await page.getByRole("button", { name: "Procesar video" }).click();
    await started.promise;
    assert.equal(await page.getByRole("button", { name: "Cargando video…" }).isDisabled(), true);
    assert.equal(await page.locator("input[type=file]").isDisabled(), true);
    failed.resolve(reply({ detail: "Error de carga simulado" }, 503));
    await page.getByRole("alert").waitFor();
    assert.match(await page.getByRole("alert").textContent(), /Error de carga simulado/);
    assert.equal(await page.getByRole("button", { name: "Procesar video" }).isDisabled(), false);
    assert.equal(await page.locator("input[type=file]").isDisabled(), false);
  });

  await t.test("selecting another analysis resets index and stops playback", async (t) => {
    const { page } = await fixture(t);
    await page.getByRole("slider").fill("1");
    await page.locator('.viewer img[src="/api/v1/analyses/a/frames/1/image"]').waitFor();
    await page.getByRole("button", { name: "Reproducir frames" }).click();
    await page.waitForFunction(() => window.timerCounts().playback === 1);
    await page.evaluate(() => window.advanceVideoTimers(true));
    await page.locator('.viewer img[src="/api/v1/analyses/a/frames/2/image"]').waitFor();
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.locator('.viewer img[src="/api/v1/analyses/b/frames/0/image"]').waitFor();
    assert.equal(await page.getByRole("slider").inputValue(), "0");
    await page.getByRole("button", { name: "Reproducir frames" }).waitFor();
    assert.equal(await page.evaluate(() => window.timerCounts().playback), 0);
    await page.evaluate(() => window.advanceVideoTimers(true));
    assert.equal(await page.getByRole("slider").inputValue(), "0");
  });

  await t.test("React unmount stops timers and ignores in-flight history/frame/upload results", async (t) => {
    const { page, mock, calls, flush, selectFile } = await fixture(t);
    const pending = [deferred(), deferred(), deferred()];
    const started = [deferred(), deferred(), deferred()];
    mock.history = () => { started[0].resolve(); return pending[0].promise; };
    mock.frames = () => { started[1].resolve(); return pending[1].promise; };
    mock.upload = () => { started[2].resolve(); return pending[2].promise; };
    await page.getByRole("button", { name: "Reproducir frames" }).click();
    await page.waitForFunction(() => window.timerCounts().playback === 1);
    await page.evaluate(() => window.advanceVideoTimers(false));
    await page.getByRole("slider").fill("1");
    await selectFile();
    await page.getByRole("button", { name: "Procesar video" }).click();
    await Promise.all(started.map((item) => item.promise));
    // Seeking paused the earlier playback; start it again to verify unmount
    // clears an active interval as well as the in-flight polling cycle.
    await page.getByRole("button", { name: "Reproducir frames" }).click();
    await page.waitForFunction(() => window.timerCounts().playback === 1);
    const settled = await page.evaluate(() => window.apiSettled);
    const before = { ...calls };
    await page.evaluate(() => window.unmountVideo());
    assert.deepEqual(await page.evaluate(() => window.timerCounts()), { poll: 0, playback: 0 });
    pending[0].resolve(reply([job("stale")]));
    pending[1].resolve(reply(frame("a", 1)));
    pending[2].resolve(reply(job("late-upload", 0), 202));
    await flush(settled + 3);
    await page.evaluate(() => { window.advanceVideoTimers(false); window.advanceVideoTimers(true); });
    assert.deepEqual(calls, before);
    assert.deepEqual(await page.evaluate(() => window.timerCounts()), { poll: 0, playback: 0 });
    assert.equal(await page.locator("#root").textContent(), "");
  });
});
