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
  summary: null, schema_version: "vacca-video-job-v1",
  ...(frames ? { highlighted_frame: {
    frame_index: 17, sample_index: 1, timestamp_ms: 1200, detection_count: 2,
    mean_confidence: 0.875, detections: [{ confidence: 0.9 }, { confidence: 0.85 }],
    image_url: `/api/v1/analyses/${id}/frames/1/image`
  } } : {})
});
const frame = (id, index) => ({ items: [{
  image_url: `/api/v1/analyses/${id}/frames/${index}/image`, timestamp_ms: (id === "b" ? 2000 : 1000) + index * 200,
  detection_count: index + 1
}], total: 3 });

const track = (id, trackId, status, warnings = []) => ({
  track_id: trackId, status, warnings, observations: 12, clip_frames: 48, first_timestamp_ms: 500, last_timestamp_ms: 2500,
  segments: [{}, ...(warnings.includes("long_gap_review_required") ? [{}] : [])],
  clip_url: `/api/v1/analyses/${id}/tracks/${trackId}/clip`, clip_media_type: "video/webm",
  best_crop_url: `/api/v1/analyses/${id}/tracks/${trackId}/best-crop`,
  best_original_url: `/api/v1/analyses/${id}/tracks/${trackId}/best-original`
});
const tracks = (id) => ({
  schema_version: "vacca-video-tracks-v1", status: "completed", stop_reason: "end_of_video",
  tracks: [{ ...track(id, 1, "candidate"), best_masked_url: `/api/v1/analyses/${id}/tracks/1/best-masked` }, { ...track(id, 7, "review", ["long_gap_review_required", "other_detection_in_crop_at_best_frame"]), possible_continuation_of: 1 }]
});
const passage = (id, sequence, { original = true, bcs = { status: "evaluated", bcs_category: sequence, reason: null, input_representation: "original_only", model_status: "experimental_not_approved" }, warnings = [] } = {}) => ({
  sequence, track_id: sequence + 10, status: "passage_evidence_candidate", start_frame: sequence, end_frame: sequence + 2,
  start_timestamp_ms: sequence * 1000, end_timestamp_ms: (sequence + 2) * 1000, end_exclusive: true,
  transition_method: "initial_confidence", candidate: { status: original ? "available" : "not_available", bcs_eligibility: "not_assessed", bcs_status: "not_evaluated", ...(original ? { frame_index: sequence * 10, timestamp_ms: sequence * 1000 } : {}), best_original_url: original ? `/api/v1/analyses/${id}/passages/${sequence}/best-original` : null }, warnings, bcs
});
const passageList = (id, passages = []) => ({ schema_version: "vacca-video-passages-v1", status: "completed", bcs_status: "completed", evidence_status: "provisional_evidence_not_unique_animal_identity", identity_semantics: "temporal_track_id_within_video", calibration: "uncalibrated_confidence_proxy_not_visibility_or_bcs", passages, unresolved_count: 1, unresolved_tracks: [{ track_id: 91, first_frame: 2, last_frame: 4, first_timestamp_ms: 200, last_timestamp_ms: 400, possible_continuation_of: null, possible_continued_by: null, status: "unresolved_passage_attribution" }], warnings: ["untracked_detection_attribution_unresolved"] });

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
  const zipBytes = Buffer.from([0x50, 0x4b, 0x03, 0x04, 0x66, 0x72, 0x61, 0x6d, 0x65]);
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
    const errors = [], calls = { history: 0, upload: 0, frames: 0, passages: 0 };
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
        if (delay !== 2000 && delay !== 3000) return timeout(callback, delay, ...args);
        const id = next--;
        timers.set(id, { callback: () => callback(...args), repeat: false, delay });
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
        poll: [...timers.values()].filter((timer) => timer.delay === 2000).length,
        passage: [...timers.values()].filter((timer) => timer.delay === 3000).length,
        playback: [...timers.values()].filter((timer) => timer.repeat).length
      });
      window.advanceVideoTimers = (repeat, delay) => {
        for (const [id, timer] of [...timers]) {
          if (timer.repeat !== repeat || (delay && timer.delay !== delay)) continue;
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
      frames: (id, index) => reply(frame(id, index)),
      tracks: (id) => reply(tracks(id)),
      passages: () => reply(passageList("a")), passageImageFails: false
    };
    await page.route("**/api/v1/**", async (route) => {
      const request = route.request(), endpoint = new URL(request.url());
      if (endpoint.pathname.endsWith("/frames.zip")) {
        return route.fulfill({ status: 200, contentType: "application/zip", headers: {
          "content-disposition": `attachment; filename="frames-${endpoint.pathname.split("/")[4]}.zip"`
        }, body: zipBytes });
      }
      if (endpoint.pathname.endsWith("/results")) {
        return route.fulfill({ status: 200, contentType: "application/x-ndjson",
          headers: { "content-disposition": `attachment; filename="${endpoint.pathname.split("/")[4]}.jsonl"` },
          body: Buffer.from('{"frame_index":17}\n') });
      }
      if (/\/tracks\/\d+\/clip$/.test(endpoint.pathname)) {
        calls.clips = (calls.clips ?? 0) + 1;
        return route.fulfill({ status: 204 });
      }
      if (endpoint.pathname.endsWith("/tracks")) {
        calls.tracks = (calls.tracks ?? 0) + 1;
        const result = await mock.tracks(endpoint.pathname.split("/")[4]);
        return route.fulfill({ status: result.status, contentType: "application/json", body: JSON.stringify(result.body) });
      }
      if (endpoint.pathname.endsWith("/passages")) {
        calls.passages++;
        const result = await mock.passages(endpoint.pathname.split("/")[4]);
        return route.fulfill({ status: result.status, contentType: "application/json", body: JSON.stringify(result.body) });
      }
      if (/\/frames\/\d+\/image$/.test(endpoint.pathname) || endpoint.pathname.endsWith("/candidate/image")
        || /\/(tracks\/\d+\/(best-crop|best-original|best-masked)|passages\/\d+\/best-original)$/.test(endpoint.pathname)) {
        if (endpoint.pathname.includes("/passages/") && mock.passageImageFails) return route.fulfill({ status: 404 });
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

  await t.test("downloads keep binary payloads and the highlight seeks without changing BCS", async (t) => {
    const { page } = await fixture(t);
    const highlight = page.getByRole("region", { name: "Frame destacado por detecciones" });
    await highlight.getByText(/confianza promedio 87\.5 %/).waitFor();
    await highlight.getByRole("button", { name: "Ver frame destacado" }).click();
    await page.locator('.viewer img[src="/api/v1/analyses/a/frames/1/image"]').waitFor();
    assert.equal(await page.getByRole("slider").inputValue(), "1");
    assert.equal(await page.getByRole("region", { name: "BCS del frame seleccionado" }).count(), 0);
    for (const [name, signature] of [["Descargar frames (ZIP)", "PK"], ["Descargar detecciones (JSONL)", "{\"frame_index\":17}"]]) {
      const [download] = await Promise.all([
        page.waitForEvent("download"),
        page.getByRole("link", { name }).click()
      ]);
      assert.match(download.suggestedFilename(), name.includes("ZIP") ? /^frames-a\.zip$/ : /^a\.jsonl$/);
      const stream = await download.createReadStream();
      const chunks = [];
      for await (const chunk of stream) chunks.push(chunk);
      assert.ok(Buffer.concat(chunks).includes(Buffer.from(signature)));
    }
  });

  await t.test("track gallery separates review tracks, toggles clips and isolates failures", async (t) => {
    const { page, mock } = await fixture(t);
    const gallery = page.getByRole("region", { name: "Trayectorias por vaca" });
    await gallery.getByText("2 trayectorias · 1 para revisión").waitFor();
    const cards = gallery.getByRole("listitem").filter({ hasText: /^Trayectoria \d/ });
    assert.equal(await cards.count(), 2);
    const review = gallery.getByRole("listitem").filter({ hasText: "Trayectoria 7" });
    await review.getByText("Revisar").waitFor();
    await review.getByText("Ausencia larga: trayectoria fragmentada").waitFor();
    await review.getByText("Otra vaca dentro del recorte").waitFor();
    assert.match(await review.textContent(), /0\.50–2\.50 s · 12 observaciones · 2 tramos/);
    await review.getByText("Puede ser el mismo animal que la trayectoria 1.").waitFor();
    assert.equal(await gallery.getByRole("listitem").filter({ hasText: "Trayectoria 1" }).getByText("Candidata").count(), 1);
    assert.equal(await review.getByAltText("Mejor recorte de la trayectoria 7").getAttribute("src"), "/api/v1/analyses/a/tracks/7/best-crop");
    assert.equal(await review.getByRole("link", { name: "Frame original" }).getAttribute("href"), "/api/v1/analyses/a/tracks/7/best-original");
    await review.getByRole("button", { name: "Ver clip" }).click();
    assert.equal(await review.locator("video").getAttribute("src"), "/api/v1/analyses/a/tracks/7/clip");
    assert.equal(await review.getByRole("button", { name: "Ver recorte" }).getAttribute("aria-expanded"), "true");
    await review.getByRole("button", { name: "Ver recorte" }).click();
    assert.equal(await review.locator("video").count(), 0);
    const candidate = gallery.getByRole("listitem").filter({ hasText: "Trayectoria 1" });
    assert.equal(await candidate.getByAltText("Silueta de la trayectoria 1, sin el fondo").getAttribute("src"), "/api/v1/analyses/a/tracks/1/best-masked");
    await candidate.getByRole("button", { name: "Ver con fondo" }).click();
    assert.equal(await candidate.getByAltText("Mejor recorte de la trayectoria 1").getAttribute("src"), "/api/v1/analyses/a/tracks/1/best-crop");
    await candidate.getByRole("button", { name: "Ver silueta" }).waitFor();
    assert.equal(await review.getByRole("button", { name: /silueta|fondo/ }).count(), 0, "No silhouette toggle without an accepted mask");

    const single = { ...tracks("a"), tracks: [{ ...track("a", 3, "review", ["short_track_review_required"]), clip_frames: 1 }] };
    mock.tracks = () => reply(single);
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    const lone = gallery.getByRole("listitem").filter({ hasText: "Trayectoria 3" });
    await lone.getByText("Pocas observaciones").waitFor();
    assert.equal(await lone.getByRole("button", { name: "Ver clip" }).count(), 0, "One-frame clips are not offered");
    mock.tracks = (id) => reply(tracks(id));
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    await gallery.getByText("2 trayectorias · 1 para revisión").waitFor();

    mock.tracks = () => reply({ detail: "Falla simulada" }, 500);
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await gallery.getByText(/No se pudieron cargar las trayectorias/).waitFor();
    assert.equal(await page.getByRole("alert").count(), 0, "Track failures never raise the page alert");
    await page.locator('.viewer img[src="/api/v1/analyses/b/frames/0/image"]').waitFor();
  });

  await t.test("passage gallery preserves endpoint order, originals, evaluated-only categories and attribution warnings", async (t) => {
    const { page, mock } = await fixture(t);
    mock.passages = (id) => reply({ ...passageList(id, [
      passage(id, 2, { warnings: ["unknown <warning>"] }),
      passage(id, 1, { bcs: { status: "unevaluable", bcs_category: null, reason: "image_rejected", input_representation: "original_only" } }),
      passage(id, 3, { original: false, bcs: null })
    ]), bcs_status: "unavailable", bcs_reason: "passage_bcs_unreadable", warnings: ["unknown <endpoint>"] });
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    await page.getByRole("region", { name: "Pasajes del video" }).waitFor();
    const gallery = page.getByRole("region", { name: "Pasajes del video" });
    const cards = gallery.locator(".passage-card");
    assert.equal(await cards.count(), 3);
    assert.match(await cards.nth(0).textContent(), /Pasaje 2.*Categoría BCS experimental: 2/);
    assert.equal(await cards.nth(0).locator("img").getAttribute("src"), "/api/v1/analyses/a/passages/2/best-original");
    assert.match(await cards.nth(1).textContent(), /Pasaje 1.*No evaluado.*rechazó.*imagen/i);
    assert.equal(await cards.nth(1).getByText(/Categoría BCS experimental:/).count(), 0);
    assert.match(await cards.nth(2).textContent(), /Imagen original no disponible.*No evaluado/);
    assert.equal(await cards.nth(2).locator("img").count(), 0);
    assert.match(await gallery.textContent(), /BCS por pasaje no está disponible/);
    assert.match(await gallery.textContent(), /1 atribución sin resolver.*91/);
    assert.ok((await gallery.textContent()).includes("unknown <warning>") && (await gallery.textContent()).includes("unknown <endpoint>"));
    assert.equal(await page.locator("warning").count(), 0, "Unknown markup remains text, not elements");
    assert.match(await gallery.textContent(), /provisional.*no tienen alcance|provisional.*solo tienen alcance/i);
    assert.match(await gallery.textContent(), /no identifican vacas únicas/i);
    assert.equal(await page.getByRole("region", { name: "BCS del frame seleccionado" }).count(), 0);
    assert.equal(await page.getByRole("region", { name: "Trayectorias por vaca" }).locator(".track").count(), 2);
    mock.passageImageFails = true;
    mock.passages = (id) => reply(passageList(id, [passage(id, 7)]));
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    const failedImage = page.getByRole("region", { name: "Pasajes del video" }).locator(".passage-card").filter({ hasText: "Pasaje 7" });
    await failedImage.getByText("Imagen original no disponible.").waitFor();
    assert.equal(await failedImage.locator("img").count(), 0, "A failed original renders text, not substitute media");
  });

  await t.test("completed empty passages retain unresolved attribution and endpoint warnings", async (t) => {
    const { page, mock } = await fixture(t);
    mock.passages = (id) => reply({ ...passageList(id), passages: [], bcs_status: "completed", warnings: ["untracked_detection_attribution_unresolved", "unknown <endpoint>"] });
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    const gallery = page.getByRole("region", { name: "Pasajes del video" });
    await gallery.getByText(/No hay pasajes disponibles/).waitFor();
    assert.match(await gallery.textContent(), /1 atribución sin resolver.*trayectoria 91/);
    assert.match(await gallery.textContent(), /detecciones cuya atribución a un pasaje no se resolvió/);
    assert.ok((await gallery.textContent()).includes("unknown <endpoint>"));
    assert.equal(await gallery.getByText("Pasadas detectadas: 0", { exact: true }).count(), 1);
    assert.equal(await gallery.locator(".passage-card").count(), 0, "The empty state has an explicit empty-response setup");
  });

  await t.test("non-empty passage gallery remains distinct beside global candidate BCS", async (t) => {
    const { page, mock, flush } = await fixture(t);
    const candidate = { frame_index: 17, timestamp_ms: 1200, detector_confidence: 0.9 };
    mock.history = () => reply([{ ...job("a"), summary: { candidate, bcs: { status: "completed", bcs_category: 4, model_status: "experimental_not_approved", package_id: "fixture", reason: null } } }, job("b")]);
    mock.passages = (id) => reply(passageList(id, [passage(id, 4)]));
    const settled = await page.evaluate(() => window.apiSettled);
    await page.evaluate(() => window.advanceVideoTimers(false, 2000));
    await flush(settled + 1);
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    const gallery = page.getByRole("region", { name: "Pasajes del video" });
    await gallery.getByText("Pasaje 4").waitFor();
    await page.getByRole("region", { name: "BCS del frame seleccionado" }).getByText("Categoría experimental BCS: 4 / 5", { exact: true }).waitFor();
    assert.match(await gallery.textContent(), /provisional/);
    assert.match(await gallery.textContent(), /no identifican vacas únicas/i);
    assert.equal(await page.getByRole("region", { name: "Trayectorias por vaca" }).locator(".track").count(), 2);
  });

  await t.test("completed passage count excludes unresolved tracks and labels each supplied best frame with its BCS", async (t) => {
    const { page, mock } = await fixture(t);
    mock.passages = (id) => reply(passageList(id, [
      passage(id, 2),
      passage(id, 1, { bcs: { status: "unevaluable", bcs_category: null, reason: "image_rejected", input_representation: "original_only" } })
    ]));
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    const gallery = page.getByRole("region", { name: "Pasajes del video" });
    await gallery.getByText("Pasadas detectadas: 2", { exact: true }).waitFor();
    const cards = gallery.locator(".passage-card");
    assert.equal(await cards.count(), 2);
    assert.match(await cards.nth(0).textContent(), /Mejor frame seleccionado para este pasaje: frame 20 · 2\.00 s/);
    assert.match(await cards.nth(0).textContent(), /Categoría BCS experimental: 2 \/ 5/);
    assert.match(await cards.nth(1).textContent(), /Mejor frame seleccionado para este pasaje: frame 10 · 1\.00 s/);
    assert.match(await cards.nth(1).textContent(), /No evaluado: El servicio rechazó la imagen original/);
    assert.equal(await cards.nth(1).getByText(/Categoría BCS experimental:/).count(), 0);
    assert.match(await gallery.textContent(), /no calibrados ni clínicos/);
  });

  await t.test("pending and unavailable passage states keep count unknown and escape unknown reasons", async (t) => {
    const { page, mock } = await fixture(t);
    for (const [status, reason] of [["disabled", "unknown <reason>"], ["unavailable", "unknown <unavailable>"]]) {
      mock.passages = () => reply({ status, reason, passages: [], bcs_status: "not_evaluated" });
      await page.getByRole("button", { name: /b\.mp4/ }).click();
      await page.getByRole("button", { name: /a\.mp4/ }).click();
      const gallery = page.getByRole("region", { name: "Pasajes del video" });
      await page.waitForFunction((value) => document.querySelector(".passages")?.textContent.includes(value), reason);
      assert.equal(await gallery.getByText(/Pasadas detectadas:/).count(), 0, `${status} count stays unknown`);
      assert.equal(await page.locator("reason, unavailable").count(), 0, "Unknown reasons remain text, not markup");
    }
  });

  await t.test("passage endpoint errors are not empty success, retry works and empty completion is explicit", async (t) => {
    const { page, mock } = await fixture(t);
    mock.passages = (id) => reply({ status: "failed", reason: "video_not_completed", passages: [], bcs_status: "not_evaluated" });
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("region", { name: "Pasajes del video" }).getByText(/El endpoint informó.*video no se completó/i).waitFor();
    assert.equal(await page.getByRole("region", { name: "Pasajes del video" }).getByText(/Pasadas detectadas:/).count(), 0);
    mock.passages = (id) => reply(passageList(id));
    await page.getByRole("region", { name: "Pasajes del video" }).getByRole("button", { name: "Reintentar" }).click();
    await page.getByRole("region", { name: "Pasajes del video" }).getByText(/No hay pasajes disponibles/).waitFor();
  });

  await t.test("pending passage refresh ignores job completion, discards stale selections, retries and stops on unmount", async (t) => {
    const { page, mock, flush, calls } = await fixture(t);
    const oldA = deferred(), oldAStarted = deferred();
    let aRequests = 0;
    mock.passages = (id) => id === "a" && ++aRequests === 1 ? (oldAStarted.resolve(), oldA.promise) : reply(passageList(id, [passage(id, id === "a" ? 7 : 8)]));
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("region", { name: "Pasajes del video" }).getByText("Pasaje 8").waitFor();
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    await oldAStarted.promise;
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    const returnedToA = page.getByRole("region", { name: "Pasajes del video" });
    await returnedToA.getByText("Pasaje 7").waitFor();
    const beforeOldA = await page.evaluate(() => window.apiSettled);
    oldA.resolve(reply(passageList("a", [passage("a", 99)])));
    await flush(beforeOldA + 1);
    assert.equal(await returnedToA.getByText("Pasaje 99").count(), 0, "A→B→A ignores the first A request");
    const pending = deferred(), started = deferred();
    mock.passages = (id) => id === "b" ? (started.resolve(), pending.promise) : reply(passageList(id, [passage(id, 8)]));
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await started.promise;
    assert.equal(await page.getByRole("region", { name: "Pasajes del video" }).getByText(/Pasadas detectadas:/).count(), 0, "Loading count is unknown");
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    const gallery = page.getByRole("region", { name: "Pasajes del video" });
    await gallery.getByText("Pasaje 8").waitFor();
    const settled = await page.evaluate(() => window.apiSettled);
    pending.resolve(reply(passageList("b", [passage("b", 99)])));
    await flush(settled + 1);
    assert.equal(await gallery.getByText("Pasaje 99").count(), 0);
    const lateError = deferred(), errorStarted = deferred();
    mock.passages = (id) => id === "b" ? (errorStarted.resolve(), lateError.promise) : reply(passageList(id, [passage(id, 8)]));
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await errorStarted.promise;
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    await gallery.getByText("Pasaje 8").waitFor();
    const beforeLateError = await page.evaluate(() => window.apiSettled);
    lateError.resolve(reply({ detail: "Falla obsoleta" }, 500));
    await flush(beforeLateError + 1);
    assert.equal(await page.getByRole("alert").count(), 0, "Stale errors stay out of the selected analysis UI");
    const beforePending = calls.passages;
    mock.passages = (id) => reply({ status: "pending", passages: [], bcs_status: "not_evaluated" });
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await gallery.getByText(/Actualizando pasajes/).waitFor();
    assert.equal(await gallery.getByText(/Pasadas detectadas:/).count(), 0, "Pending count is unknown");
    assert.equal(calls.passages, beforePending + 1, "The selected completed job triggers its first independent passage request");
    assert.equal(await page.evaluate(() => window.timerCounts().passage), 1, "Pending response schedules one passage refresh");
    mock.passages = (id) => reply(passageList(id, [passage(id, 4)]));
    await page.evaluate(() => window.advanceVideoTimers(false, 3000));
    await gallery.getByText("Pasaje 4").waitFor();
    assert.equal(calls.passages, beforePending + 2, "Only one pending refresh ran before terminal settlement");
    assert.equal(await page.evaluate(() => window.timerCounts().passage), 0, "Terminal response stops passage polling");
    mock.passages = () => reply({ detail: "Falla simulada" }, 500);
    await page.getByRole("button", { name: /a\.mp4/ }).click();
    await gallery.getByRole("button", { name: "Reintentar" }).waitFor();
    assert.equal(await gallery.getByText(/Pasadas detectadas:/).count(), 0, "Request-error count is unknown");
    mock.passages = (id) => reply(passageList(id));
    await gallery.getByRole("button", { name: "Reintentar" }).click();
    await gallery.getByText(/No hay pasajes disponibles/).waitFor();
    mock.passages = (id) => reply({ status: "pending", passages: [], bcs_status: "not_evaluated" });
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await gallery.getByText(/Actualizando pasajes/).waitFor();
    assert.equal(await page.evaluate(() => window.timerCounts().passage), 1, "A pending selected passage request owns a scheduled refresh");
    const requestsBeforeUnmount = calls.passages;
    await page.evaluate(() => window.unmountVideo());
    assert.equal(await page.evaluate(() => window.timerCounts().passage), 0, "Unmount clears the scheduled passage timeout");
    await page.evaluate(() => window.advanceVideoTimers(false, 3000));
    assert.equal(calls.passages, requestsBeforeUnmount, "Unmount prevents the scheduled refresh request");
  });

  await t.test("pending tracks show detections first and refresh until tracking finishes", async (t) => {
    const { page, mock, calls } = await fixture(t);
    await page.getByRole("region", { name: "Trayectorias por vaca" }).getByText("2 trayectorias · 1 para revisión").waitFor();
    let answered = 0;
    mock.tracks = (id) => (++answered === 1
      ? reply({ schema_version: "vacca-video-tracks-v1", status: "pending", tracks: [] })
      : reply(tracks(id)));
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    const gallery = page.getByRole("region", { name: "Trayectorias por vaca" });
    await gallery.getByText(/Calculando trayectorias/).waitFor();
    await page.locator('.viewer img[src="/api/v1/analyses/b/frames/0/image"]').waitFor();
    await page.evaluate(() => window.advanceVideoTimers(false, 3000));
    await gallery.getByText("2 trayectorias · 1 para revisión").waitFor({ timeout: 6000 });
    assert.equal(answered, 2);
    assert.equal(await page.evaluate(() => window.timerCounts().poll), 1, "Track refresh never uses the history poll timer");
    assert.ok(calls.tracks >= 3);
  });

  await t.test("React unmount stops timers and ignores in-flight history/frame/upload results", async (t) => {
    const { page, mock, calls, flush, selectFile } = await fixture(t);
    const pending = [deferred(), deferred(), deferred(), deferred()];
    const started = [deferred(), deferred(), deferred(), deferred()];
    mock.history = () => { started[0].resolve(); return pending[0].promise; };
    mock.frames = () => { started[1].resolve(); return pending[1].promise; };
    mock.upload = () => { started[2].resolve(); return pending[2].promise; };
    mock.passages = () => { started[3].resolve(); return pending[3].promise; };
    await page.getByRole("button", { name: "Reproducir frames" }).click();
    await page.waitForFunction(() => window.timerCounts().playback === 1);
    await page.evaluate(() => window.advanceVideoTimers(false));
    await page.getByRole("slider").fill("1");
    await selectFile();
    await page.getByRole("button", { name: "Procesar video" }).click();
    await Promise.all(started.slice(0, 3).map((item) => item.promise));
    // Seeking paused the earlier playback; start it again to verify unmount
    // clears an active interval as well as the in-flight polling cycle.
    await page.getByRole("button", { name: "Reproducir frames" }).click();
    await page.waitForFunction(() => window.timerCounts().playback === 1);
    const settled = await page.evaluate(() => window.apiSettled);
    await page.getByRole("button", { name: /b\.mp4/ }).click();
    await started[3].promise;
    const before = { ...calls };
    await page.evaluate(() => window.unmountVideo());
    assert.deepEqual(await page.evaluate(() => window.timerCounts()), { poll: 0, passage: 0, playback: 0 });
    pending[0].resolve(reply([job("stale")]));
    pending[1].resolve(reply(frame("a", 1)));
    pending[2].resolve(reply(job("late-upload", 0), 202));
    pending[3].resolve(reply(passageList("a", [passage("a", 99)])));
    await flush(settled + 4);
    await page.evaluate(() => { window.advanceVideoTimers(false); window.advanceVideoTimers(true); });
    assert.deepEqual(calls, before);
    assert.deepEqual(await page.evaluate(() => window.timerCounts()), { poll: 0, passage: 0, playback: 0 });
    assert.equal(await page.locator("#root").textContent(), "");
  });
});
