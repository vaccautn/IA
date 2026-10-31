## Exploration: IA web passage validation

### Current State
The IA web accepts local MP4/MOV uploads through `POST /api/v1/analyses`, polls analysis jobs, and renders sampled annotated frames, one global candidate BCS result, and tracks. Its API client has no `/passages` method, and the UI never renders the endpoint's ordered provisional passages. The backend already provides `GET /api/v1/analyses/{job_id}/passages` and per-sequence image routes. The response distinguishes pending, failed, disabled, and unavailable states; completed responses contain ordered passage entries, original-image URLs when available, authoritative BCS outcomes/reasons, and unresolved-attribution warnings.

Existing evidence is substantial: `outputs/5cow-validation-20261028/passages-response.json` records five provisional passages in sequence with categories 4, 4, 3, 3, 3 and original-image URLs; `validation-summary.json` records HTTP 200 and successful original-image URL checks. This is a validation result, not evidence of animal identity or clinical validity. The existing UI integration test bundles the real React app and hook with a simulated API; backend passage route tests use synthetic artifacts. Both are suitable for adding deterministic coverage without model inference.

### Affected Areas
- `web/src/api.js` — add a client call for the existing passage-list endpoint; passage original URLs can be rendered directly from the response.
- `web/src/useVideoAnalysis.js` — load/reset passage state for the selected completed analysis and refresh while the endpoint reports pending, without conflating it with history/tracks polling or global-candidate BCS.
- `web/src/components.jsx` and `web/src/style.css` — render passages in endpoint sequence, each with its original image and experimental category or a readable no-evaluation reason; show endpoint/unresolved warnings and non-clinical/provisional disclaimers.
- `web/ui.test.mjs` — extend the rendered-app harness to verify endpoint calls, order, original image URLs, category and unevaluable reason rendering, and pending/error/selection behavior.
- `tests/test_video_passages_api.py`, `src/vacca_video/webapp.py`, `src/vacca_video/schemas.py` — existing endpoint contract and response allowlist to reference/retain; no backend change is currently indicated.

### Approaches
1. **Render the existing passage endpoint in the IA harness (recommended)** — Add narrowly scoped UI/API state and an ordered result section, consuming existing endpoint statuses, BCS outcomes/reasons, and original-image URLs.
   - Pros: Directly evidences the actual implemented API behavior, preserves sequence and per-passage semantics, reuses backend routes and browser test harness, avoids new dependencies and model execution.
   - Cons: Requires separate pending/error states and request lifecycle handling; images may be absent for passages without a captured original.
   - Effort: Medium.

2. **Extend the existing global candidate or tracks panels to represent passages** — Fold passage entries into currently rendered candidate/tracks state.
   - Pros: Fewer visible sections in the UI.
   - Cons: Conflates distinct API concepts and existing non-equivalent selection rules; risks implying tracks/global candidate are the authoritative ordered passage results.
   - Effort: Medium, with higher semantic risk.

### Recommendation
Choose approach 1 and keep the change strictly within the disposable IA validation web. Reuse `videoApi`/the existing React hook and rendered UI test harness; display sequence, timing/track reference as provisional within-video evidence, full original image, category only when `bcs.status === "evaluated"`, and a reason/status otherwise. Handle list-level statuses explicitly and avoid claiming unique-animal identity, visibility/eligibility, calibration, or clinical validity. Do not change backend routes or schemas, add dependencies, connect R2/backend, or modify generated validation outputs. The web lockfile is `web/package-lock.json`; Node commands must run with `web` as working directory (not IA root). Strict TDD is feasible and appropriate based on the existing `node --test` rendered UI suite and synthetic Python endpoint tests; the prior init's unconfigured/stale TDD note is not evidence against current test coverage.

### Risks
- Passage BCS outcomes are experimental categories and can be unevaluable; the UI must never render a category for a missing/non-evaluated result or present it as a clinical score.
- `track_id` is temporal and only scoped to the video; neither tracks nor passage sequence establish individual-animal identity. Unresolved tracks/warnings should not be hidden if the UI implies completeness.
- The global job can be completed while passage processing is still pending; passage polling and cancellation must be tied to selected-job lifecycle and must not block existing detections/results.
- UI tests can deterministically prove request/render behavior using mocked endpoint responses, but cannot prove live model availability. A later manual smoke test with a prepared local video/model may be useful; it is not required for exploration and must avoid mutating saved validation outputs.
- Preserve the nine already-dirty IA Markdown files; only this named exploration artifact is authorized to be written in this phase.

### Ready for Proposal
Yes. Propose the smallest front-end-only addition that consumes the existing `/passages` and passage-original routes, with rendered UI tests for completed, pending, unavailable/error, category/reason, order, and selected-job changes. Keep backend/R2/productization, production architecture, new dependencies, identity guarantees, and clinical-validity claims out of scope. Use `cd web; npm test` and `cd web; npm run build` for web verification, plus focused existing Python API tests only if the implementation touches backend contracts; do not run model inference for the UI tests.
