# Design: IA Web Passage Validation

## Technical Approach

Extend the local frontend with one API method, one request lifecycle, and one passage section. Implement `specs/ia-web-passage-validation/spec.md`; preserve global candidate BCS, tracks, backend, dependencies, models, and outputs. New UI copy uses professional Spanish.

## Architecture Decisions

| Option | Tradeoff | Decision |
|---|---|---|
| Extend files / gallery infrastructure | No abstraction layer needed | Extend five files; `App` already spreads hook state into `Results`. |
| Endpoint-owned / job-owned completion | Job completion alone cannot describe passage readiness | Request for every selected ID; continue only while response `status === "pending"`, regardless of job status. |
| Existing request owner / abort alone | Abort may be ignored by transport | Reuse `createRequestOwner`, cancel and clear timeout on cleanup; tag state with selected ID to hide mismatched data immediately. |
| Manual / automatic retries | Automatic failures can loop indefinitely | Stop on request failure; section-local retry counter restarts effect without reselecting ID. |
| Original image error fallback / substitute media | Image availability can change between listing and GET | On image error show unavailable text only; never substitute crops, masks, candidate, or preview images. |

## Data Flow

`History/upload selection → hook effect → videoApi.passages → GET /analyses/{id}/passages → owned state → Results/Passages`

After settlement: `pending → timeout → request`; terminal/error → no timeout. Reuse `TRACKS_POLL_INTERVAL_MS` (3000 ms); no overlapping requests. Selection/retry/unmount cancels owner and timeout. A→B→A creates fresh owners.

### Completion evidence

`src/vacca_video/worker.py:155–169` invokes `bcs.py:131–197` before publishing final job status. `webapp.py:152–175` keeps processing jobs pending even with completed tracking metadata. BCS-in-flight means processing job plus pending passages. Also test completed job plus pending passages (running tracking metadata is supported); job completion never stops polling.

Passages GET returns **200**, including pending/failed/disabled/unavailable bodies; upload POST returns **202**. Preserve `response.ok` handling. Aggregate `bcs_status` has no pending value: completed passages with `not_evaluated` is terminal.

## File Changes

Implementation forecast; paths relative to repository root.

| File | Action | Description |
|---|---|---|
| `web/src/api.js` | Modify | Add encoded-ID, abortable `passages(id, signal)`. |
| `web/src/useVideoAnalysis.js` | Modify | Independent passage data/loading/error, ID ownership, refresh cleanup and `retryPassages`. |
| `web/src/components.jsx` | Modify | Wire `Results`; add passage section/cards, translations and image fallback. |
| `web/src/style.css` | Modify | Small scoped responsive gallery styles using existing tokens. |
| `web/ui.test.mjs` | Modify | Add passage/image routes, fixtures, counters and lifecycle/rendering cases. |

## Interfaces / Contracts

Hook adds `passages`, `passagesLoading`, `passagesError`, `retryPassages`; internal snapshot carries `jobId`. Errors remain section-local and never impersonate successful empty results.

Consume `PassageListResponse` unchanged (`schemas.py:265–278`). Preserve array order; key by job/sequence. Show sequence, interval, video-scoped track ID, supplied original, endpoint/card warnings and unresolved count/details. Render category only for `bcs.status === "evaluated"` with supplied `bcs_category`. Distinguish absent BCS, unevaluable reason, aggregate unavailable warning, empty completion, and missing/failed image. Ignore deprecated `candidate.bcs_status`. Translate known reasons/warnings; unknown strings remain escaped React text. Label evidence provisional, uncalibrated, non-identifying and non-clinical. Key image state by job/sequence/URL.

## Testing Strategy

| Layer | Coverage / approach |
|---|---|
| Rendered integration, RED→GREEN | Existing real App/hook harness with simulated API: ordering, originals, evaluated/unevaluable/missing results, warnings, unknown text, empty/terminal/errors, retry, pending after job completion, stale success/error, switch-back and unmount cleanup. Mock passages before history fallback; mock original images explicitly. Keep defaults terminal so existing timer assertions survive. |
| Actual API contract | Existing synthetic FastAPI tests prove HTTP/body statuses, artifact gates and BCS publication; browser mocks alone do not prove endpoint behavior. |
| Manual E2E | Later, only with an authorized already-running local API and existing job: compare gallery/order/images/reasons with actual GET. Not executed or claimed here. |

Commands, using installed Node v22.16.0/dependencies and existing virtualenv:

- RED then GREEN: `npm --prefix web test`
- Actual routes/coordinator: `.\.venv\Scripts\python.exe -m pytest tests/test_video_passages_api.py tests/test_video_passage_bcs.py -q`

Harness bundles in memory; no build outputs needed. No commands above run during design.

## Migration / Rollout

No migration required. Local-only, reversible frontend patch. Preserve nine preexisting dirty docs. Forecast ≤1100 changed lines including tests and all planning; reserve 100 of the 1200-line budget. No chain assigned; tasks must reforecast before apply.

## Open Questions

None blocking. Live endpoint/browser verification remains authorization-dependent.
