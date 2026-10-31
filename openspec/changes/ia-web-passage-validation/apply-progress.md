# Apply Progress: IA Web Passage Validation

**Mode:** Strict TDD · **Artifact store:** hybrid · **Scope:** tasks 1.1–4.4
**Delivery:** `exception-ok`, explicit `size:exception` accepted up to 1,200 changed lines; one bounded local UI unit, no chain. Existing 192 unrelated dirty documentation lines are preserved and excluded.

## Completed Tasks

- [x] 1.1 Rendered gallery and result-state acceptance tests.
- [x] 1.2 Pending lifecycle, retry, stale response, selection and unmount tests.
- [x] 2.1 Abortable, encoded-ID passages API method.
- [x] 2.2 Independently owned passage state, selected-ID tagging, retry and 3-second pending-only refresh.
- [x] 2.3 Ordered original-image gallery, evaluation/reason states, warnings and provisional limits; candidate BCS/tracks remain separate.
- [x] 3.1 Rendered UI suite.
- [x] 3.2 Frontend build and existing synthetic API/coordinator suites.
- [x] 3.3 Concise local-use guidance in `docs/video-web.md`.
- [x] 4.1 Rendered count, best-frame label, and per-passage result assertions (test-first).
- [x] 4.2 Unknown count for pending/errors/terminal-unavailable statuses; escaped reasons and non-clinical caveat.
- [x] 4.3 Completed-only passage count and best-frame details from supplied candidate metadata.
- [x] 4.4 Full rendered UI suite GREEN without supplemental harnesses.

## TDD Cycle Evidence

| Task | Test file / layer | Safety net | RED | GREEN | TRIANGULATE | REFACTOR |
|---|---|---|---|---|---|---|
| 1.1 | `web/ui.test.mjs`, rendered integration | 11 baseline tests passed | New gallery/status assertions failed before feature existed | 16/16 final Node test records passed | Ordered non-empty passages, evaluated/unevaluable/missing BCS, warnings, image fallback and empty/error outcomes | Re-ran rendered suite after cleanup |
| 1.2 | `web/ui.test.mjs`, rendered integration | 11 baseline tests passed | Pending/stale behavior lacked a rendered section before implementation | 16/16 final Node test records passed | Pending refresh after completed job, stale success/error and same-ID switch-back, selection, retry and scheduled-timeout unmount | Re-ran rendered suite after cleanup |
| 2.1 | `web/ui.test.mjs`, API integration through App | 11 baseline tests passed | Passage endpoint assertions failed before route/state existed | 16/16 final Node test records passed | Encoded endpoint path and response/error behavior exercised through mocked HTTP | No further refactor required |
| 2.2 | `web/ui.test.mjs`, rendered integration | 11 baseline tests passed | Request ownership/pending cases failed before passage lifecycle existed | 16/16 final Node test records passed | Terminal vs pending, retries, stale selection responses and unmount exercised | Reused `createRequestOwner`; no parallel polling abstraction |
| 2.3 | `web/ui.test.mjs`, rendered integration | 11 baseline tests passed | Rendering assertions failed before gallery existed | 16/16 final Node test records passed | Evaluated-only categories, supplied originals, missing/error images, unknown warnings and section separation | Styles scoped to the existing module |
| 3.1 | `web/ui.test.mjs`, verification | N/A | N/A — verification task | 16 tests passed (15 scenarios + suite) | Existing scenarios included in same run | N/A |
| 3.2 | Build + synthetic Python tests | N/A | N/A — verification task | Build passed; 118 passed, 2 skipped | Existing API/BCS scenarios | N/A |
| 3.3 | `docs/video-web.md`, documentation only | N/A | N/A — no runtime behavior | Usage guidance added after UI passed | N/A | Kept addition concise and preserved existing hunks |

## Verification

- `npm --prefix web test` — **18 passed, 0 failed** (17 rendered scenarios plus suite).
- `npm --prefix web run build` — passed.
- `.venv\Scripts\python.exe -m pytest tests/test_video_passages_api.py tests/test_video_passage_bcs.py -q` — **118 passed, 2 skipped**.
- `git diff --check` — passed.
- Baseline before edits: rendered UI **11 passed**; focused Python **118 passed, 2 skipped**.
- No backend/schema, dependency, model, output-data, R2 or source API changes. No commits, staging, pushes or PRs.

## Phase 4 — Count and Best-Frame Refinement (Strict TDD)

| Task | Test file / layer | Safety net | RED | GREEN | TRIANGULATE / REFACTOR |
|---|---|---|---|---|---|
| 4.1 | `web/ui.test.mjs`, rendered integration | 16 Node test records passed | `npm --prefix web test`: empty-completed test failed its expected-zero count assertion (`0 !== 1`); new two-passage test timed out waiting for `Pasadas detectadas: 2` before production edits. | Same command: 18 passed, 0 failed. | Completed count uses two passages plus one unresolved track; empty completion shows zero; both supplied originals carry their own frame metadata and BCS/reason. |
| 4.2 | `web/ui.test.mjs`, rendered integration | 16 Node test records passed | Disabled/unavailable reason, escaping, pending/error/no-count, and caveat assertions were written before production changes. Status/reason/caveat behavior already passed; no source change was needed for these clauses. | 18 passed, 0 failed. | Includes pending, request error, failed, disabled, unavailable, raw unknown reason text, no markup element, and explicit “no calibrados ni clínicos” assertion. |
| 4.3 | `web/src/components.jsx`, rendered integration | Same test-first RED above | The two-passage scenario failed at the count wait before reaching its per-card frame assertions; source had neither count nor best-frame label. | 18 passed, 0 failed after minimal display-only component change. | Shows actual `frame_index`/`timestamp_ms` only when supplied and original URL exists; no global highlight/crop/mask/selection changes. |
| 4.4 | `npm --prefix web test` | N/A verification task | N/A — run full existing harness after changes | **18 passed, 0 failed** | No stdin/data-URL/supplemental harness used. |

**Exact RED excerpts (`npm --prefix web test`, before `components.jsx` edits):**

```text
not ok 10 - completed empty passages retain unresolved attribution and endpoint warnings
error: Expected values to be strictly equal:
0 !== 1

not ok 12 - completed passage count excludes unresolved tracks and labels each supplied best frame with its BCS
error: locator.waitFor: Timeout 5000ms exceeded.
- waiting for getByRole('region', { name: 'Pasajes del video' }).getByText('Pasadas detectadas: 2', { exact: true }) to be visible
# tests 18
# pass 15
# fail 3
```

**Exact GREEN excerpt (after the component change):**

```text
# tests 18
# suites 0
# pass 18
# fail 0
# cancelled 0
# skipped 0
# todo 0
```

## Corrective Retry #1 — Fresh TDD Evidence

The previous apply-progress above is retained. This correction is test-first and limited to the verifier findings.

| Correction | Test file / layer | Safety net | RED (captured before source fix) | GREEN | TRIANGULATE / REFACTOR |
|---|---|---|---|---|---|
| Keep unresolved-attribution details and endpoint warnings visible for empty completed responses | `web/ui.test.mjs`, rendered integration | 14 Node test records passed before corrective test edits | `npm --prefix web test` failed in `completed empty passages retain unresolved attribution and endpoint warnings`: `The input did not match the regular expression /1 atribución sin resolver.*trayectoria 91/.` Actual section text contained the explicit empty state but omitted attribution and warnings. TAP: `# tests 16`, `# pass 14`, `# fail 2` (the failed subtest and parent suite). | `npm --prefix web test`: **16 passed, 0 failed** after moving response metadata/warnings outside the nonempty list branch. | Empty response uses a nonempty unresolved-track/warning fixture; companion nonempty passage test; preserved provisional/identity disclaimers. Shared metadata renders for any non-pending response. |
| Replace weak image, polling and timer checks; exercise global candidate beside nonempty passages | `web/ui.test.mjs`, rendered integration | Same 14-record baseline | The empty-response failure above was the only intended red; strengthened image/lifecycle/integration assertions exercise already-correct behaviors. | Same **16 passed, 0 failed** run. | Failed-image case has only original URLs and targets Pasaje 7; fixture uses actual `/best-original` route shape. Request count delta is measured around pending fetch; controlled 3000 ms timer proves pending→terminal stop and scheduled-timeout cleanup on unmount. Also tests A→B→A late success, stale HTTP error, disclaimer text, and global candidate + nonempty gallery + tracks. |

**Exact RED excerpt:**

```text
not ok 10 - completed empty passages retain unresolved attribution and endpoint warnings
error: The input did not match the regular expression /1 atribución sin resolver.*trayectoria 91/.
actual: 'Pasajes del videoEvidencia temporal provisional; los identificadores solo tienen alcance dentro de este video.No hay pasajes disponibles para este análisis.Resultados experimentales, no calibrados ni clínicos; no identifican vacas únicas ni confirman visibilidad o elegibilidad. BCS utiliza solo la imagen original completa cuando está disponible; no se sustituyen recortes ni máscaras.'
# tests 16
# pass 14
# fail 2
```

**Exact GREEN excerpt (`npm --prefix web test`, after the source fix):**

```text
# tests 16
# suites 0
# pass 16
# fail 0
# cancelled 0
# skipped 0
# todo 0
```

**Corrective scope:** `web/src/components.jsx` and `web/ui.test.mjs` only. The gallery now renders endpoint warnings and unresolved details independently of passage count for non-pending responses; pending responses remain in their refresh state. Existing 192 unrelated dirty documentation lines remain untouched. No backend, schema, API source, dependencies, models, R2, or generated data changed.

## Boundary and Review Budget

Only the disposable local frontend, rendered tests, one usage paragraph, and SDD apply artifacts were changed. Rollback is limited to the frontend feature and its tests/docs; existing unrelated documentation hunks remain intact. The implementation is comfortably below the expressly authorized 1,200-line maximum; no chain or review-budget escalation was needed.

## Files Changed

- `web/src/api.js` — passages endpoint method.
- `web/src/useVideoAnalysis.js` — isolated request lifecycle.
- `web/src/components.jsx`, `web/src/style.css` — passage gallery.
- `web/ui.test.mjs` — behavioral, lifecycle, count, best-frame, and status coverage.
- `docs/video-web.md` — local-use guidance including count and per-passage best-frame semantics.
- `openspec/changes/ia-web-passage-validation/tasks.md` — task checkboxes.
