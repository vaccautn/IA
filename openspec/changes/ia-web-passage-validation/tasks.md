# Tasks: IA Web Passage Validation

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~801 audited feature lines + ≤180 focused UI/test refinement lines = ≤981 total; 1,200-line exception |
| 400-line budget risk | High |
| Chained PRs recommended | No — one bounded work unit fits the explicitly approved 1,200-line review envelope |
| Suggested split | Single local validation-UI work unit |
| Delivery strategy | auto-forecast |
| Chain strategy | size-exception (explicitly accepted bounded local work; no chain) |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: size-exception
400-line budget risk: High

The prior verification footprint was 777 lines; the amended spec adds 24 lines, so the current audited feature footprint is about 801. Allow up to 180 additional changed lines for this focused UI/test refinement and its task amendment (projected total ≤981, at least 219 below the accepted 1,200 cap). The default 400-line budget risk remains High; the user explicitly accepted `size:exception` for this bounded local work, with no chain, commits, or PRs. Preserve and exclude the 192-line pre-existing documentation baseline.

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Finish count/best-frame labels and remaining rendered regressions in the local UI | Single bounded work unit | Accepted size:exception; no chain, commits, backend/R2/dependency work |

## Phase 1: Rendered UI RED tests

- [x] 1.1 In `web/ui.test.mjs`, reuse existing fixture/routes and add focused failing rendered-App tests for ordered originals/evaluated versus missing or unevaluable results, labels/warnings, empty/error states, and separation from candidate BCS/tracks; combine assertions and avoid exhaustive combinations.
- [x] 1.2 Add compact failing lifecycle tests in `web/ui.test.mjs` for pending refresh independent of job status, retry, stale response after selection change/switch-back, and unmount cleanup; keep defaults terminal to protect timer assertions.

## Phase 2: Minimal GREEN UI implementation

- [x] 2.1 Add `videoApi.passages(id, signal)` in `web/src/api.js`; encode the ID, pass the abort signal, and preserve existing successful HTTP response handling.
- [x] 2.2 Add independently owned passage state and `retryPassages` in `web/src/useVideoAnalysis.js`; use request-owner cleanup, selected-ID tagging, and the existing 3-second interval only while passages remain pending.
- [x] 2.3 In `web/src/components.jsx` and `web/src/style.css`, render the ordered passage gallery, supplied originals only, evaluated categories only, distinct missing/unavailable/reason states, warnings, unresolved attribution, and concise provisional/non-clinical labels; preserve global candidate and tracks sections.

## Phase 3: Verification and bounded documentation

- [x] 3.1 Run `npm --prefix web test`; make the RED cases pass and verify existing rendered-app regressions, including request ownership, pending completion, and image failure fallback.
- [x] 3.2 Run `npm --prefix web run build` and existing synthetic route/coordinator checks from the repository root: `.venv\Scripts\python.exe -m pytest tests/test_video_passages_api.py tests/test_video_passage_bcs.py -q`.
- [x] 3.3 After the gallery works, add a concise local-use instruction to `docs/video-web.md` explaining how to inspect ordered passage evidence and its provisional limits; preserve all pre-existing doc hunks.

## Phase 4: Passage count and best-frame refinement (RED → GREEN)

- [x] 4.1 In `web/ui.test.mjs`, add rendered RED assertions for `Pasadas detectadas: N` and completed-empty `: 0` from `passages.length` (not unresolved tracks); require “Mejor frame seleccionado para este pasaje” beside each supplied original and its BCS or reason.
- [x] 4.2 In the same existing harness, cover count unknown for pending, request error, and failed/disabled/unavailable responses; assert disabled/unavailable reasons and escaped unknown reasons plus the explicit non-clinical caveat.
- [x] 4.3 In `web/src/components.jsx`, render `Pasadas detectadas: N` only for completed results and “Mejor frame seleccionado para este pasaje” for each supplied original; retain BCS/reason and provisional-scope behavior.
- [x] 4.4 Run `npm --prefix web test`; confirm the new RED cases pass without auxiliary stdin/data-URL test harnesses.
