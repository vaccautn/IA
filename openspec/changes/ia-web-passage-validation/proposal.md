# Proposal: IA Web Passage Validation

## Intent

Expose the existing ordered `/passages` results in the disposable local IA validation web so reviewers can inspect original passage images and experimental BCS outcomes—or understandable reasons when a passage was not evaluated. This makes actual endpoint behavior observable without mistaking provisional, video-scoped evidence for animal identity or clinical validation.

## Scope

### In Scope
- Add frontend API/state handling for the selected analysis's existing passages endpoint, including pending refresh and reset/cancellation when selection changes.
- Render passages in endpoint order with original images when supplied, experimental BCS only when evaluated, readable non-evaluation reasons otherwise, and endpoint/unresolved warnings.
- Clearly label results and track references as provisional and identity-limited; retain existing global candidate BCS and tracks as distinct concepts.
- Extend rendered UI tests for ordering, images, evaluated/unevaluable results, pending/terminal/error states, and analysis selection changes.

### Out of Scope
- Changes to backend routes/schemas, R2 or remote services, production architecture/productization, dependencies, models, or generated validation outputs/data.
- Claims or inference about unique-animal identity, clinical validity, calibration, or visibility/eligibility.

## Capabilities

### New Capabilities
- `ia-web-passage-validation`: Local validation UI for viewing ordered provisional passage evidence and endpoint status.

### Modified Capabilities
- None. No existing OpenSpec specs are present.

## Approach

Reuse the existing video API client, analysis hook, React UI, and rendered-app test harness. Consume the backend's existing passage list and original-image URLs; keep passage request lifecycle separate from global analysis, candidate BCS, and tracks. Do not infer semantics beyond the endpoint response.

## Affected Areas

| Area | Impact | Description |
|---|---|---|
| `web/src/api.js`, `web/src/useVideoAnalysis.js` | Modified | Fetch and manage selected-analysis passage state. |
| `web/src/components.jsx`, `web/src/style.css` | Modified | Present ordered, clearly provisional passage evidence and statuses. |
| `web/ui.test.mjs` | Modified | Verify rendering, ordering, lifecycle, and failure cases deterministically. |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Experimental categories or video-scoped tracks are mistaken for clinical results or animal identity. | Med | Label provisional status, show warnings, and render categories only for evaluated BCS outcomes. |
| Passage requests outlive or overwrite the selected analysis. | Med | Tie refresh/reset and stale-response handling to selected-job lifecycle; cover selection changes in UI tests. |

## Rollback Plan

Revert the isolated frontend passage UI/API/hook changes and their tests. The existing backend endpoint and global candidate/tracks behavior remain unchanged.

## Dependencies

- Existing `GET /api/v1/analyses/{job_id}/passages` response and passage-original image URLs; no new dependency.

## Success Criteria

- [ ] The selected analysis's passages render in endpoint order with supplied original images; only evaluated outcomes show experimental categories, while other outcomes show readable reasons/status.
- [ ] Pending, terminal, and error/unavailable states and unresolved-attribution caveats are visible without blocking or conflating global candidate/tracks results.
- [ ] Deterministic rendered UI tests cover completed, pending, failure, and selected-analysis changes without model inference or generated-data changes.
