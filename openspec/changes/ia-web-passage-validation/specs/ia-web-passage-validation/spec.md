# IA Web Passage Validation Specification

## Purpose

Specify the disposable local validation UI for inspecting the existing ordered passage endpoint without implying animal identity, clinical validity, or product readiness.

## Requirements

### Requirement: Selected analysis passage lifecycle

The UI MUST request the existing passages endpoint for the selected analysis and keep its loading, response, and request-error state separate from global analysis results, candidate BCS, and tracks. It MUST refresh while the endpoint reports `pending`, stop refreshing for terminal statuses, and ignore responses belonging to a previously selected analysis.

#### Scenario: Pending passage tracking settles

- GIVEN a selected analysis returns `pending`
- WHEN the UI refreshes its passage request
- THEN it shows a pending/refreshing state until a terminal response arrives
- AND it presents passages from the settled response

#### Scenario: Selection changes during a request

- GIVEN a passage request for analysis A is in flight
- WHEN analysis B is selected before that request resolves
- THEN A's response MUST NOT appear under B
- AND B's passage state MUST load independently

#### Scenario: Request error or terminal endpoint status

- GIVEN the endpoint request fails or returns `failed`, `disabled`, or `unavailable`
- WHEN the passage section renders
- THEN it MUST show an understandable non-success state and any supplied reason
- AND it MUST NOT present that state as an empty successful result

### Requirement: Ordered passage evidence and evaluation state

The UI MUST preserve endpoint passage order and show the original image only when `candidate.best_original_url` is supplied. It MUST show an experimental category only for a `bcs.status` of `evaluated` with a supplied category; absent BCS, `unevaluable` BCS, unavailable aggregate BCS, and unavailable images MUST remain distinct and MUST NOT be replaced by inferred values. Known reasons SHOULD be explained in Spanish; unknown reason values MUST remain visible as escaped plain text.

#### Scenario: Completed passage results

- GIVEN a completed response with passages, original-image URLs, and evaluated BCS
- WHEN the section renders
- THEN passages appear in response order with supplied originals and experimental categories only for evaluated results
- AND each passage is clearly labeled provisional, not an identified animal or clinical result

#### Scenario: Missing image or evaluation

- GIVEN a passage has no original URL, no BCS result, or an `unevaluable` BCS result
- WHEN the section renders
- THEN it shows image-unavailable or not-evaluated status and the supplied reason when present
- AND it does not show a fabricated image, category, or reason

#### Scenario: Empty completed response and BCS warning

- GIVEN a completed response has no passages, or aggregate `bcs_status` is `unavailable`
- WHEN the section renders
- THEN it shows an explicit empty state or BCS-unavailable warning respectively
- AND neither condition is presented as a successful evaluated passage

### Requirement: Completed passage count and per-passage best-frame labeling

For a completed response, the UI MUST show the number of detected passages using the endpoint's `passages` array length (the current response schema has no separate total-count field). It MUST label each supplied `candidate.best_original_url` as the best frame selected for that passage, not as a global candidate or track crop, and show that passage's evaluated BCS or its supplied non-evaluation reason. The UI MUST NOT count tracks or unresolved attributions as passages, claim unique-animal identity, or introduce a new selection algorithm, model, or schema. Until a completed response is available, the count MUST remain unknown rather than displaying zero.

#### Scenario: Completed response reports passage count and per-passage results

- GIVEN a completed response contains two passages and unresolved tracks
- WHEN the passage section renders
- THEN it displays “Pasadas detectadas: 2,” counting only the `passages` array
- AND each passage's supplied original is labeled as its best selected frame, alongside its evaluated BCS or supplied non-evaluation reason

#### Scenario: Completed response contains no passages

- GIVEN a completed response has an empty `passages` array
- WHEN the passage section renders
- THEN it explicitly displays “Pasadas detectadas: 0” and the existing empty state
- AND it does not infer a count from tracks or unresolved attributions

#### Scenario: Passage count is not yet known

- GIVEN the endpoint is pending, the request fails, or the endpoint is failed, disabled, or unavailable
- WHEN the passage section renders
- THEN it shows the applicable pending or non-success state without displaying zero as the detected-passage count

### Requirement: Attribution, endpoint warnings, and concept boundaries

The UI MUST surface endpoint warnings and unresolved passage-attribution information when present, explain that passage evidence and track identifiers are provisional and scoped to one video, and distinguish the passage gallery from the global candidate BCS and tracks sections.

#### Scenario: Unresolved attribution and warnings

- GIVEN a response includes unresolved tracks or warning values
- WHEN the completed passage section renders
- THEN it communicates the unresolved count/details and warnings without claiming unique-animal identity or visibility
- AND unknown warning values remain visible as escaped plain text

#### Scenario: Global candidate and tracks remain distinct

- GIVEN the selected analysis has global candidate BCS or tracks alongside passages
- WHEN all result sections render
- THEN the passage gallery is explicitly identified as passage-specific
- AND global candidate BCS and tracks remain separately labeled and unchanged in meaning
