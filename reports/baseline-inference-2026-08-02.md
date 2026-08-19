# Real CPU baseline inference

The verified `yolo26n` release asset completed two real inference runs after the baseline integrity hardening. Both runs passed through the manifest gates, immutable `ValidatedImage` snapshot, `UltralyticsDetector`, and `AptitudePipeline`. They produced the same semantic result: one `cow`, accepted by the aptitude rules.

## Environment

| Item | Verified value |
|---|---|
| Platform | Windows x64 |
| Python | 3.13.5 |
| PyTorch | 2.13.0+cpu |
| TorchVision | 0.28.0+cpu |
| Ultralytics | 8.4.115 |
| Setuptools | 80.9.0 |
| Device | CPU |
| CUDA available | `false` |
| Administrator process | `false` |
| Initial available memory | 2.25 GiB |
| Initial free disk | 114.13 GiB |

Installation used the official PyTorch CPU index for the `+cpu` wheels and PyPI for the remaining packages. `pip check` reported no broken requirements. `requirements-cpu.txt` pins every dependency with the SHA-256 of its Windows x64 / CPython 3.13 wheel and contains no local paths.

## Provenance and licensing

| Artifact | Source and evidence |
|---|---|
| Model | Ultralytics `v8.4.0` release asset, `yolo26n.pt`, 5,544,453 bytes, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Model license | AGPL-3.0 baseline; repository `LICENSE` and public-repository constraint apply |
| Fixture | Wikimedia Commons revision `1202221419`, USDA ARS image by Keith Weller, `PD-USGov-USDA-ARS` |
| Fixture evidence | 908,691 bytes, SHA-256 `e0972384d3151174d1450cff81bb19d1fc89519a5d1f6fc7ade5d710a89e56d8` |

The `.pt` file was loaded only after exact size and digest verification. The fixture digest and size were checked against the same immutable byte snapshot used for inference, before detector construction. The release provides no independent signature, and a matching digest does not make an unknown source trustworthy. Binary model, fixture, settings, and run outputs remain Git-ignored.

## Command

```bash
.venv/Scripts/python scripts/run_baseline.py --output outputs/baseline/hardened-run-1.json
.venv/Scripts/python scripts/run_baseline.py --output outputs/baseline/hardened-run-2.json
```

Default thresholds came from `configs/baseline_manifest.json`: model and pipeline confidence `0.25`, minimum relative area `0.10`, border margin `0.02`, framing enabled, and input size `640`. They were selected before observing the result and were not altered to force acceptance.

## Real result

| Field | Run 1 | Run 2 |
|---|---:|---:|
| Status | `ACCEPTED` | `ACCEPTED` |
| Reason | `null` | `null` |
| Animal count | 1 | 1 |
| Class | `cow` (`class_id=19`) | `cow` (`class_id=19`) |
| Confidence | 0.9302085042 | 0.9302085042 |
| Relative area | 0.4235123662 | 0.4235123662 |
| Bounding box | `[514.3575, 208.7596, 2278.6560, 1377.3263]` | same |
| Inference time | 104.4384 ms | 104.5472 ms |
| Adapter total time | 240.7501 ms | 284.1143 ms |

Semantic payloads were exactly equal after removing only the timing object.

## Limitations and next step

- This is one public fixture, not a model-quality evaluation.
- `yolo26n` is a generic pretrained COCO detector, not a VACCA-trained model.
- CPU timing is exploratory and has no production SLA.
- Dataset provenance, splits, annotation policy, and model evaluation remain pending.

Next, assemble a small licensed evaluation set containing zero, one, and multiple bovines and run the unchanged manifest-driven pipeline over it before any training decision.
