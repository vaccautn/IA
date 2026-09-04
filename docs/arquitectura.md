# Arquitectura BCS category 1..5

La migración BCS usa únicamente la fuente local original, una frontera de
filtración basada en grupos de captura derivados del nombre de archivo y el
modelo CORAL como primera arquitectura de referencia.

## Flujo

```text
data/bcs/dataset/ (3.25, 3.5, 3.75, 4.0, 4.25)
  ↓ scan + mapping + capture-group/digest validation
data/bcs-category-v1/ (train/val/test × categories 1..5)
  ↓ train_bcs_ordinal.py
outputs/bcs-category-coral-v1/
  ↓ validated hard-category serving
POST /bcs → bcs_category: 1..5
```

| Layer | Responsibility |
|---|---|
| `local_source.py` | Exact source mapping, filename identity, safe hashing and materialization. |
| `source_plan.py` | Normalized local records without animal-ID claims. |
| `category_split_plan.py` | Transitive same-digest union and deterministic group-aware 80/10/10 planning. |
| `category_snapshot.py` | Atomic snapshot publication and strict manifest/lineage validation. |
| `dataset.py` | Deterministic validation transforms and folder loading. |
| `model.py` | ResNet18 + CORAL reference model. |
| `serving.py` | New-lineage checkpoint loading and hard category prediction. |

## Leakage boundary

`GS_<series>_<view>` and `YM_<series>_<view>` share a group by prefix and
series. `L-i<index>` and `R-i<index>` share a group by index. Malformed names,
duplicate members, duplicate identities and cross-category identical digests
fail closed. Groups containing multiple labels remain intact; they are never
majority-relabeled. Same-digest groups are unioned transitively before split.

The planner balances category-count vectors against 80% train, 10% validation
and 10% test targets, while keeping every capture group and digest in one
partition. Leakage prevention takes priority over exact ratios.

## Lineage and serving

Active identifiers are `bcs-category-1-5-v1`, `bcs-local-category-source-v1`,
`bcs-category-snapshot-v1`, `bcs-category-coral-checkpoint-v1` and
`bcs-category-coral-results-v1`. Old snapshots and checkpoints are rejected
even when tensor shapes are compatible. The API emits the CORAL hard class plus
one; no fractional expectation or half-down rounding is used.

The backend source client and backend-only provenance/materialization path are
removed. Detection assets and behavior remain separate from this pipeline.
