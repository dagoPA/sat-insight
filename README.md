# sat-insight

Mapping neighborhood-scale social deprivation from Sentinel imagery, trained on municipal
aggregates only.

A bag is a Mexican municipality; its label is the five-level Social Deprivation Grade
(GRS) that CONEVAL publishes, aggregated over urban census tracts (AGEB). The model never
receives spatial supervision. Every 160 m token gets a prediction, the bag prediction is
the mean of its tokens, and the per-token predictions form a map that is scored against
tract-level grades held out from training entirely. A fully supervised oracle trained on
the tract labels bounds what the frozen features support, so the result is stated as the
fraction of that upper bound that aggregate supervision recovers.

## Layout

- `src/satinsight/`: installable library. STAC catalog queries, windowed COG reads, annual
  median composites (Sentinel-1 RTC and Sentinel-2 L2A), tiling into bags, frozen DOFA
  encoding, auxiliary product layers, the label-proportion head, evaluation, and the
  figure guard.
- `scripts/`: experiment drivers, one per analysis, plus `reproduce.sh` with the canonical
  sequence from composites to figures.
- `tests/`: network-free tests over pure logic. `uv run pytest`.
- `docs/manuscript/`: LaTeX sources, figures, per-panel source data, and
  `canonical_results.json`, the file every figure is checked against.
- `data/`: composites, tiles, vectors, labels and results; regenerable and ignored by git.

## Conventions

Everything runs through [uv](https://docs.astral.sh/uv/): `uv run ...`, `uv add ...`.
Format and lint with `uv run ruff format . && uv run ruff check .` before a change closes.
Satellite data comes from Microsoft Planetary Computer through windowed reads of the
cloud-optimized GeoTIFFs; no whole-scene downloads.

Data tables use English column names throughout: `city`, `municipality`, `grade`,
`population`, and the five grade labels `Very low` to `Very high`. `cvegeo` stays as the
official INEGI tract key. Tables written before this schema are converted in place by
`scripts/migrate_schema.py`.

## Reproducing

`scripts/reproduce.sh` lists the canonical run in order (run it with `sh`; file modes are not kept by the Overleaf sync). From the frozen vectors on,
everything reproduces in a few hours on one GPU; the compositing and encoding steps
before that take days of downloading.

Each figure of the paper maps to one driver in `scripts/`:

| Figure | Driver |
| --- | --- |
| 1, study design | `fig1_design.py` |
| 2, supervision efficiency | `fig2_curve.py` |
| 3, prediction against localization | `fig3_dissociation.py` |
| 4, external validity | `fig4_validation.py` |
| 5, free products and uncertainty | `fig5_incumbents.py` |
| graphical abstract | `graphical_abstract.py` |

Drivers 2 to 5 recompute every quantity they draw from the per-seed artifacts and stop if
one departs from `docs/manuscript/canonical_results.json`, so a stale artifact fails the
build instead of quietly redrawing the page. `source_data.py` exports the per-panel
source data.

`package_benchmark.py` builds the frozen benchmark (DOFA vectors, labels, splits,
evaluation protocol) under `dist/benchmark` for deposit on Zenodo, so heads can be trained
in minutes without the satellite pipeline. `satinsight probe` verifies the live data
access the pipeline needs.
