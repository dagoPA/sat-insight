# sat-insight

Spatial disaggregation of social deprivation from weakly supervised satellite imagery.

A bag is a Mexican municipality; its label is the five-class ordinal Social Deprivation
Grade (GRS) that CONEVAL publishes, an aggregate over census units (AGEB). The model never
receives spatial supervision. The per-instance predictions form a map, validated against
AGEB-level grades that are entirely held out from training.

The headline quantity under construction is the supervision-efficiency curve: the fraction
of a fully supervised ceiling that aggregate supervision recovers, as a function of how
many aggregates exist, their size, and the level they are published at. Mexico can
calibrate that curve because it publishes ground truth at both levels at once.

## Layout

- `src/satinsight/`: installable library: STAC catalog queries, windowed COG reads,
  annual median composites (Sentinel-1 RTC and Sentinel-2 L2A), tiling into bags, frozen
  DOFA encoding, label-proportion models, evaluation.
- `herramientas/`: experiment drivers (Spanish filenames, English code).
- `tests/`: network-free tests over pure logic. `uv run pytest`.
- `data/`: regenerable; ignored by git.

## Conventions

Everything runs through [uv](https://docs.astral.sh/uv/): `uv run ...`, `uv add ...`.
Format and lint with `uv run ruff format . && uv run ruff check .` before any change
closes. Satellite data comes from Microsoft Planetary Computer via windowed reads; no
whole-scene downloads.

## Reproducing

Each figure and table of the paper maps to a driver in `herramientas/`. A frozen benchmark
(DOFA vectors, labels, splits, evaluation protocol) is being packaged so heads can be
trained in minutes without the satellite pipeline; until it is published, `satinsight
probe` verifies the live data access the pipeline needs.
