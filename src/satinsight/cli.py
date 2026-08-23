"""Command line interface."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from satinsight import aoi as modulo_aoi
from satinsight.agebs import CITIES, agebs_of_city, grade_summary
from satinsight.baseline import compare, fold_summary, transfer_diagnostics
from satinsight.catalog import (
    COLLECTION_S1,
    COLLECTION_S2,
    by_cloud_cover,
    cloud_summary,
    open_catalogue,
    search,
)
from satinsight.composite import composite_s1, composite_s2
from satinsight.download import DATA_ROOT
from satinsight.figures import (
    ageb_panel,
    agebs_by_city_map,
    contrast_panel,
    modality_panel,
    national_map,
)
from satinsight.pipeline import SCALES, SENSORS, features_of_all
from satinsight.raster import percentiles, read_window, stretch, to_db
from satinsight.render import save_rgb

CENSUS_PERIOD = "2020-01-01/2020-12-31"

log = logging.getLogger("satinsight")


def cmd_aoi(_: argparse.Namespace) -> int:
    """Lists the available pilot boxes."""
    for key, area in sorted(modulo_aoi.PILOT.items()):
        height, width = area.approximate_shape()
        print(f"{key:<12} {area.name:<22} {area.state:<18} ~{width}x{height} px @10 m")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Reports how many scenes are available over an AOI."""
    area = modulo_aoi.obtener(args.aoi)
    catalogue = open_catalogue()
    print(f"{area.name} · {args.period}")

    scenes_s2 = search(COLLECTION_S2, area.bbox, args.period, catalogue)
    summary = cloud_summary(scenes_s2)
    print(f"  Sentinel-2  {summary['scenes']:>4} scenes")
    print(f"              median cloud {summary['median']}%")
    print(f"              {summary['pct_over_50']}% above 50% cloud")
    print(f"              {summary['pct_over_80']}% above 80%")

    scenes_s1 = search(COLLECTION_S1, area.bbox, args.period, catalogue)
    print(f"  Sentinel-1  {len(scenes_s1):>4} scenes")
    return 0


def cmd_panels(args: argparse.Namespace) -> int:
    """Downloads a sample and renders the four inspection panels."""
    area = modulo_aoi.obtener(args.aoi)
    destination = Path(args.output)
    catalogue = open_catalogue()
    stats: dict[str, object] = {
        "aoi_clave": area.key,
        "aoi_nombre": area.name,
        "aoi_bbox": list(area.bbox),
        "period": args.period,
        "resolucion_px_m": 10,
    }

    log.info("consultando Sentinel-2")
    scenes_s2 = search(COLLECTION_S2, area.bbox, args.period, catalogue)
    stats["s2"] = cloud_summary(scenes_s2)
    ordenadas = by_cloud_cover(scenes_s2)
    despejada, nublada = ordenadas[0], ordenadas[-1]

    log.info("fecha despejada: %s", despejada.datetime.date())
    bandas = [read_window(despejada.assets[b].href, area.bbox) for b in ("B04", "B03", "B02")]
    forma = bandas[0].shape
    save_rgb(*(stretch(b) for b in bandas), destination / "s2_despejada.png")
    stats["s2_despejada"] = {
        "fecha": str(despejada.datetime.date()),
        "nubes": round(despejada.properties["eo:cloud_cover"], 1),
    }

    log.info("fecha nublada: %s", nublada.datetime.date())
    bandas = [read_window(nublada.assets[b].href, area.bbox, forma) for b in ("B04", "B03", "B02")]
    save_rgb(*(stretch(b) for b in bandas), destination / "s2_nublada.png")
    stats["s2_nublada"] = {
        "fecha": str(nublada.datetime.date()),
        "nubes": round(nublada.properties["eo:cloud_cover"], 1),
    }

    log.info("componiendo Sentinel-2")
    composite, usadas = composite_s2(scenes_s2, area.bbox, forma, max_scenes=args.max_s2)
    save_rgb(
        *(stretch(composite[b]) for b in ("B04", "B03", "B02")),
        destination / "s2_compuesto.png",
    )
    stats["s2_compuesto"] = {"scenes_used": usadas}

    log.info("consultando y componiendo Sentinel-1")
    scenes_s1 = search(COLLECTION_S1, area.bbox, args.period, catalogue)
    sar, meta = composite_s1(scenes_s1, area.bbox, forma, max_scenes=args.max_s1)
    vv_db, vh_db = to_db(sar["vv"]), to_db(sar["vh"])
    save_rgb(
        stretch(vv_db), stretch(vh_db), stretch(vv_db - vh_db), destination / "s1_compuesto.png"
    )
    stats["s1"] = {
        "scenes_available": meta["scenes_available"],
        "scenes_used": meta["scenes_used"],
        "orbit": meta["orbit"],
        "vv_db_p5_p95": list(percentiles(vv_db)),
        "vh_db_p5_p95": list(percentiles(vh_db)),
    }
    stats["aoi_px"] = [forma[1], forma[0]]

    (destination / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def cmd_agebs(args: argparse.Namespace) -> int:
    """Summarises the AGEB of each pilot city and their distribution of grades."""
    total = 0
    for key in args.cities or sorted(CITIES):
        agebs = agebs_of_city(key)
        total += len(agebs)
        people = int(agebs["poblacion"].sum())
        print(f"\n{CITIES[key].name} · {len(agebs)} AGEB · {people:,} people")
        print(grade_summary(agebs).to_string())
    print(f"\ntotal: {total} AGEB")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Extracts the per-AGEB texture of one sensor and leaves it on disk."""
    from satinsight.agebs import cities_by_size

    catalogue = cities_by_size(stratify=True)
    keys = args.cities or sorted(
        p.stem.replace(f"_{args.sensor}", "")
        for p in (DATA_ROOT / "composites").glob(f"*_{args.sensor}.tif")
    )
    table = features_of_all(
        args.sensor,
        tuple(keys),
        max_scenes=args.max_scenes,
        scale=args.scale,
        catalogue=catalogue,
    )
    destination = Path(args.output or DATA_ROOT / f"rasgos_{args.sensor}_{args.scale}.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(destination, index=False)
    print(f"{len(table)} AGEB × {table.shape[1]} columns → {destination}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Runs the phase one comparison over an already extracted feature table."""
    origin = Path(args.features or DATA_ROOT / f"rasgos_{args.sensor}.parquet")
    if not origin.exists():
        print(f"{origin} is missing. Run first: satinsight features {args.sensor}", file=sys.stderr)
        return 1

    table = pd.read_parquet(origin)
    detail = compare(table, estandarizar=args.estandarizar)
    print(f"\n{len(table)} AGEB · leave-one-city-out partition\n")
    print(fold_summary(detail).to_string(index=False))
    print("\nper fold:\n")
    columnas = ["split", "modelo", "ciudad_prueba", "n_prueba", "kappa", "spearman"]
    print(detail[columnas].round(3).to_string(index=False))

    if args.output:
        detail.to_csv(args.output, index=False)
        print(f"\ndetail → {args.output}")
    return 0


def cmd_diagnostics(args: argparse.Namespace) -> int:
    """Reporta, por split de features, si describen la ciudad o el rezago."""
    origin = Path(args.features or DATA_ROOT / f"rasgos_{args.sensor}.parquet")
    if not origin.exists():
        print(f"{origin} is missing. Run first: satinsight features {args.sensor}", file=sys.stderr)
        return 1

    table = pd.read_parquet(origin)
    print(f"\n{len(table)} AGEB · {table['ciudad'].nunique()} cities\n")
    print("Ratio of the variance the city explains to the one the grade explains.")
    print("Above one, the feature describes where it was measured more than what.\n")

    for split in ("cobertura", "densidad", "textura"):
        detail = transfer_diagnostics(table, split).dropna(subset=["ratio"])
        if detail.empty:
            continue
        low, mid, high = detail["ratio"].quantile([0.25, 0.5, 0.75])
        worst = ", ".join(detail.head(3)["feature"])
        print(f"  {split:<10} n={len(detail):<3} quartiles {low:>6.1f} /{mid:>6.1f} /{high:>6.1f}")
        print(f"  {'':<10} worst: {worst}")

    print(
        "\nSe lee junto con la reliability por mitades, nunca solo: un rasgo que es ruido "
        "comes out with a low ratio\nbecause noise correlates with nothing."
    )
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """Regenerates the phase one figures from the composite cache."""
    destination = Path(args.output)
    modality_panel(args.city, destination / "f1_brazos.png")
    ageb_panel(args.city, destination / "f2_agebs.png")
    for sensor in SENSORS:
        contrast_panel(args.city, sensor, destination / f"f3_contraste_{sensor}.png")
    national_map(destination / "f4_nacional.png")
    agebs_by_city_map(destination / "f5_agebs_ciudades.png")
    print(f"figures of {args.city} → {destination}")
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    """Reports how many cities of the set already have both composites.

    National compositing takes days and outlives the session that launched it, so it takes
    falta poder consultarla desde cualquier otra.
    """
    from satinsight.agebs import cities_by_size
    from satinsight.cache import composite_path

    catalogue = cities_by_size(stratify=not args.no_stratify)
    root = DATA_ROOT / "composites"
    complete, partial, missing = [], [], []
    for key in catalogue:
        hechos = [s for s in SENSORS if composite_path(key, s, root).exists()]
        destination = complete if len(hechos) == len(SENSORS) else partial if hechos else missing
        destination.append(key)

    size = sum(p.stat().st_size for p in root.glob("*.tif")) / 1e9 if root.exists() else 0
    print(f"\n{len(complete)} of {len(catalogue)} cities complete")
    print(f"{len(partial)} partial · {len(missing)} not started · {size:.1f} GB on disk")
    if partial:
        print(f"\nin progress: {', '.join(partial[:8])}")
    if args.detail and missing:
        print(f"\npending: {', '.join(missing)}")
    return 0


def cmd_bags(args: argparse.Namespace) -> int:
    """Tesela cities y arma sus bags MIL, sin codificar los parches."""
    from satinsight.agebs import cities_by_size
    from satinsight.dataset import build_city

    catalogue = cities_by_size(stratify=True)
    keys = args.cities or [
        p.stem.replace(f"_{args.sensor}", "")
        for p in sorted((DATA_ROOT / "composites").glob(f"*_{args.sensor}.tif"))
    ]
    done = 0
    for key in keys:
        try:
            outputs = build_city(
                key, args.sensor, encoder=None, size=args.size, catalogue=catalogue
            )
        except Exception as e:
            print(f"FALLO {key}: {type(e).__name__}: {e}")
            continue
        bags = pd.read_parquet(outputs["bags"])
        instances = pd.read_parquet(outputs["instances"])
        print(f"{key}: {len(bags)} bags, {len(instances)} instances")
        done += 1
    print(f"\n{done} de {len(keys)} cities")
    return 0


def cmd_partition(args: argparse.Namespace) -> int:
    """Writes the spatial partition of the national set."""
    from satinsight.dataset import build_split

    partition = build_split(force=args.force, proportions=tuple(args.proportions))
    summary = partition.groupby("split").agg(
        cities=("ciudad", "size"),
        agebs=("n_agebs", "sum"),
        rezago_medio=("stratum_value", "mean"),
    )
    summary["porcentaje"] = 100 * summary.cities / summary.cities.sum()
    print(summary.round(2).to_string())
    return 0


def cmd_vectors(args: argparse.Namespace) -> int:
    """Codifica los parches con el modelo fundacional congelado."""
    from satinsight.agebs import cities_by_size
    from satinsight.dataset import build_city
    from satinsight.encoders import DofaEncoder

    encoder = DofaEncoder()
    catalogue = cities_by_size(stratify=True)
    keys = args.cities or [
        p.stem.replace(f"_{args.sensor}", "")
        for p in sorted((DATA_ROOT / "instances").glob(f"*_{args.sensor}.parquet"))
    ]
    for key in keys:
        try:
            outputs = build_city(
                key, args.sensor, encoder=encoder, catalogue=catalogue, force=args.force
            )
            print(f"{key}: {outputs['vectors'].name}")
        except Exception as e:
            print(f"FALLO {key}: {type(e).__name__}: {e}")
    return 0


def cmd_reliability(args: argparse.Namespace) -> int:
    """Mide sobre todas las cities si cada rasgo se reproduce al partir la AGEB en dos."""
    from satinsight.pipeline import reliability_of_cities

    summary = reliability_of_cities(args.sensor, tuple(args.cities) or None)
    destination = Path(args.output or DATA_ROOT / f"fiabilidad_{args.sensor}.csv")
    summary.to_csv(destination, index=False)
    print(f"{len(summary)} features sobre {int(summary.cities.max())} cities → {destination}")
    print(summary.head(5).round(3).to_string(index=False))
    print("  ...")
    print(summary.tail(5).round(3).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satinsight", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("aoi", help="lists the pilot boxes").set_defaults(func=cmd_aoi)

    probe = sub.add_parser("probe", help="counts scenes available over an AOI")
    probe.add_argument("aoi", help="box key, for example tuxtla")
    probe.add_argument("--period", default=CENSUS_PERIOD)
    probe.set_defaults(func=cmd_probe)

    panels = sub.add_parser("panels", help="renders the inspection panels")
    panels.add_argument("aoi", help="box key, for example tuxtla")
    panels.add_argument("--period", default=CENSUS_PERIOD)
    panels.add_argument("--output", default="docs/figs", help="destination folder")
    panels.add_argument("--max-s2", type=int, default=36, help="maximum scenes of the S2 composite")
    panels.add_argument("--max-s1", type=int, default=24, help="maximum scenes of the S1 composite")
    panels.set_defaults(func=cmd_panels)

    agebs = sub.add_parser("agebs", help="summarises the AGEB and their grades per city")
    agebs.add_argument("cities", nargs="*", help="city keys; empty runs them all")
    agebs.set_defaults(func=cmd_agebs)

    features = sub.add_parser("features", help="extracts the per-AGEB features of one sensor")
    features.add_argument("sensor", choices=SENSORS)
    features.add_argument("cities", nargs="*", help="city keys; empty runs them all")
    features.add_argument("--output", help="path of the output parquet")
    features.add_argument(
        "--scale", default="fixed", choices=SCALES, help="how texture is quantised"
    )
    features.add_argument(
        "--max-scenes",
        type=int,
        help="maximum scenes of the composite; lowering it shortens the download",
    )
    features.set_defaults(func=cmd_features)

    base = sub.add_parser("baseline", help="runs the phase one comparison")
    base.add_argument("sensor", choices=SENSORS)
    base.add_argument("--features", help="parquet of already extracted features")
    base.add_argument("--output", help="csv to leave the per-fold detail in")
    base.add_argument(
        "--estandarizar",
        action="store_true",
        help="centres each feature within its city; runs as an ablation",
    )
    base.set_defaults(func=cmd_baseline)

    diag = sub.add_parser(
        "diagnostics", help="measures whether features describe the city or the deprivation"
    )
    diag.add_argument("sensor", choices=SENSORS)
    diag.add_argument("--features", help="parquet of already extracted features")
    diag.set_defaults(func=cmd_diagnostics)

    progress = sub.add_parser("progress", help="how many cities have both composites")
    progress.add_argument("--detail", action="store_true", help="lists the pending ones")
    progress.add_argument("--no-stratify", action="store_true", help="only the 81 largest")
    progress.set_defaults(func=cmd_progress)

    figs = sub.add_parser("figures", help="regenerates the phase one figures")
    figs.add_argument("city", help="city key, for example tapachula")
    figs.add_argument("--output", default="docs/figs", help="destination folder")
    figs.set_defaults(func=cmd_figures)

    bags = sub.add_parser("bags", help="tiles cities and assembles their MIL bags")
    bags.add_argument("sensor", choices=("s2", "s1"))
    bags.add_argument("cities", nargs="*", help="keys; empty runs those already composited")
    bags.add_argument("--size", type=int, default=224, help="side of the window in pixels")
    bags.set_defaults(func=cmd_bags)

    partition = sub.add_parser("partition", help="deals the cities into train, validation and test")
    partition.add_argument(
        "--proportions",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="how cities are dealt between the three sets",
    )
    partition.add_argument(
        "--force", action="store_true", help="rebuilds a partition already written"
    )
    partition.set_defaults(func=cmd_partition)

    vectors = sub.add_parser("vectors", help="encodes the patches with the foundation model")
    vectors.add_argument("sensor", choices=("s2", "s1"))
    vectors.add_argument("cities", nargs="*")
    vectors.add_argument("--force", action="store_true")
    vectors.set_defaults(func=cmd_vectors)

    reliability = sub.add_parser(
        "reliability", help="measures whether each feature reproduces between halves"
    )
    reliability.add_argument("sensor", choices=SENSORS)
    reliability.add_argument("cities", nargs="*", help="keys; empty runs every composited one")
    reliability.add_argument("--output", help="path of the output csv")
    reliability.set_defaults(func=cmd_reliability)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
