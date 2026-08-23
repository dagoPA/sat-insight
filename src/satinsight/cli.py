"""Interfaz de línea de comandos."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from satinsight import aoi as modulo_aoi
from satinsight.agebs import CITIES, agebs_of_city, grade_summary
from satinsight.baseline import comparar, diagnostico_transferencia, resumen
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
from satinsight.figuras import (
    mapa_agebs_por_ciudad,
    mapa_nacional,
    panel_agebs,
    panel_brazos,
    panel_contraste,
)
from satinsight.pipeline import SCALES, SENSORS, features_of_all
from satinsight.raster import percentiles, read_window, stretch, to_db
from satinsight.render import save_rgb

CENSUS_PERIOD = "2020-01-01/2020-12-31"

log = logging.getLogger("satinsight")


def cmd_aoi(_: argparse.Namespace) -> int:
    """Lista los recuadros piloto disponibles."""
    for clave, area in sorted(modulo_aoi.PILOT.items()):
        alto, ancho = area.approximate_shape()
        print(f"{clave:<12} {area.nombre:<22} {area.entidad:<18} ~{ancho}x{alto} px @10 m")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Reporta cuántas escenas hay disponibles sobre un AOI."""
    area = modulo_aoi.obtener(args.aoi)
    catalogue = open_catalogue()
    print(f"{area.nombre} · {args.period}")

    escenas_s2 = search(COLLECTION_S2, area.bbox, args.period, catalogue)
    resumen = cloud_summary(escenas_s2)
    print(f"  Sentinel-2  {resumen['escenas']:>4} escenas")
    print(f"              nubes mediana {resumen['mediana']}%")
    print(f"              {resumen['pct_mayor_50']}% por encima del 50% de nubes")
    print(f"              {resumen['pct_mayor_80']}% por encima del 80%")

    escenas_s1 = search(COLLECTION_S1, area.bbox, args.period, catalogue)
    print(f"  Sentinel-1  {len(escenas_s1):>4} escenas")
    return 0


def cmd_panels(args: argparse.Namespace) -> int:
    """Descarga una muestra y renderiza los cuatro paneles de inspección."""
    area = modulo_aoi.obtener(args.aoi)
    destino = Path(args.salida)
    catalogue = open_catalogue()
    stats: dict[str, object] = {
        "aoi_clave": area.clave,
        "aoi_nombre": area.nombre,
        "aoi_bbox": list(area.bbox),
        "period": args.period,
        "resolucion_px_m": 10,
    }

    log.info("consultando Sentinel-2")
    escenas_s2 = search(COLLECTION_S2, area.bbox, args.period, catalogue)
    stats["s2"] = cloud_summary(escenas_s2)
    ordenadas = by_cloud_cover(escenas_s2)
    despejada, nublada = ordenadas[0], ordenadas[-1]

    log.info("fecha despejada: %s", despejada.datetime.date())
    bandas = [read_window(despejada.assets[b].href, area.bbox) for b in ("B04", "B03", "B02")]
    forma = bandas[0].shape
    save_rgb(*(stretch(b) for b in bandas), destino / "s2_despejada.png")
    stats["s2_despejada"] = {
        "fecha": str(despejada.datetime.date()),
        "nubes": round(despejada.properties["eo:cloud_cover"], 1),
    }

    log.info("fecha nublada: %s", nublada.datetime.date())
    bandas = [read_window(nublada.assets[b].href, area.bbox, forma) for b in ("B04", "B03", "B02")]
    save_rgb(*(stretch(b) for b in bandas), destino / "s2_nublada.png")
    stats["s2_nublada"] = {
        "fecha": str(nublada.datetime.date()),
        "nubes": round(nublada.properties["eo:cloud_cover"], 1),
    }

    log.info("componiendo Sentinel-2")
    compuesto, usadas = composite_s2(escenas_s2, area.bbox, forma, max_scenes=args.max_s2)
    save_rgb(
        *(stretch(compuesto[b]) for b in ("B04", "B03", "B02")),
        destino / "s2_compuesto.png",
    )
    stats["s2_compuesto"] = {"scenes_used": usadas}

    log.info("consultando y componiendo Sentinel-1")
    escenas_s1 = search(COLLECTION_S1, area.bbox, args.period, catalogue)
    sar, meta = composite_s1(escenas_s1, area.bbox, forma, max_scenes=args.max_s1)
    vv_db, vh_db = to_db(sar["vv"]), to_db(sar["vh"])
    save_rgb(stretch(vv_db), stretch(vh_db), stretch(vv_db - vh_db), destino / "s1_compuesto.png")
    stats["s1"] = {
        "scenes_available": meta["scenes_available"],
        "scenes_used": meta["scenes_used"],
        "orbit": meta["orbit"],
        "vv_db_p5_p95": list(percentiles(vv_db)),
        "vh_db_p5_p95": list(percentiles(vh_db)),
    }
    stats["aoi_px"] = [forma[1], forma[0]]

    (destino / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def cmd_agebs(args: argparse.Namespace) -> int:
    """Resume las AGEB de cada ciudad piloto y su distribución de grados."""
    total = 0
    for clave in args.cities or sorted(CITIES):
        agebs = agebs_of_city(clave)
        total += len(agebs)
        habitantes = int(agebs["poblacion"].sum())
        print(f"\n{CITIES[clave].name} · {len(agebs)} AGEB · {habitantes:,} hab")
        print(grade_summary(agebs).to_string())
    print(f"\ntotal: {total} AGEB")
    return 0


def cmd_rasgos(args: argparse.Namespace) -> int:
    """Extrae la textura por AGEB de un sensor y la deja en disco."""
    from satinsight.agebs import cities_by_size

    catalogue = cities_by_size(stratify=True)
    claves = args.cities or sorted(
        p.stem.replace(f"_{args.sensor}", "")
        for p in (DATA_ROOT / "compuestos").glob(f"*_{args.sensor}.tif")
    )
    tabla = features_of_all(
        args.sensor,
        tuple(claves),
        max_scenes=args.max_scenes,
        scale=args.scale,
        catalogue=catalogue,
    )
    destino = Path(args.salida or DATA_ROOT / f"rasgos_{args.sensor}_{args.scale}.parquet")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(destino, index=False)
    print(f"{len(tabla)} AGEB × {tabla.shape[1]} columnas → {destino}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Corre la comparación de la fase 1 sobre una tabla de rasgos ya extraída."""
    origen = Path(args.features or DATA_ROOT / f"rasgos_{args.sensor}.parquet")
    if not origen.exists():
        print(f"falta {origen}. Corre primero: satinsight rasgos {args.sensor}", file=sys.stderr)
        return 1

    tabla = pd.read_parquet(origen)
    detalle = comparar(tabla, estandarizar=args.estandarizar)
    print(f"\n{len(tabla)} AGEB · partición dejando una ciudad fuera\n")
    print(resumen(detalle).to_string(index=False))
    print("\npor fold:\n")
    columnas = ["split", "modelo", "ciudad_prueba", "n_prueba", "kappa", "spearman"]
    print(detalle[columnas].round(3).to_string(index=False))

    if args.salida:
        detalle.to_csv(args.salida, index=False)
        print(f"\ndetalle → {args.salida}")
    return 0


def cmd_diagnostico(args: argparse.Namespace) -> int:
    """Reporta, por split de rasgos, si describen la ciudad o el rezago."""
    origen = Path(args.features or DATA_ROOT / f"rasgos_{args.sensor}.parquet")
    if not origen.exists():
        print(f"falta {origen}. Corre primero: satinsight rasgos {args.sensor}", file=sys.stderr)
        return 1

    tabla = pd.read_parquet(origen)
    print(f"\n{len(tabla)} AGEB · {tabla['ciudad'].nunique()} cities\n")
    print("Razón entre la varianza que explica la ciudad y la que explica el grado.")
    print("Por encima de uno, el rasgo describe dónde se midió más que qué se midió.\n")

    for split in ("cobertura", "densidad", "textura"):
        detalle = diagnostico_transferencia(tabla, split).dropna(subset=["razon"])
        if detalle.empty:
            continue
        bajo, medio, alto = detalle["razon"].quantile([0.25, 0.5, 0.75])
        peores = ", ".join(detalle.head(3)["feature"])
        print(
            f"  {split:<10} n={len(detalle):<3} cuartiles {bajo:>6.1f} /{medio:>6.1f} /{alto:>6.1f}"
        )
        print(f"  {'':<10} peores: {peores}")

    print(
        "\nSe lee junto con la fiabilidad por mitades, nunca solo: un rasgo que es ruido "
        "sale con razón baja\nporque el ruido no correlaciona con nada."
    )
    return 0


def cmd_figuras(args: argparse.Namespace) -> int:
    """Regenera las figuras de la fase 1 desde el caché de compuestos."""
    destino = Path(args.salida)
    panel_brazos(args.ciudad, destino / "f1_brazos.png")
    panel_agebs(args.ciudad, destino / "f2_agebs.png")
    for sensor in SENSORS:
        panel_contraste(args.ciudad, sensor, destino / f"f3_contraste_{sensor}.png")
    mapa_nacional(destino / "f4_nacional.png")
    mapa_agebs_por_ciudad(destino / "f5_agebs_ciudades.png")
    print(f"figuras de {args.ciudad} → {destino}")
    return 0


def cmd_avance(args: argparse.Namespace) -> int:
    """Reporta cuántas cities del split tienen ya sus dos compuestos.

    La composición nacional tarda días y sobrevive a la sesión que la lanzó, así que hace
    falta poder consultarla desde cualquier otra.
    """
    from satinsight.agebs import cities_by_size
    from satinsight.cache import composite_path

    catalogue = cities_by_size(stratify=not args.sin_estratificar)
    root = DATA_ROOT / "compuestos"
    completas, a_medias, faltan = [], [], []
    for clave in catalogue:
        hechos = [s for s in SENSORS if composite_path(clave, s, root).exists()]
        destino = completas if len(hechos) == len(SENSORS) else a_medias if hechos else faltan
        destino.append(clave)

    size = sum(p.stat().st_size for p in root.glob("*.tif")) / 1e9 if root.exists() else 0
    print(f"\n{len(completas)} de {len(catalogue)} cities completas")
    print(f"{len(a_medias)} a medias · {len(faltan)} sin empezar · {size:.1f} GB en disco")
    if a_medias:
        print(f"\nen curso: {', '.join(a_medias[:8])}")
    if args.detalle and faltan:
        print(f"\npendientes: {', '.join(faltan)}")
    return 0


def cmd_bolsas(args: argparse.Namespace) -> int:
    """Tesela cities y arma sus bolsas MIL, sin codificar los parches."""
    from satinsight.agebs import cities_by_size
    from satinsight.dataset import build_city

    catalogue = cities_by_size(stratify=True)
    claves = args.cities or [
        p.stem.replace(f"_{args.sensor}", "")
        for p in sorted((DATA_ROOT / "compuestos").glob(f"*_{args.sensor}.tif"))
    ]
    hechas = 0
    for clave in claves:
        try:
            salidas = build_city(
                clave, args.sensor, encoder=None, size=args.size, catalogue=catalogue
            )
        except Exception as e:
            print(f"FALLO {clave}: {type(e).__name__}: {e}")
            continue
        bolsas = pd.read_parquet(salidas["bolsas"])
        instancias = pd.read_parquet(salidas["instancias"])
        print(f"{clave}: {len(bolsas)} bolsas, {len(instancias)} instancias")
        hechas += 1
    print(f"\n{hechas} de {len(claves)} cities")
    return 0


def cmd_particion(args: argparse.Namespace) -> int:
    """Escribe la partición espacial del split nacional."""
    from satinsight.dataset import build_split

    particion = build_split(force=args.force, proportions=tuple(args.proportions))
    resumen = particion.groupby("split").agg(
        cities=("ciudad", "size"),
        agebs=("n_agebs", "sum"),
        rezago_medio=("stratum_value", "mean"),
    )
    resumen["porcentaje"] = 100 * resumen.cities / resumen.cities.sum()
    print(resumen.round(2).to_string())
    return 0


def cmd_vectores(args: argparse.Namespace) -> int:
    """Codifica los parches con el modelo fundacional congelado."""
    from satinsight.agebs import cities_by_size
    from satinsight.dataset import build_city
    from satinsight.encoders import DofaEncoder

    encoder = DofaEncoder()
    catalogue = cities_by_size(stratify=True)
    claves = args.cities or [
        p.stem.replace(f"_{args.sensor}", "")
        for p in sorted((DATA_ROOT / "instancias").glob(f"*_{args.sensor}.parquet"))
    ]
    for clave in claves:
        try:
            salidas = build_city(
                clave, args.sensor, encoder=encoder, catalogue=catalogue, force=args.force
            )
            print(f"{clave}: {salidas['vectores'].name}")
        except Exception as e:
            print(f"FALLO {clave}: {type(e).__name__}: {e}")
    return 0


def cmd_fiabilidad(args: argparse.Namespace) -> int:
    """Mide sobre todas las cities si cada rasgo se reproduce al partir la AGEB en dos."""
    from satinsight.pipeline import reliability_of_cities

    resumen = reliability_of_cities(args.sensor, tuple(args.cities) or None)
    destino = Path(args.salida or DATA_ROOT / f"fiabilidad_{args.sensor}.csv")
    resumen.to_csv(destino, index=False)
    print(f"{len(resumen)} rasgos sobre {int(resumen.cities.max())} cities → {destino}")
    print(resumen.head(5).round(3).to_string(index=False))
    print("  ...")
    print(resumen.tail(5).round(3).to_string(index=False))
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satinsight", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="registro detallado")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("aoi", help="lista los recuadros piloto").set_defaults(func=cmd_aoi)

    probe = sub.add_parser("probe", help="cuenta escenas disponibles sobre un AOI")
    probe.add_argument("aoi", help="clave del recuadro, por ejemplo tuxtla")
    probe.add_argument("--period", default=CENSUS_PERIOD)
    probe.set_defaults(func=cmd_probe)

    panels = sub.add_parser("panels", help="renderiza los paneles de inspección")
    panels.add_argument("aoi", help="clave del recuadro, por ejemplo tuxtla")
    panels.add_argument("--period", default=CENSUS_PERIOD)
    panels.add_argument("--salida", default="docs/figs", help="carpeta de destino")
    panels.add_argument("--max-s2", type=int, default=36, help="escenas máximas del compuesto S2")
    panels.add_argument("--max-s1", type=int, default=24, help="escenas máximas del compuesto S1")
    panels.set_defaults(func=cmd_panels)

    agebs = sub.add_parser("agebs", help="resume las AGEB y sus grados por ciudad")
    agebs.add_argument("cities", nargs="*", help="claves de ciudad; vacío corre todas")
    agebs.set_defaults(func=cmd_agebs)

    rasgos = sub.add_parser("rasgos", help="extrae los rasgos por AGEB de un sensor")
    rasgos.add_argument("sensor", choices=SENSORS)
    rasgos.add_argument("cities", nargs="*", help="claves de ciudad; vacío corre todas")
    rasgos.add_argument("--salida", help="ruta del parquet de salida")
    rasgos.add_argument(
        "--scale", default="fija", choices=SCALES, help="cómo se cuantiza la textura"
    )
    rasgos.add_argument(
        "--max-escenas",
        type=int,
        help="escenas máximas del compuesto; bajarlo acorta la descarga",
    )
    rasgos.set_defaults(func=cmd_rasgos)

    base = sub.add_parser("baseline", help="corre la comparación de la fase 1")
    base.add_argument("sensor", choices=SENSORS)
    base.add_argument("--rasgos", help="parquet de rasgos ya extraído")
    base.add_argument("--salida", help="csv donde dejar el detalle por fold")
    base.add_argument(
        "--estandarizar",
        action="store_true",
        help="centra cada rasgo dentro de su ciudad; corre como ablación",
    )
    base.set_defaults(func=cmd_baseline)

    diag = sub.add_parser("diagnostico", help="mide si los rasgos describen la ciudad o el rezago")
    diag.add_argument("sensor", choices=SENSORS)
    diag.add_argument("--rasgos", help="parquet de rasgos ya extraído")
    diag.set_defaults(func=cmd_diagnostico)

    avance = sub.add_parser("avance", help="cuántas cities llevan sus dos compuestos")
    avance.add_argument("--detalle", action="store_true", help="lista las pendientes")
    avance.add_argument("--sin-stratify", action="store_true", help="solo las 81 mayores")
    avance.set_defaults(func=cmd_avance)

    figs = sub.add_parser("figuras", help="regenera las figuras de la fase 1")
    figs.add_argument("ciudad", help="clave de ciudad, por ejemplo tapachula")
    figs.add_argument("--salida", default="docs/figs", help="carpeta de destino")
    figs.set_defaults(func=cmd_figuras)

    bolsas = sub.add_parser("bolsas", help="tesela cities y arma sus bolsas MIL")
    bolsas.add_argument("sensor", choices=("s2", "s1"))
    bolsas.add_argument("cities", nargs="*", help="claves; vacío corre las ya compuestas")
    bolsas.add_argument("--size", type=int, default=224, help="lado de la ventana en píxeles")
    bolsas.set_defaults(func=cmd_bolsas)

    particion = sub.add_parser("particion", help="reparte las cities en prueba y pliegues")
    particion.add_argument(
        "--proportions",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="reparto de cities entre los tres conjuntos",
    )
    particion.add_argument("--force", action="store_true", help="rehace una partición ya escrita")
    particion.set_defaults(func=cmd_particion)

    vectores = sub.add_parser("vectores", help="codifica los parches con el modelo fundacional")
    vectores.add_argument("sensor", choices=("s2", "s1"))
    vectores.add_argument("cities", nargs="*")
    vectores.add_argument("--force", action="store_true")
    vectores.set_defaults(func=cmd_vectores)

    fiab = sub.add_parser("fiabilidad", help="mide si cada rasgo se reproduce entre mitades")
    fiab.add_argument("sensor", choices=SENSORS)
    fiab.add_argument("cities", nargs="*", help="claves; vacío corre todas las compuestas")
    fiab.add_argument("--salida", help="ruta del csv de salida")
    fiab.set_defaults(func=cmd_fiabilidad)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
