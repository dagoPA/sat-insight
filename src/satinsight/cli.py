"""Interfaz de línea de comandos."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from satinsight import aoi as modulo_aoi
from satinsight.agebs import CIUDADES, agebs_de_ciudad, resumen_grados
from satinsight.baseline import comparar, diagnostico_transferencia, resumen
from satinsight.catalog import (
    COLECCION_S1,
    COLECCION_S2,
    abrir_catalogo,
    buscar,
    por_nubosidad,
    resumen_nubes,
)
from satinsight.composite import compuesto_s1, compuesto_s2
from satinsight.figuras import (
    mapa_agebs_por_ciudad,
    mapa_nacional,
    panel_agebs,
    panel_brazos,
    panel_contraste,
)
from satinsight.ingesta import RAIZ_DATOS
from satinsight.pipeline import ESCALAS, SENSORES, rasgos_de_todas
from satinsight.raster import a_db, estirar, leer_ventana, percentiles
from satinsight.render import guardar_rgb

PERIODO_CENSO = "2020-01-01/2020-12-31"

log = logging.getLogger("satinsight")


def cmd_aoi(_: argparse.Namespace) -> int:
    """Lista los recuadros piloto disponibles."""
    for clave, area in sorted(modulo_aoi.PILOTO.items()):
        alto, ancho = area.forma_aproximada()
        print(f"{clave:<12} {area.nombre:<22} {area.entidad:<18} ~{ancho}x{alto} px @10 m")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Reporta cuántas escenas hay disponibles sobre un AOI."""
    area = modulo_aoi.obtener(args.aoi)
    catalogo = abrir_catalogo()
    print(f"{area.nombre} · {args.periodo}")

    escenas_s2 = buscar(COLECCION_S2, area.bbox, args.periodo, catalogo)
    resumen = resumen_nubes(escenas_s2)
    print(f"  Sentinel-2  {resumen['escenas']:>4} escenas")
    print(f"              nubes mediana {resumen['mediana']}%")
    print(f"              {resumen['pct_mayor_50']}% por encima del 50% de nubes")
    print(f"              {resumen['pct_mayor_80']}% por encima del 80%")

    escenas_s1 = buscar(COLECCION_S1, area.bbox, args.periodo, catalogo)
    print(f"  Sentinel-1  {len(escenas_s1):>4} escenas")
    return 0


def cmd_panels(args: argparse.Namespace) -> int:
    """Descarga una muestra y renderiza los cuatro paneles de inspección."""
    area = modulo_aoi.obtener(args.aoi)
    destino = Path(args.salida)
    catalogo = abrir_catalogo()
    stats: dict[str, object] = {
        "aoi_clave": area.clave,
        "aoi_nombre": area.nombre,
        "aoi_bbox": list(area.bbox),
        "periodo": args.periodo,
        "resolucion_px_m": 10,
    }

    log.info("consultando Sentinel-2")
    escenas_s2 = buscar(COLECCION_S2, area.bbox, args.periodo, catalogo)
    stats["s2"] = resumen_nubes(escenas_s2)
    ordenadas = por_nubosidad(escenas_s2)
    despejada, nublada = ordenadas[0], ordenadas[-1]

    log.info("fecha despejada: %s", despejada.datetime.date())
    bandas = [leer_ventana(despejada.assets[b].href, area.bbox) for b in ("B04", "B03", "B02")]
    forma = bandas[0].shape
    guardar_rgb(*(estirar(b) for b in bandas), destino / "s2_despejada.png")
    stats["s2_despejada"] = {
        "fecha": str(despejada.datetime.date()),
        "nubes": round(despejada.properties["eo:cloud_cover"], 1),
    }

    log.info("fecha nublada: %s", nublada.datetime.date())
    bandas = [leer_ventana(nublada.assets[b].href, area.bbox, forma) for b in ("B04", "B03", "B02")]
    guardar_rgb(*(estirar(b) for b in bandas), destino / "s2_nublada.png")
    stats["s2_nublada"] = {
        "fecha": str(nublada.datetime.date()),
        "nubes": round(nublada.properties["eo:cloud_cover"], 1),
    }

    log.info("componiendo Sentinel-2")
    compuesto, usadas = compuesto_s2(escenas_s2, area.bbox, forma, max_escenas=args.max_s2)
    guardar_rgb(
        *(estirar(compuesto[b]) for b in ("B04", "B03", "B02")),
        destino / "s2_compuesto.png",
    )
    stats["s2_compuesto"] = {"escenas_usadas": usadas}

    log.info("consultando y componiendo Sentinel-1")
    escenas_s1 = buscar(COLECCION_S1, area.bbox, args.periodo, catalogo)
    sar, meta = compuesto_s1(escenas_s1, area.bbox, forma, max_escenas=args.max_s1)
    vv_db, vh_db = a_db(sar["vv"]), a_db(sar["vh"])
    guardar_rgb(
        estirar(vv_db), estirar(vh_db), estirar(vv_db - vh_db), destino / "s1_compuesto.png"
    )
    stats["s1"] = {
        "escenas_disponibles": meta["escenas_disponibles"],
        "escenas_usadas": meta["escenas_usadas"],
        "orbita": meta["orbita"],
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
    for clave in args.ciudades or sorted(CIUDADES):
        agebs = agebs_de_ciudad(clave)
        total += len(agebs)
        habitantes = int(agebs["poblacion"].sum())
        print(f"\n{CIUDADES[clave].nombre} · {len(agebs)} AGEB · {habitantes:,} hab")
        print(resumen_grados(agebs).to_string())
    print(f"\ntotal: {total} AGEB")
    return 0


def cmd_rasgos(args: argparse.Namespace) -> int:
    """Extrae la textura por AGEB de un sensor y la deja en disco."""
    from satinsight.agebs import ciudades_por_tamano

    catalogo = ciudades_por_tamano(estratificar=True)
    claves = args.ciudades or sorted(
        p.stem.replace(f"_{args.sensor}", "")
        for p in (RAIZ_DATOS / "compuestos").glob(f"*_{args.sensor}.tif")
    )
    tabla = rasgos_de_todas(
        args.sensor,
        tuple(claves),
        max_escenas=args.max_escenas,
        escala=args.escala,
        catalogo=catalogo,
    )
    destino = Path(args.salida or RAIZ_DATOS / f"rasgos_{args.sensor}_{args.escala}.parquet")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(destino, index=False)
    print(f"{len(tabla)} AGEB × {tabla.shape[1]} columnas → {destino}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Corre la comparación de la fase 1 sobre una tabla de rasgos ya extraída."""
    origen = Path(args.rasgos or RAIZ_DATOS / f"rasgos_{args.sensor}.parquet")
    if not origen.exists():
        print(f"falta {origen}. Corre primero: satinsight rasgos {args.sensor}", file=sys.stderr)
        return 1

    tabla = pd.read_parquet(origen)
    detalle = comparar(tabla, estandarizar=args.estandarizar)
    print(f"\n{len(tabla)} AGEB · partición dejando una ciudad fuera\n")
    print(resumen(detalle).to_string(index=False))
    print("\npor pliegue:\n")
    columnas = ["conjunto", "modelo", "ciudad_prueba", "n_prueba", "kappa", "spearman"]
    print(detalle[columnas].round(3).to_string(index=False))

    if args.salida:
        detalle.to_csv(args.salida, index=False)
        print(f"\ndetalle → {args.salida}")
    return 0


def cmd_diagnostico(args: argparse.Namespace) -> int:
    """Reporta, por conjunto de rasgos, si describen la ciudad o el rezago."""
    origen = Path(args.rasgos or RAIZ_DATOS / f"rasgos_{args.sensor}.parquet")
    if not origen.exists():
        print(f"falta {origen}. Corre primero: satinsight rasgos {args.sensor}", file=sys.stderr)
        return 1

    tabla = pd.read_parquet(origen)
    print(f"\n{len(tabla)} AGEB · {tabla['ciudad'].nunique()} ciudades\n")
    print("Razón entre la varianza que explica la ciudad y la que explica el grado.")
    print("Por encima de uno, el rasgo describe dónde se midió más que qué se midió.\n")

    for conjunto in ("cobertura", "densidad", "textura"):
        detalle = diagnostico_transferencia(tabla, conjunto).dropna(subset=["razon"])
        if detalle.empty:
            continue
        bajo, medio, alto = detalle["razon"].quantile([0.25, 0.5, 0.75])
        peores = ", ".join(detalle.head(3)["rasgo"])
        print(
            f"  {conjunto:<10} n={len(detalle):<3} "
            f"cuartiles {bajo:>6.1f} /{medio:>6.1f} /{alto:>6.1f}"
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
    for sensor in SENSORES:
        panel_contraste(args.ciudad, sensor, destino / f"f3_contraste_{sensor}.png")
    mapa_nacional(destino / "f4_nacional.png")
    mapa_agebs_por_ciudad(destino / "f5_agebs_ciudades.png")
    print(f"figuras de {args.ciudad} → {destino}")
    return 0


def cmd_avance(args: argparse.Namespace) -> int:
    """Reporta cuántas ciudades del conjunto tienen ya sus dos compuestos.

    La composición nacional tarda días y sobrevive a la sesión que la lanzó, así que hace
    falta poder consultarla desde cualquier otra.
    """
    from satinsight.agebs import ciudades_por_tamano
    from satinsight.cache import ruta_compuesto

    catalogo = ciudades_por_tamano(estratificar=not args.sin_estratificar)
    raiz = RAIZ_DATOS / "compuestos"
    completas, a_medias, faltan = [], [], []
    for clave in catalogo:
        hechos = [s for s in SENSORES if ruta_compuesto(clave, s, raiz).exists()]
        destino = completas if len(hechos) == len(SENSORES) else a_medias if hechos else faltan
        destino.append(clave)

    tamano = sum(p.stat().st_size for p in raiz.glob("*.tif")) / 1e9 if raiz.exists() else 0
    print(f"\n{len(completas)} de {len(catalogo)} ciudades completas")
    print(f"{len(a_medias)} a medias · {len(faltan)} sin empezar · {tamano:.1f} GB en disco")
    if a_medias:
        print(f"\nen curso: {', '.join(a_medias[:8])}")
    if args.detalle and faltan:
        print(f"\npendientes: {', '.join(faltan)}")
    return 0


def cmd_bolsas(args: argparse.Namespace) -> int:
    """Tesela ciudades y arma sus bolsas MIL, sin codificar los parches."""
    from satinsight.agebs import ciudades_por_tamano
    from satinsight.dataset import build_city

    catalogo = ciudades_por_tamano(estratificar=True)
    claves = args.ciudades or [
        p.stem.replace(f"_{args.sensor}", "")
        for p in sorted((RAIZ_DATOS / "compuestos").glob(f"*_{args.sensor}.tif"))
    ]
    hechas = 0
    for clave in claves:
        try:
            salidas = build_city(
                clave, args.sensor, encoder=None, size=args.tamano, catalogo=catalogo
            )
        except Exception as e:
            print(f"FALLO {clave}: {type(e).__name__}: {e}")
            continue
        bolsas = pd.read_parquet(salidas["bolsas"])
        instancias = pd.read_parquet(salidas["instancias"])
        print(f"{clave}: {len(bolsas)} bolsas, {len(instancias)} instancias")
        hechas += 1
    print(f"\n{hechas} de {len(claves)} ciudades")
    return 0


def cmd_particion(args: argparse.Namespace) -> int:
    """Escribe la partición espacial del conjunto nacional."""
    from satinsight.dataset import build_split

    particion = build_split(forzar=args.forzar, proporciones=tuple(args.proporciones))
    resumen = particion.groupby("conjunto").agg(
        ciudades=("ciudad", "size"),
        agebs=("tamano", "sum"),
        rezago_medio=("estrato_valor", "mean"),
    )
    resumen["porcentaje"] = 100 * resumen.ciudades / resumen.ciudades.sum()
    print(resumen.round(2).to_string())
    return 0


def cmd_vectores(args: argparse.Namespace) -> int:
    """Codifica los parches con el modelo fundacional congelado."""
    from satinsight.agebs import ciudades_por_tamano
    from satinsight.dataset import build_city
    from satinsight.encoders import DofaEncoder

    encoder = DofaEncoder()
    catalogo = ciudades_por_tamano(estratificar=True)
    claves = args.ciudades or [
        p.stem.replace(f"_{args.sensor}", "")
        for p in sorted((RAIZ_DATOS / "instancias").glob(f"*_{args.sensor}.parquet"))
    ]
    for clave in claves:
        try:
            salidas = build_city(
                clave, args.sensor, encoder=encoder, catalogo=catalogo, forzar=args.forzar
            )
            print(f"{clave}: {salidas['vectores'].name}")
        except Exception as e:
            print(f"FALLO {clave}: {type(e).__name__}: {e}")
    return 0


def cmd_fiabilidad(args: argparse.Namespace) -> int:
    """Mide sobre todas las ciudades si cada rasgo se reproduce al partir la AGEB en dos."""
    from satinsight.pipeline import fiabilidad_de_ciudades

    resumen = fiabilidad_de_ciudades(args.sensor, tuple(args.ciudades) or None)
    destino = Path(args.salida or RAIZ_DATOS / f"fiabilidad_{args.sensor}.csv")
    resumen.to_csv(destino, index=False)
    print(f"{len(resumen)} rasgos sobre {int(resumen.ciudades.max())} ciudades → {destino}")
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
    probe.add_argument("--periodo", default=PERIODO_CENSO)
    probe.set_defaults(func=cmd_probe)

    panels = sub.add_parser("panels", help="renderiza los paneles de inspección")
    panels.add_argument("aoi", help="clave del recuadro, por ejemplo tuxtla")
    panels.add_argument("--periodo", default=PERIODO_CENSO)
    panels.add_argument("--salida", default="docs/figs", help="carpeta de destino")
    panels.add_argument("--max-s2", type=int, default=36, help="escenas máximas del compuesto S2")
    panels.add_argument("--max-s1", type=int, default=24, help="escenas máximas del compuesto S1")
    panels.set_defaults(func=cmd_panels)

    agebs = sub.add_parser("agebs", help="resume las AGEB y sus grados por ciudad")
    agebs.add_argument("ciudades", nargs="*", help="claves de ciudad; vacío corre todas")
    agebs.set_defaults(func=cmd_agebs)

    rasgos = sub.add_parser("rasgos", help="extrae los rasgos por AGEB de un sensor")
    rasgos.add_argument("sensor", choices=SENSORES)
    rasgos.add_argument("ciudades", nargs="*", help="claves de ciudad; vacío corre todas")
    rasgos.add_argument("--salida", help="ruta del parquet de salida")
    rasgos.add_argument(
        "--escala", default="fija", choices=ESCALAS, help="cómo se cuantiza la textura"
    )
    rasgos.add_argument(
        "--max-escenas",
        type=int,
        help="escenas máximas del compuesto; bajarlo acorta la descarga",
    )
    rasgos.set_defaults(func=cmd_rasgos)

    base = sub.add_parser("baseline", help="corre la comparación de la fase 1")
    base.add_argument("sensor", choices=SENSORES)
    base.add_argument("--rasgos", help="parquet de rasgos ya extraído")
    base.add_argument("--salida", help="csv donde dejar el detalle por pliegue")
    base.add_argument(
        "--estandarizar",
        action="store_true",
        help="centra cada rasgo dentro de su ciudad; corre como ablación",
    )
    base.set_defaults(func=cmd_baseline)

    diag = sub.add_parser("diagnostico", help="mide si los rasgos describen la ciudad o el rezago")
    diag.add_argument("sensor", choices=SENSORES)
    diag.add_argument("--rasgos", help="parquet de rasgos ya extraído")
    diag.set_defaults(func=cmd_diagnostico)

    avance = sub.add_parser("avance", help="cuántas ciudades llevan sus dos compuestos")
    avance.add_argument("--detalle", action="store_true", help="lista las pendientes")
    avance.add_argument("--sin-estratificar", action="store_true", help="solo las 81 mayores")
    avance.set_defaults(func=cmd_avance)

    figs = sub.add_parser("figuras", help="regenera las figuras de la fase 1")
    figs.add_argument("ciudad", help="clave de ciudad, por ejemplo tapachula")
    figs.add_argument("--salida", default="docs/figs", help="carpeta de destino")
    figs.set_defaults(func=cmd_figuras)

    bolsas = sub.add_parser("bolsas", help="tesela ciudades y arma sus bolsas MIL")
    bolsas.add_argument("sensor", choices=("s2", "s1"))
    bolsas.add_argument("ciudades", nargs="*", help="claves; vacío corre las ya compuestas")
    bolsas.add_argument("--tamano", type=int, default=224, help="lado de la ventana en píxeles")
    bolsas.set_defaults(func=cmd_bolsas)

    particion = sub.add_parser("particion", help="reparte las ciudades en prueba y pliegues")
    particion.add_argument(
        "--proporciones",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="reparto de ciudades entre los tres conjuntos",
    )
    particion.add_argument("--forzar", action="store_true", help="rehace una partición ya escrita")
    particion.set_defaults(func=cmd_particion)

    vectores = sub.add_parser("vectores", help="codifica los parches con el modelo fundacional")
    vectores.add_argument("sensor", choices=("s2", "s1"))
    vectores.add_argument("ciudades", nargs="*")
    vectores.add_argument("--forzar", action="store_true")
    vectores.set_defaults(func=cmd_vectores)

    fiab = sub.add_parser("fiabilidad", help="mide si cada rasgo se reproduce entre mitades")
    fiab.add_argument("sensor", choices=SENSORES)
    fiab.add_argument("ciudades", nargs="*", help="claves; vacío corre todas las compuestas")
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
