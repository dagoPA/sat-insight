"""Incrusta los paneles descargados en la propuesta como data URIs JPEG."""

import base64
import io
import json
from pathlib import Path

from PIL import Image

FIGS = Path("docs/figs")
DOC = Path("docs/propuesta.html")
MARK = "  <!-- MARCADOR_DATOS -->"

stats = json.loads((FIGS / "stats.json").read_text())


def data_uri(name, quality=82):
    img = Image.open(FIGS / name).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"  {name}: {len(b64) // 1024} KB en base64")
    return f"data:image/jpeg;base64,{b64}"


panels = [
    ("s2_nublada.png", "Sentinel-2 · fecha única",
     f"{stats['s2_fecha_nublada']} · {stats['s2_nubes_nublada']}% de nubes",
     "Inservible. Ni la traza urbana ni la altura de los edificios sobreviven a la obstrucción.", "b"),
    ("s2_despejada.png", "Sentinel-2 · fecha única",
     f"{stats['s2_fecha_despejada']} · {stats['s2_nubes_despejada']}% de nubes",
     "La misma zona en la mejor fecha del año. La traza se lee con detalle.", "b"),
    ("s2_compuesto.png", "Sentinel-2 · compuesto 2020",
     f"mediana de {stats['s2_escenas_en_compuesto']} escenas · máscara SCL",
     "Sin rastro de nube. Es el resultado que desarma el argumento de la nubosidad.", "b"),
    ("s1_compuesto.png", "Sentinel-1 RTC · compuesto 2020",
     f"mediana de {stats['s1_escenas_en_compuesto']} escenas · {stats['s1_orbita']}",
     "Falso color VV/VH/ratio: magenta = rebote doble de estructura construida, verde = vegetación.", "a"),
]

cards = []
for fname, titulo, meta, nota, arm in panels:
    cards.append(f"""        <figure class="datacard {arm}">
          <img src="{data_uri(fname)}" alt="{titulo} sobre el AOI de Tuxtla Gutiérrez" loading="lazy">
          <figcaption>
            <p class="datacard-t">{titulo}</p>
            <p class="datacard-m">{meta}</p>
            <p class="datacard-n">{nota}</p>
          </figcaption>
        </figure>""")

seccion = f"""  <!-- ============ DATOS ============ -->
  <section class="section">
    <div class="rail">Datos</div>
    <div class="body">
      <h2>Los cuatro tipos de dato, sobre la misma manzana</h2>
      <p>
        Todo lo que sigue se descargó del catálogo STAC de Planetary Computer sobre un recuadro de
        {stats['aoi_px'][0]}&nbsp;×&nbsp;{stats['aoi_px'][1]} píxeles a {stats['resolucion_px_m']}&nbsp;m
        en {stats['aoi_nombre']} — una de las zonas más nubladas del país, escogida a propósito para
        poner a prueba el argumento de la nubosidad.
      </p>

      <div class="datagrid">
{chr(10).join(cards)}
      </div>

      <div class="statbar">
        <div class="stat b"><span class="stat-n">{stats['s2_escenas_2020']}</span><span class="stat-l">escenas S2<br>en 2020</span></div>
        <div class="stat b"><span class="stat-n">{stats['s2_pct_mayor_50_nubes']}%</span><span class="stat-l">de ellas con<br>&gt;50% de nubes</span></div>
        <div class="stat b"><span class="stat-n">{stats['s2_pct_mayor_80_nubes']}%</span><span class="stat-l">con más del<br>80% de nubes</span></div>
        <div class="stat a"><span class="stat-n">{stats['s1_escenas_2020']}</span><span class="stat-l">escenas S1<br>en 2020</span></div>
      </div>

      <h3>Qué demuestran estos cuatro paneles</h3>
      <p>
        Más de la mitad de las fechas Sentinel-2 del año quedan inutilizables por nubes, y aun así el
        compuesto anual del tercer panel sale impecable. <strong>Basta una fracción de fechas despejadas
        repartidas en doce meses para reconstruir la escena completa.</strong> Ese resultado sostiene lo
        que ya se había advertido: para un producto estático, la nubosidad se resuelve con compositing, y
        el argumento se cae como justificación para elegir radar.
      </p>
      <p>
        Lo que el cuarto panel sí sostiene es la otra línea de argumentación. Sentinel-1 entrega
        retrodispersión calibrada en un rango de {stats['s1_vv_db_p5_p95'][0]} a
        {stats['s1_vv_db_p5_p95'][1]}&nbsp;dB en VV y de {stats['s1_vh_db_p5_p95'][0]} a
        {stats['s1_vh_db_p5_p95'][1]}&nbsp;dB en VH: una magnitud física con las mismas unidades en
        Chiapas, en Bahía y en Antioquia. El compuesto óptico, en cambio, depende de qué fechas del año
        quedaron despejadas, y esas fechas cambian con el clima local — un sesgo que se cuela justo en la
        comparación entre países.
      </p>
      <p>
        Las {stats['s1_escenas_2020']} escenas Sentinel-1 disponibles en 2020 sobre este recuadro también
        liquidan el riesgo de cobertura que figuraba pendiente de verificar: con S1A y S1B operando, el
        sureste mexicano quedó bien cubierto.
      </p>
      <p>
        Queda a la vista, eso sí, el costo de esa elección. Puestos lado a lado, el panel de radar se ve
        visiblemente más basto que el óptico: la traza de calles que en Sentinel-2 se distingue manzana por
        manzana, en Sentinel-1 aparece como un patrón grueso. Es exactamente el handicap de resolución
        anticipado, ahora medido sobre datos reales, y la razón de montar los dos brazos en paralelo.
      </p>
    </div>
  </section>
"""

doc = DOC.read_text()
assert MARK in doc, "marcador ausente"
DOC.write_text(doc.replace(MARK, seccion))
print(f"\nSección insertada · documento {len(DOC.read_text()) // 1024} KB")
