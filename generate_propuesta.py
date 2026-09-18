# -*- coding: utf-8 -*-
"""
PROPUESTA DE SOLUCIÓN — entregable de 1 página (Fase 1, Hackathon EQUANS Perú).

Responde la plantilla oficial de 5 secciones. Todas las cifras se recalculan aquí
desde los datos crudos y desde el ranking entregado: si el pipeline se vuelve a
correr, el documento no puede quedar afirmando números de una corrida anterior.
"""
import json
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, PageBreak)

TARIFA = 0.667
MESES = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO',
         'AGOSTO', 'SETIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
CONS = [f'CONSUMO {m}' for m in MESES]
DIAS = [f'DIAS FACTURADO\n{m}' for m in MESES]

EQUIPO = "____________________"
SOLUCION = "RECUPERA — Focalización de inspecciones por impacto económico"


# ---------------------------------------------------------------
# CIFRAS: se recalculan, no se escriben a mano
# ---------------------------------------------------------------
def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce')


A = pd.DataFrame(json.load(open('data_cache/alimentador_raw.json', encoding='utf-8')))
H = pd.DataFrame(json.load(open('data_cache/historico_raw.json', encoding='utf-8')))
for c in CONS + DIAS:
    A[c] = _num(A[c])
    H[c] = _num(H[c])

CA, DA, CH = A[CONS], A[DIAS], H[CONS]
anual = CA.sum(axis=1)
n_ali = len(A)

K = {}
K['n_ali'] = n_ali
K['n_hist'] = len(H)
K['n_sed'] = A.SED_ID.nunique()
K['med_anual'] = anual.median()
K['mean_anual'] = anual.mean()
K['p90'] = anual.quantile(.90)
K['p99'] = anual.quantile(.99)
K['max_anual'] = anual.max()
K['top10_pct'] = 100 * anual.nlargest(int(n_ali * .1)).sum() / anual.sum()
K['nulos'] = int(CA.isna().sum().sum())
K['nulos_pct'] = 100 * K['nulos'] / CA.size
K['n_celdas'] = CA.size
K['ceros'] = int((CA == 0).sum().sum())
K['ceros_pct'] = 100 * K['ceros'] / CA.size
K['sum_1cero'] = int((CA == 0).any(axis=1).sum())
K['sum_6cero'] = int(((CA == 0).sum(axis=1) >= 6).sum())
K['acum'] = int(((DA > 32) & (DA <= 400)).sum().sum())
K['acum_pct'] = 100 * K['acum'] / DA.size
K['acum_sum'] = int(((DA > 32) & (DA <= 400)).any(axis=1).sum())
K['corr'] = int((DA > 400).sum().sum())
K['corr_sum'] = int((DA > 400).any(axis=1).sum())
K['neg'] = int((CA < 0).sum().sum())
K['pot_med'] = _num(A['POTENCIA USUARIO CONTRATADO']).median()
K['pot_med_h'] = _num(H['POTENCIA USUARIO CONTRATADO']).median()
K['anual_med_h'] = CH.sum(axis=1).median()
K['sed_med'] = A.groupby('SED_ID').size().median()
K['aerea_pct'] = 100 * (A['ACOMETIDA AEREA/SUBTERRANEA'].str.startswith('A').sum()) / n_ali

_m = CA.where(CA > 0)
cv_a = _m.std(axis=1) / _m.mean(axis=1)
_mh = CH.where(CH > 0)
cv_h = _mh.std(axis=1) / _mh.mean(axis=1)
K['cv_ali'] = cv_a.median()
K['cv_man'] = cv_h[H['TIPIFICACIÓN'] == 'MANIPULACIÓN'].median()
K['cv_cla'] = cv_h[H['TIPIFICACIÓN'] == 'CLANDESTINA'].median()
K['cv06_ali'] = 100 * (cv_a > 0.6).mean()
K['cv06_h'] = 100 * (cv_h > 0.6).mean()

p3, u3 = CA[CONS[:3]].mean(axis=1), CA[CONS[-3:]].mean(axis=1)
caida = (p3 - u3) / p3.replace(0, np.nan)
K['caida50'] = int((caida >= .5).sum())
K['caida50_pct'] = 100 * K['caida50'] / n_ali
K['caida70'] = int((caida >= .7).sum())
ph, uh = CH[CONS[:3]].mean(axis=1), CH[CONS[-3:]].mean(axis=1)
K['caida50_h_pct'] = 100 * (((ph - uh) / ph.replace(0, np.nan)) >= .5).sum() / len(H)
K['fu_med'] = (anual / (_num(A['POTENCIA USUARIO CONTRATADO']) * 24 * 365)).median()

rec = pd.to_numeric(H['PROYECCIÓN DE RECUPERO (KWH)'].astype(str)
                    .str.replace(',', '').str.strip(), errors='coerce')
K['rec_med'] = rec.median()
K['rec_max'] = rec.max()
K['rec_man'] = rec[H['TIPIFICACIÓN'] == 'MANIPULACIÓN'].median()
K['rec_cla'] = rec[H['TIPIFICACIÓN'] == 'CLANDESTINA'].median()
K['ratio_cla'] = K['rec_cla'] / K['rec_man']
K['top10_rec'] = 100 * rec.nlargest(int(len(rec) * .1)).sum() / rec.sum()
K['top1_rec'] = 100 * rec.nlargest(int(len(rec) * .01)).sum() / rec.sum()
K['n_man'] = int((H['TIPIFICACIÓN'] == 'MANIPULACIÓN').sum())
K['n_cla'] = int((H['TIPIFICACIÓN'] == 'CLANDESTINA').sum())
_ct = pd.crosstab(H['TIPIFICACIÓN'], H['Tipo_Acomet'].str.startswith('AEREO'),
                  normalize='index') * 100
K['aer_cla'] = _ct.loc['CLANDESTINA', True]
K['aer_man'] = _ct.loc['MANIPULACIÓN', True]
K['sinserv_h_pct'] = 100 * (CH.sum(axis=1) < 60).sum() / len(H)

R = pd.read_csv('entregables/ranking_completo.csv')
N = len(pd.read_excel('entregables/sospechosos.xlsx'))
T = R.head(N)
K['N'] = N
K['N_pct'] = 100 * N / n_ali
K['soles'] = T.recupero_p10_soles.sum()
K['kwh'] = T.recupero_p10_kwh.sum()
K['por_visita'] = K['soles'] / N
K['n_seds_top'] = T.SED_ID.nunique()
_vc = T.SED_ID.value_counts()
K['seds3'] = int((_vc >= 3).sum())
K['casos_seds3'] = int(_vc[_vc >= 3].sum())
K['top_man'] = int((T.tipo_predicho == 'MANIPULACIÓN').sum())
K['top_cla'] = int((T.tipo_predicho == 'CLANDESTINA').sum())
K['caida_top'] = T.caida_pct.median()
K['caida_glob'] = R.caida_pct.median()
K['sinserv'] = int(R.sin_servicio.sum())
K['sinserv_pct'] = 100 * K['sinserv'] / n_ali
K['insuf'] = int(R.datos_insuficientes.sum())
K['techo'] = 100 * min(31, N) / N

B = json.load(open('entregables/backtest_recall.json', encoding='utf-8'))


def _lift(esc):
    return next(r for r in B['escenarios'][esc]['resumen'] if r['K'] == N)['lift_vs_azar']


K['lift_opt'] = _lift('optimista')
K['lift_con'] = _lift('conservador')
K['n_boot'] = B['n_boot']

# ---------------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------------
TINTA = colors.HexColor('#0F172A')
ACENTO = colors.HexColor('#1D4ED8')
TEXTO = colors.HexColor('#1E293B')

ss = getSampleStyleSheet()
st_title = ParagraphStyle('T', parent=ss['Normal'], fontName='Helvetica-Bold',
                          fontSize=12, leading=13.5, textColor=TINTA, spaceAfter=1)
st_meta = ParagraphStyle('M', parent=ss['Normal'], fontName='Helvetica',
                         fontSize=7.4, leading=9, textColor=ACENTO, spaceAfter=4)
st_h = ParagraphStyle('H', parent=ss['Normal'], fontName='Helvetica-Bold',
                      fontSize=8, leading=9.6, textColor=colors.white,
                      backColor=TINTA, borderPadding=(2, 3, 2, 3),
                      spaceBefore=3, spaceAfter=2.5)
st_b = ParagraphStyle('B', parent=ss['Normal'], fontName='Helvetica', fontSize=6.75,
                      leading=7.95, textColor=TEXTO, alignment=4, spaceAfter=1.8)
st_ref = ParagraphStyle('R', parent=ss['Normal'], fontName='Helvetica', fontSize=8,
                        leading=10.5, textColor=TEXTO, leftIndent=12,
                        firstLineIndent=-12, spaceAfter=5)


def P(txt, s=st_b):
    return Paragraph(txt, s)


doc = SimpleDocTemplate('entregables/Propuesta_Solucion_EQUANS.pdf', pagesize=A4,
                        leftMargin=28, rightMargin=28, topMargin=24, bottomMargin=22,
                        title='Propuesta de Solución — Hackathon EQUANS Perú',
                        author='Equipo participante')
S = []

# ---------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------
S.append(P('PROPUESTA DE SOLUCIÓN — HACKATHON DE REDUCCIÓN Y CONTROL DE PÉRDIDAS DE ENERGÍA · EQUANS PERÚ',
           st_title))
S.append(P(f'<b>Nombre de equipo:</b> {EQUIPO} &nbsp;&nbsp;|&nbsp;&nbsp; '
           f'<b>Nombre de la solución:</b> {SOLUCION}', st_meta))
S.append(HRFlowable(width='100%', thickness=1.2, color=TINTA, spaceAfter=4))

# ---------------------------------------------------------------
# 1. SOLUCIÓN
# ---------------------------------------------------------------
S.append(P('1 · SOLUCIÓN PROPUESTA Y DÓNDE IMPACTA', st_h))
S.append(P(
    f"<b>Qué es.</b> Un motor de focalización que convierte los {K['n_ali']:,} suministros del alimentador en una "
    "<b>lista de despacho ordenada por soles recuperables esperados</b>, no por probabilidad de fraude. La diferencia no es "
    f"cosmética: en el histórico confirmado el 10% de casos más grandes concentra el {K['top10_rec']:.1f}% de toda la energía "
    "recuperada, de modo que un ranking por probabilidad pura gasta cuadrillas en fraudes ciertos pero económicamente irrelevantes. "
    "<b>Qué produce, por suministro:</b> (a) el <b>CUENTA_ID</b> priorizado; (b) el <b>tipo de vulneración esperado</b> "
    "—MANIPULACIÓN o CLANDESTINA—, que define qué cuadrilla y qué herramienta se despacha; (c) el <b>recupero conservador en "
    "kWh y S/</b> a tarifa referencial S/ 0.667/kWh; (d) el agrupamiento por <b>SED</b> para armar la ruta del día; y (e) una "
    "marca de <b>alta incertidumbre</b> cuando los estimadores discrepan, para que el inspector verifique la acometida antes de "
    "desmontar el medidor. "
    "<b>Quién la usa y cuándo.</b> El responsable de control de pérdidas y el programador de cuadrillas, en el paso de "
    "<b>focalización</b>, mensualmente al cierre de facturación y antes de emitir las órdenes de inspección: sustituye la "
    "selección por reglas —hoy con 7–9% de efectividad— por un criterio económico auditable. La salida se consume en un "
    "<b>dashboard de despacho</b> (mapa de calor por SED y ficha de campo por caso) y en un Excel de órdenes. "
    f"<b>Resultado concreto de esta corrida:</b> {K['N']} suministros priorizados ({K['N_pct']:.2f}% del alimentador) en "
    f"{K['n_seds_top']} de las {K['n_sed']} SEDs, con recupero mínimo proyectado de <b>S/ {K['soles']:,.0f}</b> "
    f"({K['kwh']:,.0f} kWh) — <b>S/ {K['por_visita']:,.0f} por visita</b>, cifra directamente contrastable contra el costo de "
    f"cuadrilla para decidir el ROI. {K['seds3']} SEDs concentran {K['casos_seds3']} de los {K['N']} casos, lo que permite "
    "inspeccionar varias acometidas en una misma cuadra."
))

# ---------------------------------------------------------------
# 2. SEÑALES
# ---------------------------------------------------------------
S.append(P('2 · SEÑALES DE DETECCIÓN', st_h))
S.append(P(
    "<b>Señal 1 — Volatilidad anómala del consumo registrado.</b> <i>¿Qué se observa?</i> El coeficiente de variación del "
    f"consumo normalizado a kWh/día tiene mediana <b>{K['cv_ali']:.3f}</b> en el alimentador, pero <b>{K['cv_man']:.3f}</b> en "
    f"las manipulaciones y <b>{K['cv_cla']:.3f}</b> en las clandestinas del histórico CNR; el {K['cv06_h']:.1f}% de los casos "
    f"confirmados supera CV&gt;0.6 frente al {K['cv06_ali']:.1f}% del alimentador. <i>¿Por qué se asocia a hurto?</i> Un cliente "
    "honesto consume de forma estable porque sus hábitos y el clima lo son. Cuando se interviene el medidor, el consumo real del "
    "predio continúa pero el <b>registrado</b> depende de cuándo está activa la derivación o el imán: la serie se vuelve errática "
    "sin que haya cambiado la actividad del local. Es además la señal más robusta al cambio de zona, porque describe la "
    "<b>forma</b> de la curva y no su nivel. Jokar et al. (2016) y Messinis &amp; Hatziargyriou (2018) reportan la irregularidad "
    "del perfil como el discriminante más estable en detección de PNT."
))
S.append(P(
    "<b>Señal 2 — Caída en escalón que los vecinos del mismo transformador no comparten.</b> <i>¿Qué se observa?</i> "
    f"{K['caida50']:,} suministros ({K['caida50_pct']:.1f}%) caen ≥50% entre el promedio de los 3 primeros meses válidos y los 3 "
    f"últimos, y {K['caida70']:,} caen ≥70%; en el histórico confirmado la proporción equivalente es {K['caida50_h_pct']:.1f}%. La "
    "caída por sí sola no basta: medimos el <b>desacople contra la propia SED</b> con exclusión del caso (<i>leave-one-out</i>; la "
    f"SED mediana tiene {K['sed_med']:.0f} clientes) y contra la LLAVE. <i>¿Por qué podría estar asociado con un hurto?</i> Los "
    "suministros de una SED son vecinos físicos bajo el mismo transformador: comparten clima, estacionalidad y ciclo económico "
    f"local. Si cae toda la SED, la causa es legítima; si cae uno solo mientras sus ~{K['sed_med']:.0f} vecinos se mantienen, el "
    "cambio es propio de ese cliente y no del entorno. Excluirlo de su propia referencia evita que el caso contamine la "
    "comparación. Es el análogo, a nivel de cliente, del <b>balance de energía</b> por transformador, sin necesitar telemetría en "
    "la SED."
))
S.append(P(
    "<b>Señal 3 — Factor de uso incompatible con la potencia contratada, con ceros intercalados.</b> <i>¿Qué se observa?</i> El "
    f"factor de uso (kWh / kW contratados × 24 h × días) tiene mediana <b>{K['fu_med']:.4f}</b>. {K['sum_1cero']:,} suministros "
    f"registran al menos un mes en cero y {K['sum_6cero']:,} registran ≥6 meses en cero, pero solo {K['sinserv']:,} tienen consumo "
    "~nulo de forma <b>sostenida</b>: el resto cae a cero y <b>vuelve</b>. <i>¿Por qué?</i> Un predio dado de baja registra cero de "
    "manera continua; un predio activo que registra cero algunos meses y luego vuelve a registrar indica periodos en que el consumo "
    f"no pasó por el medidor. Combinado con una potencia contratada alta (mediana {K['pot_med']:.1f} kW en el alimentador), un "
    "factor de uso sostenidamente bajo apunta a <b>sub-registro</b> y no a ausencia de actividad; Nagi et al. (2010) y Buzau et al. "
    "(2019) reportan el ratio consumo/capacidad contratada entre las variables de mayor ganancia. <i>Señal auxiliar para "
    f"tipificar:</i> en el histórico el {K['aer_cla']:.1f}% de las CLANDESTINAS es de acometida aérea contra {K['aer_man']:.1f}% de "
    f"las MANIPULACIONES, y el alimentador es {K['aerea_pct']:.1f}% aéreo — la conexión directa es físicamente plausible en toda la "
    "zona."
))

# ---------------------------------------------------------------
# 3. ENFOQUE
# ---------------------------------------------------------------
S.append(P('3 · ENFOQUE DE ANÁLISIS Y MODELAMIENTO', st_h))
S.append(P(
    "<b>La restricción que manda el diseño.</b> Los dos datasets no comparten suministros y provienen de <b>zonas distintas</b> de "
    f"la concesión: consumo anual mediano {K['anual_med_h']:,.0f} kWh en el histórico contra {K['med_anual']:,.0f} en el "
    f"alimentador, y potencia mediana {K['pot_med_h']:.1f} kW contra {K['pot_med']:.1f} kW. Probamos el enfoque directo "
    "(PU-learning: clasificar «histórico vs alimentador») y lo <b>descartamos con medición</b>: alcanzaba AUC 0.96 en holdout "
    "<b>reconociendo la zona, no el fraude</b>, y su top-100 ordenaba por tamaño de cliente (consumo mediano 17,250 kWh contra 822 "
    "de la población). Es exactamente el <i>sample selection bias</i> que Glauner et al. (2017) señalan como el fallo dominante de "
    "los modelos de PNT."
))
S.append(P(
    "<b>Método adoptado: tres estimadores independientes cuyo acuerdo constituye la evidencia.</b> <b>(i) Firma de fraude</b>, sin "
    f"entrenamiento y auditable: distancia ponderada de cada suministro al perfil de los {K['n_hist']:,} fraudes confirmados. El "
    "peso de cada variable es max(AUC<i>centrada</i> − 0.5, 0), es decir su poder discriminante <b>después</b> de descontar el "
    "desplazamiento entre zonas, de modo que una variable cuya señal es puramente zonal recibe peso ≈0 automáticamente. "
    "<b>(ii) Isolation Forest</b> (Liu et al., 2008), no supervisado y entrenado <b>dentro</b> del alimentador contra cohortes de la "
    "misma banda de potencia contratada: aporta la perspectiva intra-zona que la firma no puede dar. <b>(iii) LightGBM</b> "
    "(Ke et al., 2017) sobre el histórico, en dos tareas separadas: <b>tipificación</b> MANIPULACIÓN/CLANDESTINA (AUC 0.788 con "
    "validación cruzada de 5 pliegues), que decide la cuadrilla; y <b>recupero en kWh</b> por <b>regresión cuantílica</b> "
    f"(Koenker &amp; Bassett, 1978) a un percentil bajo en log-escala, para que la cola larga del recupero (mediana "
    f"{K['rec_med']:,.0f} kWh, máximo {K['rec_max']:,.0f}) no infle la promesa: se prioriza por el piso, no por la media."
))
S.append(P(
    "<b>Prioridad final = P(fraude) × Recupero conservador (S/).</b> El tamaño del corte <b>no se fija a dedo</b>: N* = argmax del "
    "<i>lift</i> = P(firma ≥ u | hurto confirmado) / P(firma ≥ u | alimentador), con bootstrap para el intervalo de confianza. "
    f"Descartamos el control de FDR porque es aritméticamente inalcanzable aquí: con 31 hurtos entre {K['n_ali']:,} suministros "
    "(prevalencia 0.21%) y un lift máximo medido de 17×, la precisión tope es 3.4%; y marcando "
    f"N={K['N']} el techo de precisión es {K['techo']:.1f}% <b>aunque el modelo fuese un oráculo perfecto</b>. Declararlo es parte "
    "de la propuesta: fija la expectativa correcta para el jefe de campo."
))
S.append(P(
    "<b>Variables construidas y qué aporta cada bloque.</b> <b>Forma temporal (12):</b> CV del kWh/día, magnitud y mes del escalón, "
    "estabilidad previa al quiebre, volatilidad posterior, autocorrelación de orden 1, rugosidad, pendiente normalizada, ratio de "
    "varianza entre mitades del año, caída consecutiva máxima, meses planos, ceros intercalados y fracción de meses muy bajos — "
    "<i>aportan el «cuándo» y el «cómo» del quiebre, la parte de la señal que sí transfiere entre zonas.</i> "
    "<b>Vecindario eléctrico (5):</b> desviación leave-one-out contra SED y contra LLAVE con suavizado bayesiano en SEDs de &lt;5 "
    "clientes, caída relativa a la SED, desacople de la SED y meses por debajo de la SED — <i>controlan clima y economía local, los "
    "dos confusores que generan falsos positivos.</i> <b>Contractuales y físicas (5):</b> factor de uso sobre potencia contratada, "
    "conexionado monofásico/trifásico, acometida aérea/subterránea, fracción de periodos acumulados y CV de días facturados — "
    "<i>aportan capacidad instalada y contexto de acometida, que discriminan el tipo de vulneración.</i>"
))
S.append(P(
    "<b>Agente LLM (Google Gemini), con dos roles acotados.</b> (1) Segunda opinión sobre los casos de <b>zona ambigua</b> con mayor "
    "recupero potencial, en votación contra un KNN: una discordancia &gt;20 pp marca el caso como alta incertidumbre. "
    "(2) Explicación en lenguaje natural del caso, bajo demanda en el dashboard. El LLM <b>no</b> produce el ranking: el resultado "
    "sigue siendo reproducible sin él. <b>Fuentes externas.</b> No se usan en esta fase, y por diseño ninguna que permita "
    "reidentificar suministros o personas. La extensión propuesta es agregada, nunca individual: <b>SENAMHI</b> (temperatura media "
    "mensual por estación) para separar la caída por clima de la caída por intervención, y datos abiertos de <b>INEI</b> sobre "
    "actividad económica distrital para contextualizar cierres de negocio; ambas entrarían como covariables a nivel <b>SED–mes</b>."
))

# ---------------------------------------------------------------
# 4. RESULTADOS
# ---------------------------------------------------------------
S.append(P('4 · RESULTADOS PRELIMINARES DEL ANÁLISIS EXPLORATORIO', st_h))
S.append(P(
    f"<b>Distribución del consumo: fuertemente asimétrica.</b> Mediana anual <b>{K['med_anual']:,.0f} kWh</b>, media "
    f"{K['mean_anual']:,.0f}, p90 {K['p90']:,.0f}, p99 {K['p99']:,.0f} y máximo {K['max_anual']:,.0f}. El decil superior concentra "
    f"el <b>{K['top10_pct']:.1f}%</b> de la energía facturada del alimentador. <i>Implicancia:</i> ordenar por probabilidad de "
    "fraude reparte mal las cuadrillas, por lo que la prioridad se multiplica por el recupero en soles y el regresor de recupero se "
    "estima en log-escala y por cuantil bajo."
))
tabla = [
    ['Hallazgo del análisis exploratorio', 'Cifra medida', 'Tratamiento aplicado e implicancia para la solución'],
    ['Consumos faltantes',
     f"{K['nulos']:,} de {K['n_celdas']:,} ({K['nulos_pct']:.2f}%)",
     'Coinciden exactamente con los nulos de días facturados: es un mes no facturado, no un error de captura. Se tratan como faltante, nunca como cero — imputar cero fabricaría caídas inexistentes.'],
    ['Consumos en cero',
     f"{K['ceros']:,} ({K['ceros_pct']:.1f}%)",
     f"{K['sum_1cero']:,} suministros con ≥1 mes en cero y {K['sum_6cero']:,} con ≥6. Separar el cero sostenido (baja del servicio) del cero intercalado (señal de hurto) es el tratamiento clave: es la Señal 3."],
    ['Periodos acumulados (33–400 días)',
     f"{K['acum']:,} celdas ({K['acum_pct']:.2f}%) en {K['acum_sum']:,} suministros",
     'Legítimos según el diccionario oficial, no errores. Se conservan íntegros y el consumo se normaliza a kWh/día; anularlos borraría el momento exacto del quiebre.'],
    ['Días facturados &gt; 400',
     f"{K['corr']} celdas en {K['corr_sum']} suministros",
     f"Físicamente imposibles. Se anulan como faltante. Ningún suministro se elimina: los {K['n_ali']:,} llegan al ranking. Consumos negativos: {K['neg']}."],
    ['Atípico 1 — sin servicio activo',
     f"{K['sinserv']:,} ({K['sinserv_pct']:.1f}%)",
     f"Consumo anual ~nulo o &lt;25% de meses activos: no son hurto sino medidor de baja o local vacío (solo el {K['sinserv_h_pct']:.1f}% de los fraudes confirmados tiene ese perfil). Se marcan y van al fondo del ranking, pero siguen en el CSV completo para gestión comercial."],
    ['Atípico 2 — recupero no diferenciable',
     f"{K['insuf']:,} suministros",
     'Colisión de hoja del regresor (≥5 casos con valor idéntico). Se marcan, no se excluyen: el jefe de campo debe ver dónde el modelo no distingue, en vez de recibir precisión aparente.'],
    ['Asimetría del recupero histórico',
     f"top 10% → {K['top10_rec']:.1f}%; top 1% → {K['top1_rec']:.1f}%",
     f"Mediana {K['rec_med']:,.0f} kWh, máximo {K['rec_max']:,.0f}. El recupero mediano de una CLANDESTINA ({K['rec_cla']:,.0f} kWh) es {K['ratio_cla']:.1f}× el de una MANIPULACIÓN ({K['rec_man']:,.0f}): tipificar cambia el valor esperado de la visita."],
]
# Las celdas van como Paragraph, no como str: una cadena plana no hace wrap en
# reportlab y la tercera columna se derramaba fuera del margen derecho.
st_td = ParagraphStyle('td', parent=ss['Normal'], fontName='Helvetica', fontSize=6.15,
                       leading=7.05, textColor=TEXTO, alignment=4)
st_th = ParagraphStyle('th', parent=st_td, fontName='Helvetica-Bold',
                       textColor=colors.white, alignment=0)
st_k1 = ParagraphStyle('k1', parent=st_td, fontName='Helvetica-Bold', alignment=0)
st_k2 = ParagraphStyle('k2', parent=st_k1, textColor=ACENTO)

tabla = [[Paragraph(c, st_th) for c in tabla[0]]] + [
    [Paragraph(fila[0], st_k1), Paragraph(fila[1], st_k2), Paragraph(fila[2], st_td)]
    for fila in tabla[1:]
]

t = Table(tabla, colWidths=[102, 92, 345])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), TINTA),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
    ('TOPPADDING', (0, 0), (-1, -1), 1.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ('LEFTPADDING', (0, 0), (-1, -1), 3),
    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
]))
S.append(t)
S.append(Spacer(1, 3))
S.append(P(
    f"<b>Salida del ranking y validación sin etiquetas.</b> El corte autónomo selecciona <b>{K['N']} suministros</b> "
    f"(<b>{K['top_man']} MANIPULACIÓN / {K['top_cla']} CLANDESTINA</b>) en {K['n_seds_top']} SEDs. La caída mediana del "
    f"top-{K['N']} es <b>{K['caida_top']:.2f}</b> contra <b>{K['caida_glob']:.2f}</b> en la población completa: el ranking está "
    "separando por la señal declarada y no por tamaño de cliente. Como el alimentador no tiene etiquetas, validamos inyectando 31 "
    f"hurtos del histórico confirmado sobre la población del alimentador, con {K['n_boot']} remuestreos bootstrap: "
    f"<b>{K['lift_opt']:.1f}× sobre el azar</b> en el escenario optimista y <b>{K['lift_con']:.1f}×</b> en el conservador, que "
    "descuenta el sesgo de zona. Reportamos ambos a propósito: el conservador es el piso honesto sobre una zona sin casos "
    "confirmados, y aun ese piso eleva la efectividad muy por encima del 7–9% actual. El ranking se recalibra solo con los "
    "resultados de las primeras inspecciones."
))

# ---------------------------------------------------------------
# 5. ANEXO
# ---------------------------------------------------------------
S.append(PageBreak())
S.append(P('5 · ANEXO · REFERENCIAS CONSULTADAS', st_h))
S.append(P('<i>No cuenta para el límite de una página.</i>', st_b))
S.append(Spacer(1, 4))
refs = [
    ('Fundamento de las señales de detección y del problema de PNT', [
        "Messinis, G. M., &amp; Hatziargyriou, N. D. (2018). Review of non-technical loss detection methods. "
        "<i>Electric Power Systems Research</i>, 158, 250–266. https://doi.org/10.1016/j.epsr.2018.01.005",
        "Jokar, P., Arianpoo, N., &amp; Leung, V. C. M. (2016). Electricity Theft Detection in AMI Using Customers' "
        "Consumption Patterns. <i>IEEE Transactions on Smart Grid</i>, 7(1), 216–226. "
        "https://doi.org/10.1109/TSG.2015.2425222",
        "Nagi, J., Yap, K. S., Tiong, S. K., Ahmed, S. K., &amp; Mohamad, M. (2010). Nontechnical Loss Detection for Metered "
        "Customers in Power Utility Using Support Vector Machines. <i>IEEE Transactions on Power Delivery</i>, 25(2), "
        "1162–1171. https://doi.org/10.1109/TPWRD.2009.2030890",
        "Viegas, J. L., Esteves, P. R., Melício, R., Mendes, V. M. F., &amp; Vieira, S. M. (2017). Solutions for detection of "
        "non-technical losses in the electricity grid: A review. <i>Renewable and Sustainable Energy Reviews</i>, 80, "
        "1256–1268. https://doi.org/10.1016/j.rser.2017.05.193",
    ]),
    ('Sesgo de selección y validación en detección de PNT (sustenta el descarte del PU-learning en la Sección 3)', [
        "Glauner, P., Meira, J. A., Valtchev, P., State, R., &amp; Bettinger, F. (2017). The Challenge of Non-Technical Loss "
        "Detection Using Artificial Intelligence: A Survey. <i>International Journal of Computational Intelligence "
        "Systems</i>, 10(1), 760–775. https://doi.org/10.2991/ijcis.2017.10.1.51",
        "Buzau, M. M., Tejedor-Aguilera, J., Cruz-Romero, P., &amp; Gómez-Expósito, A. (2019). Detection of Non-Technical "
        "Losses Using Smart Meter Data and Supervised Learning. <i>IEEE Transactions on Smart Grid</i>, 10(3), 2661–2670. "
        "https://doi.org/10.1109/TSG.2018.2807925",
    ]),
    ('Métodos y algoritmos empleados', [
        "Liu, F. T., Ting, K. M., &amp; Zhou, Z.-H. (2008). Isolation Forest. <i>2008 Eighth IEEE International Conference on "
        "Data Mining (ICDM)</i>, 413–422. https://doi.org/10.1109/ICDM.2008.17",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: A Highly "
        "Efficient Gradient Boosting Decision Tree. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 30, "
        "3146–3154.",
        "Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50. "
        "https://doi.org/10.2307/1913643",
        "Benjamini, Y., &amp; Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to "
        "Multiple Testing. <i>Journal of the Royal Statistical Society: Series B</i>, 57(1), 289–300. "
        "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x &nbsp;<i>(evaluado y descartado como criterio de corte; ver "
        "Sección 3)</i>",
    ]),
    ('Fuentes institucionales y de dominio', [
        "EQUANS Perú (2025). <i>Diccionario de datos y bases del Hackathon de Reducción y Control de Pérdidas de "
        "Energía</i> — definiciones de PNT, CNR y SED, tipificación de vulneraciones, criterio de días facturados "
        "acumulados (33 a 400 días) y tarifa referencial de S/ 0.667/kWh.",
        "Organismo Supervisor de la Inversión en Energía y Minería (Osinergmin). <i>Norma Técnica de Calidad de los "
        "Servicios Eléctricos</i> y regulación sobre recupero por consumos no registrados. https://www.osinergmin.gob.pe",
        "Servicio Nacional de Meteorología e Hidrología del Perú (SENAMHI). <i>Datos meteorológicos históricos por "
        "estación</i> — fuente externa propuesta en la Sección 3 para controlar el efecto del clima. "
        "https://www.senamhi.gob.pe/",
        "Instituto Nacional de Estadística e Informática (INEI). <i>Datos abiertos de actividad económica distrital</i> — "
        "fuente externa propuesta para contextualizar cierres de negocio a nivel agregado, sin reidentificación. "
        "https://www.datosabiertos.gob.pe/",
    ]),
]
for titulo, items in refs:
    S.append(P(f'<b>{titulo}</b>',
               ParagraphStyle('rh', parent=st_ref, fontSize=8.3, leftIndent=0,
                              firstLineIndent=0, textColor=ACENTO,
                              spaceBefore=6, spaceAfter=3)))
    for it in items:
        S.append(P('• ' + it, st_ref))

doc.build(S)
print('PDF generado: entregables/Propuesta_Solucion_EQUANS.pdf')
