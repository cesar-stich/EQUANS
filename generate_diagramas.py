# -*- coding: utf-8 -*-
"""
Diagramas de flujo de los Algoritmos 1 y 2 del pipeline EQUANS, como imagenes PNG
listas para el PDF de 1 pagina y el pitch. Reusa la paleta de generate_propuesta.py.
"""
import os
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

OUT_DIR = 'entregables/diagramas'
os.makedirs(OUT_DIR, exist_ok=True)

FONT = 'DejaVu Sans'
TINTA = '#0F172A'
TEXTO = '#1E293B'

KIND = {
    'start':    dict(face='#0F172A', edge='#0F172A', text='#FFFFFF', sub='#CBD5E1'),
    'step':     dict(face='#EFF6FF', edge='#1D4ED8', text='#0F172A', sub='#1E3A5F'),
    'decision': dict(face='#FFFBEB', edge='#B45309', text='#78350F', sub='#78350F'),
    'model':    dict(face='#F5F3FF', edge='#6D28D9', text='#3B0764', sub='#4C1D95'),
    'formula':  dict(face='#ECFDF5', edge='#047857', text='#064E3B', sub='#065F46'),
    'warn':     dict(face='#FEF2F2', edge='#B91C1C', text='#7F1D1D', sub='#7F1D1D'),
    'output':   dict(face='#E0F2FE', edge='#0369A1', text='#0C4A6E', sub='#0C4A6E'),
}


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def n_lines(text, width):
    return max(1, len(textwrap.wrap(text, width=width, break_long_words=False)))


def draw_box(ax, cx, cy, w, h, title, subtitle=None, kind='step',
             fs_title=10.5, fs_sub=8.4, lw=1.3, title_lines=1):
    c = KIND[kind]
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=0.09",
                          linewidth=lw, edgecolor=c['edge'], facecolor=c['face'],
                          zorder=3)
    ax.add_patch(box)
    if subtitle:
        title_y = cy + h / 2 - 0.10 - (title_lines - 1) * 0.095 - 0.09
        ax.text(cx, cy + h / 2 - 0.10, title, ha='center', va='top',
                 fontsize=fs_title, fontweight='bold', color=c['text'],
                 family=FONT, linespacing=1.35, zorder=4)
        ax.text(cx, cy - h / 2 + 0.10, subtitle, ha='center', va='bottom',
                 fontsize=fs_sub, color=c['sub'], family=FONT, linespacing=1.42, zorder=4)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                 fontsize=fs_title, fontweight='bold', color=c['text'],
                 family=FONT, linespacing=1.3, zorder=4)
    return cy - h / 2, cy + h / 2


def diamond(ax, cx, cy, dw, dh, title, kind='decision', fs=10):
    c = KIND[kind]
    pts = [(cx, cy + dh / 2), (cx + dw / 2, cy), (cx, cy - dh / 2), (cx - dw / 2, cy)]
    poly = Polygon(pts, closed=True, linewidth=1.3, edgecolor=c['edge'],
                    facecolor=c['face'], zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, title, ha='center', va='center', fontsize=fs, fontweight='bold',
             color=c['text'], family=FONT, linespacing=1.3, zorder=4)
    return cy - dh / 2, cy + dh / 2


def arrow(ax, x1, y1, x2, y2, color=TEXTO, lw=1.3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                         linewidth=lw, color=color, shrinkA=1, shrinkB=1, zorder=2)
    ax.add_patch(a)


def elbow_arrow(ax, x1, y1, x2, y2, color=TEXTO, lw=1.3):
    """Vertical-then-horizontal-then-vertical connector, arrowhead at the end."""
    ymid = (y1 + y2) / 2
    ax.plot([x1, x1, x2, x2], [y1, ymid, ymid, y2], color=color, lw=lw,
             solid_capstyle='round', zorder=2)
    a = FancyArrowPatch((x2, ymid), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                         linewidth=lw, color=color, shrinkA=0, shrinkB=1, zorder=2)
    ax.add_patch(a)


def phase_header(ax, cx, y_top, w, text, color_edge, color_face='#F1F5F9', h=0.46, fs=9.6):
    # zorder por debajo de las flechas (2): un conector nunca debe cruzar "detrás" de
    # una cabecera y desaparecer bajo su relleno opaco.
    rect = Rectangle((cx - w / 2, y_top - h), w, h, facecolor=color_face,
                      edgecolor=color_edge, linewidth=0, zorder=0.5)
    ax.add_patch(rect)
    ax.add_patch(Rectangle((cx - w / 2, y_top - h), 0.09, h, facecolor=color_edge,
                            edgecolor='none', zorder=0.6))
    ax.text(cx - w / 2 + 0.28, y_top - h / 2, text, ha='left', va='center',
             fontsize=fs, fontweight='bold', color=color_edge, family=FONT, zorder=0.7)
    return y_top - h


def note(ax, cx, cy, text, fs=8.0, color='#64748B', style='italic'):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=color,
             family=FONT, fontstyle=style, linespacing=1.35)


def finalize(fig, ax, W, H, path, title, subtitle):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.text(W / 2, H - 0.35, title, ha='center', va='top',
            fontsize=15.5, fontweight='bold', color=TINTA, family=FONT)
    ax.text(W / 2, H - 0.80, subtitle, ha='center', va='top',
            fontsize=10, color='#1D4ED8', family=FONT, style='italic')
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white', pad_inches=0.25)
    plt.close(fig)
    print(f"Generado: {path}")


# =====================================================================
# DIAGRAMA 1 — ALGORITMO 1: LIMPIEZA MINIMA DE DATOS
# =====================================================================
W1, H1 = 9.4, 16.3
fig1, ax1 = plt.subplots(figsize=(W1, H1))
cx = W1 / 2

y = H1 - 1.15  # deja espacio para titulo/subtitulo

# INICIO
h = 1.05
sub_ini = wrap("Datos crudos: Alimentador (14,951) + Histórico CNR (4,659), en Supabase", 44)
b, t = draw_box(ax1, cx, y - h / 2, 6.0, h, "INICIO", sub_ini,
                 kind='start', fs_title=11, fs_sub=8.6, title_lines=1)
prev_b, prev_cx = b, cx
y = b - 0.55

h = 1.05
sub = wrap("Se leen los 14,951 suministros del alimentador y los 4,659 casos confirmados del histórico CNR", 46)
b, t = draw_box(ax1, cx, y - h / 2, 6.9, h, "1 · Carga de datasets", sub, kind='step')
arrow(ax1, cx, prev_b, cx, t)
prev_b = b
y = b - 0.55

h = 1.15
sub = wrap("Se corrige el salto de línea de 'DIAS FACTURADO' y se homogenizan los 12 pares de columnas mensuales (enero a diciembre) en ambos datasets", 46)
b, t = draw_box(ax1, cx, y - h / 2, 6.9, h, "2 · Normalización de encabezados", sub, kind='step')
arrow(ax1, cx, prev_b, cx, t)
prev_b = b
y = b - 0.55

h = 1.05
sub = wrap("Días facturados, consumo mensual (kWh) y potencia contratada (kW) pasan de texto a número", 46)
b, t = draw_box(ax1, cx, y - h / 2, 6.9, h, "3 · Conversión a formato numérico", sub, kind='step')
arrow(ax1, cx, prev_b, cx, t)
prev_b = b
y = b - 0.65

dh = 1.15
dcy = y - dh / 2
db, dt = diamond(ax1, cx, dcy, 5.0, dh, "¿Días facturados\n> 400 en ese mes?", fs=10.2)
arrow(ax1, cx, prev_b, cx, dt)
y = db - 0.60

# Ramas
branch_h = 1.75
branch_cy = y - branch_h / 2
cx_no, cx_si = cx - 2.75, cx + 2.75

sub_no = wrap("Periodos de 33 a 400 días son acumulados legítimos (falta de lectura o refacturación), según el diccionario oficial de EQUANS. No son errores.", 33)
bno, tno = draw_box(ax1, cx_no, branch_cy, 4.0, branch_h, "NO → se conserva intacto", sub_no, kind='step', fs_title=9.6, fs_sub=8.0)

sub_si = wrap("Único caso físicamente imposible: implicaría más de un año facturado en un solo registro mensual. Se anula solo ese mes, nunca el suministro completo.", 33)
bsi, tsi = draw_box(ax1, cx_si, branch_cy, 4.0, branch_h, "SÍ → se anula ese mes", sub_si, kind='warn', fs_title=9.6, fs_sub=8.0)

# Conectores del diamante a las ramas
ax1.plot([cx - 2.5, cx_no], [dcy, tno], color='#059669', lw=1.3, zorder=2)
arrow(ax1, cx_no, tno + 0.32, cx_no, tno, color='#059669')
ax1.text(cx - 2.65, dcy + 0.30, "NO", ha='center', fontsize=8.6, fontweight='bold', color='#059669', family=FONT)

ax1.plot([cx + 2.5, cx_si], [dcy, tsi], color='#B91C1C', lw=1.3, zorder=2)
arrow(ax1, cx_si, tsi + 0.32, cx_si, tsi, color='#B91C1C')
ax1.text(cx + 2.65, dcy + 0.30, "SÍ", ha='center', fontsize=8.6, fontweight='bold', color='#B91C1C', family=FONT)

y = bno - 0.55  # bno == bsi

# Merge
h = 1.2
sub = wrap("Potencia contratada y recupero histórico (S/) quedan en formato numérico. No se imputan ceros donde falta una lectura.", 46)
b, t = draw_box(ax1, cx, y - h / 2, 6.9, h, "4 · Variables limpias listas", sub, kind='step')
elbow_arrow(ax1, cx_no, bno, cx, t)
elbow_arrow(ax1, cx_si, bsi, cx, t)
prev_b = b
y = b - 0.55

# SALIDA
h = 1.35
sub = wrap("14,951 + 4,659 registros 100% conservados. Ningún suministro se elimina ni se toca una señal de fraude → pasa intacto al Algoritmo 2.", 44)
b, t = draw_box(ax1, cx, y - h / 2, 7.4, h, "SALIDA", sub, kind='start', fs_title=11, fs_sub=8.6)
arrow(ax1, cx, prev_b, cx, t)
prev_b = b
y = b - 0.55

# Nota final
h = 1.05
note_box_sub = wrap("Por qué importa: borrar o imputar con cero estos registros destruiría exactamente las caídas de consumo que delatan el hurto. El Algoritmo 1 solo corrige lo imposible, nunca lo sospechoso.", 62)
rect = FancyBboxPatch((cx - 4.5, y - h), 9.0, h, boxstyle="round,pad=0,rounding_size=0.09",
                       linewidth=1.0, edgecolor='#94A3B8', facecolor='#F8FAFC',
                       linestyle=(0, (5, 3)), zorder=3)
ax1.add_patch(rect)
ax1.text(cx, y - h / 2, note_box_sub, ha='center', va='center', fontsize=8.3,
          color='#475569', family=FONT, linespacing=1.4, style='italic')
y = y - h - 0.3

finalize(fig1, ax1, W1, H1, f'{OUT_DIR}/diagrama_algoritmo1_limpieza.png',
         "ALGORITMO 1 — LIMPIEZA MÍNIMA DE DATOS",
         "Solo se corrige lo físicamente imposible. No se toca ninguna señal de fraude ni se elimina ningún suministro.")

print(f"Diagrama 1 · altura de contenido usada hasta y={y:.2f} de H1={H1}")


# =====================================================================
# DIAGRAMA 2 — ALGORITMO 2: FEATURES + MODELOS + RANKING ECONÓMICO
# =====================================================================
W2, H2 = 14.2, 39.6
fig2, ax2 = plt.subplots(figsize=(W2, H2))
CX = W2 / 2

y = H2 - 1.15

# INICIO
h = 1.05
sub = wrap("Datos limpios del Algoritmo 1: 14,951 suministros del alimentador + 4,659 casos confirmados del histórico", 48)
b, t = draw_box(ax2, CX, y - h / 2, 8.0, h, "INICIO", sub, kind='start', fs_title=11, fs_sub=8.6)
prev_b = b
y = b - 0.55

# ---- FASE A: Ingeniería de variables ----
arrow(ax2, CX, prev_b, CX, y)
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE A · INGENIERÍA DE VARIABLES — qué se mide de cada suministro", '#1D4ED8')
prev_b = y
y -= 0.28

h = 2.15
sub = wrap("Consumo promedio diario (kWh/día) y su variabilidad · caída de consumo (primeros 3 meses vs. últimos 3) · alternancia y factor de uso frente a la potencia contratada · desviación contra los vecinos de la misma SED y LLAVE · + 19 variables adicionales de forma temporal y de vecindario", 66)
b, t = draw_box(ax2, CX, y - h / 2, 11.4, h, "Construcción de variables por suministro", sub, kind='step', fs_sub=8.6)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.55

# ---- FASE B: Cohortes de comparación ----
arrow(ax2, CX, prev_b, CX, y)
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE B · COHORTES DE COMPARACIÓN — ¿contra quién se compara cada cliente?", '#1D4ED8')
prev_b = y
y -= 0.28

h = 1.75
sub = wrap("Se intentó agrupar por 'giro comercial' inferido de la curva de consumo y se descartó tras medirlo: acertaba solo 31%, peor que asumir siempre la categoría más común (90.7%). Se usa potencia contratada — dato objetivo y declarado, no inferido — en 4 bandas.", 68)
b, t = draw_box(ax2, CX, y - h / 2, 11.4, h, "Agrupación por potencia contratada", sub, kind='step', fs_sub=8.6)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.60

# ---- FASE C: Cuatro modelos en paralelo ----
arrow(ax2, CX, prev_b, CX, y)
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE C · CUATRO MODELOS EN PARALELO — cada uno mide algo distinto", '#6D28D9')
header_c_bottom = y
y -= 0.55

box_w, gap = 3.05, 0.35
step = box_w + gap
start_left = (W2 - (4 * box_w + 3 * gap)) / 2
cxs = [start_left + box_w / 2 + i * step for i in range(4)]

h = 2.55
titles = [
    ("Isolation Forest", "No supervisado. Busca consumos atípicos dentro de cada cohorte de potencia → puntaje de anomalía de 0 a 1."),
    ("Firma de fraude", "Sin entrenar: distancia de cada suministro al perfil de los 4,659 hurtos confirmados, solo por FORMA de la curva (nunca por nivel, para no confundir zona con fraude)."),
    ("LightGBM — tipo de hurto", "Supervisado y calibrado con el histórico. Clasifica MANIPULACIÓN vs. CLANDESTINA → decide qué cuadrilla se despacha."),
    ("LightGBM — recupero esperado", "Regresión por cuantil (percentil 25) sobre el histórico. Estima el recupero conservador en kWh, sin dejarse arrastrar por los casos extremos."),
]
model_bottoms, model_tops = [], []
for cxm, (ttl, sb) in zip(cxs, titles):
    sb_w = wrap(sb, 27)
    b, t = draw_box(ax2, cxm, y - h / 2, box_w, h, ttl, sb_w, kind='model', fs_title=9.4, fs_sub=7.7)
    elbow_arrow(ax2, cxm, header_c_bottom, cxm, t)
    model_bottoms.append(b)
    model_tops.append(t)
y = min(model_bottoms) - 0.55

# Los 4 modelos convergen en el borde superior de la barra de fase (nunca a través de
# ella): si el punto medio del codo cae dentro de la banda de la cabecera, el rectángulo
# opaco de la cabecera tapa el tramo del conector que pasa por detrás.
header_d_top = y
for cxm, bm in zip(cxs, model_bottoms):
    elbow_arrow(ax2, cxm, bm, CX, header_d_top, color='#6D28D9', lw=1.0)

# ---- FASE D: Combinación de señales y segunda opinión ----
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE D · COMBINACIÓN DE SEÑALES Y SEGUNDA OPINIÓN", '#B45309')
header_d_bottom = y
prev_b = header_d_bottom
y -= 0.28

h = 1.35
sub = wrap("= 75% Firma de fraude + 25% Puntaje de anomalía (Isolation Forest)", 46)
b, t = draw_box(ax2, CX, y - h / 2, 8.4, h, "Probabilidad de fraude", sub, kind='formula', fs_sub=8.6)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.60

dh = 1.35
dcy = y - dh / 2
db, dt = diamond(ax2, CX, dcy, 5.6, dh, "¿El caso está en\nzona de duda?", fs=10.2)
arrow(ax2, CX, prev_b, CX, dt)
ax2.text(CX, db - 0.20, "(cerca del umbral de decisión, o anomalía extrema con probabilidad baja)",
          ha='center', fontsize=7.8, color='#78350F', family=FONT, style='italic')
y = db - 0.55

branch_h = 1.95
branch_cy = y - branch_h / 2
cx_no2, cx_si2 = CX - 3.7, CX + 3.7

sub_no2 = wrap("Es la mayoría de los 14,951 casos: la evidencia ya es clara y no hace falta una segunda opinión.", 32)
bno2, tno2 = draw_box(ax2, cx_no2, branch_cy, 4.6, branch_h, "NO → se usa tal cual", sub_no2, kind='step', fs_title=9.4, fs_sub=8.0)

sub_si2 = wrap("Agente LLM (Gemini) + modelo KNN sobre el histórico. El LLM solo se consulta en los casos dudosos de mayor recupero potencial (control de costo). Si discrepan fuerte → se marca 'ALTA INCERTIDUMBRE' para el inspector.", 40)
bsi2, tsi2 = draw_box(ax2, cx_si2, branch_cy, 5.4, branch_h, "SÍ → doble voto de segunda opinión", sub_si2, kind='warn', fs_title=9.2, fs_sub=7.9)

ax2.plot([CX - 2.8, cx_no2], [dcy, tno2], color='#059669', lw=1.3, zorder=2)
arrow(ax2, cx_no2, tno2 + 0.32, cx_no2, tno2, color='#059669')
ax2.text(CX - 3.0, dcy + 0.30, "NO", ha='center', fontsize=8.6, fontweight='bold', color='#059669', family=FONT)

ax2.plot([CX + 2.8, cx_si2], [dcy, tsi2], color='#B45309', lw=1.3, zorder=2)
arrow(ax2, cx_si2, tsi2 + 0.32, cx_si2, tsi2, color='#B45309')
ax2.text(CX + 3.0, dcy + 0.30, "SÍ", ha='center', fontsize=8.6, fontweight='bold', color='#B45309', family=FONT)

y = min(bno2, bsi2) - 0.55

h = 1.05
b, t = draw_box(ax2, CX, y - h / 2, 6.4, h, "Probabilidad final de fraude", kind='formula')
elbow_arrow(ax2, cx_no2, bno2, CX, t)
elbow_arrow(ax2, cx_si2, bsi2, CX, t)
prev_b = b
y = b - 0.55

# ---- FASE E: Priorización económica ----
arrow(ax2, CX, prev_b, CX, y)
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE E · PRIORIZACIÓN ECONÓMICA — el criterio #1 del reto", '#047857')
prev_b = y
y -= 0.28

h = 1.55
sub = wrap("Se ordena por soles recuperables esperados, no solo por sospecha de fraude — evita gastar cuadrillas en hurtos ciertos pero económicamente irrelevantes.", 56)
b, t = draw_box(ax2, CX, y - h / 2, 10.4, h, "PRIORIDAD = Probabilidad final × Recupero estimado (S/)", sub, kind='formula', fs_title=11, fs_sub=8.4, lw=2.0)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.55

h = 1.65
sub = wrap("Consumo casi nulo todo el año (medidor de baja o local vacío) → no es hurto, no hay nada que recuperar. Se marca y baja al fondo del ranking, pero sigue visible en el CSV completo.", 58)
b, t = draw_box(ax2, CX, y - h / 2, 10.4, h, "Exclusión: suministros 'sin servicio activo'", sub, kind='warn', fs_sub=8.4)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.55

h = 1.05
sub = wrap("14,951 suministros ordenados de mayor a menor prioridad económica", 48)
b, t = draw_box(ax2, CX, y - h / 2, 8.0, h, "Ranking final", sub, kind='step', fs_sub=8.6)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.55

# ---- FASE F: Corte autónomo ----
arrow(ax2, CX, prev_b, CX, y)
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE F · CORTE AUTÓNOMO — dónde termina la lista, sin fijarlo a mano", '#0369A1')
prev_b = y
y -= 0.28

h = 1.95
sub = wrap("Umbral de la firma de fraude donde la razón de verosimilitud (evidencia en hurtos confirmados vs. población normal) es máxima, validado con remuestreo bootstrap. Se recalcula con cualquier dato nuevo: el tamaño de la lista no es fijo.", 60)
b, t = draw_box(ax2, CX, y - h / 2, 10.8, h, "Punto de corte automático (N óptimo)", sub, kind='step', fs_sub=8.4)
arrow(ax2, CX, prev_b, CX, t)
prev_b = b
y = b - 0.60

# ---- FASE G: Entregables ----
arrow(ax2, CX, prev_b, CX, y)
y = phase_header(ax2, CX, y, W2 - 0.6, "FASE G · ENTREGABLES", '#0369A1')
header_g_bottom = y
y -= 0.55

g_w, g_gap = 3.9, 0.4
g_step = g_w + g_gap
g_start_left = (W2 - (3 * g_w + 2 * g_gap)) / 2
g_cxs = [g_start_left + g_w / 2 + i * g_step for i in range(3)]

h = 1.75
g_titles = [
    ("Excel de sospechosos", "CUENTA_ID priorizados + tipo de hurto + recupero estimado, en soles y kWh."),
    ("Ranking completo (CSV)", "Los 14,951 suministros del alimentador, con trazabilidad total."),
    ("Datos del dashboard", "Mapa de calor por SED para armar la ruta de la cuadrilla."),
]
g_bottoms = []
for gcx, (ttl, sb) in zip(g_cxs, g_titles):
    sb_w = wrap(sb, 28)
    b, t = draw_box(ax2, gcx, y - h / 2, g_w, h, ttl, sb_w, kind='output', fs_title=9.6, fs_sub=8.0)
    elbow_arrow(ax2, gcx, header_g_bottom, gcx, t)
    g_bottoms.append(b)
y = min(g_bottoms) - 0.55

# SALIDA
h = 1.35
sub = wrap("Lista de inspección ordenada por impacto económico recuperable, lista para el jefe de campo.", 46)
b, t = draw_box(ax2, CX, y - h / 2, 8.6, h, "SALIDA", sub, kind='start', fs_title=11, fs_sub=8.6)
for gcx, gb in zip(g_cxs, g_bottoms):
    elbow_arrow(ax2, gcx, gb, CX, t)
y = b - 0.4

# ---- Leyenda ----
h = 1.3
legend_items = [
    ('step', 'Paso del proceso'), ('model', 'Modelo de machine learning'),
    ('decision', 'Punto de decisión'), ('formula', 'Fórmula / resultado clave'),
    ('warn', 'Exclusión o alerta'), ('output', 'Entregable'),
]
leg_y = y - h / 2
box_leg = FancyBboxPatch((0.3, leg_y - h / 2), W2 - 0.6, h, boxstyle="round,pad=0,rounding_size=0.06",
                          linewidth=0.8, edgecolor='#CBD5E1', facecolor='#FAFAFA', zorder=3)
ax2.add_patch(box_leg)
n_items = len(legend_items)
slot_w = (W2 - 1.0) / n_items
for i, (k, label) in enumerate(legend_items):
    c = KIND[k]
    lx = 0.6 + i * slot_w
    ax2.add_patch(Rectangle((lx, leg_y - 0.12), 0.32, 0.24, facecolor=c['face'],
                             edgecolor=c['edge'], linewidth=1.1, zorder=4))
    ax2.text(lx + 0.44, leg_y, label, ha='left', va='center', fontsize=7.6,
              color='#334155', family=FONT, zorder=4)
y = leg_y - h / 2 - 0.3

finalize(fig2, ax2, W2, H2, f'{OUT_DIR}/diagrama_algoritmo2_priorizacion.png',
         "ALGORITMO 2 — PRIORIZACIÓN POR IMPACTO ECONÓMICO",
         "Convierte cada suministro en un score = probabilidad de hurto × recupero esperado en soles.")

print(f"Diagrama 2 · altura de contenido usada hasta y={y:.2f} de H2={H2}")
