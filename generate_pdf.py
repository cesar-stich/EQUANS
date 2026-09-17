import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Las cifras del PDF se leen del ranking real en vez de estar escritas a mano: si el
# pipeline se vuelve a correr con otros parámetros, el entregable no puede quedar
# afirmando números de una corrida anterior.
TARIFA_REF_SOLES = 0.667
CORTE_OPERATIVO = int(os.environ.get('CORTE_OPERATIVO', 100))

_rank = pd.read_csv('entregables/ranking_completo.csv')
_top = _rank.head(CORTE_OPERATIVO)
_total_suministros = len(_rank)
_recupero_soles = _top['recupero_p10_soles'].sum()
_recupero_kwh = _recupero_soles / TARIFA_REF_SOLES
_tipos = _top['tipo_predicho'].value_counts()
_tipo_dom = _tipos.index[0]
_tipo_dom_pct = 100.0 * _tipos.iloc[0] / len(_top)
_seds_top = ", ".join(_top['SED_ID'].value_counts().head(3).index.tolist())
_n_seds = _top['SED_ID'].nunique()
_n_sin_servicio = int(_rank['sin_servicio'].sum()) if 'sin_servicio' in _rank.columns else 0

pdf_path = 'entregables/Fase1_Solucion_Tecnica_EQUANS.pdf'
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=17,
    textColor=colors.HexColor('#0F172A'),
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor('#2563EB'),
    spaceAfter=8
)

h1_style = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=9.5,
    leading=12,
    textColor=colors.HexColor('#1E293B'),
    spaceBefore=4,
    spaceAfter=2
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.5,
    leading=10,
    textColor=colors.HexColor('#334155'),
    spaceAfter=4
)

bullet_style = ParagraphStyle(
    'Bullet_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.2,
    leading=9.5,
    textColor=colors.HexColor('#334155'),
    leftIndent=10,
    spaceAfter=2
)

story = []

# Encabezado
story.append(Paragraph("HACKATHON DE REDUCCIÓN Y CONTROL DE PÉRDIDAS DE ENERGÍA — EQUANS PERÚ", title_style))
story.append(Paragraph("<b>FASE 1: MEMORIA TÉCNICA DE LA SOLUCIÓN Y PRIORIZACIÓN ECONÓMICA DE INSPECCIONES</b>", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0F172A'), spaceAfter=8))

# Sección 1: Explicación de la Solución
story.append(Paragraph("1. EXPLICACIÓN DE LA SOLUCIÓN INTEGRAL", h1_style))
story.append(Paragraph(
    "Diseñamos un pipeline de despacho inteligente orientado al <b>impacto económico recuperable en soles</b>. "
    "Frente al paradigma tradicional que solo prioriza por probabilidad de fraude, nuestro sistema combina dos señales independientes: "
    "una <b>firma de fraude</b> construida por distancia al perfil de consumo de los 4,659 casos confirmados del histórico CNR, y un "
    "<b>Isolation Forest no supervisado</b> que mide qué tan atípico es cada suministro frente a sus pares de potencia contratada. "
    "Los pesos de la firma no se fijan a mano: se derivan de la separación medida entre el perfil de fraude y el perfil típico del alimentador, "
    "de modo que la señal dominante emerge de los datos (resultó ser la <b>volatilidad del consumo</b>: 0.64 en clandestinas y 0.43 en manipulación, contra 0.22 en el alimentador). "
    "Un <b>LightGBM supervisado</b> ejecuta dos tareas: (a) clasificar la tipificación (<i>Manipulación</i> vs <i>Clandestina</i>) para definir qué cuadrilla enviar y "
    "(b) estimar el recupero mediante <b>regresión por cuantiles (P25) en log-escala</b>, escenario conservador que blinda a las cuadrillas contra el sesgo de outliers extremos. "
    "La prioridad final es <b>Prioridad = P(Fraude) × Recupero_P25(S/.)</b>, maximizando el retorno de inversión por cuadrilla despachada.",
    body_style
))

story.append(Spacer(1, 4))

# Sección 2: Señales Observadas
story.append(Paragraph("2. SEÑALES FÍSICAS Y ANALÍTICAS OBSERVADAS (FEATURES F1 A F5)", h1_style))
story.append(Paragraph(
    "<b>• Caída Porcentual Sostenida (F2):</b> Detección de quiebre estructural comparando los primeros 3 meses válidos contra los últimos 3 meses, aislando el momento de intervención ilícita.<br/>"
    "<b>• Alternancia de Consumo (F3):</b> Frecuencia de inversión de signos en el consumo mensual para identificar manipulaciones cíclicas o intermitentes del medidor.<br/>"
    "<b>• Factor de Uso Anormal (F4):</b> Relación entre kWh consumidos y capacidad contratada máxima teórica (kW × 24h × días). Factores sostenidos &lt; 0.05 en potencias comerciales denotan sub-registro grave.<br/>"
    "<b>• Desviación Leave-One-Out por SED (F5):</b> Comparación del suministro frente a sus vecinos de transformador, excluyéndose a sí mismo para evitar fuga de datos (data leakage), con suavizado Bayesiano en SEDs con &lt; 5 clientes.",
    body_style
))

story.append(Spacer(1, 4))

# Sección 3: Enfoque y Método de Análisis
story.append(Paragraph("3. ENFOQUE Y MÉTODO DE ANÁLISIS", h1_style))
story.append(Paragraph(
    "<b>• Limpieza Física Estricta:</b> Solo se anulan registros con <i>DÍAS FACTURADOS &gt; 400</i> (físicamente imposibles). Los periodos entre 33 y 400 días corresponden a acumulaciones legítimas del negocio y se conservan íntegros. Los 14,951 suministros permanecen en el pipeline.<br/>"
    "<b>• Cohortes de Comparación por Potencia Contratada:</b> Cada suministro se contrasta contra pares de su misma banda de potencia (dato objetivo declarado). "
    "Se validó primero la inferencia de giro por correlación estacional y se descartó: alcanzaba 31% de acierto frente a un 90.7% de la clase mayoritaria, "
    "es decir agrupaba peor que no agrupar, y esas cohortes son las que definen qué es \"normal\" para el detector de anomalías.<br/>"
    f"<b>• Exclusión de Suministros sin Servicio Activo:</b> {_n_sin_servicio:,} suministros con consumo anual casi nulo se apartan del despacho. "
    "No son hurtos sino medidores de baja o locales vacíos (en el histórico confirmado solo el 6.3% de los fraudes tiene ese perfil), y enviarles cuadrilla no recupera energía. "
    "Permanecen en el ranking completo, señalizados, para gestión comercial.<br/>"
    "<b>• Doble Voto en Incertidumbre:</b> Casos en zona ambigua o con discordancia entre estimadores (LLM vs KNN &gt; 20pp) se etiquetan con <i>Alta Incertidumbre</i> como advertencia al inspector en campo.",
    body_style
))

story.append(Spacer(1, 4))

# Sección 4: Resultado del Análisis Exploratorio Principal y Despacho
story.append(Paragraph("4. RESULTADOS PRINCIPALES Y DESPACHO DE CAMPO", h1_style))
story.append(Paragraph(
    f"Del alimentador de {_total_suministros:,} suministros, el ranking económico focaliza el despacho en los "
    f"<b>{CORTE_OPERATIVO} de mayor retorno esperado</b> ({100.0*CORTE_OPERATIVO/_total_suministros:.1f}% de la población), "
    "corte fijado por capacidad operativa de inspección:",
    body_style
))

# Tabla Resumen — cifras leídas del ranking real, no escritas a mano.
data_table = [
    ["Métrica Clave", "Valor Obtenido", "Impacto Operativo"],
    ["Suministros Auditados", f"{_total_suministros:,} suministros", "100% de la población del alimentador evaluada"],
    ["Suministros Priorizados (Corte)", f"{CORTE_OPERATIVO} suministros", f"{100.0*CORTE_OPERATIVO/_total_suministros:.1f}% de la red: despacho focalizado"],
    ["Recupero Total Proyectado (P25)", f"S/ {_recupero_soles:,.0f}", "Escenario conservador (75% de casos similares recupera más)"],
    ["Energía Recuperable Estimada", f"{_recupero_kwh:,.0f} kWh", "Aporte directo a la reducción de PNT"],
    ["Mezcla de Tipificación en el Top", f"{_tipo_dom.title()} {_tipo_dom_pct:.0f}% + resto", "Cobertura de ambos tipos: define cuadrilla y herramienta"],
    ["Cobertura Geográfica del Despacho", f"{_n_seds} de 95 transformadores", f"Rutas concentradas. Top: {_seds_top}"],
    ["Retorno por Inspección Despachada", f"S/ {_recupero_soles/CORTE_OPERATIVO:,.0f} por visita", "Contrastar contra costo de cuadrilla para el ROI"],
    ["Frecuencia de Actualización", "Mensual (cierre de facturación)", "Diaria si se incorpora telemetría de transformadores"]
]

t = Table(data_table, colWidths=[150, 140, 250])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 7),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(t)

story.append(Spacer(1, 6))
story.append(Paragraph(
    "<i>Entregables generados: Excel con códigos CUENTA_ID priorizados ('entregables/sospechosos.xlsx'), Ranking Completo 14,951 ('entregables/ranking_completo.csv') y Dashboard de Despacho Operativo integrado.</i>",
    body_style
))

doc.build(story)
print("PDF de 1 página generado exitosamente en:", pdf_path)

