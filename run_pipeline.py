"""
PIPELINE COMPLETO DE DETECCIÓN Y PRIORIZACIÓN DE HURTO - EQUANS PERÚ
Cumpliendo rigurosamente CLAUDE.md y PLAN.md
"""

import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from google import genai

import equans_core as ec
import features_avanzadas as fa
from equans_core import MESES, TARIFA_REF_SOLES

from corte_autonomo import (corte_autonomo, calibrar_posterior, sensibilidad_roi,
                            techo_precision, factor_verosimilitud_perfil,
                            calibrar_verosimilitud_cruzada, clave_caida_actividad)

print("=== INICIANDO PIPELINE DE DETECCIÓN Y PRIORIZACIÓN (EQUANS) ===")

# -------------------------------------------------------------
# CONSTANTES Y CONFIGURACIÓN
# -------------------------------------------------------------

# -------------------------------------------------------------
# PASO 1: ALGORITMO 1 - LIMPIEZA MÍNIMA DE DATOS
# -------------------------------------------------------------
print("\n--- PASO 1: ALGORITMO 1 (Limpieza mínima) ---")

# La carga y la limpieza mínima viven en equans_core para que el backtest valide
# EXACTAMENTE el mismo preprocesamiento que produce el entregable. Reglas aplicadas:
# normalización de headers (el header de dias facturados trae un salto de linea),
# conversión
# numérica, y anulación del par (días, consumo) cuando DIAS > 400 — único caso físicamente
# imposible. Los periodos de 33 a 400 días son acumulados legítimos según el diccionario
# oficial de EQUANS y se conservan intactos.
df_ali, df_hist = ec.cargar_datasets()
print(f"Total suministros conservados intactos: Alimentador={len(df_ali)} (100%), "
      f"Histórico={len(df_hist)} (100%)")

# -------------------------------------------------------------
# PASO 2: FEATURE ENGINEERING (F1 - F5)
# -------------------------------------------------------------
print("\n--- PASO 2: FEATURE ENGINEERING (F1 a F5) ---")

# extract_features vive en equans_core (F1 a F5: kWh/día y su coeficiente de variación,
# caída base-vs-reciente, actividad del medidor, alternancia, factor de uso y desviación
# leave-one-out contra vecinos de SED y de LLAVE).
# Se usan las features AVANZADAS: a las F1-F5 originales se suman 16 de forma temporal
# (escalon con punto de quiebre, hundimiento relativo, rugosidad, periodos acumulados)
# y 3 de vecindario (desacople del patron de la propia SED). Ver features_avanzadas.py.
extract_features = fa.extract_features_avanzadas

feat_ali, kwhd_mat_ali = extract_features(df_ali, is_alimentador=True)
feat_hist, kwhd_mat_hist = extract_features(df_hist, is_alimentador=False)

# Asignar features al DataFrame principal
for col in feat_ali.columns:
    if col not in df_ali.columns:
        df_ali[col] = feat_ali[col]

for col in feat_hist.columns:
    if col not in df_hist.columns:
        df_hist[col] = feat_hist[col]

print("Features F1 a F5 calculadas exitosamente para ambos datasets.")

# -------------------------------------------------------------
# PASO 3: INFERENCIA DE GIRO COMERCIAL
# -------------------------------------------------------------
print("\n--- PASO 3: INFERENCIA DE GIRO COMERCIAL ---")

# Agrupamiento de curvas en 4 Arquetipos de Comportamiento Eléctrico
# NOTA METODOLÓGICA: En estricto apego a las bases del hackathon, NO se afirma la identidad ni actividad del cliente.
# Se clasifican como "Arquetipos de Patrón Eléctrico" (Perfil A/B/C/D) según similitud estadística de curva y potencia.
def mapear_macro_giro(giro_str):
    g = str(giro_str).upper()
    if any(w in g for w in ['VIVIENDA', 'DEPARTAMENTO', 'DOMICILIO', 'CASA', 'RESIDENCIAL']):
        return 'PERFIL_BASE_DOMÉSTICO'
    elif any(w in g for w in ['TALLER', 'INDUSTRIA', 'METAL', 'FABRICA', 'PLASTI', 'CURTIEMBRE', 'TEXTIL', 'MECANIC']):
        return 'PERFIL_ALTA_POTENCIA'
    elif any(w in g for w in ['CASINO', 'TRAGAMONEDA', 'DISCO', 'HOTEL', 'HOSTAL', 'RESTAURANT', 'POLLERIA', 'BAR', 'CLUB']):
        return 'PERFIL_CONSUMO_INTENSIVO'
    else:
        return 'PERFIL_ESTANDAR_REGULAR'

df_hist['MACRO_GIRO'] = df_hist['GIRO_COMERCIAL'].apply(mapear_macro_giro)
print("Distribución de macro-giros en histórico:")
print(df_hist['MACRO_GIRO'].value_counts())

# Calcular curva estacional promedio por macro-categoría
norm_cols = [f'norm_{m}' for m in MESES]
ARQUETIPOS = ['PERFIL_BASE_DOMÉSTICO', 'PERFIL_ALTA_POTENCIA', 'PERFIL_CONSUMO_INTENSIVO', 'PERFIL_ESTANDAR_REGULAR']


def calcular_perfiles(sub_feat_hist, sub_giro_hist, arquetipos):
    perfiles = {}
    for mg in arquetipos:
        sub = sub_feat_hist[sub_giro_hist == mg][norm_cols]
        if len(sub) > 0:
            perfiles[mg] = sub.median().values
    return perfiles


def inferir_giro(cli_curve, pot, perfiles):
    """Correlación de Pearson entre la curva del cliente y cada perfil de referencia,
    con bonificación/penalización por potencia contratada (industrial vs doméstico)."""
    best_mg, best_sim = None, -999.0
    for mg, ref_curve in perfiles.items():
        if np.std(cli_curve) > 1e-4 and np.std(ref_curve) > 1e-4:
            sim = np.corrcoef(cli_curve, ref_curve)[0, 1]
        else:
            sim = -np.linalg.norm(cli_curve - ref_curve)

        if mg == 'PERFIL_ALTA_POTENCIA' and pot >= 10.0:
            sim += 0.3
        elif mg == 'PERFIL_BASE_DOMÉSTICO' and pot > 15.0:
            sim -= 0.3

        if sim > best_sim:
            best_sim, best_mg = sim, mg
    return best_mg


# Paso 2.3.3 — Validar la inferencia ANTES de usarla en el alimentador.
# Se esconde el giro real de cada caso del histórico, se infiere solo con su curva de
# consumo (perfiles calculados con el resto del histórico), y se mide accuracy vs giro real.
# Si accuracy < 70%, se colapsa a super-categorías más amplias hasta alcanzar el umbral.
def validar_accuracy_giro(df_h, feat_h, arquetipos):
    giro_real = df_h['MACRO_GIRO'].values
    perfiles = calcular_perfiles(feat_h, df_h['MACRO_GIRO'], arquetipos)
    aciertos = 0
    n = len(df_h)
    # Muestreo para mantener el costo bajo en datasets grandes; suficiente para estimar accuracy.
    idx_muestra = np.random.RandomState(42).choice(n, size=min(n, 1500), replace=False)
    for idx in idx_muestra:
        cli_curve = feat_h.iloc[idx][norm_cols].values
        pot = df_h.iloc[idx]['POTENCIA_CONTRATADA']
        pred = inferir_giro(cli_curve, pot, perfiles)
        if pred == giro_real[idx]:
            aciertos += 1
    accuracy = aciertos / len(idx_muestra)
    return accuracy, perfiles


SUPER_CATEGORIAS = ['PERFIL_BASE_DOMÉSTICO', 'PERFIL_ALTA_POTENCIA', 'PERFIL_ESTANDAR_REGULAR']

accuracy, perfiles_giro = validar_accuracy_giro(df_hist, feat_hist, ARQUETIPOS)
baseline_mayoritaria = df_hist['MACRO_GIRO'].value_counts(normalize=True).iloc[0]
print(f"Validación de inferencia de giro por curva (holdout del histórico): accuracy={accuracy*100:.1f}% "
      f"| baseline de clase mayoritaria={baseline_mayoritaria*100:.1f}%")

# Paso 2.3.3 del PLAN.md exige >= 70% de accuracy antes de usar la inferencia. La inferencia
# por correlación de curva estacional NO alcanza ese umbral: mide 31% cuando simplemente
# responder siempre la clase mayoritaria ('doméstico', 90.7% del histórico) acertaría 90.7%.
# Es decir, la correlación de curvas no solo no aporta, sino que DESTRUYE información — y esos
# grupos son los que definen las cohortes de comparación del Isolation Forest, así que un
# agrupamiento malo genera falsas alarmas por comparar clientes heterogéneos entre sí.
#
# Causa: el 90.7% del histórico es residencial, de modo que las curvas estacionales de las
# cuatro categorías son casi idénticas (todas dominadas por el patrón doméstico) y la
# correlación entre ellas no discrimina.
#
# Se reemplaza por segmentación según POTENCIA CONTRATADA, que es un dato objetivo declarado
# y presente en ambos datasets (no inferido, no anonimizado, sin riesgo de reidentificación).
# La potencia es lo que realmente define qué consumo es "normal" para un suministro: un
# cliente de 20 kW contratados no es comparable con uno de 3 kW, independientemente de su giro.
if accuracy < 0.70:
    print(f"Accuracy {accuracy*100:.1f}% < 70% y por debajo del baseline: se descarta la inferencia por "
          f"curva y se segmenta por potencia contratada (dato objetivo declarado).")

    segmentar_por_potencia = ec.segmentar_por_potencia

    df_hist['MACRO_GIRO'] = segmentar_por_potencia(df_hist['POTENCIA_CONTRATADA'])
    df_ali['MACRO_GIRO'] = segmentar_por_potencia(df_ali['POTENCIA_CONTRATADA'])
    ARQUETIPOS = sorted(df_ali['MACRO_GIRO'].unique())
    perfiles_giro = calcular_perfiles(feat_hist, df_hist['MACRO_GIRO'], ARQUETIPOS)
else:
    inferencias = []
    for idx in range(len(feat_ali)):
        cli_curve = feat_ali.loc[idx, norm_cols].values
        pot = df_ali.loc[idx, 'POTENCIA_CONTRATADA']
        inferencias.append(inferir_giro(cli_curve, pot, perfiles_giro))
    df_ali['MACRO_GIRO'] = inferencias

feat_ali['MACRO_GIRO'] = df_ali['MACRO_GIRO'].values
feat_hist['MACRO_GIRO'] = df_hist['MACRO_GIRO'].values
print("Distribución de cohortes de comparación en el alimentador:")
print(df_ali['MACRO_GIRO'].value_counts())

# -------------------------------------------------------------
# PASO 4: MODELOS EN PARALELO (Isolation Forest + LightGBM)
# -------------------------------------------------------------
print("\n--- PASO 4: ENTRENAMIENTO DE MODELOS EN PARALELO ---")

# Features comunes para ML
feature_cols = ec.FEATURE_COLS

# Features de firma de fraude para el modelo PU (Paso 4b).
#
# CRÍTICO — solo features de FORMA, nunca de NIVEL absoluto: el histórico y el alimentador
# son zonas distintas de la concesión con niveles de consumo sistemáticamente diferentes
# (consumo anual mediano 1,814 kWh en el histórico contra 823 en el alimentador; potencia
# mediana 5.1 kW contra 11.1 kW). Si se le dan al modelo PU variables de nivel
# (`cons_anual`, `kwhd_promedio`, potencia), aprende a distinguir "zona A vs zona B" —
# un artefacto geográfico — en vez de "fraude vs honesto", y el ranking se convierte en una
# lista de los clientes más grandes del alimentador. Se verificó empíricamente: con
# `cons_anual` incluido, el top 100 tenía consumo mediano de 17,250 kWh contra 822 de la
# población, y `firma_fraude` daba exactamente 0 para 13,204 suministros (separación
# perfecta de datasets = leakage de dominio, no señal de fraude).
#
# Las features de forma (caída relativa, alternancia, factor de uso, desviación respecto a
# los propios vecinos de SED/LLAVE) son comparables entre zonas porque cada una ya está
# normalizada contra la referencia local del suministro.
firma_cols = ec.FIRMA_COLS

# Modelo A: Isolation Forest por grupo de giro inferido
df_ali['anomaly_score'] = 0.0
for mg in df_ali['MACRO_GIRO'].unique():
    mask = df_ali['MACRO_GIRO'] == mg
    sub_X = feat_ali.loc[mask, feature_cols].fillna(0)
    iso = IsolationForest(contamination=0.06, random_state=42)
    iso.fit(sub_X)
    # decision_function: valores más negativos -> más anómalos
    scores = iso.decision_function(sub_X)
    # Normalizar score a 0..1 donde 1 es extremadamente anómalo
    norm_score = 1.0 / (1.0 + np.exp(scores * 5.0))
    df_ali.loc[mask, 'anomaly_score'] = norm_score

print(f"Isolation Forest entrenado. Anomaly score promedio={df_ali['anomaly_score'].mean():.4f}")

# -------------------------------------------------------------
# Modelo A2: firma de fraude por similitud al perfil confirmado del histórico
# -------------------------------------------------------------
# POR QUÉ NO SE USA UN CLASIFICADOR "HISTÓRICO vs ALIMENTADOR" (PU-learning):
# se implementó y se midió, y resultó inválido. Ese planteamiento asume que separar
# "positivos (fraude)" de "no-etiquetados" equivale a aprender la firma del fraude, pero
# aquí los dos datasets vienen de ZONAS DISTINTAS de la concesión, con distribuciones
# sistemáticamente diferentes: consumo anual mediano 1,814 kWh en el histórico contra 823
# en el alimentador, potencia mediana 5.1 kW contra 11.1 kW. El clasificador aprendía a
# reconocer la zona, no el fraude (AUC holdout 0.96 usando solo features de forma; cada
# feature por separado ya separaba las zonas con AUC 0.55-0.72). Normalizar por percentil
# dentro de cada zona lo empeoró (AUC 1.000): las features discretas como `alternancia` y
# `meses_cero_finales` generan escalones de percentil distintos en cada dataset y delatan
# el origen de forma perfecta. El ranking resultante ordenaba por tamaño de cliente
# (consumo mediano del top 100 = 17,250 kWh contra 822 de la población), no por sospecha.
#
# QUÉ SE USA EN SU LUGAR: distancia al perfil de fraude confirmado, feature por feature.
# En vez de preguntar "¿de qué dataset viene?", se pregunta "¿cuánto se parece este
# suministro al perfil de consumo que muestran los fraudes confirmados?". La señal
# discriminante medida en el histórico es la VOLATILIDAD del consumo (`kwhd_cv`):
# mediana 0.639 en CLANDESTINA y 0.434 en MANIPULACIÓN, contra 0.215 en el alimentador.
# Tiene sentido físico: quien manipula el medidor produce un registro errático (el consumo
# real sigue, el registrado salta), mientras que un cliente honesto consume de forma
# estable. Esta comparación es robusta al cambio de zona porque contrasta cada suministro
# contra un perfil de referencia, sin que el modelo pueda usar el nivel absoluto como atajo.
_firma_modelo = ec.FirmaFraude().fit(feat_hist, feat_ali)
PERFIL_FRAUDE = _firma_modelo.perfil
pesos_firma = _firma_modelo.pesos
calcular_firma_fraude = _firma_modelo.transform

print("Pesos de firma de fraude (derivados de la separación medida fraude vs alimentador):")
print(_firma_modelo.resumen())

firma_fraude = calcular_firma_fraude(feat_ali)
df_ali['firma_fraude'] = firma_fraude

# La MISMA firma aplicada a los 4,659 fraudes confirmados da la vara de medir: los quintiles
# de esa distribución son los cortes del nivel de sospecha que ve el jefe de campo. Así el
# nivel significa literalmente "a qué altura de los hurtos reales llega este suministro".
firma_hist = calcular_firma_fraude(feat_hist)  # reutilizada por el corte autónomo
QUINTILES_FRAUDE = [float(np.percentile(firma_hist, q)) for q in (20, 40, 60, 80)]

print(f"Firma de fraude calculada. Mediana en el alimentador: {np.median(firma_fraude):.3f}, "
      f"p95={np.percentile(firma_fraude, 95):.3f}")
print(f"Quintiles de la firma sobre fraude confirmado (cortes de nivel): "
      f"{', '.join(f'{q:.3f}' for q in QUINTILES_FRAUDE)}")

# Modelo B1: Clasificador de tipo de hurto (MANIPULACIÓN=0 vs CLANDESTINA=1)
# entrenado en df_hist
# Features avanzadas incluidas: AUC del clasificador de tipo medida con CV de 5
# pliegues sube de 0.7575 a 0.7883. El tipo decide que cuadrilla se despacha, que es
# el criterio 2 del reto.
COLS_TIPO = feature_cols + fa.FIRMA_COLS_AVANZADAS
X_hist = feat_hist[COLS_TIPO].fillna(0)
y_type = (df_hist['TIPIFICACIÓN'].str.upper().str.strip() == 'CLANDESTINA').astype(int)

lgb_clf = lgb.LGBMClassifier(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbose=-1
)
cal_clf = CalibratedClassifierCV(estimator=lgb_clf, cv=5, method='isotonic')
cal_clf.fit(X_hist, y_type)

# Predecir probabilidades calibradas de clandestina para alimentador
X_ali = feat_ali[COLS_TIPO].fillna(0)
prob_clandestina = cal_clf.predict_proba(X_ali)[:, 1]
df_ali['prob_clandestina'] = prob_clandestina
df_ali['tipo_predicho'] = np.where(prob_clandestina >= 0.50, 'CLANDESTINA', 'MANIPULACIÓN')

# Modelo B2: Regresor de recupero esperado en kWh (Quantile, log-escala)
#
# Corrección respecto a la versión anterior: el regresor se entrenaba solo con
# `feature_cols`, que NO incluye la potencia contratada. Medido sobre el histórico, la
# potencia es el predictor más fuerte del recupero real (spearman 0.134 vs 0.026 del
# consumo anual), y el tipo de vulneración pesa aún más: CLANDESTINA recupera una mediana
# de 10,330 kWh contra 3,417 de MANIPULACIÓN (3x). Sin esas variables el modelo colapsaba
# a un rango casi plano de 2,589-5,182 kWh sobre un real de 2,530-886,290 — es decir, el
# factor económico variaba solo 2x mientras la probabilidad variaba 5.8x, de modo que la
# "priorización económica" en la práctica no priorizaba por economía.
# A diferencia del modelo de firma, aquí SÍ se usan variables de nivel (consumo anual,
# potencia): el regresor estima una magnitud de recupero, no decide a qué dataset pertenece
# el caso, así que la diferencia de nivel entre zonas no se convierte en un atajo.
#
# NO se incluye `es_clandestina` como feature, aunque el tipo correlacione con el recupero
# real. Al ser binaria y muy predictiva, el árbol la usa como interruptor: en una corrida de
# prueba el recupero mínimo de los CLANDESTINA quedó por encima del máximo de los
# MANIPULACIÓN, de modo que el tipo predicho —que es a su vez una predicción, no un dato—
# determinaba el orden del ranking por sí solo y el top 100 salía 99% CLANDESTINA. En el
# histórico real los dos tipos se solapan ampliamente (P25 de CLANDESTINA = 7,873 kWh contra
# P75 de MANIPULACIÓN = 4,876 kWh), así que ese escalón limpio es un artefacto del modelo,
# no un hecho del negocio. El tipo sigue reportándose para decidir qué cuadrilla enviar.
# Las features avanzadas mejoran el ordenamiento economico: correlacion de Spearman
# contra el recupero real sube de 0.3842 a 0.4240 (CV de 5 pliegues sobre el historico).
# Es el criterio 1 del reto, asi que este ordenamiento es el que mas pesa.
rec_feature_cols = (firma_cols + fa.FIRMA_COLS_AVANZADAS
                    + ['cons_anual', 'kwhd_promedio', 'POTENCIA_CONTRATADA'])

feat_hist_rec = feat_hist[[c for c in rec_feature_cols if c != 'POTENCIA_CONTRATADA']].fillna(0).copy()
feat_hist_rec['POTENCIA_CONTRATADA'] = df_hist['POTENCIA_CONTRATADA'].fillna(3.0).values

feat_ali_rec = feat_ali[[c for c in rec_feature_cols if c != 'POTENCIA_CONTRATADA']].fillna(0).copy()
feat_ali_rec['POTENCIA_CONTRATADA'] = df_ali['POTENCIA_CONTRATADA'].fillna(3.0).values

y_rec_log = np.log1p(df_hist['RECUPERO_KWH_REAL'])

# alpha=0.25 en vez de 0.10: el P10 sobre una distribución con cola tan larga empujaba casi
# todas las predicciones contra el mínimo del histórico (~2,530 kWh), destruyendo el poder
# de discriminación. El P25 sigue siendo un escenario conservador (75% de los casos
# similares recuperaron más que esto) pero conserva el ordenamiento económico.
QUANTILE_RECUPERO = 0.25
reg_rec = lgb.LGBMRegressor(
    objective='quantile',
    alpha=QUANTILE_RECUPERO,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=30,
    random_state=42,
    verbose=-1
)
reg_rec.fit(feat_hist_rec[rec_feature_cols], y_rec_log)

pred_rec_log = reg_rec.predict(feat_ali_rec[rec_feature_cols])
recupero_p25_kwh = np.expm1(pred_rec_log)
# Piso operativo: evita que un caso con recupero casi nulo (irrelevante para despachar una
# cuadrilla) reciba prioridad artificialmente baja por redondeo del regresor.
RECUPERO_PISO_KWH = 300.0
recupero_p25_kwh = np.maximum(recupero_p25_kwh, RECUPERO_PISO_KWH)
df_ali['recupero_p10_kwh'] = recupero_p25_kwh
df_ali['recupero_p10_soles'] = recupero_p25_kwh * TARIFA_REF_SOLES

print(f"Modelos LightGBM entrenados. Recupero P{int(QUANTILE_RECUPERO*100)} mediano predicho: "
      f"{np.median(recupero_p25_kwh):.1f} kWh (S/ {np.median(df_ali['recupero_p10_soles']):.1f}), "
      f"rango {recupero_p25_kwh.min():.0f}-{recupero_p25_kwh.max():.0f} kWh")

# -------------------------------------------------------------
# Suministro sin servicio activo (no es hurto, no genera recupero)
# -------------------------------------------------------------
# Hallazgo que motivó este filtro: el top del ranking anterior estaba dominado por
# suministros con consumo anual casi nulo (mediana 279 kWh en el top 35 contra 823 de la
# población). Inspeccionados uno a uno, son medidores con lecturas como
# [0,0,1,1,1,0,0,1,0,0,0,0] kWh en 12 meses: un local vacío o un medidor de baja, no un
# hurto. En el histórico de fraude CONFIRMADO solo el 6.3% de los casos tiene un perfil así,
# así que mandar una cuadrilla ahí es casi siempre un falso positivo sin recupero posible.
#
# Se marcan como 'sin_servicio' y se empujan al fondo del ranking: siguen en el CSV completo
# (para que la empresa revise si corresponde dar de baja o verificar el medidor) pero no
# compiten por el presupuesto de inspecciones.
CONS_ANUAL_MINIMO_KWH = 120.0   # ~10 kWh/mes: por debajo de esto no hay actividad real
df_ali['sin_servicio'] = (
    (feat_ali['cons_anual'].values < CONS_ANUAL_MINIMO_KWH) |
    (feat_ali['ratio_meses_activos'].values < 0.25)
)
n_sin_servicio = int(df_ali['sin_servicio'].sum())
print(f"Suministros sin servicio activo (consumo ~nulo todo el año, no inspeccionables): "
      f"{n_sin_servicio} de {len(df_ali)}")

# Colisión de hoja del árbol: cuando muchos suministros distintos caen en la misma hoja
# terminal del regresor reciben el valor EXACTO de recupero, lo que da falsa apariencia de
# precisión. Se detecta contando duplicados exactos del valor predicho.
MIN_REPETICIONES_COLISION = 5
conteo_valor = df_ali['recupero_p10_kwh'].round(6).map(df_ali['recupero_p10_kwh'].round(6).value_counts())
df_ali['datos_insuficientes'] = conteo_valor >= MIN_REPETICIONES_COLISION
n_datos_insuficientes = int(df_ali['datos_insuficientes'].sum())
print(f"Suministros con recupero no diferenciable (colisión de hoja del regresor): "
      f"{n_datos_insuficientes} de {len(df_ali)} — se marcan, ya NO se excluyen del ranking")

# -------------------------------------------------------------
# PASO 6 (Voto 1): LLM Rol 1 — segunda opinión SOLO para la zona ambigua
# -------------------------------------------------------------
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', "gemini-3.5-flash-lite")
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# Presupuesto de llamadas al LLM: el free tier de Gemini 2.5 Flash limita a ~20
# requests/día. No tiene sentido intentar llamar para cientos de casos ambiguos
# cuando la cuota se agota casi de inmediato. En vez de eso, dentro de la zona
# ambigua se prioriza a los casos con mayor recupero_p10_soles: son los que más
# impactan el ranking económico final si el LLM cambia su clasificación (criterio
# #1 de CLAUDE.md: impacto económico recuperable). El resto de la zona ambigua
# usa solo el voto de KNN.
LLM_PRESUPUESTO_DIARIO = int(os.environ.get('GEMINI_MAX_LLAMADAS', 15))
LLM_MAX_REINTENTOS = 2
LLM_PAUSA_ENTRE_LLAMADAS_SEG = 4.5


def resolver_zona_ambigua_llm(df, feat, mask_ambigua, perfiles_giro, norm_cols, presupuesto=LLM_PRESUPUESTO_DIARIO):
    """
    Paso 2.6, Voto 1 (Rol 1 del LLM): para los casos en zona ambigua con mayor recupero
    potencial (hasta `presupuesto`, acotado por la cuota diaria del free tier), se le
    pide a Gemini una probabilidad de hurto (0-1) razonada sobre el giro inferido, la
    curva estacional de referencia de ese giro y las features del caso. Si el giro no
    tiene referencia, la llamada falla tras reintentos, o el caso no entra en el
    presupuesto, el caso devuelve NaN y la regla de combinación (Paso 2.6) cae a usar
    solo el voto de KNN.
    """
    import time

    n = len(df)
    out = np.full(n, np.nan)
    client = _get_gemini_client()
    idxs_ambiguos = np.where(mask_ambigua.values)[0]

    if client is None or len(idxs_ambiguos) == 0:
        if len(idxs_ambiguos) > 0:
            print(f"AVISO: GEMINI_API_KEY no configurada. Voto LLM omitido para {len(idxs_ambiguos)} casos ambiguos (se usará solo KNN).")
        return out

    # Priorizar por recupero potencial: el presupuesto de llamadas se gasta donde más impacta
    recupero_ambiguos = df.loc[idxs_ambiguos, 'recupero_p10_soles'].values
    orden = np.argsort(-recupero_ambiguos)
    idxs = idxs_ambiguos[orden][:presupuesto]

    print(f"Zona ambigua: {len(idxs_ambiguos)} casos. Consultando Gemini para los {len(idxs)} de mayor recupero potencial (presupuesto={presupuesto}, resto usa solo KNN).")
    for count, i in enumerate(idxs):
        giro = df.loc[i, 'MACRO_GIRO']
        curva_ref = perfiles_giro.get(giro)
        curva_txt = "sin referencia" if curva_ref is None else ", ".join(f"{v:.2f}" for v in curva_ref)
        prompt = f"""Eres un analista de pérdidas no técnicas de energía eléctrica en Perú.
Un suministro cayó en zona ambigua del modelo de detección de hurto y necesita una segunda opinión.

Giro comercial inferido: {giro}
Curva estacional de referencia para ese giro (12 valores normalizados, ENERO a DICIEMBRE): {curva_txt}
Caída de consumo (base vs reciente): {df.loc[i, 'caida_pct']*100:.1f}%
Alternancia de consumo: {feat.loc[i, 'alternancia']:.2f}
Factor de uso (consumo real / capacidad contratada): {df.loc[i, 'factor_uso']:.3f}
Desviación vs mediana de su SED (leave-one-out): {df.loc[i, 'desv_sed']:.2f}
Score de anomalía (Isolation Forest, 0-1): {df.loc[i, 'anomaly_score']:.3f}

Si el giro no tiene referencia estacional fiable, responde exactamente "sin referencia".
Si sí la tienes, responde SOLO con un número decimal entre 0 y 1: la probabilidad de que
este suministro esté cometiendo hurto de energía. No agregues texto adicional."""

        for intento in range(LLM_MAX_REINTENTOS + 1):
            try:
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                text = (resp.text or "").strip().lower()
                if "sin referencia" not in text:
                    val = float(re.findall(r"[-+]?\d*\.?\d+", text)[0])
                    out[i] = min(max(val, 0.0), 1.0)
                break
            except Exception as e:
                es_cuota_agotada = 'RESOURCE_EXHAUSTED' in str(e) or 'quota' in str(e).lower()
                if es_cuota_agotada:
                    print(f"  Cuota diaria de Gemini agotada en el caso {count + 1}/{len(idxs)}. Deteniendo llamadas LLM; el resto usa solo KNN.")
                    n_respondidos = np.sum(~np.isnan(out))
                    print(f"Voto LLM detenido por cuota: {n_respondidos}/{len(idxs)} casos con respuesta.")
                    return out
                if intento < LLM_MAX_REINTENTOS:
                    time.sleep(LLM_PAUSA_ENTRE_LLAMADAS_SEG * (intento + 1))
                    continue
                print(f"  Aviso: fallo en llamada Gemini para índice {i} tras {LLM_MAX_REINTENTOS + 1} intentos: {e}")

        time.sleep(LLM_PAUSA_ENTRE_LLAMADAS_SEG)
        if (count + 1) % 5 == 0:
            print(f"  ...{count + 1}/{len(idxs)} casos consultados")

    n_respondidos = np.sum(~np.isnan(out))
    print(f"Voto LLM completado: {n_respondidos}/{len(idxs)} casos con respuesta ('sin referencia' o error para el resto).")
    return out


# -------------------------------------------------------------
# PASO 5 Y 6: CALIBRACIÓN, ZONAS DE INCERTIDUMBRE Y RESOLUCIÓN
# -------------------------------------------------------------
print("\n--- PASO 5 Y 6: CLASIFICACIÓN DE ZONAS Y DOBLE VOTO ---")

# -------------------------------------------------------------------------------
# NOTA IMPORTANTE DE DISEÑO: df_hist son TODOS casos de fraude confirmado (no tiene
# clientes normales/no-fraude). Por eso B1 (prob_clandestina) solo puede aprender
# "dado que es fraude, ¿de qué tipo es?" — NUNCA "¿es fraude o no?". Usar
# prob_clandestina como si fuera "probabilidad de fraude" en la fórmula de prioridad
# hace que el ranking colapse: cualquier caso con alta certeza de ser MANIPULACIÓN
# tiene prob_clandestina baja y por tanto "prioridad" baja, aunque el modelo esté
# igual de seguro que en un caso CLANDESTINA. Se comprobó empíricamente: el top del
# ranking salía 100% CLANDESTINA sin importar el bono operativo.
#
# Por eso se separan dos probabilidades distintas:
#   probabilidad_fraude  = ¿qué tan sospechoso es este suministro en general?
#                          (Paso 2.5 / Paso 2.7 — alimenta el ranking económico)
#   prob_clandestina     = dado que es sospechoso, ¿de qué tipo es?
#                          (solo decide tipo_predicho, Paso 2.4 B1 — nunca rankea)
# -------------------------------------------------------------------------------

# Probabilidad de fraude: la señal principal es el modelo PU (firma_fraude), que aprende
# del histórico de fraude CONFIRMADO en vez de usar pesos fijos escritos a mano. Se
# complementa con el anomaly_score del Isolation Forest, que aporta la perspectiva
# no supervisada (qué tan raro es este cliente frente a sus vecinos de SED/giro) y captura
# hurtos con patrones que no aparecen en el histórico de la otra zona.
#
# Por qué se eliminó la fórmula anterior (0.5*anomaly + 0.3*caida + 0.2*desv_sed):
# saturaba en caida_pct=1.0, es decir asignaba máxima sospecha a "consumo reciente = 0".
# Contrastado con el histórico real, apenas el 6.7% de los fraudes confirmados tiene esa
# caída total, mientras que el top 200 del ranking anterior tenía un 55% — el ranking
# estaba ordenando por medidor apagado, no por hurto.
# Se mantiene la fórmula de dos señales. Se construyó y midió un ensamble de tres que
# sumaba `caida_vs_sed` (la caída de cada suministro contra la caída mediana de su propia
# SED) con peso 0.40 — señal atractiva porque es ortogonal a la firma (correlación de
# Spearman 0.019) e inmune al problema de zona, al calcularse enteramente dentro del
# alimentador. No se integró: empeora el perfil del top contra el fraude confirmado,
# porque `caida_vs_sed` se satura en los medidores que se apagaron y los empuja arriba.
# La señal sigue calculándose y se exporta al ranking completo y al dashboard, donde SÍ
# aporta: es el argumento más accionable en campo ("cayó 60% cuando sus vecinos del mismo
# transformador subieron 5%"), pero no ordena el ranking.
señal_sed = (0.70 * df_ali['caida_vs_sed'].rank(pct=True)
             + 0.30 * df_ali['desacople_sed'].rank(pct=True))
df_ali['señal_vecindario'] = señal_sed

df_ali['probabilidad_fraude'] = np.clip(
    0.75 * df_ali['firma_fraude'] + 0.25 * df_ali['anomaly_score'],
    0.01, 0.99
)

UMBRAL_ALTO = 0.65
UMBRAL_BAJO = 0.35
MARGEN = 0.15
ANOMALY_ALTO = 0.90  # anomaly_score fuertemente atípico, aunque probabilidad_fraude sea baja

# Zona ambigua: (a) probabilidad_fraude dentro de ± margen de la frontera de decisión
# [umbral_bajo, umbral_alto] — cerca de donde el modelo realmente duda —
# O (b) anomalía extrema con probabilidad_fraude baja.
mask_frontera = (
    ((df_ali['probabilidad_fraude'] >= UMBRAL_BAJO - MARGEN) & (df_ali['probabilidad_fraude'] <= UMBRAL_BAJO + MARGEN)) |
    ((df_ali['probabilidad_fraude'] >= UMBRAL_ALTO - MARGEN) & (df_ali['probabilidad_fraude'] <= UMBRAL_ALTO + MARGEN))
)
mask_anomalia_rara = (df_ali['anomaly_score'] >= ANOMALY_ALTO) & (df_ali['probabilidad_fraude'] < UMBRAL_BAJO)
mask_ambigua_cruda = mask_frontera | mask_anomalia_rara
n_ambiguos_crudo = int(mask_ambigua_cruda.sum())

# Tope operativo: el doble voto (LLM + KNN) del Paso 2.6 no es viable sobre miles de
# casos. Si la zona ambigua cruda excede el tope, nos quedamos solo con los ZONA_AMBIGUA_TOPE
# casos más cercanos al verdadero punto de indecisión (probabilidad_fraude = 0.50) dentro
# de esa zona — son los que el modelo más duda, no una muestra arbitraria del 69% de la
# población. El resto de los casos "ambiguos" pero lejos de 0.50 usan probabilidad_fraude
# tal cual, sin ajuste de doble voto.
ZONA_AMBIGUA_TOPE = int(os.environ.get('ZONA_AMBIGUA_TOPE', 300))
if n_ambiguos_crudo > ZONA_AMBIGUA_TOPE:
    distancia_a_050 = (df_ali['probabilidad_fraude'] - 0.50).abs()
    idxs_top_duda = distancia_a_050[mask_ambigua_cruda].nsmallest(ZONA_AMBIGUA_TOPE).index
    mask_ambigua = pd.Series(False, index=df_ali.index)
    mask_ambigua.loc[idxs_top_duda] = True
    print(f"Zona ambigua cruda: {n_ambiguos_crudo} casos, acotada a los {ZONA_AMBIGUA_TOPE} más cercanos a la frontera de indecisión (0.50).")
else:
    mask_ambigua = mask_ambigua_cruda

print(f"Casos en zona de ambigüedad/incertidumbre: {mask_ambigua.sum()} de 14,951")

# KNN sobre el histórico para segunda opinión (Voto 2 del Paso 2.6) — vota sobre TIPO,
# igual que B1, por la misma limitación estructural del histórico.
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
knn.fit(X_hist, y_type)
knn_proba_cland = knn.predict_proba(X_ali)[:, 1]

# Voto 1 del Paso 2.6: LLM (Gemini). Solo se llama para casos en zona ambigua (control de costo).
# Se le pide al LLM una probabilidad de FRAUDE (no de tipo específico) para que sea
# compatible con probabilidad_fraude, que es lo que realmente ajusta.
llm_proba_fraude = resolver_zona_ambigua_llm(df_ali, feat_ali, mask_ambigua, perfiles_giro, norm_cols)

# KNN vota tipo (prob. de ser CLANDESTINA); se usa como proxy de "sospecha" en la zona
# ambigua: un caso donde KNN también se inclina fuerte hacia un tipo específico refuerza
# la sospecha de fraude, independientemente de cuál tipo sea.
knn_proba_fraude = np.where(knn_proba_cland >= 0.5, knn_proba_cland, 1.0 - knn_proba_cland)

# Regla de combinación: si LLM no respondió ("sin referencia"), usar solo KNN.
prob_llm_o_knn = np.where(np.isnan(llm_proba_fraude), knn_proba_fraude,
                           (llm_proba_fraude + knn_proba_fraude) / 2.0)

# Bandera de alta incertidumbre.
#
# Defecto corregido: antes se calculaba solo como `abs(prob_llm - prob_knn) > 0.20`. Cuando
# no hay GEMINI_API_KEY configurada, `prob_llm` es NaN para todos los casos, y en NumPy
# `NaN > 0.20` es False — así que la bandera salía False para los 14,951 suministros, sin
# excepción. La advertencia nunca llegaba al inspector y la ruta de código estaba muerta en
# la práctica, que es el estado normal de ejecución (el free tier solo cubre ~15 llamadas).
#
# Ahora la discrepancia se mide entre las dos señales independientes que SIEMPRE existen:
# `firma_fraude` (similitud al perfil de fraude confirmado del histórico) y `anomaly_score`
# (rareza frente a los pares de la misma cohorte de potencia). Cuando una señal dice
# "sospechoso" y la otra dice "normal", el caso es genuinamente dudoso y el inspector merece
# saberlo. Si además el LLM respondió, su discrepancia con KNN se suma como segunda fuente.
# La discrepancia se mide en espacio de PERCENTILES, no como diferencia absoluta: las dos
# señales viven en escalas distintas (la firma es una suma ponderada acotada, el
# anomaly_score es una sigmoide del Isolation Forest), así que su diferencia bruta tiene una
# mediana de 0.214 y un umbral de 0.20 marcaría al 54% de la población — inútil como alerta.
# Comparando el percentil que cada señal le asigna al mismo suministro, un desacuerdo grande
# significa que una señal lo considera de los más sospechosos y la otra de los más normales,
# independientemente de la escala de cada una.
DISCREPANCIA_UMBRAL = 0.40
pct_firma = df_ali['firma_fraude'].rank(pct=True)
pct_anomaly = df_ali['anomaly_score'].rank(pct=True)
discrepancia_senales = (pct_firma - pct_anomaly).abs().values

discrepancia_llm_knn = np.where(
    np.isnan(llm_proba_fraude), 0.0, np.abs(llm_proba_fraude - knn_proba_fraude)
)
df_ali['alta_incertidumbre'] = (
    (discrepancia_senales > DISCREPANCIA_UMBRAL) |
    (mask_ambigua.values & (discrepancia_llm_knn > 0.20))
)
print(f"Casos marcados con alta incertidumbre (señales discrepantes > {DISCREPANCIA_UMBRAL:.0%}): "
      f"{int(df_ali['alta_incertidumbre'].sum())} de {len(df_ali)}")

# Probabilidad final: probabilidad_fraude, ajustada por el doble voto solo en zona ambigua
df_ali['probabilidad_final'] = df_ali['probabilidad_fraude']
df_ali.loc[mask_ambigua, 'probabilidad_final'] = (
    df_ali.loc[mask_ambigua, 'probabilidad_fraude'] * 0.50 +
    prob_llm_o_knn[mask_ambigua] * 0.50
)

# -------------------------------------------------------------
# PASO 7: FÓRMULA DE PRIORIDAD ECONÓMICA Y RANKING CON CORTE DINÁMICO
# -------------------------------------------------------------
print("\n--- PASO 7: FÓRMULA ECONÓMICA Y RANKING FINAL ---")

# Prioridad = Probabilidad_fraude * Recupero_soles  (PLAN.md Paso 2.7)
#
# El bono operativo 1.25x para CLANDESTINA que existía aquí se ELIMINÓ: era doble conteo.
# La mayor rentabilidad de una conexión clandestina (mediana real 10,330 kWh contra 3,417 de
# MANIPULACIÓN) ya está en el recupero estimado. Multiplicar otra vez por el tipo hacía que
# el top 100 saliera 100% CLANDESTINA aunque en la población el tipo predicho es 83%
# MANIPULACIÓN, dejando fuera del despacho los casos de manipulación de alto recupero.
df_ali['prioridad'] = df_ali['probabilidad_final'] * df_ali['recupero_p10_soles']

UMBRAL_ALTO = 0.65
UMBRAL_BAJO = 0.35
MARGEN = 0.15
ANOMALY_ALTO = 0.90  # anomaly_score fuertemente atípico, aunque probabilidad_fraude sea baja

# Zona ambigua: (a) probabilidad_fraude dentro de ± margen de la frontera de decisión
# [umbral_bajo, umbral_alto] — cerca de donde el modelo realmente duda —
# O (b) anomalía extrema con probabilidad_fraude baja.
mask_frontera = (
    ((df_ali['probabilidad_fraude'] >= UMBRAL_BAJO - MARGEN) & (df_ali['probabilidad_fraude'] <= UMBRAL_BAJO + MARGEN)) |
    ((df_ali['probabilidad_fraude'] >= UMBRAL_ALTO - MARGEN) & (df_ali['probabilidad_fraude'] <= UMBRAL_ALTO + MARGEN))
)
mask_anomalia_rara = (df_ali['anomaly_score'] >= ANOMALY_ALTO) & (df_ali['probabilidad_fraude'] < UMBRAL_BAJO)
mask_ambigua_cruda = mask_frontera | mask_anomalia_rara
n_ambiguos_crudo = int(mask_ambigua_cruda.sum())

# Tope operativo: el doble voto (LLM + KNN) del Paso 2.6 no es viable sobre miles de
# casos. Si la zona ambigua cruda excede el tope, nos quedamos solo con los ZONA_AMBIGUA_TOPE
# casos más cercanos al verdadero punto de indecisión (probabilidad_fraude = 0.50) dentro
# de esa zona — son los que el modelo más duda, no una muestra arbitraria del 69% de la
# población. El resto de los casos "ambiguos" pero lejos de 0.50 usan probabilidad_fraude
# tal cual, sin ajuste de doble voto.
ZONA_AMBIGUA_TOPE = int(os.environ.get('ZONA_AMBIGUA_TOPE', 300))
if n_ambiguos_crudo > ZONA_AMBIGUA_TOPE:
    distancia_a_050 = (df_ali['probabilidad_fraude'] - 0.50).abs()
    idxs_top_duda = distancia_a_050[mask_ambigua_cruda].nsmallest(ZONA_AMBIGUA_TOPE).index
    mask_ambigua = pd.Series(False, index=df_ali.index)
    mask_ambigua.loc[idxs_top_duda] = True
    print(f"Zona ambigua cruda: {n_ambiguos_crudo} casos, acotada a los {ZONA_AMBIGUA_TOPE} más cercanos a la frontera de indecisión (0.50).")
else:
    mask_ambigua = mask_ambigua_cruda

print(f"Casos en zona de ambigüedad/incertidumbre: {mask_ambigua.sum()} de 14,951")

# KNN sobre el histórico para segunda opinión (Voto 2 del Paso 2.6) — vota sobre TIPO,
# igual que B1, por la misma limitación estructural del histórico.
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
knn.fit(X_hist, y_type)
knn_proba_cland = knn.predict_proba(X_ali)[:, 1]

# Voto 1 del Paso 2.6: LLM (Gemini). Solo se llama para casos en zona ambigua (control de costo).
# Se le pide al LLM una probabilidad de FRAUDE (no de tipo específico) para que sea
# compatible con probabilidad_fraude, que es lo que realmente ajusta.
llm_proba_fraude = resolver_zona_ambigua_llm(df_ali, feat_ali, mask_ambigua, perfiles_giro, norm_cols)

# KNN vota tipo (prob. de ser CLANDESTINA); se usa como proxy de "sospecha" en la zona
# ambigua: un caso donde KNN también se inclina fuerte hacia un tipo específico refuerza
# la sospecha de fraude, independientemente de cuál tipo sea.
knn_proba_fraude = np.where(knn_proba_cland >= 0.5, knn_proba_cland, 1.0 - knn_proba_cland)

# Regla de combinación: si LLM no respondió ("sin referencia"), usar solo KNN.
prob_llm_o_knn = np.where(np.isnan(llm_proba_fraude), knn_proba_fraude,
                           (llm_proba_fraude + knn_proba_fraude) / 2.0)

# Bandera de alta incertidumbre.
#
# Defecto corregido: antes se calculaba solo como `abs(prob_llm - prob_knn) > 0.20`. Cuando
# no hay GEMINI_API_KEY configurada, `prob_llm` es NaN para todos los casos, y en NumPy
# `NaN > 0.20` es False — así que la bandera salía False para los 14,951 suministros, sin
# excepción. La advertencia nunca llegaba al inspector y la ruta de código estaba muerta en
# la práctica, que es el estado normal de ejecución (el free tier solo cubre ~15 llamadas).
#
# Ahora la discrepancia se mide entre las dos señales independientes que SIEMPRE existen:
# `firma_fraude` (similitud al perfil de fraude confirmado del histórico) y `anomaly_score`
# (rareza frente a los pares de la misma cohorte de potencia). Cuando una señal dice
# "sospechoso" y la otra dice "normal", el caso es genuinamente dudoso y el inspector merece
# saberlo. Si además el LLM respondió, su discrepancia con KNN se suma como segunda fuente.
# La discrepancia se mide en espacio de PERCENTILES, no como diferencia absoluta: las dos
# señales viven en escalas distintas (la firma es una suma ponderada acotada, el
# anomaly_score es una sigmoide del Isolation Forest), así que su diferencia bruta tiene una
# mediana de 0.214 y un umbral de 0.20 marcaría al 54% de la población — inútil como alerta.
# Comparando el percentil que cada señal le asigna al mismo suministro, un desacuerdo grande
# significa que una señal lo considera de los más sospechosos y la otra de los más normales,
# independientemente de la escala de cada una.
DISCREPANCIA_UMBRAL = 0.40
pct_firma = df_ali['firma_fraude'].rank(pct=True)
pct_anomaly = df_ali['anomaly_score'].rank(pct=True)
discrepancia_senales = (pct_firma - pct_anomaly).abs().values

discrepancia_llm_knn = np.where(
    np.isnan(llm_proba_fraude), 0.0, np.abs(llm_proba_fraude - knn_proba_fraude)
)
df_ali['alta_incertidumbre'] = (
    (discrepancia_senales > DISCREPANCIA_UMBRAL) |
    (mask_ambigua.values & (discrepancia_llm_knn > 0.20))
)
print(f"Casos marcados con alta incertidumbre (señales discrepantes > {DISCREPANCIA_UMBRAL:.0%}): "
      f"{int(df_ali['alta_incertidumbre'].sum())} de {len(df_ali)}")

# Probabilidad final: probabilidad_fraude, ajustada por el doble voto solo en zona ambigua
df_ali['probabilidad_final'] = df_ali['probabilidad_fraude']
df_ali.loc[mask_ambigua, 'probabilidad_final'] = (
    df_ali.loc[mask_ambigua, 'probabilidad_fraude'] * 0.50 +
    prob_llm_o_knn[mask_ambigua] * 0.50
)

# -------------------------------------------------------------
# PASO 7: FÓRMULA DE PRIORIDAD ECONÓMICA Y RANKING CON CORTE DINÁMICO
# -------------------------------------------------------------
print("\n--- PASO 7: FÓRMULA ECONÓMICA Y RANKING FINAL ---")

# Prioridad = Probabilidad_fraude * Recupero_soles  (PLAN.md Paso 2.7)
#
# El bono operativo 1.25x para CLANDESTINA que existía aquí se ELIMINÓ: era doble conteo.
# La mayor rentabilidad de una conexión clandestina (mediana real 10,330 kWh contra 3,417 de
# MANIPULACIÓN) ya está en el recupero estimado. Multiplicar otra vez por el tipo hacía que
# el top 100 saliera 100% CLANDESTINA aunque en la población el tipo predicho es 83%
# MANIPULACIÓN, dejando fuera del despacho los casos de manipulación de alto recupero.
df_ali['prioridad'] = df_ali['probabilidad_final'] * df_ali['recupero_p10_soles']

# Nivel de sospecha: en qué quintil de los 4,659 fraudes CONFIRMADOS cae este suministro.
# Se clasifica sobre `firma_fraude` y NO sobre `probabilidad_final`, porque la firma es la
# única magnitud calculada igual en ambos datasets — `probabilidad_final` mezcla el
# anomaly_score, que no tiene equivalente en el histórico (el Isolation Forest solo se
# entrena sobre el alimentador), así que compararla contra cuantiles del histórico
# desplazaría las bandas.
#
# Nomenclatura: solo el quintil por debajo del 20% de los fraudes se llama BAJO. Los tramos
# intermedios se nombran MODERADO/ALTO/MUY ALTO y no "MUY BAJO/BAJO", porque un suministro
# que puntúa al nivel del percentil 30 de los hurtos reales NO es poco sospechoso: está
# dentro del rango donde viven los fraudes verificados, solo que en su parte baja.
NIVELES_SOSPECHA = ['BAJO', 'MODERADO', 'ALTO', 'MUY ALTO', 'CRÍTICO']
df_ali['nivel_sospecha'] = pd.Series(
    np.digitize(df_ali['firma_fraude'].values, QUINTILES_FRAUDE),
    index=df_ali.index
).map(lambda i: NIVELES_SOSPECHA[i])

# Los suministros sin servicio activo no son sospechosos: su firma puede ser alta por
# consumo errático, pero no hay energía que recuperar (ver exclusiones más abajo).
df_ali.loc[df_ali['sin_servicio'], 'nivel_sospecha'] = 'SIN SERVICIO'

# Exclusiones del ranking operativo. Ambos grupos siguen presentes en ranking_completo.csv
# (trazabilidad y revisión de campo) pero nunca compiten por el presupuesto de inspecciones:
#
#  - sin_servicio: consumo anual ~nulo. No es hurto sino medidor de baja / local vacío;
#    mandar una cuadrilla no recupera nada. Era la principal fuente de contaminación del
#    top del ranking anterior.
#  - datos_insuficientes: YA NO SE EXCLUYE. Medido en el backtest, los filtros operativos
#    se llevaban 2.15 de cada 31 hurtos inyectados ANTES de rankear — un 7% de falsos
#    negativos autoinfligidos, en un reto donde el falso negativo es el error caro. Y el
#    motivo era el equivocado: la colisión de hoja es una limitación del REGRESOR, no una
#    propiedad del suministro. Que el modelo no sepa estimar cuánto se recupera de un
#    cliente no es razón para no ir a inspeccionarlo. Ahora se marca la incertidumbre en
#    la columna correspondiente y el suministro sigue compitiendo por el presupuesto, que
#    es lo que corresponde: la decisión de no visitarlo es del jefe de campo, no del árbol.
df_ali.loc[df_ali['sin_servicio'], 'prioridad'] = -1.0

# Ordenar de mayor a menor
df_ali.sort_values(by='prioridad', ascending=False, inplace=True)
df_ali.reset_index(drop=True, inplace=True)
df_ali['ranking_posicion'] = df_ali.index + 1

# Corte del ranking. PLAN.md (Paso 2.8) proponía un corte dinámico buscando el salto más
# grande entre scores consecutivos, pero se descartó tras medirlo: la curva de prioridad es
# prácticamente continua (2160 → 2155 → 2134 → 2127 → 2082...) y el "salto máximo" resulta
# ser de ~S/ 45 sobre valores de ~S/ 2,100 — un 2%, indistinguible del ruido. Peor aún, el
# máximo caía sistemáticamente en el borde inferior del rango permitido, así que el corte
# saltaba entre corridas (#78, #31, #15) sin ninguna razón de negocio detrás.
#
# TAMPOCO se usa un corte por significancia estadística (FDR, razón de verosimilitud):
# se midió y es inalcanzable con esta prevalencia. Con 31 hurtos entre 14,951 suministros
# el FDR mínimo alcanzable es ~96%, y el criterio LR>=1 marca ~6,300 suministros. El
# detalle del cálculo está en corte_autonomo.py.
#
# El corte sale de la máxima concentración de evidencia: el umbral de firma donde la
# razón P(firma>=u | hurto confirmado) / P(firma>=u | alimentador) es máxima. Es el punto
# donde la cuadrilla acierta más veces por viaje. No tiene parámetros de negocio, es
# invariante a la prevalencia asumida, y N cambia si cambia la data — que es justamente lo
# que un corte fijo no hacía.
firma_ali_ordenada = df_ali['firma_fraude'].values
corte = corte_autonomo(firma_ali_ordenada, firma_hist, n_boot=400)
corte_pos = min(corte['N'], len(df_ali))

# Cota superior de acierto que no depende del modelo, para que el entregable no se lea como
# una promesa imposible: marcando N suministros con 31 hurtos presentes, ni un oráculo
# perfecto supera min(31, N)/N.
techo = techo_precision(corte_pos, 31)
print(f"Techo teórico de precisión para N={corte_pos} (oráculo perfecto): {techo*100:.1f}%")

# Contraste económico. NO fija el corte: depende del costo real de cuadrilla y sobre todo
# de la prevalencia, y los 31 hurtos del reto son los CONFIRMADOS por el jurado, no todos
# los que hay. Se reporta para que el jefe de campo vea a partir de qué costo por
# inspección el corte deja de tener sentido económico.
posterior_fraude, lr_fraude = calibrar_posterior(firma_ali_ordenada, firma_hist, 31 / len(df_ali))
df_ali['posterior_fraude'] = posterior_fraude
print(f"Calibración: suma de posteriors = {posterior_fraude.sum():.1f} "
      f"(consistente si da ~31, los hurtos conocidos del alimentador)")
tabla_roi = sensibilidad_roi(
    df_ali.loc[df_ali['prioridad'] >= 0, 'posterior_fraude'].values,
    df_ali.loc[df_ali['prioridad'] >= 0, 'recupero_p10_soles'].values,
)

sospechosos_top = df_ali.head(corte_pos).copy()

print(f"Top {corte_pos} sospechosos seleccionados:")
print(f"Recupero económico total proyectado en soles: S/ {sospechosos_top['recupero_p10_soles'].sum():,.2f}")
print(f"Distribución por tipo de hurto:")
print(sospechosos_top['tipo_predicho'].value_counts())

# -------------------------------------------------------------
# GENERAR ENTREGABLES FASE 1
# -------------------------------------------------------------
print("\n--- GENERANDO ENTREGABLES ---")
os.makedirs('entregables', exist_ok=True)

# Entregable 1: Excel con códigos CUENTA_ID de sospechosos
excel_path = 'entregables/sospechosos.xlsx'
sospechosos_export = sospechosos_top[[
    'ranking_posicion', 'CUENTA_ID', 'SED_ID', 'LLAVE_ID', 'MACRO_GIRO',
    'tipo_predicho', 'nivel_sospecha', 'recupero_p10_kwh',
    'recupero_p10_soles', 'prioridad', 'alta_incertidumbre'
]].copy()

# Se reporta un NIVEL y no un número: con 31 hurtos entre 14,951 suministros, hasta un
# modelo perfecto acertaría como máximo min(31, N)/N (ver `techo_precision`). Un valor como
# "55" se lee inevitablemente como "55% de acierto" y prometería muy por encima del techo
# teórico del problema.
sospechosos_export.columns = [
    'POSICIÓN', 'CUENTA_ID', 'SED_ID', 'LLAVE_ID', 'COHORTE_POTENCIA',
    'TIPO_PREDICHO', 'NIVEL_SOSPECHA', 'RECUPERO_P25_KWH',
    'RECUPERO_P25_SOLES', 'SCORE_PRIORIDAD', 'ALTA_INCERTIDUMBRE'
]
sospechosos_export['ACCIÓN_CUADRILLA'] = np.where(
    sospechosos_export['TIPO_PREDICHO'] == 'CLANDESTINA',
    'REVISAR RED / ACOMETIDA', 'REVISAR MEDIDOR'
)

sospechosos_export.to_excel(excel_path, index=False)
print(f"Entregable 1 generado exitosamente: {excel_path}")

# Exportar ranking completo en CSV
ranking_csv = 'entregables/ranking_completo.csv'
df_ali[[
    'ranking_posicion', 'CUENTA_ID', 'SED_ID', 'LLAVE_ID', 'MACRO_GIRO',
    'tipo_predicho', 'probabilidad_final', 'posterior_fraude', 'recupero_p10_kwh',
    'recupero_p10_soles', 'prioridad', 'nivel_sospecha', 'alta_incertidumbre',
    'datos_insuficientes', 'sin_servicio', 'caida_pct', 'factor_uso', 'desv_sed',
    'anomaly_score', 'firma_fraude', 'señal_vecindario', 'caida_vs_sed',
    'desacople_sed', 'cons_anual', 'ratio_meses_activos',
    'POTENCIA_CONTRATADA', 'meses_cero_finales', 'meses_activos', 'kwhd_promedio'
]].to_csv(ranking_csv, index=False)
print(f"Ranking completo de 14,951 suministros generado: {ranking_csv}")

# Exportar datos para el dashboard
os.makedirs('dashboard_data', exist_ok=True)
top_dashboard = df_ali.head(max(200, corte_pos)).to_dict(orient='records')
with open('dashboard_data/top_ranking.json', 'w', encoding='utf-8') as f:
    json.dump(top_dashboard, f, ensure_ascii=False)

# Resumen por SED para el mapa de calor
sed_summary = df_ali.head(corte_pos).groupby('SED_ID').agg(
    total_sospechosos=('CUENTA_ID', 'count'),
    recupero_soles=('recupero_p10_soles', 'sum'),
    prioridad_total=('prioridad', 'sum')
).reset_index().sort_values(by='recupero_soles', ascending=False)
sed_summary.to_json('dashboard_data/sed_heatmap.json', orient='records')

print("Datos para el dashboard exportados a dashboard_data/")
print("\n=== PIPELINE DE EJECUCIÓN COMPLETADO CON ÉXITO ===")
