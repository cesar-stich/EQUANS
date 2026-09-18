"""
FEATURES AVANZADAS DE FORMA TEMPORAL

Motivación medida: la firma original quedaba sostenida en dos features (kwhd_cv con
peso 0.604 y factor_uso con 0.343, el 94.7% del total), y `factor_uso` es sospechosa
de medir la zona y no el hurto — la potencia mediana contratada es 5.1 kW en el
histórico contra 11.1 kW en el alimentador. El backtest lo confirma: al descontar el
desplazamiento de medianas entre zonas, el recall cae 3x.

QUÉ FEATURES SON LEGÍTIMAMENTE COMPARABLES ENTRE ZONAS
------------------------------------------------------
El histórico y el alimentador son zonas distintas, y el histórico es 100% positivo:
no tiene clientes honestos de su propia zona. Eso limita qué se puede comparar:

  - Features INTRA-SUMINISTRO y adimensionales (forma de la curva de 12 meses):
    comparables. Un escalón del 60% es un escalón del 60% en cualquier zona.
    Aquí vive la firma del hurto y es donde se concentra este módulo.

  - Features INTRA-SED (desviación contra vecinos): NO comparables. En el histórico
    los "vecinos" son también fraudes. Por eso el ajuste original le da peso 0.003 a
    `desv_sed`: no porque no sirva, sino porque no se puede medir contra el histórico.

  - Features de NIVEL (consumo absoluto, potencia): NO comparables. Son el vector
    directo del artefacto de zona.

La física del hurto que se busca capturar: un cliente consumía de forma estable,
en algún momento interviene el medidor, y a partir de ahí registra menos y de forma
más errática. Eso deja un ESCALÓN con fecha, no una caída difusa — y la feature
original `caida_pct` (promedio de los 3 primeros meses contra los 3 últimos) no
puede verlo si el quiebre ocurre en el medio del año.
"""

import numpy as np
import pandas as pd

import equans_core as ec

# Features nuevas, todas adimensionales (ratios, fracciones o correlaciones).
FIRMA_COLS_AVANZADAS = [
    'escalon_mag', 'escalon_mes', 'estabilidad_pre', 'volatilidad_post',
    'ratio_min_med', 'autocorr1', 'rugosidad', 'meses_planos',
    'ceros_intercalados', 'slope_norm', 'ratio_var_mitades',
    'frac_acumulados', 'dias_cv', 'caida_consecutiva_max', 'frac_muy_bajo',
    'conexionado_mono',
]


def _detectar_escalon(serie):
    """Punto de quiebre por segmentación binaria: el corte que más reduce la suma de
    cuadrados intra-segmento. Devuelve (magnitud relativa, posición relativa,
    cv antes, cv después).

    Magnitud = (media_antes - media_después) / media_antes. Positiva cuando el
    consumo BAJA, que es el sentido del hurto. Es un ratio, así que no arrastra el
    nivel de consumo de la zona.
    """
    v = serie[~np.isnan(serie)]
    n = len(v)
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0

    mejor_sce, mejor_k = np.inf, None
    for k in range(2, n - 1):
        izq, der = v[:k], v[k:]
        sce = izq.var() * len(izq) + der.var() * len(der)
        if sce < mejor_sce:
            mejor_sce, mejor_k = sce, k

    if mejor_k is None:
        return 0.0, 0.0, 0.0, 0.0

    pre, post = v[:mejor_k], v[mejor_k:]
    m_pre, m_post = pre.mean(), post.mean()
    mag = (m_pre - m_post) / m_pre if m_pre > 1e-9 else 0.0
    cv_pre = pre.std() / (m_pre + 1e-6) if m_pre > 1e-9 else 0.0
    cv_post = post.std() / (m_post + 1e-6) if m_post > 1e-9 else 0.0

    # Estabilidad previa: un cliente que era REGULAR y cayó de golpe es mucho más
    # sospechoso que uno que ya venía errático. Se codifica como 1/(1+cv) para que
    # valores altos signifiquen "venía estable".
    return float(np.clip(mag, -3.0, 1.0)), mejor_k / n, float(1.0 / (1.0 + cv_pre)), float(cv_post)


def extract_features_avanzadas(df, incluir_base=True, is_alimentador=None):
    """Features base (F1-F5) + forma temporal + vecindario SED. Índice RangeIndex 0..n-1.

    `is_alimentador` se acepta por compatibilidad con la firma de ec.extract_features
    y no se usa: ninguna feature cambia de definición según el dataset, que es
    justamente lo que las hace comparables entre zonas.
    """
    feat, kwhd = ec.extract_features(df) if incluir_base else (pd.DataFrame(index=df.index), None)
    if kwhd is None:
        _, kwhd = ec.extract_features(df)

    n = len(df)
    dias_mat = np.column_stack([df[f'DIAS_{m}'].values for m in ec.MESES]).astype(float)

    escalon_mag = np.zeros(n)
    escalon_mes = np.zeros(n)
    estabilidad_pre = np.zeros(n)
    volatilidad_post = np.zeros(n)
    ratio_min_med = np.zeros(n)
    autocorr1 = np.zeros(n)
    rugosidad = np.zeros(n)
    meses_planos = np.zeros(n)
    ceros_inter = np.zeros(n)
    slope_norm = np.zeros(n)
    ratio_var_mit = np.zeros(n)
    caida_cons_max = np.zeros(n)
    frac_muy_bajo = np.zeros(n)

    for i in range(n):
        fila = kwhd[i]
        v = fila[~np.isnan(fila)]
        if len(v) == 0:
            continue

        escalon_mag[i], escalon_mes[i], estabilidad_pre[i], volatilidad_post[i] = \
            _detectar_escalon(fila)

        med = np.median(v)
        mu = v.mean()
        if med > 1e-9:
            ratio_min_med[i] = v.min() / med
            # Meses hundidos a menos de la mitad de lo habitual del propio cliente:
            # captura el hurto intermitente, que la caída global promedia y borra.
            frac_muy_bajo[i] = np.mean(v < 0.5 * med)

        if len(v) >= 3 and v.std() > 1e-9:
            vc = v - mu
            denom = np.sum(vc * vc)
            if denom > 1e-12:
                autocorr1[i] = float(np.sum(vc[:-1] * vc[1:]) / denom)
            d = np.diff(v)
            # Razón de Von Neumann: mide cuán "rugosa" es la serie frente a su propia
            # varianza. Un medidor intervenido salta; uno honesto es suave.
            rugosidad[i] = float(np.sum(d * d) / (2.0 * denom / len(v) * len(v) + 1e-12))
            x = np.arange(len(v), dtype=float)
            slope = np.polyfit(x, v, 1)[0]
            slope_norm[i] = float(slope / (mu + 1e-6))
            mitad = len(v) // 2
            if mitad >= 2:
                v1, v2 = v[:mitad], v[mitad:]
                ratio_var_mit[i] = float((v2.var() + 1e-9) / (v1.var() + 1e-9))
            # Mayor caída relativa de un mes al siguiente.
            caidas = np.where(v[:-1] > 1e-9, (v[:-1] - v[1:]) / v[:-1], 0.0)
            caida_cons_max[i] = float(np.clip(caidas.max(), -3.0, 1.0)) if len(caidas) else 0.0

        if len(v) >= 2:
            # Lecturas idénticas consecutivas: medidor detenido o lectura estimada.
            meses_planos[i] = float(np.mean(np.abs(np.diff(v)) < 1e-9))

        # Ceros en medio del año (no la baja definitiva, que ya mide meses_cero_finales).
        idx_pos = np.flatnonzero(v > 0)
        if len(idx_pos) >= 2:
            ceros_inter[i] = float(np.sum(v[idx_pos[0]:idx_pos[-1] + 1] <= 0) / len(v))

    feat['escalon_mag'] = escalon_mag
    feat['escalon_mes'] = escalon_mes
    feat['estabilidad_pre'] = estabilidad_pre
    feat['volatilidad_post'] = volatilidad_post
    feat['ratio_min_med'] = ratio_min_med
    feat['autocorr1'] = autocorr1
    feat['rugosidad'] = np.clip(rugosidad, 0, 10)
    feat['meses_planos'] = meses_planos
    feat['ceros_intercalados'] = ceros_inter
    feat['slope_norm'] = np.clip(slope_norm, -2, 2)
    feat['ratio_var_mitades'] = np.clip(ratio_var_mit, 0, 20)
    feat['caida_consecutiva_max'] = caida_cons_max
    feat['frac_muy_bajo'] = frac_muy_bajo

    # Periodos facturados: valores de 33 a 400 días son acumulados LEGÍTIMOS según el
    # diccionario de EQUANS, así que no se limpian. Pero su FRECUENCIA sí es señal: un
    # medidor al que no se le puede tomar lectura con regularidad es un medidor de
    # acceso difícil, y el acceso difícil es precondición de la manipulación sostenida.
    with np.errstate(invalid='ignore'):
        validos = ~np.isnan(dias_mat)
        n_val = validos.sum(axis=1)
        feat['frac_acumulados'] = np.where(n_val > 0,
                                           np.nansum(dias_mat > 33, axis=1) / np.maximum(n_val, 1), 0.0)
        med_d = np.nanmedian(dias_mat, axis=1)
        std_d = np.nanstd(dias_mat, axis=1)
        feat['dias_cv'] = np.nan_to_num(np.where(med_d > 0, std_d / med_d, 0.0), nan=0.0)

    # Tipo de conexionado: presente y comparable en ambos datasets. Medido sobre los
    # 4,659 casos confirmados, 'M' concentra 81.9% de los fraudes contra 68.6% de la
    # población del alimentador (lift 1.19x) y 'T' va al revés (18.1% vs 31.4%).
    if 'TIPO_CONEXIONADO' in df.columns:
        conx = df['TIPO_CONEXIONADO'].astype(str).str.upper().str.strip()
        feat['conexionado_mono'] = (conx == 'M').astype(float).values
    else:
        feat['conexionado_mono'] = 0.0

    feat = features_vecindario_sed(df, feat, kwhd)
    return feat.fillna(0.0), kwhd


# -------------------------------------------------------------
# SEÑAL INTRA-ALIMENTADOR: desviación temporal contra la propia SED
# -------------------------------------------------------------
# Esta es la única familia de señales INMUNE al problema de zona, porque no usa el
# histórico para nada: compara a cada suministro contra sus vecinos físicos, los que
# cuelgan del mismo transformador. Es además el nivel de análisis que el propio reto
# señala como el más útil.
#
# La idea: si en julio TODA la SED baja el consumo, eso es estacionalidad o un evento
# del barrio, no hurto. Lo sospechoso es el suministro que se desacopla de sus
# vecinos. Al restar la curva mediana de la SED se controlan de una vez la
# estacionalidad, el clima y cualquier evento local — sin necesidad de modelarlos.
#
# `caida_vs_sed` es la versión de esto que más directamente apunta al hurto: la caída
# del cliente MENOS la caída mediana de su SED. Un cliente que cae 60% en una SED que
# cayó 55% no tiene nada de raro; uno que cae 60% donde el resto subió 5%, sí.

def features_vecindario_sed(df, feat, kwhd):
    """Desviación del patrón TEMPORAL respecto de los vecinos de la misma SED.

    Requiere una SED con vecinos suficientes para que la mediana signifique algo; por
    debajo de MIN_VECINOS la señal se anula en vez de inventarse un valor.
    """
    MIN_VECINOS = 8

    n = len(df)
    sed = pd.Series(df['SED_ID'].astype(str).values)

    # Curva de forma: consumo de cada mes dividido por el promedio del propio cliente.
    # Al normalizar por su propio nivel, un cliente grande y uno chico de la misma SED
    # se vuelven comparables y solo queda la FORMA de su año.
    with np.errstate(invalid='ignore', divide='ignore'):
        nivel = np.nanmean(kwhd, axis=1)
        forma = kwhd / np.where(nivel[:, None] > 1e-9, nivel[:, None], np.nan)

    desacople = np.zeros(n)
    caida_vs_sed = np.zeros(n)
    meses_bajo_sed = np.zeros(n)

    # Caída propia de cada cliente (primer tercio contra último tercio del año).
    with np.errstate(invalid='ignore'):
        base = np.nanmean(kwhd[:, :4], axis=1)
        fin = np.nanmean(kwhd[:, -4:], axis=1)
    caida = np.where(base > 1e-9, (base - fin) / base, np.nan)

    for _, idx in pd.DataFrame({'s': sed}).groupby('s', sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < MIN_VECINOS:
            continue

        sub = forma[idx]
        with np.errstate(invalid='ignore'):
            # Mediana leave-one-out aproximada: con 8+ vecinos, el efecto de excluirse
            # a sí mismo sobre la mediana es despreciable y ahorra un orden de magnitud
            # de cómputo en el bootstrap.
            perfil_sed = np.nanmedian(sub, axis=0)
            resid = sub - perfil_sed[None, :]
            # Desacople: distancia media absoluta entre la forma del cliente y la de
            # su vecindario, en unidades de la propia forma.
            desacople[idx] = np.nan_to_num(np.nanmean(np.abs(resid), axis=1), nan=0.0)
            # Meses en que el cliente está claramente por debajo del patrón del barrio.
            meses_bajo_sed[idx] = np.nan_to_num(np.nanmean(resid < -0.35, axis=1), nan=0.0)

            caida_sed = np.nanmedian(caida[idx])
            caida_vs_sed[idx] = np.nan_to_num(caida[idx] - caida_sed, nan=0.0)

    feat['desacople_sed'] = np.clip(desacople, 0, 5)
    feat['caida_vs_sed'] = np.clip(caida_vs_sed, -2, 2)
    feat['meses_bajo_sed'] = meses_bajo_sed
    return feat


FEATURES_SED = ['desacople_sed', 'caida_vs_sed', 'meses_bajo_sed']
