"""
NÚCLEO COMPARTIDO DEL PIPELINE EQUANS

Contiene las piezas sin efectos secundarios que necesitan tanto el pipeline de
producción (`run_pipeline.py`) como los módulos de corte autónomo y validación
(`corte_autonomo.py`, `backtest_recall.py`).

Se extrajo a un módulo aparte para que la validación mida EXACTAMENTE el mismo
instrumento que genera el entregable. Si el backtest reimplementara las features
por su cuenta, validaría un modelo distinto al que se entrega y el número de
recall no significaría nada.
"""

import json
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# CONSTANTES
# -------------------------------------------------------------
MESES = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
         'JULIO', 'AGOSTO', 'SETIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
TARIFA_REF_SOLES = 0.667

# Features comunes para ML (cohortes del Isolation Forest)
FEATURE_COLS = ['kwhd_promedio', 'kwhd_cv', 'caida_pct', 'alternancia',
                'factor_uso', 'desv_sed', 'desv_llave']

# Features de firma de fraude para el modelo de similitud.
#
# CRÍTICO — solo features de FORMA, nunca de NIVEL absoluto: el histórico y el
# alimentador son zonas distintas de la concesión con niveles de consumo
# sistemáticamente diferentes (consumo anual mediano 1,814 kWh en el histórico
# contra 823 en el alimentador; potencia mediana 5.1 kW contra 11.1 kW). Si se le
# dan variables de nivel (`cons_anual`, `kwhd_promedio`, potencia), el modelo
# aprende a distinguir "zona A vs zona B" — un artefacto geográfico — en vez de
# "fraude vs honesto".
FIRMA_COLS = [
    'kwhd_cv', 'caida_pct', 'alternancia', 'factor_uso',
    'desv_sed', 'desv_llave', 'ratio_meses_activos', 'meses_cero_finales'
]


# -------------------------------------------------------------
# CARGA Y LIMPIEZA (Algoritmo 1)
# -------------------------------------------------------------
def normalize_cols(df):
    new_cols = {}
    for c in df.columns:
        clean = c.replace('\n', '_').replace(' ', '_').strip()
        new_cols[c] = clean
    return df.rename(columns=new_cols)


def find_col(df, keyword, df_name, verbose=True):
    matches = [c for c in df.columns if keyword in c]
    if not matches:
        raise KeyError(f"No se encontró ninguna columna con '{keyword}' en {df_name}. "
                       f"Columnas disponibles: {list(df.columns)}")
    if len(matches) > 1 and verbose:
        print(f"AVISO: múltiples columnas con '{keyword}' en {df_name}: {matches}. "
              f"Usando la primera: {matches[0]}")
    return matches[0]


def cargar_datasets(cache_dir='data_cache', verbose=True):
    """Algoritmo 1: carga desde caché + limpieza mínima. No toca señales de fraude."""
    with open(f'{cache_dir}/alimentador_raw.json', 'r', encoding='utf-8') as f:
        df_ali = pd.DataFrame(json.load(f))
    with open(f'{cache_dir}/historico_raw.json', 'r', encoding='utf-8') as f:
        df_hist = pd.DataFrame(json.load(f))

    if verbose:
        print(f"Cargados: Alimentador={len(df_ali)} suministros, "
              f"Histórico CNR={len(df_hist)} suministros")

    df_ali = normalize_cols(df_ali)
    df_hist = normalize_cols(df_hist)

    # Renombrar pares de meses a formato estándar DIAS_[MES] y CONS_[MES]
    for m in MESES:
        for df in (df_ali, df_hist):
            d_col = [c for c in df.columns if 'DIAS' in c and m in c]
            c_col = [c for c in df.columns if 'CONSUMO' in c and m in c]
            if d_col:
                df.rename(columns={d_col[0]: f'DIAS_{m}'}, inplace=True)
            if c_col:
                df.rename(columns={c_col[0]: f'CONS_{m}'}, inplace=True)

    for m in MESES:
        for df in (df_ali, df_hist):
            df[f'DIAS_{m}'] = pd.to_numeric(df[f'DIAS_{m}'], errors='coerce')
            df[f'CONS_{m}'] = pd.to_numeric(df[f'CONS_{m}'], errors='coerce')

    # Regla de física: si DIAS > 400 marcar como NaN ese par (dias + consumo).
    # Periodos entre 33 y 400 son acumulados legítimos según diccionario EQUANS.
    corruptos = {'ali': 0, 'hist': 0}
    for m in MESES:
        for key, df in (('ali', df_ali), ('hist', df_hist)):
            mask = df[f'DIAS_{m}'] > 400
            corruptos[key] += int(mask.sum())
            df.loc[mask, [f'DIAS_{m}', f'CONS_{m}']] = np.nan

    if verbose:
        print(f"Meses anulados por DIAS > 400: {corruptos['ali']} en alimentador, "
              f"{corruptos['hist']} en histórico")

    pot_col_ali = find_col(df_ali, 'POTENCIA', 'df_ali', verbose)
    pot_col_hist = find_col(df_hist, 'POTENCIA', 'df_hist', verbose)
    df_ali['POTENCIA_CONTRATADA'] = pd.to_numeric(df_ali[pot_col_ali], errors='coerce').round(2)
    df_hist['POTENCIA_CONTRATADA'] = pd.to_numeric(df_hist[pot_col_hist], errors='coerce').round(2)

    rec_col = find_col(df_hist, 'RECUPERO', 'df_hist', verbose)
    df_hist['RECUPERO_KWH_REAL'] = df_hist[rec_col].astype(str).str.replace(',', '').str.strip()
    df_hist['RECUPERO_KWH_REAL'] = pd.to_numeric(df_hist['RECUPERO_KWH_REAL'],
                                                 errors='coerce').fillna(0)

    return df_ali, df_hist


# -------------------------------------------------------------
# FEATURE ENGINEERING (F1 - F5)
# -------------------------------------------------------------
def extract_features(df, is_alimentador=True):
    """Requiere índice RangeIndex 0..n-1 (usa df.loc[idx, ...] posicionalmente)."""
    feat_df = pd.DataFrame(index=df.index)

    # F1: kWh/día por mes
    kwhd_matrix = np.full((len(df), len(MESES)), np.nan)

    for i, m in enumerate(MESES):
        dias = df[f'DIAS_{m}'].values
        cons = df[f'CONS_{m}'].values
        valid = (dias > 0) & (~np.isnan(dias)) & (~np.isnan(cons))
        kwhd_matrix[valid, i] = cons[valid] / dias[valid]

    with np.errstate(invalid='ignore'):
        kwhd_mean = np.nanmean(kwhd_matrix, axis=1)
        kwhd_std = np.nanstd(kwhd_matrix, axis=1)
    kwhd_mean[np.isnan(kwhd_mean)] = 0.0
    kwhd_std[np.isnan(kwhd_std)] = 0.0

    feat_df['kwhd_promedio'] = kwhd_mean
    feat_df['kwhd_cv'] = np.where(kwhd_mean > 0, kwhd_std / (kwhd_mean + 1e-4), 0)

    # F2: Caída porcentual (primeros 3 meses válidos vs últimos 3 meses válidos)
    with np.errstate(invalid='ignore'):
        base_3m = np.nanmean(kwhd_matrix[:, :3], axis=1)
        rec_3m = np.nanmean(kwhd_matrix[:, -3:], axis=1)
    caida_pct = np.where(base_3m > 0, (base_3m - rec_3m) / base_3m, 0.0)
    caida_pct = np.nan_to_num(caida_pct, nan=0.0)
    feat_df['caida_pct'] = np.clip(caida_pct, -2.0, 1.0)

    # F2b: Consumo anual total y actividad del medidor. Distingue el hurto (el
    # cliente sigue consumiendo, solo registra menos) del medidor inactivo.
    cons_matrix = np.full((len(df), len(MESES)), np.nan)
    for i, m in enumerate(MESES):
        cons_matrix[:, i] = df[f'CONS_{m}'].values
    feat_df['cons_anual'] = np.nan_to_num(np.nansum(cons_matrix, axis=1), nan=0.0)

    meses_validos = np.sum(~np.isnan(kwhd_matrix), axis=1)
    meses_con_consumo = np.nansum(kwhd_matrix > 0, axis=1)
    feat_df['meses_activos'] = meses_con_consumo
    feat_df['ratio_meses_activos'] = np.where(meses_validos > 0,
                                              meses_con_consumo / meses_validos, 0.0)

    # Meses consecutivos en cero al final del año: separa "dejó de consumir y nunca
    # volvió" (probable baja/medidor muerto) de un patrón intermitente.
    ceros_finales = []
    for row in kwhd_matrix:
        cnt = 0
        for v in row[::-1]:
            if np.isnan(v):
                continue
            if v <= 0:
                cnt += 1
            else:
                break
        ceros_finales.append(cnt)
    feat_df['meses_cero_finales'] = ceros_finales

    # F3: Alternancia de consumo
    alternancias = []
    for row in kwhd_matrix:
        valid_vals = row[~np.isnan(row)]
        alt_ratio = 0.0
        if len(valid_vals) >= 3:
            signs = np.sign(np.diff(valid_vals))
            signs = signs[signs != 0]
            if len(signs) >= 2:
                alt_ratio = np.sum(signs[:-1] != signs[1:]) / (len(signs) - 1)
        alternancias.append(alt_ratio)
    feat_df['alternancia'] = alternancias

    # F4: Factor de uso medio. Consumo máximo teórico mes = kW_contratado * 24h * días
    kw_c = df['POTENCIA_CONTRATADA'].fillna(3.0).values
    dias_mat = np.column_stack([df[f'DIAS_{m}'].values for m in MESES]).astype(float)
    cons_mat = np.column_stack([df[f'CONS_{m}'].values for m in MESES]).astype(float)
    pot = np.where((kw_c > 0) & np.isfinite(kw_c), kw_c, 3.0)
    max_teorico = pot[:, None] * 24.0 * dias_mat
    valido = (~np.isnan(dias_mat)) & (dias_mat > 0) & (~np.isnan(cons_mat)) & (cons_mat >= 0) \
        & (max_teorico > 0)
    ratios = np.where(valido, cons_mat / np.where(max_teorico > 0, max_teorico, 1.0), np.nan)
    with np.errstate(invalid='ignore'):
        factor_uso = np.nanmedian(ratios, axis=1)
    feat_df['factor_uso'] = np.nan_to_num(factor_uso, nan=0.0)

    # F5 / F5b: Desviación respecto a vecinos de la misma SED y de la misma LLAVE
    # (leave-one-out real + suavizado para grupos pequeños).
    llave_vals = df['LLAVE_ID'].values if 'LLAVE_ID' in df.columns else np.array(['LLAVE_UNK'] * len(df))
    feat_df['desv_sed'] = _desviacion_vecinos(df['SED_ID'].values, kwhd_mean)
    feat_df['desv_llave'] = _desviacion_vecinos(llave_vals, kwhd_mean)

    # Features estacionales (curva normalizada 12 meses)
    for i, m in enumerate(MESES):
        norm_m = np.where(kwhd_mean > 0, kwhd_matrix[:, i] / (kwhd_mean + 1e-4), 1.0)
        feat_df[f'norm_{m}'] = np.nan_to_num(norm_m, nan=1.0)

    return feat_df, kwhd_matrix


def _desviacion_vecinos(grupos, kwhd_mean):
    """Desviación del consumo frente a la mediana del grupo, leave-one-out y suavizada.

    Suavizado w = n/(n+10): en un grupo de 3 vecinos la mediana local es ruido, así
    que el peso se corre hacia la mediana global del alimentador.
    """
    kwhd_mean = np.asarray(kwhd_mean, dtype=float)
    n_total = len(kwhd_mean)
    global_median = float(np.median(kwhd_mean))
    desv_global = (kwhd_mean - global_median) / (global_median + 1e-3)

    # Un grupo nulo no es un grupo: cada suministro sin SED/LLAVE declarada queda
    # como singleton y se compara contra la mediana global, nunca contra los otros
    # suministros que también tienen el campo vacío.
    claves = pd.Series(grupos).astype('object')
    faltantes = claves.isna().values
    claves = claves.astype(str).values
    if faltantes.any():
        claves = claves.copy()
        claves[faltantes] = [f'__SIN_GRUPO_{i}__' for i in np.flatnonzero(faltantes)]

    out = desv_global.copy()

    df_g = pd.DataFrame({'g': claves, 'i': np.arange(n_total)})
    for _, sub in df_g.groupby('g', sort=False):
        idxs = sub['i'].values
        n = len(idxs)
        if n < 2:
            continue  # singleton: se queda con desv_global

        # Mediana leave-one-out en forma cerrada sobre el grupo ordenado: para cada
        # miembro, la mediana de los OTROS n-1. Evita recalcular n medianas por grupo,
        # que en el backtest (cientos de repeticiones) domina el tiempo de cómputo.
        orden_local = np.argsort(kwhd_mean[idxs], kind='stable')
        v = kwhd_mean[idxs][orden_local]
        p = np.arange(n)
        if n % 2 == 0:
            half = n // 2
            med_ord = np.where(p <= half - 1, v[half], v[half - 1])
        else:
            m = (n - 1) // 2
            lo = np.where(m - 1 < p, v[m - 1], v[m])
            hi = np.where(m < p, v[m], v[m + 1])
            med_ord = (lo + hi) / 2.0

        med = np.empty(n)
        med[orden_local] = med_ord

        k = kwhd_mean[idxs]
        desv_local = (k - med) / (med + 1e-3)
        w = n / (n + 10.0)
        out[idxs] = w * desv_local + (1.0 - w) * desv_global[idxs]

    return np.nan_to_num(out, nan=0.0)


# -------------------------------------------------------------
# COHORTES DE COMPARACIÓN
# -------------------------------------------------------------
def segmentar_por_potencia(pot_series):
    """Cohorte objetiva por potencia contratada (dato declarado, no inferido).

    Reemplaza a la inferencia de giro por curva estacional, que se midió en 31% de
    accuracy contra un baseline de clase mayoritaria de 90.7% (ver run_pipeline.py).
    """
    pot = pd.to_numeric(pot_series, errors='coerce').fillna(3.0)
    return pd.cut(
        pot,
        bins=[-np.inf, 4.0, 8.0, 13.0, np.inf],
        labels=['POT_BAJA', 'POT_MEDIA', 'POT_ALTA', 'POT_MUY_ALTA']
    ).astype(str)


# -------------------------------------------------------------
# FIRMA DE FRAUDE (similitud al perfil confirmado del histórico)
# -------------------------------------------------------------
class FirmaFraude:
    """Instrumento de medición: distancia al perfil de fraude confirmado.

    Se ajusta con dos referencias — el perfil mediano de los fraudes confirmados
    (histórico) y el perfil mediano de la población normal (alimentador) — y luego
    se aplica sin cambios a cualquier conjunto de features. Es un instrumento fijo,
    no un modelo que se reentrena por conjunto: eso es lo que permite comparar la
    firma del alimentador contra la del histórico en la misma escala.

    Cada feature aporta como máximo su peso (satura en el perfil de fraude), y los
    pesos suman 1, así que la firma vive en [0, 1] por construcción. La saturación
    es deliberada: evita que un valor absurdamente alto en una sola feature domine
    la suma. Como consecuencia existe un átomo de probabilidad en 1.0 (suministros
    con TODAS las features saturadas), que los módulos de corte tratan aparte.
    """

    def __init__(self, cols=None):
        self.cols = list(cols) if cols is not None else list(FIRMA_COLS)
        self.perfil = {}
        self.pesos = {}

    def fit(self, feat_fraude, feat_normal):
        for c in self.cols:
            self.perfil[c] = {
                'p50_fraude': float(feat_fraude[c].median()),
                'p50_normal': float(feat_normal[c].median()),
            }

        # Peso de cada feature = cuánto separa el perfil de fraude del perfil típico,
        # escalado por la dispersión del histórico. Una feature cuyo valor mediano es
        # igual en ambos recibe peso ~0 automáticamente.
        pesos = {}
        for c in self.cols:
            escala = float(feat_fraude[c].std())
            if not np.isfinite(escala) or escala < 1e-6:
                pesos[c] = 0.0
                continue
            sep = abs(self.perfil[c]['p50_fraude'] - self.perfil[c]['p50_normal']) / escala
            pesos[c] = float(sep)

        total = sum(pesos.values())
        self.pesos = {c: w / total for c, w in pesos.items()} if total > 0 else pesos
        return self

    def transform(self, feat):
        """0 = igual que un cliente típico del alimentador,
        1 = tan extremo como el perfil de fraude confirmado en todas las features."""
        out = np.zeros(len(feat))
        for c in self.cols:
            w = self.pesos.get(c, 0.0)
            if w <= 0:
                continue
            p_fraude = self.perfil[c]['p50_fraude']
            p_normal = self.perfil[c]['p50_normal']
            delta = p_fraude - p_normal
            if abs(delta) < 1e-9:
                continue
            out += w * np.clip((feat[c].values - p_normal) / delta, 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)

    def resumen(self):
        filas = []
        for c, w in sorted(self.pesos.items(), key=lambda kv: -kv[1]):
            filas.append(f"    {c:22s} peso={w:.3f}  "
                         f"(fraude={self.perfil[c]['p50_fraude']:.3f} vs "
                         f"alimentador={self.perfil[c]['p50_normal']:.3f})")
        return "\n".join(filas)
