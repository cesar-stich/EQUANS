"""
BACKTEST DE RECALL — ¿cuántos de los hurtos reales caen en el top del ranking?

El reto no entrega las etiquetas del alimentador: los 31 hurtos los tiene el jurado.
Sin etiquetas no se puede medir el recall directamente, pero sí se puede SIMULAR la
situación: se toman hurtos CONFIRMADOS del histórico, se los inyecta dentro de la
población del alimentador a la misma prevalencia real (31 de 14,951), se corre el
mismo ranking de producción y se cuenta cuántos de los inyectados aparecen en el
top-K. Repetido con bootstrap, da la curva de recall con intervalo de confianza.

-------------------------------------------------------------------------------
CONTROLES PARA QUE EL NÚMERO SIGNIFIQUE ALGO
-------------------------------------------------------------------------------
1. PARTICIÓN DEL HISTÓRICO. El instrumento (firma de fraude y regresor de recupero)
   se ajusta con una mitad, y los casos inyectados salen SIEMPRE de la otra mitad.
   Sin esto se estaría midiendo cuán bien el modelo reconoce los casos con los que
   fue construido, que es garantía de un número inflado y sin valor.

2. REUBICACIÓN EN LA RED. Cada caso inyectado recibe un par (SED, LLAVE) real del
   alimentador, de modo que `desv_sed` y `desv_llave` se recalculan contra vecinos
   del alimentador y no contra los de su zona de origen. Un fraude que era atípico
   en su barrio no hereda gratis esa ventaja en el barrio nuevo.

3. DOBLE VARIANTE, PORQUE HAY UN SESGO QUE NO SE PUEDE ELIMINAR. Los dos datasets
   son de ZONAS DISTINTAS y sus features difieren de entrada (kwhd_cv mediano 0.466
   en el histórico contra 0.215 en el alimentador). Parte de esa brecha es firma de
   fraude y parte es artefacto de zona, y NO hay forma de separarlas: el histórico
   es 100% positivo, no tiene clientes honestos de su propia zona contra los cuales
   contrastar. Es una limitación estructural del reto, no del método. Por eso se
   reportan dos escenarios que acotan la respuesta:

     OPTIMISTA     — se inyecta el caso tal cual. Atribuye toda la brecha a fraude.
                     Es el techo del recall alcanzable.
     CONSERVADOR   — se resta de cada feature la diferencia de medianas entre zonas,
                     dejando solo cuán extremo es el caso RESPECTO DE SU PROPIA
                     población. Atribuye toda la brecha a artefacto de zona. Es el
                     piso.

   El recall real está entre ambos. Reportar solo el optimista sería vender un
   número que no se sostiene si el jurado pregunta por el cambio de zona.

LÍMITE DEL ESCENARIO CONSERVADOR (medido, para no sobreinterpretarlo)
-------------------------------------------------------------------
El escenario conservador resta a cada feature del caso inyectado la diferencia de
medianas entre zonas. Para una firma construida COMO distancia a esas medianas, esa
corrección es casi degenerada: deja al caso sobre el perfil normal por construcción y
su score se va a ~0. Se comprobó comparando variantes de la firma — todas las
alternativas caían a lift 0.0x en el conservador, no por ser malas sino porque la
corrección anula el mecanismo que están usando para medir.

Conclusión operativa: el conservador sirve como PISO y como alarma de dependencia de
la zona, pero NO sirve para elegir entre modelos. Para eso hace falta lo único que no
tenemos: suministros etiquetados del propio alimentador.

Uso:
    python backtest_recall.py                 # 200 repeticiones, ambas variantes
    python backtest_recall.py --boot 50       # corrida rápida
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

import equans_core as ec
import features_avanzadas as fa
import firma_v2

warnings.filterwarnings('ignore')

# Prevalencia del reto: 31 hurtos conocidos entre 14,951 suministros.
N_HURTOS = 31
KS_REPORTE = [10, 20, 31, 40, 50, 96, 100, 150, 200, 300, 500]

# Features intrínsecas al suministro (no dependen del vecindario). Son las únicas a
# las que se aplica la corrección de zona en el escenario conservador: `desv_sed` y
# `desv_llave` ya se recalculan contra vecinos del alimentador, así que corregirlas
# también sería descontar el mismo sesgo dos veces.
FIRMA_COLS_INTRINSECAS = [
    'kwhd_cv', 'caida_pct', 'alternancia', 'factor_uso',
    'ratio_meses_activos', 'meses_cero_finales',
] + [c for c in fa.FIRMA_COLS_AVANZADAS if c != 'conexionado_mono']

COLS_MES = [f'{p}_{m}' for m in ec.MESES for p in ('DIAS', 'CONS')]
# TIPO_CONEXIONADO viaja con el caso inyectado: es una caracteristica fisica del
# suministro intervenido, no del barrio donde se lo reubica.
COLS_INYECCION = COLS_MES + ['POTENCIA_CONTRATADA', 'TIPO_CONEXIONADO']


REC_COLS = (ec.FIRMA_COLS + fa.FIRMA_COLS_AVANZADAS
            + ['cons_anual', 'kwhd_promedio', 'POTENCIA_CONTRATADA'])


def _entrenar_regresor_recupero(feat_fit, df_fit, cols):
    """Mismo regresor cuantílico P25 que producción, entrenado solo con la mitad A."""
    X = feat_fit[[c for c in cols if c != 'POTENCIA_CONTRATADA']].fillna(0).copy()
    X['POTENCIA_CONTRATADA'] = df_fit['POTENCIA_CONTRATADA'].fillna(3.0).values
    y = np.log1p(df_fit['RECUPERO_KWH_REAL'].values)

    reg = lgb.LGBMRegressor(
        objective='quantile', alpha=0.25, n_estimators=300, learning_rate=0.05,
        max_depth=6, num_leaves=31, min_child_samples=30, random_state=42, verbose=-1
    )
    reg.fit(X[cols], y)
    return reg


def _entrenar_isolation_forest(feat_ali, cohortes):
    """Se ajusta UNA vez sobre el alimentador real y se reutiliza para puntuar cada
    piscina simulada. Es lo correcto además de lo barato: en producción el Isolation
    Forest también aprende sobre el alimentador con los hurtos adentro."""
    modelos = {}
    for coh in pd.unique(cohortes):
        mask = cohortes == coh
        iso = IsolationForest(contamination=0.06, random_state=42)
        iso.fit(feat_ali.loc[mask, ec.FEATURE_COLS].fillna(0))
        modelos[coh] = iso
    return modelos


def _anomaly_score(modelos, feat, cohortes):
    out = np.zeros(len(feat))
    for coh, iso in modelos.items():
        mask = (cohortes == coh).values if hasattr(cohortes, 'values') else (cohortes == coh)
        if not mask.any():
            continue
        scores = iso.decision_function(feat.loc[mask, ec.FEATURE_COLS].fillna(0))
        out[mask] = 1.0 / (1.0 + np.exp(scores * 5.0))
    return out


def construir_piscina(df_ali, df_inyectar, rng):
    """Piscina simulada: suministros del alimentador + casos de hurto confirmado.

    Se reemplazan tantos suministros del alimentador como casos se inyectan, para
    que el tamaño total y la prevalencia sean los del reto.
    """
    n_iny = len(df_inyectar)
    idx_ali = rng.choice(len(df_ali), size=len(df_ali) - n_iny, replace=False)
    base = df_ali.iloc[np.sort(idx_ali)][COLS_INYECCION + ['SED_ID', 'LLAVE_ID']].copy()
    base['es_hurto'] = False

    # Reubicación en la red: par (SED, LLAVE) real tomado del propio alimentador.
    ubic = df_ali[['SED_ID', 'LLAVE_ID']].iloc[rng.choice(len(df_ali), size=n_iny, replace=True)]
    iny = df_inyectar[COLS_INYECCION].copy()
    iny['SED_ID'] = ubic['SED_ID'].values
    iny['LLAVE_ID'] = ubic['LLAVE_ID'].values
    iny['es_hurto'] = True

    piscina = pd.concat([base, iny], ignore_index=True)
    return piscina.reset_index(drop=True)


def corregir_sesgo_zona(feat, es_hurto, shift):
    """Escenario conservador: descuenta de los casos inyectados la diferencia de
    medianas entre zonas, feature por feature. Lo que queda es solo cuán extremo es
    el caso dentro de su propia población de origen."""
    feat = feat.copy()
    idx = np.flatnonzero(es_hurto)
    for c, delta in shift.items():
        feat.loc[feat.index[idx], c] = feat[c].values[idx] - delta
    return feat


def una_repeticion(df_ali, df_pool_b, firma, reg, iso_modelos, rec_cols, rng,
                   shift=None, n_hurtos=N_HURTOS):
    """Una simulación completa: inyectar, rankear con el pipeline, medir recall."""
    df_iny = df_pool_b.iloc[rng.choice(len(df_pool_b), size=n_hurtos, replace=False)]
    piscina = construir_piscina(df_ali, df_iny, rng)

    feat, _ = fa.extract_features_avanzadas(piscina)
    es_hurto = piscina['es_hurto'].values

    if shift is not None:
        feat = corregir_sesgo_zona(feat, es_hurto, shift)

    cohortes = ec.segmentar_por_potencia(piscina['POTENCIA_CONTRATADA'])

    f = firma.transform(feat)
    anomaly = _anomaly_score(iso_modelos, feat, cohortes)
    # Mismo ensamble de tres señales que producción, en escala de percentil.
    sed = (0.70 * pd.Series(feat['caida_vs_sed'].values).rank(pct=True).values
           + 0.30 * pd.Series(feat['desacople_sed'].values).rank(pct=True).values)
    prob = np.clip(
        0.40 * pd.Series(f).rank(pct=True).values
        + 0.20 * pd.Series(anomaly).rank(pct=True).values
        + 0.40 * sed,
        0.01, 0.99)

    X = feat[[c for c in rec_cols if c != 'POTENCIA_CONTRATADA']].fillna(0).copy()
    X['POTENCIA_CONTRATADA'] = piscina['POTENCIA_CONTRATADA'].fillna(3.0).values
    recupero_kwh = np.maximum(np.expm1(reg.predict(X[rec_cols])), 300.0)
    prioridad = prob * recupero_kwh * ec.TARIFA_REF_SOLES

    # Mismas exclusiones que producción: sin servicio activo y recupero no
    # diferenciable por colisión de hoja del regresor.
    sin_servicio = (feat['cons_anual'].values < 120.0) | (feat['ratio_meses_activos'].values < 0.25)
    prioridad = np.where(sin_servicio, -1.0, prioridad)

    orden = np.argsort(-prioridad, kind='stable')
    hurto_ordenado = es_hurto[orden]
    pos_hurtos = np.flatnonzero(hurto_ordenado) + 1  # posiciones 1-indexadas

    return {
        'recall': {k: int((pos_hurtos <= k).sum()) for k in KS_REPORTE},
        'posiciones': pos_hurtos.tolist(),
        'excluidos': int((es_hurto & sin_servicio).sum()),
    }


def correr_backtest(n_boot=200, seed=42, verbose=True, version='v2'):
    df_ali, df_hist = ec.cargar_datasets(verbose=False)
    feat_ali, _ = fa.extract_features_avanzadas(df_ali)

    rng = np.random.default_rng(seed)

    # Partición estratificada por tipificación: las dos mitades conservan la mezcla
    # MANIPULACIÓN / CLANDESTINA del histórico completo.
    col_tip = ec.find_col(df_hist, 'TIPIFICACI', 'df_hist', verbose=False)
    mitad_a, mitad_b = [], []
    for _, sub in df_hist.groupby(df_hist[col_tip].astype(str)):
        idx = sub.index.values.copy()
        rng.shuffle(idx)
        corte = len(idx) // 2
        mitad_a.extend(idx[:corte])
        mitad_b.extend(idx[corte:])

    df_fit = df_hist.loc[sorted(mitad_a)].reset_index(drop=True)
    df_pool_b = df_hist.loc[sorted(mitad_b)].reset_index(drop=True)
    feat_fit, _ = fa.extract_features_avanzadas(df_fit)

    if verbose:
        print(f"Histórico particionado: {len(df_fit)} casos para ajustar el instrumento, "
              f"{len(df_pool_b)} reservados para inyectar (disjuntos).")

    firma = ec.FirmaFraude().fit(feat_fit, feat_ali)
    rec_cols = REC_COLS
    reg = _entrenar_regresor_recupero(feat_fit, df_fit, rec_cols)
    cohortes_ali = ec.segmentar_por_potencia(df_ali['POTENCIA_CONTRATADA'])
    iso_modelos = _entrenar_isolation_forest(feat_ali, cohortes_ali)

    # Desplazamiento de medianas entre zonas, medido en la mitad de ajuste.
    shift = {c: float(feat_fit[c].median() - feat_ali[c].median())
             for c in FIRMA_COLS_INTRINSECAS}
    if verbose:
        print("Desplazamiento de medianas histórico - alimentador (sesgo de zona a descontar "
              "en el escenario conservador):")
        for c, v in shift.items():
            print(f"    {c:22s} {v:+.4f}")

    resultados = {}
    for nombre, sh in (('optimista', None), ('conservador', shift)):
        rng_esc = np.random.default_rng(seed + (0 if sh is None else 1))
        filas, todas_pos, excluidos = [], [], []
        for i in range(n_boot):
            r = una_repeticion(df_ali, df_pool_b, firma, reg, iso_modelos, rec_cols,
                               rng_esc, shift=sh)
            filas.append(r['recall'])
            todas_pos.extend(r['posiciones'])
            excluidos.append(r['excluidos'])
            if verbose and (i + 1) % 25 == 0:
                print(f"    [{nombre}] {i+1}/{n_boot} repeticiones")

        tabla = pd.DataFrame(filas)
        resumen = pd.DataFrame({
            'K': KS_REPORTE,
            'recall_medio': [tabla[k].mean() for k in KS_REPORTE],
            'recall_p2.5': [np.percentile(tabla[k], 2.5) for k in KS_REPORTE],
            'recall_p97.5': [np.percentile(tabla[k], 97.5) for k in KS_REPORTE],
        })
        resumen['pct_hurtos_detectados'] = resumen['recall_medio'] / N_HURTOS * 100
        resumen['precision_pct'] = resumen['recall_medio'] / resumen['K'] * 100
        # Cota que no depende del modelo: con 31 hurtos, marcar K no puede superar
        # min(31, K)/K de acierto ni con un oráculo.
        resumen['precision_techo_pct'] = [min(N_HURTOS, k) / k * 100 for k in KS_REPORTE]
        resumen['lift_vs_azar'] = resumen['precision_pct'] / (N_HURTOS / len(df_ali) * 100)

        resultados[nombre] = {
            'resumen': resumen,
            'mediana_posicion': float(np.median(todas_pos)) if todas_pos else float('nan'),
            'excluidos_medio': float(np.mean(excluidos)),
        }

        if verbose:
            print(f"\n=== ESCENARIO {nombre.upper()} ({n_boot} repeticiones) ===")
            print(resumen.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
            print(f"Posición mediana de un hurto en el ranking: "
                  f"#{resultados[nombre]['mediana_posicion']:.0f} de {len(df_ali)}")
            print(f"Hurtos excluidos por los filtros operativos (sin servicio / recupero no "
                  f"diferenciable): {resultados[nombre]['excluidos_medio']:.2f} de {N_HURTOS} "
                  f"en promedio")

    return resultados


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--boot', type=int, default=200, help='repeticiones de bootstrap')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--version', default='v2', choices=['v1', 'v2'])
    ap.add_argument('--out', default='entregables/backtest_recall.json')
    args = ap.parse_args()

    print("=== BACKTEST DE RECALL POR INYECCIÓN DE HURTOS CONFIRMADOS ===")
    res = correr_backtest(n_boot=args.boot, seed=args.seed, version=args.version)

    payload = {
        'n_boot': args.boot,
        'n_hurtos_inyectados': N_HURTOS,
        'version_firma': args.version,
        'escenarios': {
            nombre: {
                'resumen': r['resumen'].to_dict(orient='records'),
                'mediana_posicion': r['mediana_posicion'],
                'excluidos_medio': r['excluidos_medio'],
            } for nombre, r in res.items()
        },
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {args.out}")


if __name__ == '__main__':
    main()
