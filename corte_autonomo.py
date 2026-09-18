"""
CORTE AUTÓNOMO DEL RANKING — ¿cuántos suministros se entregan?

Reemplaza la constante `CORTE_OPERATIVO = 100` por un número que sale de la data.

-------------------------------------------------------------------------------
POR QUÉ NO SE USA CONTROL DE FDR (ni ningún corte por "significancia estadística")
-------------------------------------------------------------------------------
Era el candidato natural: fijar q=10% y dejar que Benjamini-Hochberg decida N. Se
midió y es INALCANZABLE en este problema, por aritmética, no por implementación.

Con 31 hurtos entre 14,951 suministros la prevalencia es 0.21%. La precisión de
cualquier conjunto marcado es:

    precisión = pi  *  lift / (pi  *  lift + 1 - pi)        con pi = 0.0021

El lift máximo que alcanza la firma de fraude, medido sobre los 4,659 casos
confirmados, es 17x. Eso da una precisión máxima de 3.4%, es decir un FDR mínimo
alcanzable de 96.6%. Para un FDR de 10% haría falta un lift de ~430x.

Además existe una cota que no depende del modelo: marcando N suministros, la
precisión no puede superar min(31, N)/N. Para N=100 el techo es 31% (FDR 69%)
AUNQUE el modelo fuese un oráculo perfecto. Un corte por FDR devolvería N=0 en
toda corrida y no es un criterio utilizable acá.

Tampoco sirve el corte por razón de verosimilitud LR>=1 ("más parecido a un hurto
que a un cliente normal"): las dos distribuciones se solapan tanto que lo cumplen
~6,300 suministros. Correcto estadísticamente, inútil para despachar cuadrillas.

-------------------------------------------------------------------------------
QUÉ SE USA: MÁXIMA CONCENTRACIÓN DE EVIDENCIA (máximo lift)
-------------------------------------------------------------------------------
N* = el tamaño de corte donde la CONCENTRACIÓN de fraude es máxima, medida como

    lift(u) = P(firma >= u | hurto confirmado) / P(firma >= u | alimentador)

El lift es exactamente el factor por el que la inspección supera al azar, y la
precisión esperada es monótona en él, así que maximizarlo maximiza la tasa de
acierto de la cuadrilla. Es un criterio con tres propiedades que importan acá:

  1. No tiene parámetros de negocio: no necesita saber la prevalencia real ni el
     costo de inspección, que son justamente los dos números que no conocemos.
  2. Es invariante a pi: cambiar la prevalencia asumida escala la precisión pero
     no mueve el argmax. Un corte por ROI sí se mueve (ver `sensibilidad_roi`).
  3. N sale de la data. Con otro alimentador, otra firma u otro histórico, el
     máximo cae en otro lugar y N cambia solo.

El máximo se estima con bootstrap sobre el histórico para reportar intervalo de
confianza: si el IC es ancho, el jurado merece saberlo.

Una salvedad honesta que debe aparecer en el PDF: el lift se estima con los
fraudes confirmados de OTRA zona de la concesión. Se asume que la firma de un
hurto es transferible entre zonas; es la misma suposición que ya sostiene todo el
modelo, y el backtest (`backtest_recall.py`) la pone a prueba.
"""

import numpy as np
import pandas as pd

# Guardas estadísticas (no son parámetros de negocio): con menos casos que esto
# el lift se estima sobre tan pocos datos que su máximo es ruido. No fijan N, solo
# excluyen del argmax los tramos donde la estimación no significa nada.
MIN_FRAUDES_EN_COLA = 30   # casos confirmados por encima del umbral
MIN_SUMINISTROS_CORTE = 20  # suministros del alimentador marcados


def curva_lift(firma_ali, firma_hist, min_fraudes=MIN_FRAUDES_EN_COLA,
               min_n=MIN_SUMINISTROS_CORTE, max_n=None):
    """Lift para cada corte alcanzable del ranking por firma.

    Recorre los umbrales distintos presentes en la cola del alimentador. Se indexa
    por UMBRAL y no por posición porque la firma tiene empates masivos (el átomo en
    1.0 agrupa 83 suministros): cortar "en la posición 50" no está definido cuando
    83 comparten el mismo score, mientras que cortar en un umbral sí lo está.
    """
    firma_ali = np.asarray(firma_ali, dtype=float)
    firma_hist = np.asarray(firma_hist, dtype=float)
    n_ali_total = len(firma_ali)
    n_hist_total = len(firma_hist)
    if max_n is None:
        max_n = n_ali_total

    ali_sort = np.sort(firma_ali)[::-1]
    hist_sort = np.sort(firma_hist)[::-1]

    # Umbrales candidatos: valores distintos de la firma en el alimentador, de
    # mayor a menor, limitados al rango de cortes operativamente posibles.
    umbrales = np.unique(ali_sort[:max_n])[::-1]

    n_marcados = n_ali_total - np.searchsorted(ali_sort[::-1], umbrales, side='left')
    n_frau = n_hist_total - np.searchsorted(hist_sort[::-1], umbrales, side='left')

    with np.errstate(divide='ignore', invalid='ignore'):
        p_frau = n_frau / n_hist_total
        p_norm = n_marcados / n_ali_total
        lift = np.where(p_norm > 0, p_frau / p_norm, 0.0)

    out = pd.DataFrame({
        'umbral': umbrales,
        'n_marcados': n_marcados,
        'n_fraudes_sobre_umbral': n_frau,
        'cobertura_fraude': p_frau,
        'lift': lift,
    })
    out['elegible'] = (
        (out['n_fraudes_sobre_umbral'] >= min_fraudes)
        & (out['n_marcados'] >= min_n)
        & (out['n_marcados'] <= max_n)
    )
    return out


def _argmax_lift(firma_ali, firma_hist, **kw):
    curva = curva_lift(firma_ali, firma_hist, **kw)
    elegible = curva[curva['elegible']]
    if elegible.empty:
        return None, None, curva
    fila = elegible.loc[elegible['lift'].idxmax()]
    return int(fila['n_marcados']), float(fila['umbral']), curva


def corte_autonomo(firma_ali, firma_hist, n_boot=400, seed=42, verbose=True, **kw):
    """Determina N por máxima concentración de evidencia, con IC por bootstrap.

    El bootstrap remuestrea los casos confirmados del histórico (la fuente de
    incertidumbre dominante: 4,659 casos estiman la cola de la distribución de
    fraude, y en el extremo quedan pocos). El alimentador no se remuestrea: es la
    población completa que hay que rankear, no una muestra de algo mayor.
    """
    firma_ali = np.asarray(firma_ali, dtype=float)
    firma_hist = np.asarray(firma_hist, dtype=float)

    n_opt, umbral_opt, curva = _argmax_lift(firma_ali, firma_hist, **kw)
    if n_opt is None:
        raise RuntimeError(
            "Ningún corte cumple las guardas estadísticas mínimas. Revisar la firma: "
            "probablemente no separa fraude de normal en ningún tramo."
        )

    rng = np.random.default_rng(seed)
    n_boot_vals, lift_boot_vals = [], []
    for _ in range(n_boot):
        muestra = rng.choice(firma_hist, size=len(firma_hist), replace=True)
        n_b, _, curva_b = _argmax_lift(firma_ali, muestra, **kw)
        if n_b is None:
            continue
        n_boot_vals.append(n_b)
        eleg = curva_b[curva_b['elegible']]
        lift_boot_vals.append(float(eleg['lift'].max()))

    n_boot_vals = np.array(n_boot_vals)
    ic_n = (int(np.percentile(n_boot_vals, 2.5)), int(np.percentile(n_boot_vals, 97.5))) \
        if len(n_boot_vals) else (n_opt, n_opt)

    fila = curva[curva['n_marcados'] == n_opt].iloc[0]
    res = {
        'N': n_opt,
        'umbral_firma': umbral_opt,
        'lift': float(fila['lift']),
        'cobertura_fraude': float(fila['cobertura_fraude']),
        'ic95_N': ic_n,
        'n_boot_exitosos': int(len(n_boot_vals)),
        'curva': curva,
    }

    if verbose:
        print(f"Corte autónomo por máxima concentración de evidencia:")
        print(f"    N = {res['N']} suministros (umbral de firma = {res['umbral_firma']:.4f})")
        print(f"    lift = {res['lift']:.1f}x sobre inspección al azar")
        print(f"    cobertura: el umbral deja por encima al {res['cobertura_fraude']*100:.1f}% "
              f"de los hurtos confirmados del histórico")
        print(f"    IC95% de N por bootstrap ({res['n_boot_exitosos']} repeticiones): "
              f"[{ic_n[0]}, {ic_n[1]}]")

    return res


# -------------------------------------------------------------
# CALIBRACIÓN Y CONTRASTE ECONÓMICO
# -------------------------------------------------------------
def calibrar_posterior(firma_ali, firma_hist, prevalencia):
    """P(hurto | firma) calibrada, vía razón de verosimilitud isotónica.

    Se ajusta una regresión isotónica sobre la mezcla {alimentador=0, histórico=1}
    y se corrige por las proporciones artificiales de esa mezcla para recuperar la
    razón de verosimilitud f1/f0. La monotonía se impone a propósito: más firma
    nunca puede significar menos sospecha, y sin esa restricción la cola (donde hay
    pocos casos) produce oscilaciones espurias.

    Chequeo de consistencia: la suma de los posteriors sobre todo el alimentador
    debe dar ~= prevalencia x N_total. Si no da, la calibración está rota.
    """
    from sklearn.isotonic import IsotonicRegression

    firma_ali = np.asarray(firma_ali, dtype=float)
    firma_hist = np.asarray(firma_hist, dtype=float)

    z = np.concatenate([firma_ali, firma_hist])
    y = np.concatenate([np.zeros(len(firma_ali)), np.ones(len(firma_hist))])
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1 - 1e-6).fit(z, y)

    r = iso.predict(firma_ali)
    odds_mezcla = len(firma_ali) / len(firma_hist)
    lr = (r / (1.0 - r)) * odds_mezcla

    pi = float(prevalencia)
    posterior = pi * lr / (pi * lr + 1.0 - pi)
    return posterior, lr


def sensibilidad_roi(posterior, recupero_soles, costos=(50, 100, 150, 250),
                     prevalencias=(31 / 14951,), verbose=True):
    """Cuántos suministros justifican económicamente la inspección, por costo.

    Regla de Bayes: inspeccionar mientras P(hurto) x recupero x tarifa >= costo.
    Se reporta como CONTRASTE del corte por lift, no como el corte mismo, porque
    depende de dos números que no conocemos:

      - el costo real de despachar una cuadrilla, que es dato de EQUANS;
      - la prevalencia pi, y acá está el problema serio: los 31 hurtos del reto son
        los CONFIRMADOS por el jurado, no todos los que hay en el alimentador. La
        prevalencia real es más alta y desconocida, y el N que sale de esta regla
        escala casi linealmente con ella. Por eso no se usa para fijar N.
    """
    posterior = np.asarray(posterior, dtype=float)
    recupero_soles = np.asarray(recupero_soles, dtype=float)

    filas = []
    pi_base = float(np.mean(posterior)) if len(posterior) else 0.0
    for pi in prevalencias:
        escala = (pi / pi_base) if pi_base > 0 else 1.0
        post_esc = np.clip(posterior * escala, 0.0, 1.0)
        ve = post_esc * recupero_soles
        for c in costos:
            n = int((ve >= c).sum())
            top = np.sort(ve)[::-1][:n]
            filas.append({
                'prevalencia': pi,
                'costo_inspeccion': c,
                'N': n,
                'recupero_esperado_soles': float(top.sum()),
                'beneficio_neto_soles': float(top.sum() - n * c),
            })

    tabla = pd.DataFrame(filas)
    if verbose and not tabla.empty:
        print("Contraste económico (regla de Bayes: inspeccionar si VE >= costo):")
        for _, r in tabla.iterrows():
            print(f"    pi={r['prevalencia']*100:.2f}%  costo=S/{r['costo_inspeccion']:.0f}"
                  f"  ->  N={int(r['N']):5d}  neto=S/{r['beneficio_neto_soles']:,.0f}")
    return tabla


def techo_precision(n_marcados, n_hurtos_reales):
    """Cota superior de precisión que no depende del modelo: min(H, N)/N.

    Se reporta junto al entregable para que nadie lea el ranking como una promesa
    de acierto imposible: con 31 hurtos, marcar 100 suministros no puede superar
    el 31% de aciertos ni con un modelo perfecto.
    """
    return min(n_hurtos_reales, n_marcados) / max(n_marcados, 1)


# -------------------------------------------------------------
# AJUSTE POR PERFIL: verosimilitud medida en el fraude confirmado
# -------------------------------------------------------------
def factor_verosimilitud_perfil(mask_ali, mask_hist, nombre='', verbose=True):
    """Cuánto multiplica (o divide) la sospecha el pertenecer a un perfil.

        factor = P(perfil | fraude confirmado) / P(perfil | alimentador)

    Es la corrección bayesiana que corresponde: la probabilidad posterior de fraude es
    proporcional a la previa por la razón de verosimilitud del perfil observado. Un
    factor menor que 1 significa que el perfil aparece MENOS entre los fraudes reales
    que en la población general, y por lo tanto debe bajar en el ranking, no subir.

    Se usa un factor continuo y no un filtro binario a propósito: excluir de plano
    cuesta falsos negativos —el backtest mostró que los filtros duros se llevaban 2.15
    de cada 31 hurtos antes de rankear— mientras que un factor deja al suministro
    compitiendo, solo que con el peso que la evidencia le da.
    """
    p_hist = float(np.mean(mask_hist))
    p_ali = float(np.mean(mask_ali))
    factor = p_hist / p_ali if p_ali > 1e-9 else 1.0
    if verbose:
        sentido = 'penaliza' if factor < 1 else 'refuerza'
        print(f"    perfil '{nombre}': {p_hist*100:.1f}% del fraude confirmado contra "
              f"{p_ali*100:.1f}% del alimentador -> factor {factor:.2f}x ({sentido})")
    return factor


def calibrar_senal_por_verosimilitud(x_ali, x_hist, n_bins=10, suavizado=20.0,
                                     nombre='', verbose=True):
    """Convierte una señal continua en su razón de verosimilitud empírica.

    En vez de asumir que "más señal = más sospechoso" (monótono), se mide para cada
    tramo cuántas veces más frecuente es ese tramo entre los fraudes confirmados que en
    el alimentador. Devuelve una función que mapea el valor crudo a ese factor.

    Por qué hace falta: la caída de consumo NO es monótona en sospecha. Una caída del
    60% multiplica la sospecha, pero una caída del 100% sostenida la baja — es una baja
    de servicio, no un hurto. Una señal monótona no puede representar eso y termina
    poniendo los medidores apagados arriba del ranking.

    Los bins salen de los cuantiles del alimentador, así que ninguno queda vacío. El
    suavizado hacia 1.0 (peso `suavizado` casos equivalentes) evita que un bin con
    pocos fraudes produzca un factor extremo por azar.
    """
    x_ali = np.asarray(x_ali, dtype=float)
    x_hist = np.asarray(x_hist, dtype=float)

    cortes = np.unique(np.quantile(x_ali, np.linspace(0, 1, n_bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf

    idx_a = np.digitize(x_ali, cortes[1:-1])
    idx_h = np.digitize(x_hist, cortes[1:-1])

    n_bin = len(cortes) - 1
    factores = np.ones(n_bin)
    for b in range(n_bin):
        na, nh = int((idx_a == b).sum()), int((idx_h == b).sum())
        if na == 0:
            continue
        p_a = na / len(x_ali)
        p_h = nh / len(x_hist)
        crudo = p_h / p_a if p_a > 0 else 1.0
        # Suavizado hacia 1.0 proporcional a cuántos fraudes respaldan el bin.
        peso = nh / (nh + suavizado)
        factores[b] = peso * crudo + (1.0 - peso) * 1.0

    if verbose:
        print(f"    señal '{nombre}' calibrada por verosimilitud ({n_bin} tramos):")
        for b in range(n_bin):
            lo = cortes[b] if np.isfinite(cortes[b]) else float(x_ali.min())
            hi = cortes[b + 1] if np.isfinite(cortes[b + 1]) else float(x_ali.max())
            print(f"        [{lo:7.3f}, {hi:7.3f}) -> {factores[b]:.2f}x")

    def aplicar(x):
        return factores[np.digitize(np.asarray(x, dtype=float), cortes[1:-1])]

    return aplicar


def calibrar_verosimilitud_cruzada(claves_ali, claves_hist, suavizado=20.0,
                                   nombre='', verbose=True):
    """Razón de verosimilitud sobre una clave CATEGÓRICA compuesta.

    Por qué cruzada y no dos correcciones por separado: aplicar en cadena el factor de
    `caida_pct` y el del perfil de medidor apagado las hace cancelarse entre sí (1.50x
    por caer fuerte contra 0.63x por estar apagado da 0.95x, prácticamente neutral) y el
    ranking vuelve a llenarse de medidores de baja. No son señales independientes: estar
    apagado IMPLICA haber caído. Cruzarlas mide la celda real —"cayó fuerte Y sigue
    activo" contra "cayó fuerte Y quedó en cero"— que es donde vive la distinción entre
    un hurto y una baja de servicio.
    """
    claves_ali = pd.Series(claves_ali).astype(str)
    claves_hist = pd.Series(claves_hist).astype(str)

    p_a = claves_ali.value_counts(normalize=True)
    n_h = claves_hist.value_counts()
    p_h = claves_hist.value_counts(normalize=True)

    factores = {}
    for k in p_a.index:
        nh = int(n_h.get(k, 0))
        crudo = (p_h.get(k, 0.0) / p_a[k]) if p_a[k] > 0 else 1.0
        peso = nh / (nh + suavizado)
        factores[k] = peso * crudo + (1.0 - peso) * 1.0

    if verbose:
        print(f"    verosimilitud cruzada '{nombre}':")
        for k in sorted(factores, key=lambda k: -factores[k]):
            print(f"        {k:34s} {p_h.get(k, 0.0)*100:5.1f}% fraude vs "
                  f"{p_a[k]*100:5.1f}% alimentador -> {factores[k]:.2f}x")

    return lambda claves: np.array([factores.get(str(k), 1.0) for k in claves])


def clave_caida_actividad(caida_pct, meses_cero_finales, cortes=(0.0, 0.30, 0.65)):
    """Clave compuesta: tramo de caída x si el medidor quedó apagado al cierre."""
    caida_pct = np.asarray(caida_pct, dtype=float)
    apagado = np.asarray(meses_cero_finales, dtype=float) >= 3
    tramo = np.digitize(caida_pct, list(cortes))
    etiquetas = np.array(['subio_fuerte', 'estable', 'cayo_moderado', 'cayo_fuerte'])
    return np.array([f"{etiquetas[t]}|{'apagado' if a else 'activo'}"
                     for t, a in zip(tramo, apagado)])
