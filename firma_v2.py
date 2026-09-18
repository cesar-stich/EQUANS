"""
FIRMA DE FRAUDE v2 — clasificador entrenado con el sesgo de zona neutralizado

PROBLEMA QUE RESUELVE
---------------------
La firma v1 pondera cada feature por cuánto separa la mediana del fraude de la
mediana del alimentador. Medido, el 94.7% del peso terminó en dos features:

    kwhd_cv      peso 0.604    AUC 0.696  ->  0.545 al descontar el sesgo de zona
    factor_uso   peso 0.343    AUC 0.701  ->  0.536 al descontar el sesgo de zona

Es decir: las dos features que sostenían la firma son las que MÁS se desploman
cuando se les quita la diferencia de zona. La v1 estaba ordenando, en buena medida,
por "cuánto se parece este suministro a la otra zona de la concesión". El backtest
lo mostraba desde afuera (el recall caía 3x en el escenario conservador); esta es la
causa.

CÓMO SE NEUTRALIZA
------------------
El modelo se entrena con el histórico CENTRADO: a cada feature de los casos de
fraude se le resta (mediana_histórico - mediana_alimentador) antes de entrenar. Con
las dos poblaciones alineadas en locación, el clasificador ya no puede separarlas
por el desplazamiento de zona — la vía fácil queda cerrada y solo le queda apoyarse
en diferencias de forma, dispersión e interacción entre features, que es la señal
que sí transfiere de una zona a otra.

No elimina el riesgo por completo (una diferencia de dispersión también podría ser
zonal), pero lo reduce de raíz en vez de corregirlo después. Es la misma corrección
que el escenario conservador del backtest aplica para auditar, aplicada ahora dentro
del entrenamiento.

SELECCIÓN DE FEATURES
---------------------
Se miden todas por AUC después de centrar, y entran solo las que conservan señal.
Quedan fuera por construcción las features de NIVEL y las INTRA-SED: en el histórico
los vecinos de un fraude son también fraudes, así que `desv_sed` y `desv_llave` no
son medibles contra él. Esa perspectiva la aporta el Isolation Forest, que sí se
entrena dentro del alimentador.

PONDERACIÓN POR RECUPERO
------------------------
Los casos de entrenamiento se ponderan por su recupero real (raíz cuadrada, para que
la cola larga no domine: la mediana es 3,930 kWh y el máximo 886,290). El objetivo
del reto es el impacto económico recuperable, así que el modelo debe aprender mejor
la firma de los peces gordos que la de los casos marginales.

-------------------------------------------------------------------------------
INTENTO DESCARTADO: ENTRENAR UN CLASIFICADOR SOBRE EL HISTÓRICO CENTRADO
-------------------------------------------------------------------------------
Primero se probó entrenar un LightGBM con el histórico centrado (restando a cada
feature la diferencia de medianas entre zonas), con la idea de cerrarle al modelo la
vía del desplazamiento zonal. Se midió con positivos held-out y NO funciona:

    v1                          lift 16.6x @ N=96
    clasificador centrado       lift  8.4x @ N=96   (peor)
    idem, evaluado centrado     lift  155x @ N=96   (imposible: 99.5% de cobertura)

El 155x delata el problema: restar una constante no borra la diferencia, deja una
huella. El modelo aprende a reconocer "caso centrado" contra "alimentador", y al
evaluarlo con casos centrados los reconoce casi perfecto. No mide fraude, mide su
propia transformación. Y evaluado sin centrar queda peor que v1 por el desajuste
entre el espacio de entrenamiento y el de aplicación.

Se conserva escrito porque el 99.5% es justo el tipo de número que uno querría creer.

-------------------------------------------------------------------------------
LO QUE SÍ CORRIGE EL DEFECTO: PONDERAR POR AUC CENTRADA
-------------------------------------------------------------------------------
La v1 pesa cada feature por |mediana_fraude - mediana_normal| / desviación. Eso es,
literalmente, la magnitud del desplazamiento entre zonas: el criterio de ponderación
PREMIA el artefacto que se quiere evitar. De ahí que kwhd_cv y factor_uso, las dos
features que más se desploman al descontar zona, concentraran el 94.7% del peso.

La v3 mantiene intacto el mecanismo de la v1 —distancia al perfil de fraude, sin
entrenamiento, interpretable— y solo cambia de dónde sale el peso:

    peso(feature) = max(AUC_centrada - 0.5, 0)

Una feature cuya capacidad de discriminar se evapora al quitarle el desplazamiento
de zona recibe peso ~0 automáticamente, en vez del peso máximo que le daba la v1.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

import equans_core as ec
import features_avanzadas as fa

# Features candidatas: forma temporal intra-suministro (adimensionales) más las de
# la firma original que no son de nivel ni intra-SED.
CANDIDATAS = [
    'ratio_min_med', 'frac_muy_bajo', 'dias_cv', 'caida_consecutiva_max',
    'conexionado_mono', 'escalon_mes', 'escalon_mag', 'estabilidad_pre',
    'volatilidad_post', 'autocorr1', 'rugosidad', 'ceros_intercalados',
    'ratio_var_mitades', 'slope_norm', 'frac_acumulados', 'meses_planos',
    'kwhd_cv', 'caida_pct', 'factor_uso', 'alternancia',
    'ratio_meses_activos', 'meses_cero_finales',
]

# Umbral de AUC centrada para que una feature entre al modelo. 0.52 es deliberadamente
# permisivo: LightGBM sabe ignorar una feature débil, y descartar de más costaría las
# interacciones. Lo que se quiere excluir es lo que NO aporta nada tras quitar la zona.
AUC_MINIMA = 0.52


class FirmaFraudeV2:
    def __init__(self, candidatas=None, auc_minima=AUC_MINIMA, random_state=42):
        self.candidatas = list(candidatas) if candidatas else list(CANDIDATAS)
        self.auc_minima = auc_minima
        self.random_state = random_state
        self.shift = {}
        self.cols = []
        self.auc_centrada = {}
        self.modelo = None

    def _centrar(self, feat, cols=None):
        """Lleva las features del histórico al centro del alimentador."""
        cols = cols or self.cols
        X = feat[cols].astype(float).copy()
        for c in cols:
            X[c] = X[c] - self.shift.get(c, 0.0)
        return X

    def fit(self, feat_fraude, feat_normal, peso_recupero=None):
        from sklearn.metrics import roc_auc_score

        disponibles = [c for c in self.candidatas
                       if c in feat_fraude.columns and c in feat_normal.columns]

        # Desplazamiento de locación entre zonas, medido feature por feature.
        self.shift = {c: float(feat_fraude[c].median() - feat_normal[c].median())
                      for c in disponibles}

        # Selección: AUC tras centrar. Lo que sobrevive es señal de forma.
        y = np.concatenate([np.ones(len(feat_fraude)), np.zeros(len(feat_normal))])
        seleccionadas = []
        for c in disponibles:
            xf = feat_fraude[c].values.astype(float) - self.shift[c]
            xn = feat_normal[c].values.astype(float)
            auc = roc_auc_score(y, np.concatenate([xf, xn]))
            auc = max(auc, 1.0 - auc)
            self.auc_centrada[c] = float(auc)
            if auc >= self.auc_minima:
                seleccionadas.append(c)

        if not seleccionadas:
            raise RuntimeError("Ninguna feature conserva señal tras descontar el sesgo de zona.")
        self.cols = seleccionadas

        X = pd.concat([
            self._centrar(feat_fraude),
            feat_normal[self.cols].astype(float),
        ], ignore_index=True)

        # Peso por recupero (raíz cuadrada) en los positivos; los no-etiquetados pesan 1.
        if peso_recupero is not None:
            w_pos = np.sqrt(np.asarray(peso_recupero, dtype=float).clip(min=0.0))
            w_pos = w_pos / (w_pos.mean() + 1e-9)
        else:
            w_pos = np.ones(len(feat_fraude))
        w = np.concatenate([w_pos, np.ones(len(feat_normal))])

        # El alimentador se usa como clase "no etiquetada" (PU learning): es 99.8%
        # honesto, así que aproxima bien a los negativos. `scale_pos_weight` compensa
        # que haya 3.2 veces más no-etiquetados que positivos.
        self.modelo = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5, num_leaves=31,
            min_child_samples=60, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0,
            random_state=self.random_state, verbose=-1,
        )
        self.modelo.fit(X, y, sample_weight=w)
        return self

    def transform(self, feat):
        X = feat[self.cols].astype(float)
        return self.modelo.predict_proba(X)[:, 1]

    def transform_centrado(self, feat):
        """Para puntuar casos del histórico en la misma escala que el alimentador."""
        return self.modelo.predict_proba(self._centrar(feat))[:, 1]

    def resumen(self):
        imp = dict(zip(self.cols, self.modelo.feature_importances_))
        total = sum(imp.values()) or 1
        filas = []
        for c, v in sorted(imp.items(), key=lambda kv: -kv[1]):
            filas.append(f"    {c:24s} importancia={v/total:.3f}  "
                         f"AUC_centrada={self.auc_centrada[c]:.3f}")
        descartadas = [c for c in self.auc_centrada if c not in self.cols]
        txt = "\n".join(filas)
        if descartadas:
            txt += ("\n    descartadas por no conservar señal tras descontar zona: "
                    + ", ".join(descartadas))
        return txt


def construir(df_fraude, feat_fraude, feat_normal, **kw):
    """Atajo: ajusta la firma v2 ponderando por el recupero real del histórico."""
    peso = df_fraude['RECUPERO_KWH_REAL'].values if 'RECUPERO_KWH_REAL' in df_fraude else None
    return FirmaFraudeV2(**kw).fit(feat_fraude, feat_normal, peso_recupero=peso)


class FirmaFraudeV3:
    """Firma v1 con ponderación honesta: el peso lo da la señal que SOBREVIVE al
    descuento del sesgo de zona, no la magnitud del desplazamiento entre zonas.

    Mantiene la arquitectura de la v1 a propósito: no se entrena ningún clasificador,
    así que no hay forma de que el modelo memorice la separación entre datasets. Cada
    feature aporta, como máximo, su peso; los pesos suman 1; la firma vive en [0, 1] y
    sigue significando "qué tan lejos está este suministro del cliente típico, en
    dirección al perfil del fraude confirmado".
    """

    def __init__(self, candidatas=None, auc_minima=0.52):
        self.candidatas = list(candidatas) if candidatas else list(CANDIDATAS)
        self.auc_minima = auc_minima
        self.perfil = {}
        self.pesos = {}
        self.auc_centrada = {}
        self.cols = []

    def fit(self, feat_fraude, feat_normal):
        from sklearn.metrics import roc_auc_score

        disponibles = [c for c in self.candidatas
                       if c in feat_fraude.columns and c in feat_normal.columns]
        y = np.concatenate([np.ones(len(feat_fraude)), np.zeros(len(feat_normal))])

        pesos = {}
        for c in disponibles:
            p_f = float(feat_fraude[c].median())
            p_n = float(feat_normal[c].median())
            self.perfil[c] = {'p50_fraude': p_f, 'p50_normal': p_n}

            # AUC tras alinear las medianas: cuánta capacidad de discriminar queda
            # cuando se le quita a la feature el desplazamiento entre zonas.
            xf = feat_fraude[c].values.astype(float) - (p_f - p_n)
            xn = feat_normal[c].values.astype(float)
            auc = roc_auc_score(y, np.concatenate([xf, xn]))
            auc = max(auc, 1.0 - auc)
            self.auc_centrada[c] = float(auc)

            # 0.5 es el azar: una feature que no discrimina nada aporta exactamente 0.
            pesos[c] = max(auc - 0.5, 0.0) if auc >= self.auc_minima else 0.0

        total = sum(pesos.values())
        if total <= 0:
            raise RuntimeError("Ninguna feature conserva señal tras descontar el sesgo de zona.")
        self.pesos = {c: w / total for c, w in pesos.items()}
        self.cols = [c for c, w in self.pesos.items() if w > 0]
        return self

    def transform(self, feat):
        out = np.zeros(len(feat))
        for c in self.cols:
            w = self.pesos[c]
            p_f = self.perfil[c]['p50_fraude']
            p_n = self.perfil[c]['p50_normal']
            delta = p_f - p_n
            if abs(delta) < 1e-9:
                continue
            out += w * np.clip((feat[c].values.astype(float) - p_n) / delta, 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)

    def resumen(self):
        filas = [f"    {c:24s} peso={self.pesos[c]:.3f}  AUC_centrada={self.auc_centrada[c]:.3f}"
                 for c in sorted(self.cols, key=lambda k: -self.pesos[k])]
        nulas = [c for c in self.auc_centrada if self.pesos.get(c, 0) <= 0]
        txt = chr(10).join(filas)
        if nulas:
            txt += chr(10) + "    peso 0 (sin señal al descontar zona): " + ", ".join(nulas)
        return txt


def construir_v3(feat_fraude, feat_normal, **kw):
    return FirmaFraudeV3(**kw).fit(feat_fraude, feat_normal)
