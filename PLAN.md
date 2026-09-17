# PLAN.md — Pipeline técnico de detección y priorización de hurto

> Este archivo describe **qué hay que implementar y por qué cada decisión**, orientado a los criterios de evaluación del hackathon: impacto económico, clasificación del tipo de hurto, dashboard accionable y viabilidad. No es historia de conversación; es la especificación de ejecución.

---

## Estado actual

- [x] CLAUDE.md con contexto del problema ✓
- [x] Datos explorados (columnas, tipos, distribuciones) ✓
- [x] Arquitectura del pipeline diseñada ✓
- [x] Algoritmo 1 (limpieza) — implementado en `run_pipeline.py` (Paso 1) ✓
- [x] Algoritmo 2 (scoring) — implementado en `run_pipeline.py` (Pasos 2-8), incluyendo voto LLM real (Gemini) en zona ambigua ✓
- [x] Dashboard — implementado como app React/Vite (`src/components/Dashboard.tsx`), no como HTML estático autocontenido (ver nota de desviación de arquitectura abajo) ✓
- [x] Entregables Fase 1 — generados en `entregables/` (Excel, CSV, PDF, guión de pitch); **pendiente regenerar tras los fixes aplicados el 2026-09-17** (ver abajo) ⚠️

**Nota de arquitectura**: el proyecto no sigue el layout modular `src/01_limpieza.py ... 07_ranking.py` de este documento; usa un `run_pipeline.py` monolítico + `sync_dashboard.py` + `generate_pdf.py`, y un dashboard Vite/React en vez de un `dashboard/index.html` autocontenido (desviación de CLAUDE.md, que pide HTML/CSS/JS autocontenido). Funciona, pero requiere `npm run build`/`dev` en vez de abrir un archivo HTML directo en una tablet.

**Auditoría 2026-09-17**: se corrigieron varios defectos encontrados por revisión de coherencia contra este documento: F5 no era leave-one-out real (data leakage), la fórmula de prioridad usaba un blend heurístico en vez de la probabilidad calibrada de B1, el voto LLM del Paso 2.6 no existía (import muerto), la explicación del Paso 2.9 era una plantilla pregenerada en vez de una llamada real bajo demanda, el corte dinámico se sobreescribía con un valor fijo, y el dashboard exportaba siempre el CSV completo ignorando filtros.

---

## Auditoría algorítmica 2026-09-17 (segunda pasada) — reescritura del núcleo de scoring

Se auditó el algoritmo end-to-end midiendo cada supuesto contra los datos, en vez de contra la especificación. Se encontraron **cinco defectos que invalidaban el ranking**. Todos corregidos y verificados; el pipeline fue reejecutado y los entregables regenerados.

### D1 — El ranking ordenaba por medidor apagado, no por hurto (crítico)

`probabilidad_fraude` era una heurística de pesos fijos: `0.5*anomaly + 0.3*caida_pct + 0.2*desv_sed`. El término `caida_pct` satura en 1.0 cuando el consumo reciente es cero, así que dominaba la suma.

**Evidencia medida**: el top 200 del ranking tenía 110 casos (55%) con caída total, mientras que en el histórico de fraude **confirmado** solo el 6.7% presenta ese patrón. El consumo anual mediano del top 35 era 279 kWh contra 823 de la población. Inspeccionados fila por fila, esos suministros tienen lecturas tipo `[0,0,1,1,1,0,0,1,0,0,0,0]` kWh en 12 meses: son locales vacíos o medidores de baja, **no hurtos**. Mandar una cuadrilla ahí no recupera nada.

**Corrección**: la heurística de pesos a mano se reemplazó por una **firma de fraude derivada del histórico confirmado**. Para cada feature se mide la separación real entre el perfil de fraude y el perfil típico del alimentador, y ese valor *es* el peso — una feature que no separa (`alternancia`: 0.500 en ambos) recibe peso 0 automáticamente. La señal dominante que emergió de los datos es la **volatilidad del consumo** (`kwhd_cv`, peso 0.604): mediana 0.639 en CLANDESTINA, 0.434 en MANIPULACIÓN, 0.215 en el alimentador. Tiene sentido físico: quien manipula el medidor genera un registro errático, mientras que un cliente honesto consume de forma estable.

### D2 — El PU-learning aprendía la zona geográfica, no el fraude (crítico)

Se implementó primero un clasificador Positive-Unlabeled (histórico=positivo vs. alimentador=no-etiquetado). **Resultó estructuralmente inválido y se descartó.**

**Evidencia medida**: los dos datasets son de zonas distintas de la concesión con distribuciones sistemáticamente diferentes — consumo anual mediano 1,814 kWh en el histórico contra 823 en el alimentador; potencia mediana 5.1 kW contra 11.1 kW. El clasificador alcanzaba AUC holdout **0.96 usando solo features de forma**, y cada feature por separado ya distinguía las zonas con AUC 0.55–0.72. Es decir, aprendía a reconocer el origen del dato, no la firma del fraude. Normalizar por percentil dentro de cada zona lo empeoró (**AUC 1.000**): las features discretas (`alternancia`, `meses_cero_finales`) producen escalones de percentil distintos en cada dataset y delatan la zona perfectamente. El ranking resultante ordenaba por tamaño de cliente (consumo mediano del top 100 = 17,250 kWh contra 822 de la población).

**Corrección**: se abandonó el enfoque PU. La comparación por distancia a perfil (D1) es robusta al cambio de zona porque contrasta cada suministro contra un perfil de referencia, sin que el modelo pueda usar el nivel absoluto como atajo de clasificación.

### D3 — El factor económico no discriminaba (crítico)

El regresor B2 se entrenaba solo con `feature_cols`, que **no incluye la potencia contratada**, y usaba el percentil 10 sobre una distribución de cola muy larga.

**Evidencia medida**: predecía un rango de 2,589–5,182 kWh cuando el recupero real va de 2,530 a 886,290 kWh. En la fórmula `prioridad = prob × recupero`, el recupero variaba solo **2×** mientras la probabilidad variaba 5.8× — la "priorización económica" en la práctica no priorizaba por economía, que es el criterio #1 del hackathon.

**Corrección**: se añadieron al regresor las variables de nivel que sí predicen recupero (consumo anual, kWh/día, potencia contratada) y se subió el cuantil de P10 a **P25** (sigue siendo conservador: 75% de los casos similares recuperaron más). Rango predicho actual: 2,570–17,332 kWh. Aquí sí es válido usar variables de nivel, porque el regresor estima una magnitud, no decide a qué dataset pertenece el caso.

### D4 — El tipo de hurto actuaba como interruptor del ranking

Se añadió `es_clandestina` como feature del regresor (el tipo correlaciona fuerte con el recupero real: mediana 10,330 kWh contra 3,417). Por ser binaria y muy predictiva, el árbol la usó como interruptor: el recupero mínimo de los CLANDESTINA quedó **por encima** del máximo de los MANIPULACIÓN, de modo que el tipo predicho —que es a su vez una predicción, no un dato observado— determinaba el orden por sí solo. El top 100 salía **99% CLANDESTINA** aunque en la población el tipo predicho es 83% MANIPULACIÓN.

**Evidencia medida**: en el histórico real los dos tipos se **solapan ampliamente** (P25 de CLANDESTINA = 7,873 kWh contra P75 de MANIPULACIÓN = 4,876 kWh). El escalón limpio era un artefacto del modelo, no un hecho del negocio.

**Corrección**: se quitó `es_clandestina` del regresor. También se eliminó el **bono operativo 1.25×** por ser doble conteo sobre la misma señal. El tipo se sigue reportando para decidir qué cuadrilla enviar (criterio #2 del hackathon), pero ya no infla el ranking. Resultado: top 100 con **71 CLANDESTINA / 29 MANIPULACIÓN**, cobertura real de ambos tipos.

### D5 — La inferencia de giro destruía información

La inferencia por correlación de curva estacional medía **31% de accuracy**, cuando responder siempre la clase mayoritaria ('doméstico', 90.7% del histórico) acertaría **90.7%**. No solo incumplía el umbral de ≥70% que exige el Paso 2.3.3, sino que era peor que el baseline trivial. Causa: el 90.7% del histórico es residencial, así que las curvas de las cuatro categorías son casi idénticas y la correlación no discrimina. Esos grupos son los que definen las cohortes de comparación del Isolation Forest, de modo que un agrupamiento malo genera falsas alarmas por comparar clientes heterogéneos.

**Corrección**: se segmenta por **potencia contratada** (bandas ≤4 / 4–8 / 8–13 / >13 kW), un dato objetivo declarado y presente en ambos datasets, sin inferencia ni riesgo de reidentificación. La potencia es lo que realmente define qué consumo es "normal" para un suministro. La validación del Paso 2.3.3 se conserva y es la que dispara automáticamente este reemplazo cuando la accuracy no supera el umbral.

### Filtro nuevo — suministros sin servicio activo

Se marcan como `sin_servicio` los suministros con consumo anual < 120 kWh (~10 kWh/mes) **o** con menos del 25% de meses activos, y se empujan al fondo del ranking (`prioridad = -1`). En el histórico de fraude confirmado solo el 6.3% tiene un perfil así, de modo que inspeccionarlos es casi siempre un falso positivo sin recupero posible. Siguen presentes en `ranking_completo.csv` con la bandera expuesta, para que la empresa evalúe darlos de baja o revisar el medidor — pero no compiten por el presupuesto de inspecciones. Marcados: 3,437 de 14,951.

### Resultado medido tras las correcciones

| Métrica (top 100 vs. población) | Antes | Después |
|---|---|---|
| Consumo anual mediano | 279 kWh (**0.34×** la población) | 395 kWh, sin medidores muertos |
| Suministros con < 120 kWh/año en el top | 34 de 100 | **0** |
| Firma de fraude vs. población | — (no existía) | **1.74×** |
| Recupero mediano vs. población | ~1.2× | **2.12×** |
| Rango del regresor de recupero | 2,589–5,182 kWh (2× de spread) | 2,570–17,332 kWh |
| Mezcla de tipos en el top 100 | 34 CLAND / 1 MANIP | **71 CLAND / 29 MANIP** |
| Recupero total proyectado del top | ~S/ 80,000 | **S/ 456,167** |
| SEDs cubiertas en el top 100 | — | 48 |

### D6 — La sugerencia operativa se decidía por la variable equivocada

La columna "Sugerencia Operativa" del dashboard elegía entre *Revisar Medidor* y *Revisar Red / Acometida* según `alta_incertidumbre`, no según `tipo_predicho`. Son conceptos distintos: la bandera significa "las señales discrepan", mientras que la acción de cuadrilla depende del **tipo de vulneración**. El resultado era una contradicción visible en pantalla: una fila con el badge **CLANDESTINA** (derivación en la red) recomendaba *Revisar Medidor*. Afectaba a los 71 casos clandestinos del top 100, justo el criterio #2 del hackathon (clasificar el tipo define qué cuadrilla enviar).

**Corrección**: la acción ahora deriva de `tipo_predicho` — CLANDESTINA → red/acometida, MANIPULACIÓN → medidor. La leyenda de la Guía Operativa ya describía esta lógica por tipo, lo que confirma que era la intención original y que el defecto estaba en la implementación.

### D7 — La bandera de alta incertidumbre nunca se activaba

`alta_incertidumbre` se calculaba como `mask_ambigua & (abs(prob_llm - prob_knn) > 0.20)`. Sin `GEMINI_API_KEY`, `prob_llm` es NaN para todos los casos, y en NumPy `NaN > 0.20` evalúa a False. Resultado: la bandera era False en los **14,951 suministros, sin excepción**, y ninguna advertencia llegaba nunca al inspector. Como el free tier de Gemini solo cubre ~15 llamadas diarias, ese es el estado normal de ejecución, no un caso excepcional.

**Corrección**: la discrepancia se mide ahora entre las dos señales que siempre existen — `firma_fraude` y `anomaly_score` — y el voto LLM/KNN se suma como fuente adicional cuando está disponible. La comparación se hace en **espacio de percentiles**, no por diferencia absoluta: las dos señales viven en escalas distintas (mediana de la diferencia bruta = 0.214), así que un umbral de 0.20 sobre valores crudos marcaba al 54% de la población, inútil como alerta. Comparando el percentil que cada señal asigna al mismo suministro, un umbral de 0.40 marca al **15% de la población y solo 1 caso del top 100** — coherente con que el top esté formado justamente por casos donde ambas señales coinciden.

### D8 — El score se mostraba como un porcentaje que prometía el doble del techo teórico

La columna mostraba "55%" bajo el rótulo *Probabilidad Estimada de Hurto*. El valor es en realidad un índice de similitud al perfil de fraude confirmado (`0.75 × firma_fraude + 0.25 × anomaly_score`), sin calibración estadística. Renombrarlo a "Índice" fue necesario pero insuficiente: mientras se muestre como `55/100`, cualquier lector —incluido el jurado— lo interpreta como tasa de acierto.

**La aritmética que lo invalida**: hay 31 hurtos entre 14,951 suministros (0.207%). Si se inspeccionan 100 suministros y el modelo fuera **perfecto** —los 31 hurtos en las primeras 31 posiciones— la tasa de acierto sería **31%**. Ese es el techo absoluto del problema, y explica por qué la meta del hackathon (25-30%) está fijada justo debajo. Mostrar "55" sugiere una efectividad del orden del doble de lo que ningún modelo puede alcanzar aquí.

**Corrección**: se reporta un **nivel categórico** en vez del número, con cortes en los **quintiles de la firma medida sobre los 4,659 fraudes confirmados**. El nivel significa literalmente "a qué altura de los hurtos reales llega este suministro":

| Nivel | Corte | Lectura | Top 100 | Población |
|---|---|---|---|---|
| CRÍTICO | ≥ 0.938 | Supera al 80% de los hurtos confirmados | 20 | 248 |
| MUY ALTO | ≥ 0.686 | Supera al 60% | 11 | 614 |
| ALTO | ≥ 0.604 | Supera al 40% | 64 | 1,872 |
| MODERADO | ≥ 0.409 | En rango de fraude, parte baja (sobre el 20%) | 3 | 1,036 |
| BAJO | < 0.409 | Por debajo del 20% de los hurtos | 2 | 7,744 |
| SIN SERVICIO | — | Consumo ~nulo: no inspeccionable | 0 | 3,437 |

**Sobre la nomenclatura**: los tramos intermedios se llaman MODERADO/ALTO/MUY ALTO y no "MUY BAJO/BAJO/MEDIO". Un suministro que puntúa al nivel del percentil 30 de los hurtos verificados **no es poco sospechoso** — está dentro del rango donde viven los fraudes reales, solo que en su parte baja. Llamarlo "BAJO" invitaría al error inverso: descartar casos legítimamente sospechosos. Solo el quintil que queda por debajo del 20% de los fraudes lleva ese nombre.

**Defecto de coherencia corregido de paso**: los umbrales se derivaban de `firma_fraude` pero se aplicaban sobre `probabilidad_final`, que es otra magnitud (mezcla 75/25 con el `anomaly_score`). Ahora la clasificación se hace sobre `firma_fraude`, la única cantidad calculada igual en ambos datasets — el `anomaly_score` no tiene equivalente en el histórico porque el Isolation Forest solo se entrena sobre el alimentador, así que compararlo contra cuantiles del histórico desplazaba las bandas.

El nivel se calcula **una sola vez en el pipeline** y viaja como columna `nivel_sospecha` a CSV, Excel y dashboard, en vez de replicar los umbrales en Python y TypeScript por separado. El ranking y los `CUENTA_ID` no cambian: esto es presentación, no scoring. El Excel además incorpora `ACCIÓN_CUADRILLA` derivada del tipo (ver D6).

### D9 — El mapa de transformadores ordenaba al revés para despachar

El mapa de calor coloreaba cada transformador por el **promedio** de prioridad de sus suministros, con umbrales fijos (2400/2000/1500) heredados de la escala de scoring anterior. Dos consecuencias:

- **Umbrales muertos**: con la escala actual, 73 de 91 transformadores caían en el mismo color y **ninguno** en el más bajo, pese a que la leyenda mostraba cuatro niveles.
- **Orden invertido para el negocio**: el transformador 1482, con 4 suministros y S/ 16,992 recuperables, salía en rojo; el 0515, con 39 suministros y **S/ 144,555** (8.5× más), salía en naranja. La cuadrilla se desplaza a una zona y revisa varios medidores ahí, así que lo que decide la ruta es el dinero total concentrado, no el promedio por suministro.

**Corrección**: se colorea y ordena por **recupero total del transformador**, con cortes en los **cuartiles de la propia distribución** en vez de números fijos. Los cuartiles son autocalibrantes: el mapa sigue repartido aunque cambie la escala del score, que es justo lo que rompió la versión anterior.

**Corrección adicional — el mapa no cuadraba con la lista**: el heatmap se calculaba sobre los 1,000 suministros analizados, mientras la lista de despacho tiene 100. Los conteos por transformador sumaban 1,000, de modo que el supervisor leía "39 sospechosos" en un transformador del que en realidad solo iba a visitar 11.

**Un solo universo en todo el dashboard**: se eliminó el botón "Ver Ranking Analizado (Top 1,000)" y con él el tier `TODOS`. El dashboard muestra únicamente la lista de despacho, así que el mapa, la tabla, los KPIs y el Excel entregable hablan siempre del mismo conjunto: **48 transformadores, 100 suministros, S/ 456,168**. El subtítulo del mapa lo declara para que sea verificable sumando las celdas. El ranking completo de los 14,951 sigue disponible en `entregables/ranking_completo.csv` para análisis, que es donde corresponde — no en la pantalla del supervisor.

Efecto secundario: `sync_dashboard.py` ahora exporta exactamente `CORTE_OPERATIVO` registros (misma variable de entorno que el pipeline, de modo que no pueden desincronizarse) en vez de 1,000. `src/data.ts` bajó de 674 KB a 68 KB y el bundle de 848 KB a 325 KB (85 KB comprimido) — relevante porque el dashboard está pensado para abrirse desde una tablet en campo.

### D10 — Lenguaje del dashboard: de analista a técnico de campo

El dashboard hablaba en vocabulario interno del modelo. Cambios de presentación, sin efecto en el scoring:

- **Columna "Zona / Prioridad" eliminada**: dentro de la Zona Crítica todas las filas mostraban el mismo badge. El número de ranking ya va en rojo cuando es crítico.
- **"SED" → "Subestación" / "Transformador"**: el acrónimo no dice nada a quien lee el tablero. El mapa pasó de *"Matriz de Concentración por Transformador (SED)"* a *"¿En qué transformadores se está perdiendo la energía?"*, y cada celda rotula `Transf. 0515 · 39 sospechosos` en vez de `0515 / 39 sum.`.
- **Leyenda "Cómo leer esta tabla"**: explica qué es una vulneración CLANDESTINA (*"hay un cable conectado directo a la red, que no pasa por el medidor"*) frente a una MANIPULACIÓN (*"el cable entra normal, pero el medidor fue intervenido"*), con el badge de acción operativa que le corresponde a cada una, y qué significa cada nivel de sospecha (*"CRÍTICO: se parece más que 8 de cada 10 robos ya comprobados"*), aclarando que no es una tasa de acierto.
- **Diagnóstico en lenguaje natural**: antes mostraba `Desviación relativa SED: -0.24, Caída de consumo: 100.0%, Factor de uso: 0.000`. Ahora redacta el hallazgo como un informe de campo: *"Tiene 12 kW contratados. Venía registrando unos 20 kWh al mes y desde setiembre marca cero: lleva 4 meses seguidos sin registrar consumo. Aun antes de apagarse ya registraba 24% menos que el promedio de su subestación."* El promedio mensual se calcula sobre los meses activos (dividir entre 12 mezclaba los meses en cero y subestimaba el consumo real), y cuando el medidor lleva meses en cero la comparación con vecinos se redacta en pasado — decir *"registra 123% más que sus vecinos"* junto a *"lleva 3 meses en cero"* sonaba contradictorio, cuando es precisamente el argumento más fuerte.

### D11 — Contexto de la red y preguntas operativas del brief sin responder

El dashboard mostraba el ranking pero no decía **qué red** se está analizando, y dejaba sin responder dos cosas que el brief del hackathon pide de forma explícita: *"¿Cada cuánto tiempo se actualizarán las alertas?"* y la recomendación de incluir un análisis de retorno de inversión.

**Agregado**: franja de contexto con la composición del alimentador y tres fichas operativas.

| Dato | Valor |
|---|---|
| Alimentador | ALIM_311 |
| Suministros | 14,951 |
| Transformadores (SED) | 95 |
| Zonas | 29 |
| Llaves de corte | 7 |
| Tarifa | BT5 B (baja tensión) |

- **Frecuencia de actualización**: mensual, al cerrar el ciclo de facturación, porque esa es la granularidad de la lectura del medidor (el dataset son 12 pares días/consumo). Con telemetría en transformadores podría pasar a diaria — es la respuesta honesta, no un "tiempo real" que la fuente de datos no soporta.
- **Retorno por inspección**: S/ 4,562 por visita despachada (S/ 456,168 en 100 inspecciones, escenario conservador P25). Se presenta el lado del ingreso y se indica contrastarlo contra el costo de cuadrilla, que es dato de EQUANS: inventar un costo para cerrar el ROI sería fabricar la conclusión.
- **Esfuerzo que ahorra**: se inspecciona el 0.7% de la red, concentrado en 48 de 95 transformadores.

Las cifras se derivan del dataset crudo en `sync_dashboard.py` y viajan a `data.ts` como `resumenAlimentador`, no escritas a mano. **Nota sobre el brief**: la sesión verbal mencionaba 15,147 suministros, 112 paquetes de subestaciones y 4,805 casos CNR; los archivos efectivamente entregados tienen **14,951**, **95 SEDs** y **4,659 casos**. Manda el archivo, y por eso estos números se calculan y no se transcriben.

También se corrigió el rótulo de las celdas del mapa: decía "11 sospechosos" y ahora dice "11 suministros sospechosos", para que se entienda que se cuentan suministros y no incidencias.

### Corte operativo unificado en 100

El corte pasó de 35 a **100** y ahora es un solo valor en todo el sistema (`CORTE_OPERATIVO` en `run_pipeline.py`, propagado a `src/data.ts` por `sync_dashboard.py`). El dashboard se simplificó de 4 zonas a 2 (Crítica / Limpia): las zonas "Secundaria" (#36–200) y "Monitoreo" (#201–1000) cortaban por **posición fija en el ranking**, no por umbral de evidencia, de modo que el suministro #201 aparecía como sospechoso solo por haber quedado después del corte 200. Cien es defendible ante el jurado como capacidad de despacho (0.67% del alimentador) y no sobreajusta al número real de hurtos conocido (31).

---

## Arquitectura general

```
DATA_ALIMENTADOR.csv  ──┐
                         ├──► ALGORITMO 1 (limpieza) ──► data limpia ──► ALGORITMO 2 (scoring) ──► ranking.csv
DATA_HISTORICO_CNR.csv ─┘                                                     │
                                                                               ├── Excel entregable
                                                                               └── Dashboard HTML
```

El pipeline corre localmente en Python. Los archivos CSV son la fuente de verdad local (Supabase como respaldo).

---

## ALGORITMO 1 — Limpieza mínima de datos

### Objetivo y restricción de diseño

Eliminar únicamente errores físicamente imposibles, sin tocar ningún patrón de consumo real. **Un cambio brusco de consumo puede ser exactamente la señal de fraude** (el momento en que alguien empezó a robar). Filtrar eso como "ruido" destruiría la señal más valiosa.

### Paso 1.1 — Carga de datos

```python
# Separador: ';' | Encoding: 'utf-8-sig' (BOM)
# Columnas de días facturados tienen '\n' en el nombre → normalizar al leer
df_ali  = pd.read_csv('DATA_ALIMENTADOR.csv',   sep=';', encoding='utf-8-sig')
df_hist = pd.read_csv('DATA_HISTORICO_CNR.csv', sep=';', encoding='utf-8-sig')

# Renombrar columnas: 'DIAS FACTURADO\nENERO' → 'DIAS_ENERO', 'CONSUMO ENERO' → 'CONS_ENERO', etc.
MESES = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO',
         'JULIO','AGOSTO','SETIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
```

### Paso 1.2 — Única regla de limpieza: imposibilidad física

**Regla**: si `DIAS_[MES] > 400` para algún mes, ese mes se marca como corrupto.

**Por qué 400 y no 31**: El diccionario oficial dice que valores entre 33 y ~400 son **periodos acumulados por falta de lectura o refacturaciones, y son comportamiento real del negocio**, no errores. Solo > 400 es físicamente imposible (más de un año en un solo registro). Usar una regla relativa al historial del cliente (como "3× la mediana propia") confundiría un salto real de consumo con un error de sistema.

```python
# Para cada mes, si dias > 400 → marcar como NaN ese par (dias + consumo)
for mes in MESES:
    mask = df_ali[f'DIAS_{mes}'] > 400
    df_ali.loc[mask, [f'DIAS_{mes}', f'CONS_{mes}']] = np.nan
# Idem para df_hist
```

### Paso 1.3 — Anulación puntual, sin descartar clientes

Un mes marcado como corrupto se excluye únicamente del cálculo de features de ese mes específico. Los demás meses del mismo cliente se usan con normalidad. **Ningún cliente sale del pipeline** por tener meses corruptos — todos los 14,951 reciben un score.

### Paso 1.4 — Limpieza cosmética

Redondear `POTENCIA USUARIO CONTRATADO` a 2 decimales (errores de punto flotante en el archivo original, tipo `4.3000000000000007`). No afecta ningún cálculo.

### Salida del Algoritmo 1

Los mismos **14,951** suministros del alimentador y **4,659** del histórico, intactos en cantidad, solo con meses físicamente imposibles anulados. Ambos DataFrames limpios pasan al Algoritmo 2.

---

## ALGORITMO 2 — Scoring y priorización

### Paso 2.1 — Entrada

DataFrames limpios del Algoritmo 1: `df_hist` (con etiqueta `TIPIFICACIÓN` y `PROYECCIÓN DE RECUPERO (KWH)`) y `df_ali` (sin etiqueta).

---

### Paso 2.2 — Feature engineering

**Misma función aplicada a ambos datasets.** Cada feature se calcula por suministro usando sus 12 meses (excluyendo los marcados como NaN).

#### F1 — kWh/día por mes

```
kwhd_mes = CONS_mes / DIAS_mes     (para cada mes con dias > 0 y no NaN)
```

Normaliza meses de distinta duración. Base de casi todos los demás features.

#### F2 — Caída porcentual de consumo

```
base    = media(kwhd de los primeros 3 meses válidos del cliente)
reciente = media(kwhd de los últimos 3 meses válidos)
caida_pct = (base - reciente) / base     si base > 0, else NaN
```

**Por qué importa**: una caída brusca y sostenida puede marcar el momento de inicio del hurto. El histórico CNR muestra este patrón en clientes con manipulación de medidor. No se filtra como "ruido" en el Paso 1.2 precisamente para preservar esta señal.

#### F3 — Alternancia de consumo

```
alternancia = número de veces que kwhd[t] > kwhd[t-1] alternando con kwhd[t] < kwhd[t-1]
             dividido entre (meses_válidos - 1)
```

Un patrón muy alternante (sube-baja-sube-baja sin tendencia) puede indicar manipulación intermitente del medidor. Un cliente normal tiene tendencia más suave.

#### F4 — Factor de uso

```
consumo_max_teorico_mes = kW_contratados × 24h × dias_facturados_mes
factor_uso_mes = CONS_mes / consumo_max_teorico_mes
factor_uso = mediana(factor_uso_mes sobre meses válidos)
```

Un factor de uso anormalmente bajo y **sostenido** es sospechoso, especialmente en clientes industriales o comerciales con alta potencia contratada: están pagando por capacidad que supuestamente no usan, pero quizás sí la usan y no la registran.

**Por qué mediana y no media**: la mediana es robusta a un único mes atípico (p.ej. un mes con lectura acumulada residual); la media dejaría que ese mes domine el resultado anual.

#### F5 — Desviación respecto a vecinos de la misma SED

```python
# Para cada cliente i de la SED s:
mediana_sed_sin_i = mediana(kwhd_promedio de todos los clientes de SED s, excepto i)
desv_sed = (kwhd_promedio_i - mediana_sed_sin_i) / (mediana_sed_sin_i + epsilon)
```

**Por qué excluir al propio cliente del cálculo de su mediana (leave-one-out)**: si un cliente con consumo muy atípico se incluyera en la mediana de su grupo, desplazaría esa mediana hacia él y se vería menos atípico de lo que realmente es. Este sesgo se llama data leakage.

**Suavizado para grupos pequeños**: si una SED tiene < 5 clientes, la mediana de ese grupo es inestable y puede generar falsas alarmas. En ese caso:
```
desv_sed_suavizada = w × desv_sed_local + (1 - w) × desv_alimentador_global
# w = n_sed / (n_sed + 10)   → w crece con el tamaño del grupo
```

**También calcular F5 a nivel LLAVE_ID** (subnivel de la SED) para suministros con suficientes vecinos en la misma llave.

---

### Paso 2.3 — Inferencia de giro comercial para el alimentador

El histórico tiene `GIRO COMERCIAL` (71 categorías únicas con mezcla de niveles de detalle: "Viviendas de un piso", "Panaderias", "Talleres de metalmecanica", etc.). El alimentador **no tiene** esta columna. El giro importa porque define el patrón estacional esperado: un colegio cae en julio-agosto (vacaciones), una panadería no.

#### Paso 2.3.1 — Agrupar los 71 giros en categorías manejables

Antes de inferir, agrupar manualmente (o con clustering) los 71 giros del histórico en ~8-12 macro-categorías:

| Macro-categoría | Ejemplos |
|---|---|
| Residencial | Viviendas de un piso, Viviendas de dos pisos, Vivienda precaria |
| Comercio minorista | Bodegas, Farmacias, Ferreterías |
| Alimentación | Panaderías, Restaurantes, Mercados |
| Industria/taller | Talleres metalmecanica, Curtiembres, Plastiqueros |
| Entretenimiento alto consumo | Tragamonedas, Discotecas, Casinos |
| Servicios | Peluquerías, Lavanderías, Imprentas |
| Educación/salud | Clínicas, Colegios |
| Otros/especial | Campos feriales, Hoteles, Instituciones vecinales |

#### Paso 2.3.2 — Calcular curva estacional promedio por macro-categoría

Para cada macro-categoría, calcular la curva de consumo normalizada mensual (shape, no nivel):
```
curva_cat[mes] = mediana(kwhd_mes / kwhd_anual_promedio) sobre todos los clientes de esa categoría
```

#### Paso 2.3.3 — Validar la inferencia antes de usarla en el alimentador

**Por qué validar**: si la inferencia es mala, los grupos serán incorrectos y el Isolation Forest del Paso 2.4 comparará clientes heterogéneos entre sí, generando falsas alarmas.

```python
# Esconder el giro real del histórico
# Inferir el giro para cada caso del histórico usando solo su curva de consumo
# Medir accuracy vs giro real
accuracy = correcto / total
```

- Si `accuracy >= 70%` → usar las macro-categorías tal cual
- Si `accuracy < 70%` → colapsar a 4-5 super-categorías más amplias (residencial / comercial / industrial / entretenimiento) hasta alcanzar ≥ 70%

#### Paso 2.3.4 — Aplicar la inferencia al alimentador

Asignar a cada uno de los 14,951 suministros la macro-categoría cuya curva estacional más se parece a su propia curva de consumo (correlación de Pearson o distancia coseno).

---

### Paso 2.4 — Dos modelos en paralelo

#### Modelo A — Isolation Forest (no supervisado, sobre el alimentador)

**Por qué Isolation Forest**: no necesita etiquetas de fraude. Aprende qué es "normal" para cada grupo y asigna un score de anomalía a cada cliente. Complementa al LightGBM cuando hay incertidumbre.

**Por qué correrlo por grupo de giro inferido y no sobre los 14,951 mezclados**: comparar una panadería contra viviendas residenciales daría falsas alarmas por diferencias de tipo de negocio, no por fraude. Cada grupo tiene su propio "normal".

```python
from sklearn.ensemble import IsolationForest

for categoria in macro_categorias:
    subset = df_ali[df_ali['giro_inferido'] == categoria]
    features = ['caida_pct', 'alternancia', 'factor_uso', 'desv_sed', ...]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(subset[features])
    df_ali.loc[subset.index, 'anomaly_score'] = iso.decision_function(subset[features])
    # score más negativo → más anómalo
```

#### Modelo B — LightGBM (supervisado, entrenado en el histórico)

**Por qué LightGBM**: maneja bien variables mixtas (numéricas + categóricas), es robusto con datasets pequeños-medianos, y la cola larga del recupero no lo desestabiliza si se trabaja bien.

**Dos sub-modelos:**

**B1 — Clasificador de tipo de hurto (MANIPULACIÓN vs CLANDESTINA)**
```python
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

X_hist = df_hist[features]   # mismas features del paso 2.2
y_type = df_hist['TIPIFICACIÓN'].map({'MANIPULACIÓN': 0, 'CLANDESTINA': 1})

clf = lgb.LGBMClassifier(random_state=42)
# Calibración de probabilidades: la salida cruda del clasificador no es una probabilidad real
# CalibratedClassifierCV ajusta la salida para que "70%" signifique que ~70% de esos casos son fraude
clf_calibrated = CalibratedClassifierCV(clf, cv=5, method='isotonic')
clf_calibrated.fit(X_hist, y_type)
```

**Por qué calibrar**: los scores crudos de LightGBM son ordinales (ordenan casos) pero no son probabilidades reales. La calibración los convierte en frecuencias estadísticas confiables, lo que permite usar el score directamente en la fórmula de prioridad.

**B2 — Regresor de recupero esperado (kWh)**
```python
# Trabajar en log-escala para dominar la cola larga
# (recupero máximo ~886,290 kWh, el log aplana esa cola)
y_rec = np.log1p(df_hist['PROYECCIÓN DE RECUPERO (KWH)'])

# Quantile regression con percentil 10 (escenario conservador)
reg_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, random_state=42)
reg_q10.fit(X_hist, y_rec)

# Al predecir: volver a escala original
recupero_p10_kwh = np.expm1(reg_q10.predict(X_ali))
```

**Por qué un percentil bajo y no la media**: la cola larga del recupero (algunos casos con 886,290 kWh) es real pero rara. Usar la media haría que casos parecidos a esos outliers se inflen artificialmente en el ranking. Un cuantil bajo es el escenario conservador: si el modelo dice "5,000 kWh", se esperan al menos 5,000 kWh de recupero.

**Corrección 2026-09-17 — P10 → P25**: el P10 sobre esta distribución empujaba casi todas las predicciones contra el mínimo del histórico (rango resultante 2,589–5,182 kWh sobre un real de 2,530–886,290), destruyendo el poder de discriminación económica. El P25 sigue siendo conservador (75% de los casos similares recuperaron más) pero conserva el ordenamiento. Ver D3 de la auditoría. **El regresor también recibe ahora variables de nivel** (consumo anual, kWh/día, potencia contratada), que son las que realmente predicen la magnitud del recupero, y **no recibe el tipo de hurto** (ver D4).

**Por qué log-escala en el regresor**: sin log, el gradiente boosting intenta ajustar los outliers extremos y distorsiona las predicciones para el resto. Con log, la distribución es más simétrica y el modelo aprende mejor el comportamiento de los casos "normales" de fraude.

---

### Paso 2.5 — Clasificación de confianza (tres zonas)

Con la probabilidad calibrada del B1 y el anomaly_score del Isolation Forest, cada suministro cae en una zona:

| Zona | Condición | Acción |
|---|---|---|
| **Clara-alta** | prob_calibrada ≥ umbral_alto | Sigue directo al ranking |
| **Clara-baja** | prob_calibrada ≤ umbral_bajo | Sigue directo al ranking |
| **Ambigua/rara** | (a) prob cerca del umbral ± margen, O (b) anomaly_score muy alto aunque prob sea baja | Pasa al Paso 2.6 |

- Umbral inicial sugerido: `umbral_alto = 0.65`, `umbral_bajo = 0.35`, margen `± 0.15`
- La condición (b) captura casos que son raros de manera nunca vista antes, aunque el clasificador no esté seguro de que sea fraude → vale la pena la revisión adicional

---

### Paso 2.6 — Doble voto para la zona ambigua

Solo los casos en zona ambigua pasan por aquí. Los casos claros no.

**Por qué doble voto solo en zona ambigua**: el LLM tiene costo por token. Los casos claros (alta o baja probabilidad) ya tienen suficiente evidencia; gastar tokens en ellos no agrega valor. La zona ambigua es donde el ML genuinamente no puede distinguir, y es ahí donde el razonamiento de negocio (LLM) y la memoria histórica (KNN) agregan valor marginal real.

#### Voto 1 — LLM (Google Gemini API) — Rol 1

```
Prompt incluye:
- Giro comercial inferido del caso
- Tabla de patrones estacionales reales del histórico para ese giro (lo que observamos en los datos)
- Resumen de sus features: caída X%, alternancia Y, factor de uso Z
- Por qué quedó en zona ambigua (¿frontera de decisión? ¿out-of-distribution?)
- Pregunta: "¿Qué probabilidad asignarías de que este suministro esté cometiendo hurto?"
- Restricción explícita: si el giro no está en la tabla de referencia, responder "sin referencia" (no improvisar)
```

El LLM devuelve una probabilidad entre 0 y 1 + breve justificación. **No toca el recupero. No reemplaza al ML.**

#### Voto 2 — KNN sobre el histórico

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
knn.fit(X_hist, y_type)   # mismas features

# Para cada caso ambiguo del alimentador:
proba_knn = knn.predict_proba(X_caso)[0][1]   # proporción de vecinos que fueron CLANDESTINA
```

**Limitación reconocida**: el KNN hereda la misma limitación del histórico (otra zona de la concesión). Se documenta pero no se intenta resolver.

#### Regla de combinación

```python
diferencia = abs(prob_llm - prob_knn)

if diferencia <= 0.20:
    prob_ajustada = (prob_llm + prob_knn) / 2
    alta_incertidumbre = False
else:
    prob_ajustada = (prob_llm + prob_knn) / 2   # se promedia igual
    alta_incertidumbre = True   # se muestra bandera en el dashboard
```

**Por qué 20 puntos porcentuales como umbral de discrepancia**: es el punto de partida, calibrable con datos de campo reales. Por encima de ese umbral, las dos fuentes de opinión no convergen y conviene que el inspector humano lo sepa.

**Desviación de implementación — presupuesto de llamadas LLM**: el free tier de los modelos Gemini de texto (2.5 Flash, 2.5 Flash Lite, 3 Flash) tiene una cuota de 20 requests/día por modelo, muy por debajo del número típico de casos en zona ambigua. En vez de llamar al LLM para todos los casos ambiguos (lo que agota la cuota casi de inmediato y falla en cadena), el pipeline ordena la zona ambigua por `recupero_p10_soles` y solo consulta a Gemini para los N casos (configurable, default 15 — con margen de seguridad bajo el tope de 20/día) con mayor recupero potencial — son los que más impactan el ranking económico final si el LLM corrige su clasificación. El resto de la zona ambigua usa solo el voto de KNN, como ya estaba diseñado para el caso de "sin referencia". Configurable vía las variables de entorno `GEMINI_MAX_LLAMADAS` (presupuesto de llamadas) y `GEMINI_MODEL` (modelo a usar, default `gemini-2.5-flash-lite`; la cuota diaria es independiente por modelo, así que cambiar de modelo da una cuota fresca).

**Desviación de implementación — probabilidad de fraude separada de probabilidad de tipo**: se detectó empíricamente que `df_hist` contiene solo casos de fraude confirmado (no tiene clientes normales/no-fraude), así que B1 (`prob_clandestina`) solo puede aprender "dado que es fraude, ¿de qué tipo es?" — nunca "¿es fraude o no?". Usar `prob_clandestina` directamente como la "Probabilidad_calibrada" de la fórmula de prioridad (Paso 2.7) hacía que el ranking colapsara: el top completo salía 100% CLANDESTINA, porque cualquier caso con alta certeza de ser MANIPULACIÓN tiene `prob_clandestina` baja por construcción, sin importar cuán sospechoso sea en general. Por eso se introdujo `probabilidad_fraude` (basada en `anomaly_score` del Isolation Forest — la única señal no supervisada, comparada contra vecinos normales de la SED — combinada con caída de consumo y desviación de SED), que alimenta el ranking económico. `prob_clandestina` se usa exclusivamente para decidir `tipo_predicho`, nunca para ordenar el ranking.

**Desviación de implementación — tope de casos en zona ambigua**: con el margen ±0.15 sobre los umbrales 0.35/0.65, la zona ambigua capturaba ~69% de los 14,951 suministros (la distribución de `probabilidad_fraude` tiene su mediana cerca de esa banda). El doble voto LLM+KNN no es viable sobre miles de casos. Se acotó la zona ambigua a los `ZONA_AMBIGUA_TOPE` (default 300) casos más cercanos al verdadero punto de indecisión (probabilidad_fraude = 0.50) dentro de la zona ambigua cruda — los que el modelo genuinamente más duda — en vez de aplicar el doble voto a toda la banda amplia.

---

### Paso 2.7 — Fórmula de fusión final

Para **todos los 14,951 suministros**, sin excepción:

```
Prioridad = Probabilidad_fraude × Recupero_P25_soles

donde:
  Probabilidad_fraude    = 0.75 × firma_fraude + 0.25 × anomaly_score
                           firma_fraude  = distancia al perfil de fraude confirmado del
                                           histórico, con pesos por feature derivados de la
                                           separación medida (no escritos a mano). Ver D1/D2
                                           de la auditoría 2026-09-17.
                           anomaly_score = Isolation Forest por cohorte de potencia (aporta
                                           la perspectiva no supervisada: qué tan raro es
                                           este cliente frente a sus pares locales).
                           Ajustada 50/50 con el doble voto (Paso 2.6) SOLO si el caso cayó
                           en zona ambigua (Paso 2.5).
  Recupero_P25_soles     = recupero_p25_kwh × 0.667    (tarifa referencial del hackathon)
                           recupero_p25_kwh tiene un piso operativo de 300 kWh

Exclusiones (prioridad = -1, permanecen en ranking_completo.csv con su bandera):
  sin_servicio          = consumo anual < 120 kWh o < 25% de meses activos.
                          No es hurto: medidor de baja o local vacío, sin recupero posible.
  datos_insuficientes   = el regresor no diferencia su recupero (colisión de hoja del árbol).
```

**Sin bono por tipo**: el bono operativo 1.25× para CLANDESTINA se eliminó por doble conteo. Ver D4 de la auditoría.

**Por qué esta fórmula y no solo probabilidad**:
- Un caso con 90% de probabilidad pero recupero chico (vivienda pequeña) puede quedar debajo de un caso con 50% pero recupero enorme (industrial).
- Esto es correcto: la empresa quiere **maximizar dinero recuperado por cuadrilla despachada**.
- El dato operativo que lo confirma: 195 clientes industriales → 27.1 GWh recuperados vs. 3,559 clientes residenciales → 17 GWh. El industrial vale 18× más por inspección.

**Por qué se eliminó el bono operativo de 1.25× para CLANDESTINA** (estaba vigente hasta la auditoría 2026-09-17): era doble conteo. La mayor rentabilidad de CLANDESTINA ya estaba capturada por el regresor de recupero, y multiplicar otra vez por el tipo hacía que el top del ranking saliera 99-100% CLANDESTINA, dejando fuera del despacho todos los casos de MANIPULACIÓN de alto recupero. Ver D4 de la auditoría. La urgencia operativa distinta de una intervención en red se comunica al jefe de campo mediante la columna de tipo predicho y la sugerencia operativa del dashboard, no inflando el score.

**Por qué el piso de 300 kWh en recupero_p25_kwh**: evita que un caso con una predicción del regresor casi nula (irrelevante operativamente) reciba una prioridad artificialmente aplastada por ruido del modelo en vez de por evidencia real de bajo recupero.

**Por qué no se usa la salida calibrada de B1 como Probabilidad_calibrada**: `df_hist` (el histórico CNR) contiene **solo** casos de fraude confirmado — no tiene clientes normales/no-fraude. Por eso B1 (LightGBM + CalibratedClassifierCV) solo puede aprender "dado que es fraude, ¿de qué tipo es?", nunca "¿es fraude o no?". Usar su salida calibrada (`prob_clandestina`) como la probabilidad que alimenta el ranking hacía que el top completo saliera 100% CLANDESTINA: cualquier caso con alta certeza de ser MANIPULACIÓN tiene `prob_clandestina` baja por construcción matemática, sin importar cuán sospechoso sea en general — esto se comprobó empíricamente antes de corregirlo. La solución fue separar dos probabilidades: `probabilidad_fraude` (basada en `anomaly_score`, la única señal no supervisada del pipeline, que sí compara cada cliente contra vecinos normales de su SED) alimenta el ranking; `prob_clandestina` de B1 se usa exclusivamente para fijar `tipo_predicho`.

**Sesgo observado hacia CLANDESTINA en el top del ranking — explicación de dominio**: incluso con la separación anterior, el top del ranking sigue concentrado mayormente en CLANDESTINA (ej. 30 de 31 en una corrida de referencia). La razón no es un artefacto del código: en la población completa, `probabilidad_fraude` promedia 0.37 para los casos etiquetados CLANDESTINA vs. 0.24 para MANIPULACIÓN — una diferencia real y consistente con el glosario de CLAUDE.md: una conexión clandestina (deriva directa de la red, sin pasar por el medidor) produce caídas de consumo registrado más abruptas y mayor desviación respecto a los vecinos de la SED que una manipulación de medidor (alteración más sutil del registro). El anomaly_score y las señales de caída/desviación —que no usan el tipo predicho en su cálculo— capturan ese patrón de forma independiente. Sumado a que CLANDESTINA también tiene mayor recupero típico (hasta 8× según CLAUDE.md) y al bono operativo 1.25×, el ranking prioriza económicamente casos CLANDESTINA de forma consistente con el negocio. **Riesgo reconocido**: esto puede subrepresentar casos de MANIPULACIÓN de alto recupero en el corte superior del ranking; el jefe de campo debería revisar también la lista completa (no solo el corte dinámico) si quiere asegurar cobertura de ambos tipos.

**Desviación de implementación — exclusión de suministros con "datos insuficientes" (colisión de hoja del árbol)**: se detectó que el regresor B2 (LightGBM quantile) asigna el mismo valor exacto de `recupero_p10_kwh` a cientos de suministros distintos (ej. un valor se repite 845 veces, otro 574 veces) — 1,996 suministros (13.3%) caen en 31 grupos así. La causa es que esos suministros colisionan en la misma hoja terminal del árbol: el modelo no tiene señal real para diferenciarlos, así que predice igual para todos. Mostrar ese número como si fuera una predicción precisa sería engañoso — es un reflejo honesto de falta de información, no evidencia de recupero real.

Se probó primero filtrar por `factor_uso ≈ 0` (consumo casi nulo), pero ese umbral marcaba ~3,586 suministros (24% de la población) — demasiado amplio, porque `factor_uso` bajo es común y no implica colisión de por sí. Se corrigió a detectar la colisión directamente: cualquier suministro cuyo `recupero_p10_kwh` se repite ≥5 veces exactas en la población se marca `datos_insuficientes = True` (columna en `ranking_completo.csv`, expuesta para trazabilidad). Estos suministros se empujan al fondo del ranking (`prioridad = -1`) — nunca compiten por el corte operativo ni aparecen en el top 1000 del dashboard, pero siguen en el CSV completo para revisión de campo si se sospecha medidor inactivo o dañado. **Hipótesis no confirmada sobre la causa de fondo**: podrían ser medidores inactivos, cuentas dadas de baja, o acometidas sin servicio activo — no se investigó a nivel de fila porque no afecta los entregables de Fase 1.

---

### Paso 2.8 — Ranking final con corte dinámico

```python
df_ali.sort_values('prioridad', ascending=False, inplace=True)

# Corte dinámico: buscar el salto más grande entre scores consecutivos en las primeras ~150 posiciones
top150 = df_ali.head(150)
diffs  = top150['prioridad'].diff().abs()
corte  = diffs.idxmax()   # posición donde el ranking "se cae"
```

**Por qué corte dinámico y no fijo en 31 o 70**: el número 31 (hurtos reales) no se conoce al momento de la entrega. Un corte fijo puede quedar en medio de una zona donde varios casos tienen scores casi idénticos. El salto natural en el ranking es el punto donde la evidencia realmente se separa.

**Columna de tipo de hurto predicho**: para cada suministro en el ranking, se agrega la predicción de tipo (MANIPULACIÓN / CLANDESTINA) del clasificador B1, ya que el hackathon lo valora explícitamente y define qué cuadrilla y herramientas enviar.

---

### Paso 2.9 — Explicación bajo demanda (LLM Rol 2)

Solo para los suministros **por encima del corte dinámico**, y solo cuando el inspector hace clic en el dashboard (no se pregenera para los 14,951).

```
Prompt incluye:
- Posición en el ranking y score de prioridad
- Tipo predicho (MANIPULACIÓN / CLANDESTINA) y probabilidad calibrada
- Recupero estimado en kWh y soles
- Giro inferido
- Features más relevantes del caso (caída %, factor de uso, desviación de la SED)
- Si aplica: "alta incertidumbre" y qué votaron el LLM y el KNN
```

El LLM devuelve un párrafo en lenguaje natural para el inspector. **No modifica ningún número ni reordena nada.**

---

## Dashboard — especificación de pantalla

El dashboard debe responder en 30 segundos a: "¿A quién mando mañana para recuperar el máximo dinero?"

### Tarjetas resumen (parte superior)

| Tarjeta | Contenido |
|---|---|
| Total suministros | 14,951 |
| Sospechosos identificados | N (corte dinámico) |
| Recupero total estimado | S/ XXX,XXX (suma de recupero P10 de los sospechosos) |
| SEDs críticas | Top 3 SEDs por concentración de riesgo |

### Mapa de calor por SED

Cuadrícula de celdas (una por SED), coloreada por concentración de riesgo (suma de prioridad de los casos de esa SED). **No es mapa geográfico** — los datos están anonimizados, no hay coordenadas. Al hacer clic en una celda, filtra la lista de abajo a esa SED.

### Lista priorizada

Cada fila muestra, en orden:

| Campo | Detalle |
|---|---|
| Posición | #1, #2, ... |
| CUENTA_ID | código del suministro |
| SED_ID | para que la cuadrilla sepa a qué transformador ir |
| Tipo | badge de color: 🔴 CLANDESTINA / 🟠 MANIPULACIÓN |
| Probabilidad | % calibrado |
| Recupero estimado | S/ X,XXX (P10 conservador) |
| Incertidumbre | bandera "⚠️ Alta incertidumbre" si LLM y KNN discreparon > 20pp |
| Detalle | botón "Ver explicación" → dispara LLM Rol 2 |

### Filtros

- Por tipo de hurto (CLANDESTINA / MANIPULACIÓN / Todos)
- Por SED (dropdown)

### Exportar

Botón "Descargar lista CSV" — descarga la lista visible con todos los campos de la tabla.

### Diseño

Limpio y profesional. No es una demo colorida. El inspector de campo debe poder usarlo desde una tablet.

---

## Limitaciones reconocidas (documentar en el PDF de la Fase 1)

1. **Fraude estadísticamente indetectable**: un hurto tan preciso que el consumo reportado es idéntico a un patrón normal no genera señal capturable. Requiere balance de energía por SED, sensores IoT o inspección física → roadmap de fase posterior, información adicional disponible bajo solicitud.

2. **Causal 5 — no-clientes**: personas conectadas a la red sin ser clientes registrados no aparecen en ningún dataset. Requiere comparar energía que entra a cada transformador vs. suma de medidores legales → datos de subestaciones balanceadas/sobrecargadas (disponibles en etapa posterior).

3. **Dependencia del histórico de otra zona**: LightGBM y KNN se entrenan contra el histórico CNR de otros alimentadores. No hay casos confirmados del propio alimentador para validar. Esta es la limitación estructural del reto, no un defecto del diseño.

---

## Decisiones de diseño y su justificación (resumen)

| Decisión | Justificación |
|---|---|
| Fórmula de pesos fijos del PDF de referencia → descartada | Demasiado rígida; LightGBM aprende los pesos óptimos solo |
| Prioridad = prob × recupero, no solo prob | Alineado con brief: maximizar S/ recuperados por cuadrilla despachada |
| Días facturados > 33 NO son error (umbral = 400) | Diccionario oficial EQUANS: son periodos acumulados legítimos |
| Isolation Forest por grupo de giro inferido | Comparar dentro de pares homogéneos, no mezclar panaderías con viviendas |
| LLM solo en zona ambigua + explicación bajo demanda | Costo por token; el ML resuelve los casos claros |
| Percentil 25 para recupero (no la media, no P10) | Conserva el escenario conservador sin aplastar el rango contra el mínimo (D3) |
| Log-escala en el regresor de recupero | La cola larga (máx ~886k kWh) distorsiona el gradient boosting sin log |
| Corte fijo en 100 (no dinámico) | La curva de prioridad es continua: el "salto máximo" era ~2%, indistinguible del ruido, y saltaba entre corridas. 100 = capacidad de despacho, defendible y estable |
| Firma de fraude por distancia a perfil, no PU-learning | Un clasificador histórico-vs-alimentador aprende la zona geográfica, no el fraude (AUC 0.96 por artefacto de dominio) — D2 |
| Pesos de firma derivados de la separación medida | Elimina los pesos escritos a mano; una feature que no separa recibe peso 0 automáticamente (D1) |
| Cohortes por potencia contratada, no por giro inferido | La inferencia de giro por curva daba 31% de accuracy contra un baseline de 90.7% (D5) |
| Sin bono por tipo de hurto en la prioridad | Doble conteo: el recupero ya refleja la diferencia entre tipos; el bono forzaba un top 100% CLANDESTINA (D4) |
| Exclusión de suministros sin servicio activo | Consumo ~nulo no es hurto sino medidor de baja; era la principal contaminación del top del ranking |
| Umbral de discrepancia LLM/KNN = 20pp | Punto de partida calibrable; sobre ese umbral se muestra bandera en dashboard |
| Validar inferencia de giro antes de usar | Si no alcanza ≥ 70% accuracy, colapsar a super-categorías más amplias |
| Leave-one-out en mediana de la SED | Evita que el cliente atípico se camufle en su propia vara de comparación |
| Suavizado de grupos pequeños (< 5 en SED) | Grupos chicos tienen estadísticas inestables; se suavizan hacia la media global |

---

## Archivos del proyecto

```
hackathon-energia/
├── CLAUDE.md                    ← contexto del proyecto (este archivo + PLAN.md)
├── PLAN.md                      ← este archivo
├── data/
│   ├── DATA_ALIMENTADOR.csv
│   └── DATA_HISTORICO_CNR.csv
├── src/
│   ├── 01_limpieza.py           ← Algoritmo 1
│   ├── 02_features.py           ← Paso 2.2 feature engineering
│   ├── 03_giro_inferido.py      ← Paso 2.3 inferencia de giro
│   ├── 04_modelos.py            ← Paso 2.4 Isolation Forest + LightGBM
│   ├── 05_calibracion.py        ← Paso 2.5 calibración + clasificación por zona
│   ├── 06_zona_ambigua.py       ← Paso 2.6 LLM Rol 1 + KNN
│   ├── 07_ranking.py            ← Paso 2.7-2.8 fórmula final + corte dinámico
│   └── utils.py                 ← helpers compartidos (nombres de meses, etc.)
├── dashboard/
│   └── index.html               ← dashboard autocontenido HTML/CSS/JS
└── entregables/
    ├── sospechosos.xlsx          ← entregable 1: CUENTA_IDs
    └── ranking_completo.csv      ← ranking de los 14,951 con todos los scores
```

---

## Orden de implementación recomendado

1. `01_limpieza.py` → validar que los 14,951 y 4,659 salen intactos en cantidad
2. `02_features.py` → calcular F1-F5, inspeccionar distribuciones
3. `03_giro_inferido.py` → agrupar giros, validar accuracy ≥ 70%, inferir sobre alimentador
4. `04_modelos.py` → entrenar Isolation Forest + LightGBM (clasificador + regresor)
5. `05_calibracion.py` → calibrar probabilidades, clasificar zonas
6. `07_ranking.py` → fórmula de prioridad, corte dinámico, generar Excel entregable
7. `06_zona_ambigua.py` → LLM Rol 1 + KNN (puede ir en paralelo con 5-7, no es crítico path)
8. `dashboard/index.html` → conectar con ranking.csv, implementar LLM Rol 2 bajo demanda
