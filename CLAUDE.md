# CLAUDE.md — Hackathon de Reducción y Control de Pérdidas de Energía (EQUANS Perú)

## Qué es este proyecto

Pipeline de detección y priorización de hurto de energía eléctrica para el hackathon organizado por EQUANS Perú. El objetivo operativo real es: **dado un alimentador de 14,951 suministros sin etiquetar, generar un ranking de inspección ordenado por impacto económico recuperable en soles**, no solo por probabilidad de fraude. Hay 31 casos reales de hurto ocultos en ese alimentador.

La solución se compone de:
1. **Algoritmo 1** — limpieza mínima de datos (sin tocar señales de fraude)
2. **Algoritmo 2** — feature engineering + dos modelos ML en paralelo + calibración + ranking final
3. **Agente LLM** (Google Gemini API) con exactamente dos roles acotados
4. **Dashboard** interactivo para el jefe de campo

Los datos viven en **Supabase**. El pipeline corre en **Python**. El dashboard es **HTML/CSS/JS**.

---

## Entregables de Fase 1

1. **Excel** con los códigos `CUENTA_ID` de los suministros sospechosos de hurto
2. **PDF de 1 página** con 4 secciones: (1) explicación de la solución, (2) señales observadas, (3) enfoque/método de análisis, (4) resultado del análisis exploratorio principal
3. **Video/pitch de 3 minutos máx**: primero la solución, después por qué el equipo es el indicado. Énfasis en viabilidad, no en profundidad técnica

---

## Restricciones del hackathon

- **NO** intentar identificar personas ni revertir la anonimización
- **NO** cruzar la data con fuentes externas para reidentificar suministros
- **NO** divulgar la información a personas ajenas al equipo
- **NO** usar la data para entrenar modelos ajenos al evento
- **SÍ** se pueden usar fuentes externas públicas (datos abiertos del Estado, meteorológicos, actividad económica) siempre que **no** sea para reidentificar
- **Falso positivo** = mandar cuadrilla y no encontrar nada → solo incomoda, no es grave
- **Falso negativo** = no inspeccionar un pez gordo → carísimo, esto es lo que hay que evitar
- Se valora distinguir **MANIPULACIÓN** vs **CLANDESTINA** (la segunda tiene hasta 8× más recupero)

---

## Datasets

### DATA_ALIMENTADOR (en Supabase / CSV local: `DATA_ALIMENTADOR.csv`)

- **14,951 suministros** de un único alimentador crítico, **sin etiqueta**
- Separador: `;` | Encoding: UTF-8 BOM
- **95 SEDs**, 29 zonas, 7 llaves. Todos tarifa `BT5 B`
- Es el entregable que será evaluado

**Columnas exactas:**
```
CUENTA_ID | ZONA_ID | TARIFA | ACOMETIDA AEREA/SUBTERRANEA | POTENCIA USUARIO CONTRATADO |
TIPO CONEXIONADO | SET_ID | ALIMENTADOR_ID | SED_ID | LLAVE_ID |
DIAS FACTURADO\nENERO | CONSUMO ENERO |
DIAS FACTURADO\nFEBRERO | CONSUMO FEBRERO |
... (patrón igual para MARZO→DICIEMBRE, 12 pares en total)
```
Nota: el header de días facturados contiene un `\n` literal (salto de línea) en el nombre de la columna.

### DATA_HISTORICO_CNR (en Supabase / CSV local: `DATA_HISTORICO_CNR.csv`)

- **4,659 casos de hurto confirmado**, de **otra zona** de la concesión (no del alimentador crítico)
- Separador: `;` | Encoding: UTF-8 BOM
- **Todos son positivos**: en todos se encontró vulneración
- Tipificación: `MANIPULACIÓN` (3,658 casos) | `CLANDESTINA` (1,001 casos)
- Recupero: mediana ~3,930 kWh | máximo ~886,290 kWh (cola larga muy pronunciada)
- **71 giros comerciales únicos** (mezcla niveles de detalle, requieren agrupación antes de usar)
- Uso: entrenar el modelo de "firma de fraude"

**Columnas exactas:**
```
CUENTA_ID | SET_ID | ALIMENTADOR_ID | SED_ID | SERIE ID | LLAVE_ID |
DIA INTERVENCIÓN | MES INTERVENCIÓN | VULERACIÓN ENCONTRADA |
PROYECCIÓN DE RECUPERO (KWH) | TIPIFICACIÓN | GIRO COMERCIAL |
Tipo_Acomet | POTENCIA USUARIO CONTRATADO | TIPO CONEXIONADO |
DIAS FACTURADO\nENERO | CONSUMO ENERO |
... (mismo patrón hasta DICIEMBRE)
```

### Relación entre datasets

Los dos archivos **NO comparten suministros**. El histórico es de otros alimentadores. El vínculo es la estructura compartida, no los clientes. Esta es una limitación estructural del reto: no hay casos confirmados del propio alimentador para validar el modelo contra esa zona específica.

---

## Jerarquía de la red eléctrica

```
SET → ALIMENTADOR → SED → LLAVE → SUMINISTRO
```

- **SED** es el nivel más útil para el análisis: suministros de la misma SED son vecinos físicos que comparten el mismo transformador. Comparar un cliente contra su SED es una de las señales más potentes
- **LLAVE**: agrupación justo debajo de la SED
- **SUMINISTRO** (`CUENTA_ID`): la unidad de análisis, lo que se entrega en el Excel final

---

## Nota crítica sobre días facturados

Valores de `DIAS FACTURADO` **entre 33 y ~400 son periodos acumulados legítimos** (falta de lectura o refacturaciones) según el diccionario oficial de EQUANS Perú. **NO son errores**. Solo valores > 400 son físicamente imposibles (más de un año en un solo registro) y se tratan como corruptos.

---

## Jerarquía de criterios del hackathon (para priorizar decisiones de diseño)

1. **Impacto económico recuperable** (S/) — la métrica principal del negocio
2. **Clasificar tipo de hurto** (MANIPULACIÓN / CLANDESTINA) — lo que define la cuadrilla a enviar
3. **Dashboard accionable** — el jefe de campo debe poder decir "mando la camioneta 3 a la SED X mañana"
4. **Efectividad de inspección** — meta: subir del 7-9% actual al 25-30%
5. **Viabilidad / ROI** — el jurado pedirá que la solución sea implementable

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Base de datos | Supabase |
| Pipeline ML | Python (pandas, numpy, scikit-learn, LightGBM) |
| LLM | Google Gemini API |
| Dashboard | HTML/CSS/JS autocontenido |
| Entregables | Excel (.xlsx), PDF (1 página), video (3 min) |

---

## Glosario (del diccionario oficial de EQUANS Perú)

| Término | Definición |
|---|---|
| PNT | Pérdida no técnica: energía consumida pero no facturada |
| CNR | Consumo no registrado |
| SED | Subestación de distribución (transformador que alimenta una cuadra/sector) |
| Manipulación | Intervención sobre el medidor para que registre menos |
| Clandestina | Conexión directa a la red sin pasar por el medidor |
| Recupero | Energía que vuelve a facturarse tras detectar la irregularidad |
| Focalización | Seleccionar suministros con mayor probabilidad de PNT para priorizar inspecciones |
| Balance de energía | Energía que entra a una zona vs. la facturada (detecta "no-clientes") |
| BT5 B | Pliego tarifario de baja tensión: residenciales y pequeños comercios |
| Tarifa referencial | S/ 0.667 por kWh (fijada por el hackathon para calcular recupero en soles) |
