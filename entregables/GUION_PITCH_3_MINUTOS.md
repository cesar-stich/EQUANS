# GUION PARA EL VIDEO / PITCH DE 3 MINUTOS (EQUANS PERÚ)
## HACKATHON DE CONTROL Y REDUCCIÓN DE PÉRDIDAS DE ENERGÍA (PNT)

**Estructura del Pitch (3 minutos estrictos):**
- **Minuto 0:00 - 1:45**: La Solución y su Impacto Operativo Directo (Lo que resuelve el negocio).
- **Minuto 1:45 - 2:30**: Viabilidad, Robustez Técnica y Retorno de Inversión (ROI).
- **Minuto 2:30 - 3:00**: Por qué somos el equipo indicado y llamado a la acción.

---

### PARTE 1: LA SOLUCIÓN Y EL IMPACTO OPERATIVO (0:00 - 1:45)

> *"Buenas tardes, jurado de EQUANS. Hoy las cuadrillas de campo operan con una efectividad de inspección de apenas el 7 al 9%. Enviar una camioneta a revisar un medidor que no tiene nada no solo cuesta dinero y horas hombre: distrae recursos de los verdaderos focos de pérdidas no técnicas."*

> *"Para resolver esto en el alimentador crítico de 14,951 suministros, no construimos un modelo que solo predice probabilidad de fraude; construimos un **sistema de despacho de inspecciones guiado por impacto económico recuperable en soles**."*

> *"Nuestra solución integra un pipeline de tres capas:*
> 1. *Primero, una **limpieza física estricta**: respetamos los periodos acumulados entre 33 y 400 días porque son parte legítima de la operación y preservan el momento exacto en que inició la caída de consumo.*
> 2. *Segundo, un **doble modelo en paralelo**: un Isolation Forest por giro comercial para detectar anomalías relativas a negocios homogéneos, y un ensamble LightGBM supervisado que calibra la probabilidad y estima el recupero mediante **regresión por cuantiles al percentil 10 en log-escala**. Esto garantiza que prioricemos el escenario conservador más rentable, sin dejarnos engañar por outliers.*
> 3. *Tercero, clasificamos la vulneración: distinguimos **Clandestinas de Manipulaciones**, sabiendo que las clandestinas recuperan hasta 8 veces más energía.*

> *"¿El resultado operativo? Con nuestro corte dinámico natural, identificamos **35 suministros de máxima prioridad**, concentrados en transformadores críticos como **SED_2181, SED_1693 y SED_1207**, con un recupero mínimo proyectado de **más de S/ 112,000 y 168,000 kWh**."*

---

### PARTE 2: VIABILIDAD, DASHBOARD Y ROI (1:45 - 2:30)

> *"La solución no se queda en un Jupyter Notebook. Es 100% accionable hoy mismo a través de nuestro **Dashboard de Despacho Operativo**."*

> *"El jefe de operaciones puede abrir el mapa de calor por subestación y decir: 'Mañana concentro la camioneta 1 en la SED 2181 para inspeccionar 12 acometidas en la misma cuadra'. Cada caso cuenta con su diagnóstico de campo automático, advirtiendo incluso si existe alta incertidumbre para verificar la acometida antes de desmontar el medidor.*
> *Esto permite elevar la tasa de efectividad de las cuadrillas del 8% actual a **más del 25-30%**, multiplicando el ROI del programa de pérdidas."*

---

### PARTE 3: POR QUÉ SOMOS EL EQUIPO INDICADO (2:30 - 3:00)

> *"Somos el equipo indicado porque entendimos el reto no como un ejercicio abstracto de Machine Learning, sino como un **desafío de logística, ingeniería eléctrica y rentabilidad operativa**.*
> *Nuestra solución respeta la física de la red, es auditable, se integra a las bases de datos existentes de Supabase y está lista para ser desplegada en el terreno mañana.*

> *Muchas gracias."*

