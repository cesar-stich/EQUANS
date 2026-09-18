import json
import os
import pandas as pd

# Cargar el ranking completo de los 14,951 suministros generado por el pipeline
df = pd.read_csv('entregables/ranking_completo.csv')
print(f"Cargando {len(df)} suministros del ranking completo...")

# El dashboard solo muestra la lista de despacho, así que se exporta exactamente esa: el
# mismo corte que usa el pipeline para el Excel entregable, leído de la misma variable para
# que no puedan desincronizarse. Antes se exportaban 1,000 registros de los que se usaban
# 100, lo que inflaba src/data.ts a 674 KB — peso muerto en la tablet del supervisor.
# El corte lo decide el pipeline (corte_autonomo.py), no una constante. Se lee del Excel
# entregado para que el tablero del jefe de campo muestre exactamente los suministros
# que recibe el jurado, sin quedar desincronizado cuando N cambia entre corridas.
corte_operativo = len(pd.read_excel('entregables/sospechosos.xlsx'))
df_despacho = df.head(corte_operativo).copy()

# Ficha del alimentador para dar contexto en pantalla. Se deriva del dataset crudo en vez de
# escribirse a mano, para que no pueda quedar desactualizada respecto a los datos reales.
# (El brief verbal del hackathon mencionaba 15,147 suministros y 112 paquetes; el archivo
# efectivamente entregado tiene 14,951 y 95 SEDs. Manda el archivo.)
with open('data_cache/alimentador_raw.json', 'r', encoding='utf-8') as f:
    _raw = pd.DataFrame(json.load(f))
_raw.columns = [c.replace('\n', '_').replace(' ', '_').strip() for c in _raw.columns]

resumen_alimentador = {
    'alimentador': str(_raw['ALIMENTADOR_ID'].iloc[0]),
    'suministros': int(len(_raw)),
    'transformadores': int(_raw['SED_ID'].nunique()),
    'zonas': int(_raw['ZONA_ID'].nunique()),
    'llaves': int(_raw['LLAVE_ID'].nunique()),
    'tarifa': str(_raw['TARIFA'].mode().iloc[0]),
    'inspeccionar': int(corte_operativo),
    'transformadores_con_alerta': int(df_despacho['SED_ID'].nunique()),
    'recupero_soles': float(df_despacho['recupero_p10_soles'].sum()),
}

MESES_ES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
            'julio', 'agosto', 'setiembre', 'octubre', 'noviembre', 'diciembre']


def explicacion_para_campo(d):
    """
    Redacta el hallazgo como se lo contarías a un ingeniero eléctrico que va a despachar
    una cuadrilla: qué se observó en el medidor, cómo se compara con los vecinos del mismo
    transformador, y qué revisar. Sin nombres de features ni valores normalizados — el
    técnico no necesita saber que 'desv_sed = -0.24', necesita saber que el suministro
    registra 24% menos que sus vecinos.
    """
    pot = float(d.get('POTENCIA_CONTRATADA') or 0)
    cons = float(d.get('cons_anual') or 0)
    ceros = int(d.get('meses_cero_finales') or 0)
    activos = int(d.get('meses_activos') or 0)
    caida = float(d.get('caida_pct') or 0)
    fuso = float(d.get('factor_uso') or 0)
    desv = float(d.get('desv_sed') or 0)
    caida_vs_sed = float(d.get('caida_vs_sed') or 0)
    frases = []

    def kw(v):
        return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"

    # Promedio sobre los meses en que SÍ registró consumo. Dividir entre 12 mezclaría los
    # meses en cero y subestimaría lo que el suministro consumía cuando estaba activo, que
    # es justamente el dato con el que el técnico compara.
    prom_activo = cons / activos if activos > 0 else 0.0
    hay_corte = ceros >= 2

    # 1) Qué tiene contratado contra lo que registra, y el patrón en el tiempo.
    if hay_corte:
        mes_corte = MESES_ES[max(0, 12 - ceros)]
        cabecera = (
            f"Venía registrando unos {prom_activo:,.0f} kWh al mes y desde {mes_corte} marca cero: "
            f"lleva {ceros} meses seguidos sin registrar consumo."
        )
        if pot > 0:
            cabecera = f"Tiene {kw(pot)} kW contratados. " + cabecera
        frases.append(cabecera)
    else:
        if pot > 0:
            frases.append(
                f"Tiene {kw(pot)} kW contratados y registra un promedio de {prom_activo:,.0f} kWh al mes."
            )
        else:
            frases.append(f"Registra un promedio de {prom_activo:,.0f} kWh al mes.")

        if caida >= 0.40:
            frases.append(
                f"Su consumo cayó {caida*100:.0f}% entre el inicio y el final del año, sin recuperarse."
            )
        elif caida <= -0.40:
            frases.append(
                f"Su consumo subió {abs(caida)*100:.0f}% en el año, un salto que no corresponde a variación estacional."
            )
        elif activos < 10:
            frases.append(f"Solo registra consumo en {activos} de los 12 meses del año.")

    # 1b) El argumento más fuerte para la cuadrilla: la caída de ESTE suministro contra
    # la caída típica de su propia SED. Separa el hurto individual del efecto de barrio
    # (si toda la manzana bajó, es estacionalidad o un evento local, no manipulación).
    if caida_vs_sed >= 0.30:
        vecinos = caida - caida_vs_sed
        if vecinos <= 0.05:
            frases.append(
                f"Cayó {caida*100:.0f}% mientras el resto de su subestación se mantuvo estable "
                f"({vecinos*100:+.0f}%): la caída es de este suministro, no del sector."
            )
        else:
            frases.append(
                f"Cayó {caida*100:.0f}%, muy por encima del {vecinos*100:.0f}% que cayó el "
                f"resto de su subestación."
            )

    # 2) Comparación con los vecinos del mismo transformador. Cuando el medidor ya está en
    # cero, la comparación se redacta en pasado: decir "registra 123% más que sus vecinos"
    # junto a "lleva 3 meses en cero" suena contradictorio, cuando en realidad es el
    # argumento más fuerte — consumía más que el resto y de golpe dejó de registrar.
    if hay_corte:
        if desv >= 0.5:
            frases.append(
                "Mientras estuvo activo registraba más energía que el resto de su subestación, "
                "así que el corte a cero no cuadra con un local que simplemente dejó de operar."
            )
        elif desv <= -0.15:
            frases.append(
                f"Aun antes de apagarse ya registraba {abs(desv)*100:.0f}% menos que el promedio de su subestación."
            )
    else:
        if desv <= -0.15:
            frases.append(
                f"Registra {abs(desv)*100:.0f}% menos energía que el promedio de los demás suministros de su misma subestación."
            )
        elif desv >= 2.0:
            frases.append(
                f"Registra {desv:.0f} veces más energía que el promedio de su subestación, con lecturas muy irregulares mes a mes."
            )
        elif desv >= 0.5:
            frases.append(
                f"Registra {desv*100:.0f}% más energía que el promedio de su subestación."
            )

    # 3) Capacidad contratada ociosa: el argumento más fuerte frente a un cliente.
    # Solo se menciona si el medidor sigue activo — si ya marca cero es una consecuencia
    # obvia del corte y repetirlo no agrega nada.
    if pot >= 5.0 and fuso < 0.01 and not hay_corte:
        frases.append(
            f"Está pagando por {kw(pot)} kW de capacidad pero usa menos del 1% de ella, "
            "algo poco habitual en un cliente que mantiene el servicio activo."
        )

    # 5) Qué hacer en campo, según la hipótesis.
    if str(d['tipo_predicho']).upper().startswith('CLANDEST'):
        frases.append(
            "El patrón es compatible con una conexión directa a la red: conviene rastrear el "
            "cable de acometida desde el poste antes de abrir la caja del medidor."
        )
    else:
        frases.append(
            "El patrón es compatible con una alteración del medidor: revisar sellos, borneras "
            "y contrastar la precisión del equipo."
        )

    return " ".join(frases)


ts_items = []
for idx, d in df_despacho.iterrows():
    item = {
        'suministro': str(d['CUENTA_ID']),
        'sed': str(d['SED_ID']),
        'tipo': str(d['tipo_predicho']),
        'probabilidad': round(float(d['probabilidad_final']), 3),
        'nivel_sospecha': str(d['nivel_sospecha']),
        'recupero_soles': round(float(d['recupero_p10_soles']), 2),
        'score_prioridad': round(float(d['prioridad']), 2),
        'incertidumbre_alta': bool(d.get('alta_incertidumbre', False)),
        'explicacion': explicacion_para_campo(d)
    }
    ts_items.append(item)

ts_content = (
    'import { Suministro, ResumenAlimentador } from "./types";\n\n'
    f'export const CORTE_CRITICA = {corte_operativo};\n\n'
    'export const resumenAlimentador: ResumenAlimentador = '
    + json.dumps(resumen_alimentador, indent=2, ensure_ascii=False)
    + ';\n\n'
    'export const mockData: Suministro[] = '
    + json.dumps(ts_items, indent=2, ensure_ascii=False)
    + ';\n'
)

with open('src/data.ts', 'w', encoding='utf-8') as f:
    f.write(ts_content)

print(f"Actualizado src/data.ts con {len(ts_items)} suministros priorizados del alimentador!")
