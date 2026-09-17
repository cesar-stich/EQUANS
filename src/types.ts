export interface ResumenAlimentador {
  alimentador: string;
  suministros: number;
  transformadores: number;
  zonas: number;
  llaves: number;
  tarifa: string;
  inspeccionar: number;
  transformadores_con_alerta: number;
  recupero_soles: number;
}

export interface Suministro {
  suministro: string;
  sed: string;
  tipo: 'CLANDESTINA' | 'MANIPULACIÓN';
  probabilidad: number;
  nivel_sospecha: 'BAJO' | 'MODERADO' | 'ALTO' | 'MUY ALTO' | 'CRÍTICO' | 'SIN SERVICIO';
  recupero_soles: number;
  score_prioridad: number;
  incertidumbre_alta: boolean;
  explicacion: string;
}
