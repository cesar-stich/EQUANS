export interface Suministro {
  suministro: string;
  sed: string;
  zona: string;
  tipo: 'CLANDESTINA' | 'MANIPULACIÓN';
  probabilidad: number;
  recupero_soles: number;
  score_prioridad: number;
  incertidumbre_alta: boolean;
  explicacion: string;
}
