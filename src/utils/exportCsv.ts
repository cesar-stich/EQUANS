import { Suministro } from '../types';

const TARIFA_KWH = 0.667;

export const exportToCsv = (data: Suministro[], filename = 'rutas_inspeccion_cuadrillas.csv') => {
  const headers = ['Ranking', 'Suministro', 'SED', 'Zona', 'Tipo de Hurto (Sospecha)', 'Probabilidad (%)', 'Pérdida Est. (kWh)', 'Recupero Est. (S/.)', 'Score Prioridad', 'Condición de Red'];
  
  const rows = data.map((row, index) => [
    index + 1,
    row.suministro,
    row.sed,
    row.zona,
    row.tipo,
    (row.probabilidad * 100).toFixed(1) + '%',
    (row.recupero_soles / TARIFA_KWH).toFixed(2),
    row.recupero_soles.toFixed(2),
    row.score_prioridad.toFixed(0),
    row.incertidumbre_alta ? 'ALTA INCERTIDUMBRE (Precaución)' : 'NORMAL'
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map(e => e.join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
