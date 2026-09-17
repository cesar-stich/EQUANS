import { Suministro } from '../types';
import { CORTE_CRITICA } from '../data';

const TARIFA_KWH = 0.667;

type ExportRow = Suministro & { globalRank?: number };

export const exportToCsv = (data: ExportRow[], filename = 'rutas_inspeccion_cuadrillas.csv') => {
  const headers = ['Ranking', 'Zona de Despacho', 'Suministro', 'SED', 'Tipo de Hurto (Sospecha)', 'Nivel de Sospecha', 'Pérdida Est. (kWh)', 'Recupero Est. (S/.)', 'Score Prioridad', 'Acción de Cuadrilla', 'Alerta'];

  const rows = data.map((row, index) => {
    // Usamos el ranking global del suministro (no la posición dentro de la lista filtrada/visible)
    // para que la etiqueta de zona de despacho sea correcta incluso al exportar un subconjunto filtrado.
    const rank = row.globalRank ?? index + 1;
    const zonaDespacho = rank <= CORTE_CRITICA
      ? `🔴 Zona Crítica (#1-${CORTE_CRITICA})`
      : `⚪ No Priorizado (#${CORTE_CRITICA + 1}+)`;

    return [
      rank,
      zonaDespacho,
      row.suministro,
      row.sed,
      row.tipo,
      row.nivel_sospecha,
      (row.recupero_soles / TARIFA_KWH).toFixed(2),
      row.recupero_soles.toFixed(2),
      row.score_prioridad.toFixed(0),
      row.tipo === 'CLANDESTINA' ? 'REVISAR RED / ACOMETIDA' : 'REVISAR MEDIDOR',
      row.incertidumbre_alta ? 'DUDOSO (confirmar en campo)' : ''
    ];
  });

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
