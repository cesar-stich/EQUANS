import React, { useState, useMemo } from 'react';
import { Download, AlertTriangle, Search, Filter, Zap, Map as MapIcon, ChevronRight, Activity, BatteryWarning } from 'lucide-react';
import { mockData } from '../data';
import { exportToCsv } from '../utils/exportCsv';
import { Suministro } from '../types';

const UMBRAL_SOSPECHOSO = 8000;
const UMBRAL_SED_CRITICA = 10000;
const TARIFA_KWH = 0.667;

export default function Dashboard() {
  const [selectedSed, setSelectedSed] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<string>('TODOS');

  // Cálculo de KPIs Operativos
  const kpis = useMemo(() => {
    const sospechosos = mockData.filter(d => d.score_prioridad >= UMBRAL_SOSPECHOSO);
    const recuperoTotalSoles = sospechosos.reduce((acc, curr) => acc + curr.recupero_soles, 0);
    const recuperoTotalKwh = recuperoTotalSoles / TARIFA_KWH;
    
    const sedMap = new Map<string, { count: number, totalScore: number }>();
    mockData.forEach(item => {
      const current = sedMap.get(item.sed) || { count: 0, totalScore: 0 };
      sedMap.set(item.sed, { count: current.count + 1, totalScore: current.totalScore + item.score_prioridad });
    });
    
    let sedsCriticas = 0;
    sedMap.forEach(stats => {
      if ((stats.totalScore / stats.count) >= UMBRAL_SED_CRITICA) {
        sedsCriticas++;
      }
    });

    return {
      total: mockData.length,
      sospechosos: sospechosos.length,
      recuperoSoles: recuperoTotalSoles,
      recuperoKwh: recuperoTotalKwh,
      sedsCriticas
    };
  }, []);

  // Balance de Energía por SED (Heatmap)
  const sedHeatmap = useMemo(() => {
    const map = new Map<string, { count: number, totalScore: number, seds: Suministro[] }>();
    mockData.forEach(item => {
      const current = map.get(item.sed) || { count: 0, totalScore: 0, seds: [] };
      current.seds.push(item);
      map.set(item.sed, { count: current.count + 1, totalScore: current.totalScore + item.score_prioridad, seds: current.seds });
    });
    
    return Array.from(map.entries()).map(([sed, stats]) => ({
      sed,
      count: stats.count,
      avgScore: stats.totalScore / stats.count
    })).sort((a, b) => b.avgScore - a.avgScore);
  }, []);

  const filteredData = useMemo(() => {
    return mockData
      .filter(item => {
        if (selectedSed && item.sed !== selectedSed) return false;
        if (selectedType !== 'TODOS' && item.tipo !== selectedType) return false;
        return true;
      })
      .sort((a, b) => b.score_prioridad - a.score_prioridad);
  }, [selectedSed, selectedType]);

  const getHeatmapColor = (score: number) => {
    if (score >= 15000) return 'bg-red-600 text-white border-red-700 hover:bg-red-700';
    if (score >= 10000) return 'bg-orange-500 text-white border-orange-600 hover:bg-orange-600';
    if (score >= 5000) return 'bg-amber-100 text-amber-900 border-amber-300 hover:bg-amber-200';
    return 'bg-emerald-50 text-emerald-900 border-emerald-200 hover:bg-emerald-100';
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('es-PE', { style: 'currency', currency: 'PEN', maximumFractionDigits: 0 }).format(value);
  };
  
  const formatKwh = (value: number) => {
    return new Intl.NumberFormat('es-PE', { maximumFractionDigits: 0 }).format(value) + ' kWh';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-blue-100">
      {/* Header Operativo */}
      <header className="bg-slate-900 text-white px-6 py-4 sticky top-0 z-10 shadow-md">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center shadow-inner">
              <Zap className="w-5 h-5 text-white" fill="currentColor" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">Despacho de Inspecciones Técnicas</h1>
              <p className="text-sm text-slate-300 font-medium">Panel de Control de Pérdidas No Técnicas (PNT)</p>
            </div>
          </div>
          <button 
            onClick={() => exportToCsv(filteredData)}
            className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white text-sm font-semibold rounded-md hover:bg-blue-500 transition-colors shadow-sm focus:ring-2 focus:ring-blue-400 focus:outline-none"
          >
            <Download className="w-4 h-4" />
            Exportar Rutas para Cuadrilla
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        
        {/* KPI Cards (Ingeniería / Despacho) */}
        <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
            <p className="text-sm font-semibold text-slate-500 mb-2 uppercase tracking-wide flex items-center gap-1.5">
              <Activity className="w-4 h-4" /> Suministros Auditados
            </p>
            <p className="text-3xl font-bold text-slate-900">{kpis.total}</p>
            <p className="text-xs text-slate-400 mt-2">Población total monitoreada</p>
          </div>
          
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm border-l-4 border-l-blue-600 flex flex-col justify-between">
            <p className="text-sm font-semibold text-slate-500 mb-2 uppercase tracking-wide flex items-center gap-1.5">
              <Zap className="w-4 h-4" /> Inspecciones Asignables
            </p>
            <div className="flex items-end gap-2">
              <p className="text-3xl font-bold text-blue-700">{kpis.sospechosos}</p>
            </div>
            <p className="text-xs text-slate-400 mt-2">Suministros superan score de corte ({UMBRAL_SOSPECHOSO})</p>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm border-l-4 border-l-emerald-500 flex flex-col justify-between">
            <p className="text-sm font-semibold text-slate-500 mb-2 uppercase tracking-wide flex items-center gap-1.5">
              <BatteryWarning className="w-4 h-4" /> Pérdida Detectada
            </p>
            <div>
              <p className="text-3xl font-bold text-emerald-600 leading-none">{formatKwh(kpis.recuperoKwh)}</p>
              <p className="text-sm font-semibold text-slate-600 mt-1">{formatCurrency(kpis.recuperoSoles)}</p>
            </div>
            <p className="text-xs text-slate-400 mt-2">Tarifa referencial: S/ 0.667 / kWh</p>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm border-l-4 border-l-orange-500 flex flex-col justify-between">
            <p className="text-sm font-semibold text-slate-500 mb-2 uppercase tracking-wide flex items-center gap-1.5">
              <MapIcon className="w-4 h-4" /> SEDs Críticas
            </p>
            <div className="flex items-end gap-2">
              <p className="text-3xl font-bold text-orange-600">{kpis.sedsCriticas}</p>
            </div>
            <p className="text-xs text-slate-400 mt-2">Balance de energía deficiente (Déficit)</p>
          </div>
        </section>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          
          {/* Columna Izquierda: Heatmap por Transformador */}
          <section className="xl:col-span-1 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <MapIcon className="w-4 h-4 text-slate-500" />
                Matriz de Déficit por SED
              </h2>
              {selectedSed && (
                <button 
                  onClick={() => setSelectedSed(null)}
                  className="text-xs font-bold text-blue-600 hover:text-blue-800 uppercase tracking-wide"
                >
                  Ver toda la red
                </button>
              )}
            </div>
            
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
              <p className="text-xs text-slate-500 mb-4 leading-relaxed font-medium">
                Selecciona una Subestación de Distribución (SED) para focalizar a la cuadrilla. El color refleja el índice de pérdidas no técnicas en su circuito de baja tensión.
              </p>
              
              <div className="grid grid-cols-3 sm:grid-cols-4 xl:grid-cols-3 gap-2">
                {sedHeatmap.map((sed) => {
                  const isSelected = selectedSed === sed.sed;
                  return (
                    <button
                      key={sed.sed}
                      onClick={() => setSelectedSed(isSelected ? null : sed.sed)}
                      className={`
                        p-3 flex flex-col items-center justify-center rounded-lg border-2 transition-all text-left
                        ${getHeatmapColor(sed.avgScore)}
                        ${isSelected ? 'ring-2 ring-slate-900 ring-offset-2 scale-95 shadow-inner' : 'shadow-sm opacity-90 hover:opacity-100'}
                      `}
                      title={`Score de Riesgo: ${sed.avgScore.toFixed(0)}`}
                    >
                      <span className="text-xs font-extrabold tracking-tight mb-1">{sed.sed.replace('SED_', '')}</span>
                      <span className="text-[10px] opacity-90 font-semibold">{sed.count} suministros</span>
                    </button>
                  );
                })}
              </div>

              {/* Leyenda de Ingeniería */}
              <div className="mt-6 flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-emerald-100 border border-emerald-300"></div>Balanceado</div>
                <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-amber-100 border border-amber-400"></div>Déficit Leve</div>
                <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-orange-500"></div>Fuga</div>
                <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-red-600"></div>Crítico</div>
              </div>
            </div>
          </section>

          {/* Columna Derecha: Tabla Priorizada para Despacho */}
          <section className="xl:col-span-2 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <h2 className="text-base font-bold text-slate-900">Rutas de Inspección Priorizadas</h2>
              
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-slate-400" />
                <select 
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="text-sm font-medium bg-white border border-slate-200 rounded-md py-2 pl-3 pr-8 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-700 shadow-sm"
                >
                  <option value="TODOS">Todas las Vulneraciones</option>
                  <option value="CLANDESTINA">Conexiones Clandestinas (Alto Impacto)</option>
                  <option value="MANIPULACIÓN">Manipulación de Medidor</option>
                </select>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead>
                    <tr className="bg-slate-100 border-b border-slate-200 text-slate-600 font-bold uppercase text-[11px] tracking-wider">
                      <th className="px-4 py-3 text-center">Prioridad</th>
                      <th className="px-4 py-3">Suministro</th>
                      <th className="px-4 py-3">SED / Red</th>
                      <th className="px-4 py-3">Anomalía Detectada</th>
                      <th className="px-4 py-3 text-right">Pérdida (kWh)</th>
                      <th className="px-4 py-3 text-right">Recupero (S/.)</th>
                      <th className="px-4 py-3 text-right">Score ML</th>
                      <th className="px-4 py-3">Observaciones de Campo</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredData.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="px-4 py-8 text-center text-slate-500 font-medium">
                          No se encontraron suministros con los parámetros actuales.
                        </td>
                      </tr>
                    ) : (
                      filteredData.map((row, idx) => (
                        <tr key={row.suministro} className="hover:bg-blue-50/50 transition-colors">
                          <td className="px-4 py-3 text-center">
                            <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                              idx < 3 ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600'
                            }`}>
                              {idx + 1}
                            </span>
                          </td>
                          <td className="px-4 py-3 font-bold text-slate-900">
                            {row.suministro}
                          </td>
                          <td className="px-4 py-3 font-medium text-slate-600">
                            {row.sed}
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center px-2.5 py-1 rounded-sm text-xs font-bold uppercase tracking-wide ${
                              row.tipo === 'CLANDESTINA' 
                                ? 'bg-purple-100 text-purple-800 border border-purple-200' 
                                : 'bg-blue-100 text-blue-800 border border-blue-200'
                            }`}>
                              {row.tipo}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right font-bold text-emerald-600">
                            {formatKwh(row.recupero_soles / TARIFA_KWH)}
                          </td>
                          <td className="px-4 py-3 text-right font-medium text-slate-600">
                            {formatCurrency(row.recupero_soles)}
                          </td>
                          <td className="px-4 py-3 text-right font-bold text-slate-900">
                            {row.score_prioridad.toLocaleString()}
                          </td>
                          <td className="px-4 py-3">
                            {row.incertidumbre_alta ? (
                              <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-amber-700 bg-amber-50 border border-amber-300 px-2 py-1 rounded-sm">
                                <AlertTriangle className="w-3.5 h-3.5" />
                                Precaución / Falsa Alarma
                              </span>
                            ) : (
                              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                                Confirmado
                              </span>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              <div className="bg-slate-50 border-t border-slate-200 px-4 py-3 text-xs text-slate-500 font-medium flex justify-between items-center">
                <span>Total: {filteredData.length} suministros asignados a despacho.</span>
                <span>Priorización basada en recuperación económica máxima (S/.).</span>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
