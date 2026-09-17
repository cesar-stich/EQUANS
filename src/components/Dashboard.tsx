import React, { useState, useMemo } from 'react';
import { 
  Download, 
  AlertTriangle, 
  Search, 
  Filter, 
  Zap, 
  Map as MapIcon, 
  Activity,
  BatteryWarning,
  ShieldCheck,
  CheckCircle2,
  Truck,
  Eye,
  AlertCircle,
  HelpCircle,
  TrendingUp,
  X,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { mockData, CORTE_CRITICA, resumenAlimentador } from '../data';
import { exportToCsv } from '../utils/exportCsv';
import { Suministro } from '../types';

const TOTAL_ALIMENTADOR = 14951;
const TARIFA_KWH = 0.667;

type TierFilter = 'CRITICA' | 'LIMPIA';

export default function Dashboard() {
  const [selectedTier, setSelectedTier] = useState<TierFilter>('CRITICA');
  const [selectedSed, setSelectedSed] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<string>('TODOS');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedRow, setSelectedRow] = useState<Suministro | null>(null);
  const [showSedHeatmap, setShowSedHeatmap] = useState<boolean>(true);
  const [showLeyenda, setShowLeyenda] = useState<boolean>(true);

  const kpisCritica = useMemo(() => {
    const criticos = mockData.slice(0, CORTE_CRITICA);
    const recuperoSoles = criticos.reduce((acc, curr) => acc + curr.recupero_soles, 0);
    const recuperoKwh = recuperoSoles / TARIFA_KWH;
    return {
      count: criticos.length,
      recuperoSoles,
      recuperoKwh,
      promedioSoles: recuperoSoles / criticos.length
    };
  }, []);

  // Concentración de pérdida por transformador.
  //
  // Se ordena y colorea por DINERO TOTAL recuperable en el transformador, no por el promedio
  // de sus suministros: la cuadrilla se desplaza a una zona y revisa varios medidores ahí, así
  // que lo que decide la ruta es cuánto hay para recuperar en total. Con el promedio, un
  // transformador con 4 suministros y S/ 17,000 salía más urgente que uno con 48 suministros
  // y S/ 138,000 — al revés de lo que conviene despachar.
  //
  // Los cortes de color son cuartiles de la propia distribución y no números fijos: así el
  // mapa sigue estando bien repartido aunque cambie la escala del score. Con los umbrales
  // fijos anteriores (2400/2000/1500) habían quedado 73 de 91 transformadores en el mismo
  // color y ninguno en el más bajo.
  const sedHeatmap = useMemo(() => {
    // El mapa refleja exactamente los mismos suministros que la tabla de abajo: la lista de
    // despacho. Antes se calculaba sobre los 1,000 analizados, así que los conteos por
    // transformador sumaban 1,000 mientras la lista tenía 100, y el supervisor leía "39
    // sospechosos" en un transformador del que solo iba a visitar 11.
    const universo = mockData.slice(0, CORTE_CRITICA);

    const map = new Map<string, { count: number; soles: number }>();
    universo.forEach(item => {
      const current = map.get(item.sed) || { count: 0, soles: 0 };
      map.set(item.sed, { count: current.count + 1, soles: current.soles + item.recupero_soles });
    });

    const filas = Array.from(map.entries())
      .map(([sed, s]) => ({ sed, count: s.count, soles: s.soles }))
      .sort((a, b) => b.soles - a.soles);

    const ordenados = filas.map(f => f.soles).sort((a, b) => a - b);
    const cuartil = (p: number) => ordenados[Math.floor(ordenados.length * p)] ?? 0;
    const q25 = cuartil(0.25), q50 = cuartil(0.50), q75 = cuartil(0.75);

    return filas.map(f => ({
      ...f,
      nivel: f.soles >= q75 ? 'CRITICO' : f.soles >= q50 ? 'ALTO' : f.soles >= q25 ? 'MEDIO' : 'BAJO'
    }));
  }, []);

  // Cuántos suministros están representados en el mapa: el total debe cuadrar con la lista
  // de abajo, para que el supervisor pueda sumar las celdas y llegar al mismo número.
  const totalEnMapa = useMemo(
    () => sedHeatmap.reduce((acc: number, s: { count: number }) => acc + s.count, 0),
    [sedHeatmap]
  );

  // Filtrado reactivo de la lista de despacho por transformador, tipo y búsqueda.
  const filteredData = useMemo(() => {
    return mockData
      .map((item, originalIndex) => ({ ...item, globalRank: originalIndex + 1 }))
      .filter(item => {
        // Solo la lista de despacho. La zona limpia es conceptual: tiene su propia vista.
        if (item.globalRank > CORTE_CRITICA) return false;
        if (selectedTier === 'LIMPIA') return false;

        // Filtro por transformador
        if (selectedSed && item.sed !== selectedSed) return false;

        // Filtro por Tipo de Vulneración
        if (selectedType !== 'TODOS' && item.tipo !== selectedType) return false;

        // Búsqueda por texto (Suministro, SED)
        if (searchTerm.trim() !== '') {
          const term = searchTerm.toLowerCase().trim();
          const matchSum = item.suministro.toLowerCase().includes(term);
          const matchSed = item.sed.toLowerCase().includes(term);
          if (!matchSum && !matchSed) return false;
        }

        return true;
      });
  }, [selectedTier, selectedSed, selectedType, searchTerm]);

  const COLOR_TRANSFORMADOR: Record<string, string> = {
    CRITICO: 'bg-red-600 text-white border-red-700 hover:bg-red-700',
    ALTO: 'bg-orange-500 text-white border-orange-600 hover:bg-orange-600',
    MEDIO: 'bg-amber-100 text-amber-900 border-amber-300 hover:bg-amber-200',
    BAJO: 'bg-emerald-50 text-emerald-900 border-emerald-200 hover:bg-emerald-100'
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('es-PE', { style: 'currency', currency: 'PEN', maximumFractionDigits: 0 }).format(value);
  };
  
  const formatKwh = (value: number) => {
    return new Intl.NumberFormat('es-PE', { maximumFractionDigits: 0 }).format(value) + ' kWh';
  };

  // El nivel lo calcula el pipeline (quintiles de la firma sobre los 4,659 hurtos
  // confirmados) y aquí solo se le da color: un único lugar define los umbrales.
  //
  // Se muestra el nivel y NO el número: con 31 hurtos en 14,951 suministros, hasta un modelo
  // perfecto tendría 31% de tasa de acierto en un top-100. Un "55/100" en pantalla se lee
  // como "55% de acierto" y prometería el doble del techo teórico del problema.
  const ESTILO_NIVEL: Record<string, string> = {
    'CRÍTICO': 'text-red-50 bg-red-700 border-red-800',
    'MUY ALTO': 'text-red-900 bg-red-100 border-red-300',
    'ALTO': 'text-amber-900 bg-amber-100 border-amber-300',
    'MODERADO': 'text-slate-800 bg-slate-100 border-slate-300',
    'BAJO': 'text-slate-500 bg-white border-slate-200',
    'SIN SERVICIO': 'text-slate-500 bg-slate-50 border-slate-200'
  };

  const AYUDA_NIVEL: Record<string, string> = {
    'CRÍTICO': 'Puntúa por encima del 80% de los 4,659 hurtos confirmados del histórico.',
    'MUY ALTO': 'Puntúa por encima del 60% de los hurtos confirmados.',
    'ALTO': 'Puntúa por encima del 40% de los hurtos confirmados.',
    'MODERADO': 'Dentro del rango de los hurtos confirmados, en su parte baja (sobre el 20%).',
    'BAJO': 'Por debajo del 20% de los hurtos confirmados.',
    'SIN SERVICIO': 'Consumo anual casi nulo: medidor de baja o local vacío, sin energía que recuperar.'
  };

  return (
    <div className="min-h-screen bg-white text-slate-900 font-sans selection:bg-emerald-100 pb-12">
      
      {/* Header Corporativo EQUANS */}
      <header className="bg-white text-slate-900 px-4 sm:px-6 lg:px-8 py-4 sticky top-0 z-20 border-b border-slate-200 shadow-xs">
        <div className="max-w-[1920px] w-full mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-800 rounded-lg flex items-center justify-center shadow-xs">
              <Zap className="w-5 h-5 text-white" fill="currentColor" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-slate-950">Despacho de Inspecciones Técnicas</h1>
                <span className="px-2 py-0.5 rounded text-[10px] font-extrabold bg-emerald-100 text-emerald-900 border border-emerald-300">FASE 1</span>
              </div>
              <p className="text-xs text-slate-500 font-medium">Panel de Priorización de Pérdidas No Técnicas (PNT) — EQUANS Perú</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => exportToCsv(filteredData)}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-800 text-white text-xs sm:text-sm font-semibold rounded-md hover:bg-emerald-900 transition-colors shadow-xs focus:ring-2 focus:ring-emerald-600 focus:outline-none"
            >
              <Download className="w-4 h-4" />
              Exportar Rutas para Cuadrilla (CSV)
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1920px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 bg-white">

        {/* Ficha del alimentador: qué red es la que se está analizando */}
        <section className="bg-slate-900 text-slate-100 rounded-xl px-4 py-3 shadow-2xs">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
            <div className="flex items-center gap-2 shrink-0">
              <Zap className="w-4 h-4 text-emerald-400" fill="currentColor" />
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-bold leading-none">Red analizada</p>
                <p className="text-sm font-extrabold leading-tight">Alimentador {resumenAlimentador.alimentador}</p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
              {[
                { v: resumenAlimentador.suministros.toLocaleString('es-PE'), l: 'suministros (casas y locales)' },
                { v: resumenAlimentador.transformadores, l: 'transformadores' },
                { v: resumenAlimentador.zonas, l: 'zonas' },
                { v: resumenAlimentador.llaves, l: 'llaves de corte' },
                { v: resumenAlimentador.tarifa, l: 'tarifa (baja tensión)' }
              ].map(({ v, l }) => (
                <div key={l} className="leading-tight">
                  <span className="font-extrabold text-emerald-300 text-sm">{v}</span>
                  <span className="block text-[10px] text-slate-400 font-medium">{l}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* KPI Cards Superiores */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Alimentador */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 shadow-2xs flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <p className="text-xs font-bold text-slate-500 uppercase tracking-wide flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-slate-700" /> Suministros Auditados
              </p>
              <span className="text-[10px] font-bold text-slate-500 bg-slate-200 px-2 py-0.5 rounded">100% Red</span>
            </div>
            <p className="text-3xl font-extrabold text-slate-950 mt-2">{TOTAL_ALIMENTADOR.toLocaleString()}</p>
            <p className="text-xs text-slate-500 mt-1 font-medium">Población total del alimentador</p>
          </div>
          
          {/* Zona Crítica (Despacho Inmediato) */}
          <div className="bg-emerald-50/40 p-4 rounded-xl border-2 border-emerald-700 shadow-2xs flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <p className="text-xs font-extrabold text-emerald-950 uppercase tracking-wide flex items-center gap-1.5">
                <Truck className="w-4 h-4 text-emerald-800" /> Despacho Inmediato
              </p>
              <span className="text-[10px] font-extrabold text-red-700 bg-red-100 border border-red-300 px-2 py-0.5 rounded">🔴 Zona Crítica</span>
            </div>
            <div className="flex items-baseline gap-2 mt-2">
              <p className="text-3xl font-extrabold text-emerald-900">{kpisCritica.count}</p>
              <span className="text-xs font-bold text-emerald-800">suministros clave</span>
            </div>
            <p className="text-xs text-slate-600 mt-1 font-medium">Mayor prioridad económica del alimentador</p>
          </div>

          {/* Recupero Proyectado Zona Crítica */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 shadow-2xs border-l-4 border-l-emerald-600 flex flex-col justify-between">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide flex items-center gap-1.5">
              <BatteryWarning className="w-4 h-4 text-emerald-700" /> Recupero Zona Crítica
            </p>
            <div className="mt-2">
              <p className="text-2xl font-extrabold text-slate-950">{formatCurrency(kpisCritica.recuperoSoles)}</p>
              <p className="text-xs font-bold text-emerald-800 mt-0.5">{formatKwh(kpisCritica.recuperoKwh)}</p>
            </div>
            <p className="text-xs text-slate-500 mt-1 font-medium">Tarifa contractual: S/ 0.667 / kWh</p>
          </div>

          {/* Zonas con más dinero concentrado: por dónde empezar la ruta */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 shadow-2xs border-l-4 border-l-slate-400 flex flex-col justify-between">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide flex items-center gap-1.5">
              <MapIcon className="w-4 h-4 text-emerald-700" /> Transformadores más críticos
            </p>
            <div className="mt-2 space-y-1">
              {sedHeatmap.slice(0, 3).map((s, i) => (
                <div key={s.sed} className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-900">{i + 1}. Transf. {s.sed.replace('SED_', '')}</span>
                  <span className="text-slate-500 font-medium">{formatCurrency(s.soles)}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-1 font-medium">Donde más energía hay por recuperar en una sola salida</p>
          </div>
        </section>

        {/* ========================================================= */}
        {/* BANNER INTERACTIVO: ESTRATIFICACIÓN OPERATIVA (2 ZONAS) */}
        {/* ========================================================= */}
        <section className="bg-slate-50 border border-slate-200 rounded-xl p-4 sm:p-5 shadow-2xs">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-200">
            <div>
              <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-800"></span>
                Estratificación de la Red: Zona de Despacho vs. Zona Limpia
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Selecciona una zona para filtrar la tabla y ver la asignación de recursos recomendada para la cuadrilla.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

            {/* TIER 1: ZONA CRÍTICA */}
            <button
              onClick={() => setSelectedTier('CRITICA')}
              className={`p-3.5 rounded-lg border-2 text-left transition-all relative flex flex-col justify-between ${
                selectedTier === 'CRITICA'
                  ? 'bg-red-50/90 border-red-600 ring-2 ring-red-500 shadow-xs'
                  : 'bg-white border-slate-200 hover:border-red-400 hover:bg-red-50/30'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="inline-flex items-center gap-1 text-[11px] font-extrabold text-red-900 bg-red-100 border border-red-300 px-2 py-0.5 rounded">
                    🔴 1. ZONA CRÍTICA
                  </span>
                  <span className="text-xs font-extrabold text-red-700">Puestos #1 al #{CORTE_CRITICA}</span>
                </div>
                <h3 className="text-xs font-bold text-slate-900 mb-1">Lista de Inspección para Despacho</h3>
                <p className="text-[11px] text-slate-600 leading-snug">
                  Los {CORTE_CRITICA} suministros de mayor impacto económico recuperable del alimentador. Recupero promedio de <strong>S/ {kpisCritica.promedioSoles.toLocaleString('es-PE', { maximumFractionDigits: 0 })}</strong> por suministro.
                </p>
              </div>
              <div className="mt-3 pt-2 border-t border-red-200 flex items-center justify-between">
                <span className="text-[10px] font-extrabold text-red-800 uppercase flex items-center gap-1">
                  <Truck className="w-3 h-3" /> Despacho Inmediato
                </span>
                <span className="text-[11px] font-bold text-slate-900">{CORTE_CRITICA} sum.</span>
              </div>
            </button>

            {/* TIER 2: ZONA LIMPIA */}
            <button
              onClick={() => setSelectedTier('LIMPIA')}
              className={`p-3.5 rounded-lg border-2 text-left transition-all relative flex flex-col justify-between ${
                selectedTier === 'LIMPIA'
                  ? 'bg-emerald-50/90 border-emerald-600 ring-2 ring-emerald-500 shadow-xs'
                  : 'bg-white border-slate-200 hover:border-emerald-400 hover:bg-emerald-50/30'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="inline-flex items-center gap-1 text-[11px] font-extrabold text-emerald-900 bg-emerald-100 border border-emerald-300 px-2 py-0.5 rounded">
                    🟢 2. ZONA LIMPIA
                  </span>
                  <span className="text-xs font-extrabold text-emerald-800">#{CORTE_CRITICA + 1} al #14,951</span>
                </div>
                <h3 className="text-xs font-bold text-slate-900 mb-1">No Priorizados para esta Ronda</h3>
                <p className="text-[11px] text-slate-600 leading-snug">
                  Resto del alimentador: consumo continuo y sin señales suficientes para justificar despacho de cuadrilla en esta fase.
                </p>
              </div>
              <div className="mt-3 pt-2 border-t border-emerald-200 flex items-center justify-between">
                <span className="text-[10px] font-extrabold text-emerald-800 uppercase flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3" /> {(TOTAL_ALIMENTADOR - CORTE_CRITICA).toLocaleString()} No Priorizados
                </span>
                <span className="text-[11px] font-bold text-slate-900">{(((TOTAL_ALIMENTADOR - CORTE_CRITICA) / TOTAL_ALIMENTADOR) * 100).toFixed(1)}% Red</span>
              </div>
            </button>

          </div>
        </section>

        {/* CUERPO PRINCIPAL: MAPA POR SED + TABLA */}
        {(() => {
          // Renderizador de la Matriz de Déficit por SED
          const renderHeatmap = () => (
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
                <div>
                  <h2 className="text-sm sm:text-base font-bold text-slate-900 flex items-center gap-2">
                    <MapIcon className="w-4 h-4 text-emerald-800" />
                    ¿En qué transformadores se está perdiendo la energía?
                    <span className="text-xs font-semibold text-slate-500">({sedHeatmap.length} transformadores)</span>
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5 font-medium leading-relaxed">
                    {selectedSed
                      ? `Mostrando solo el transformador ${selectedSed.replace('SED_', '')}. Haz clic de nuevo para ver todos.`
                      : `Los ${totalEnMapa} suministros de esta lista están repartidos en ${sedHeatmap.length} transformadores. Cada cuadro es uno, y alimenta una cuadra o sector: haz clic para ver solo los suministros de esa zona y armar la ruta de la cuadrilla.`}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {selectedSed && (
                    <button 
                      onClick={() => setSelectedSed(null)}
                      className="text-xs font-bold text-emerald-800 hover:text-emerald-950 uppercase tracking-wide underline"
                    >
                      Ver todas ({sedHeatmap.length})
                    </button>
                  )}
                  <button 
                    onClick={() => setShowSedHeatmap(!showSedHeatmap)}
                    className="flex items-center gap-1 text-xs font-bold text-slate-700 bg-white border border-slate-300 rounded px-2.5 py-1 hover:bg-slate-50 transition-colors shadow-2xs"
                  >
                    {showSedHeatmap ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    <span>{showSedHeatmap ? 'Plegar Mapa' : 'Desplegar Mapa'}</span>
                  </button>
                </div>
              </div>

              {showSedHeatmap && (
                <>
                  <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 xl:grid-cols-12 max-h-[220px] gap-2 overflow-y-auto pr-1">
                    {sedHeatmap.map((sed) => {
                      const isSelected = selectedSed === sed.sed;
                      return (
                        <button
                          key={sed.sed}
                          onClick={() => setSelectedSed(isSelected ? null : sed.sed)}
                          className={`
                            p-2 flex flex-col items-center justify-center rounded-lg border-2 transition-all text-center
                            ${COLOR_TRANSFORMADOR[sed.nivel]}
                            ${isSelected ? 'ring-2 ring-slate-900 ring-offset-2 scale-95 shadow-inner font-extrabold' : 'shadow-2xs opacity-90 hover:opacity-100 hover:scale-102'}
                          `}
                          title={`Transformador ${sed.sed.replace('SED_', '')} — ${sed.count} suministro(s) bajo sospecha, ${formatCurrency(sed.soles)} recuperables en total`}
                        >
                          <span className="text-[9px] opacity-75 font-semibold leading-none">Transformador</span>
                          <span className="text-sm font-extrabold tracking-tight leading-tight">{sed.sed.replace('SED_', '')}</span>
                          <span className="text-[10px] opacity-90 font-semibold leading-tight mt-0.5">
                            {sed.count} suministro{sed.count === 1 ? '' : 's'}
                            <span className="block">sospechoso{sed.count === 1 ? '' : 's'}</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {/* Leyenda: qué representa el color y qué dice cada cuadro */}
                  <div className="mt-3 pt-3 border-t border-slate-100 space-y-2">
                    <p className="text-[11px] text-slate-600 leading-relaxed">
                      <strong className="text-slate-800">El color indica cuánto dinero hay para recuperar en todo ese transformador</strong>,
                      sumando sus suministros sospechosos. Los rojos son las zonas donde más conviene mandar una cuadrilla,
                      porque en un solo desplazamiento se revisan varios medidores.
                    </p>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[10px] font-bold text-slate-600">
                      <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-emerald-100 border border-emerald-300"></div>Poco por recuperar</div>
                      <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-amber-100 border border-amber-400"></div>Moderado</div>
                      <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-orange-500"></div>Bastante</div>
                      <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded bg-red-600"></div>Máxima concentración de pérdida</div>
                    </div>
                    <p className="text-[10px] text-slate-500 leading-relaxed">
                      En cada cuadro: arriba el código del transformador, abajo cuántos de sus suministros están bajo sospecha.
                      Pasa el cursor por encima para ver el monto recuperable.
                    </p>
                  </div>
                </>
              )}
            </div>
          );

          // Contenedor y tabla priorizada
          const renderTableSection = () => (
            <div className="space-y-4">
              {/* Barra de Filtros, Búsqueda y Selector de Modo de Ancho */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white p-3.5 rounded-xl border border-slate-200 shadow-2xs">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input 
                    type="text"
                    placeholder="Buscar suministro (ej: SUM_009955 o 9955) o SED..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-700 text-slate-900"
                  />
                  {searchTerm && (
                    <button 
                      onClick={() => setSearchTerm('')}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex items-center gap-1.5">
                    <Filter className="w-4 h-4 text-slate-400 shrink-0" />
                    <select 
                      value={selectedType}
                      onChange={(e) => setSelectedType(e.target.value)}
                      className="text-xs font-semibold bg-slate-50 border border-slate-200 rounded-md py-1.5 pl-2.5 pr-7 focus:outline-none focus:ring-2 focus:ring-emerald-700 text-slate-700 shadow-2xs"
                    >
                      <option value="TODOS">Todas las Vulneraciones</option>
                      <option value="CLANDESTINA">Clandestinas (Alto Impacto)</option>
                      <option value="MANIPULACIÓN">Manipulación de Medidor</option>
                    </select>
                  </div>

                </div>
              </div>

              {/* Leyenda: cómo leer la tabla (tipos de vulneración + niveles de sospecha) */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg text-xs shadow-2xs">
                <button
                  onClick={() => setShowLeyenda(!showLeyenda)}
                  className="w-full flex items-center justify-between px-3 py-2 hover:bg-slate-100/70 transition-colors rounded-lg"
                >
                  <span className="font-bold text-slate-800 uppercase text-[11px] tracking-wide flex items-center gap-1.5">
                    <HelpCircle className="w-3.5 h-3.5 text-emerald-800" />
                    Cómo leer esta tabla — guía para el supervisor de campo
                  </span>
                  {showLeyenda ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
                </button>

                {showLeyenda && (
                  <div className="px-3 pb-3 space-y-3 border-t border-slate-200 pt-3">

                    {/* Sección 1: qué significa cada hipótesis de vulneración */}
                    <div>
                      <p className="font-bold text-slate-800 text-[11px] mb-1.5">
                        Hipótesis de vulneración — de qué manera estaría tomando la energía
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-slate-600 text-[11px] leading-relaxed">
                        <div className="bg-white p-2.5 rounded-md border border-slate-200">
                          <span className="inline-flex items-center px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase bg-slate-900 text-white mb-1.5">
                            Clandestina
                          </span>
                          <p>
                            Hay un cable conectado <strong className="text-slate-800">directo a la red</strong>, que no pasa por el medidor.
                            La energía que se roba nunca llega a registrarse, por eso el medidor puede marcar muy poco o cero
                            aunque el local siga funcionando.
                          </p>
                          <div className="mt-2 pt-2 border-t border-slate-100">
                            <p className="text-slate-500 mb-1">En la columna <em>Sugerencia Operativa</em> aparece como:</p>
                            <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-900 bg-amber-100 border border-amber-300 px-2 py-0.5 rounded">
                              <AlertTriangle className="w-3 h-3 text-amber-700" />
                              Revisar Red / Acometida
                            </span>
                            <p className="mt-1 text-amber-800">
                              Seguir el cable de acometida desde el poste hasta la caja. Requiere cuadrilla con herramienta
                              de red y corte. Es el caso que más energía recupera.
                            </p>
                          </div>
                        </div>
                        <div className="bg-white p-2.5 rounded-md border border-slate-200">
                          <span className="inline-flex items-center px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase bg-slate-100 text-slate-800 border border-slate-300 mb-1.5">
                            Manipulación
                          </span>
                          <p>
                            El cable entra normal, pero el <strong className="text-slate-800">medidor fue intervenido</strong> para que
                            registre menos de lo que realmente pasa: sellos violados, borneras puenteadas o el equipo alterado.
                            El consumo sigue apareciendo, solo que más bajo y con saltos raros.
                          </p>
                          <div className="mt-2 pt-2 border-t border-slate-100">
                            <p className="text-slate-500 mb-1">En la columna <em>Sugerencia Operativa</em> aparece como:</p>
                            <span className="inline-flex items-center text-[10px] font-bold text-emerald-900 bg-emerald-100 border border-emerald-300 px-2 py-0.5 rounded">
                              Revisar Medidor
                            </span>
                            <p className="mt-1 text-emerald-800">
                              Revisar sellos y borneras, y contrastar la precisión del medidor. No hace falta intervenir
                              la red, así que la inspección es más rápida y barata.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Sección 2: qué significan los niveles de sospecha */}
                    <div>
                      <p className="font-bold text-slate-800 text-[11px] mb-1">
                        Nivel de sospecha — cuánto se parece a un robo ya comprobado
                      </p>
                      <p className="text-slate-600 text-[11px] leading-relaxed mb-2">
                        Comparamos el comportamiento de este medidor contra <strong className="text-slate-800">4,659 casos de robo
                        que ya fueron confirmados en campo</strong> por la empresa. El nivel dice qué tan parecido es a esos casos
                        reales. No es la probabilidad de encontrar robo al llegar: es qué tan marcada está la señal.
                      </p>
                      <div className="space-y-1">
                        {[
                          { n: 'CRÍTICO', d: 'Se parece más que 8 de cada 10 robos ya comprobados. La señal es muy marcada.' },
                          { n: 'MUY ALTO', d: 'Se parece más que 6 de cada 10 robos comprobados.' },
                          { n: 'ALTO', d: 'Se parece más que 4 de cada 10 robos comprobados.' },
                          { n: 'MODERADO', d: 'Está dentro del rango de los robos comprobados, en la parte baja. Sigue siendo sospechoso.' },
                          { n: 'BAJO', d: 'Por debajo de casi todos los robos comprobados. Entra a la lista por el monto en juego, no por la señal.' }
                        ].map(({ n, d }) => (
                          <div key={n} className="flex items-start gap-2 bg-white p-1.5 rounded-md border border-slate-200">
                            <span className={`shrink-0 inline-flex items-center px-2 py-0.5 rounded text-[10px] font-extrabold border w-[76px] justify-center ${ESTILO_NIVEL[n]}`}>
                              {n}
                            </span>
                            <span className="text-slate-600 text-[11px] leading-relaxed">{d}</span>
                          </div>
                        ))}
                      </div>
                      <p className="text-slate-500 text-[11px] leading-relaxed mt-2">
                        La etiqueta <strong className="text-slate-700">⚠ Dudoso</strong> aparece cuando las dos señales que usa el
                        sistema no coinciden entre sí. Conviene documentar bien el hallazgo, salga positivo o negativo.
                      </p>
                    </div>

                  </div>
                )}
              </div>

              {/* TABLA O VISTA DE ZONA LIMPIA */}
              {selectedTier === 'LIMPIA' ? (
                <div className="bg-white rounded-xl border border-emerald-200 p-8 text-center space-y-4 shadow-2xs">
                  <div className="w-16 h-16 bg-emerald-100 text-emerald-800 rounded-full flex items-center justify-center mx-auto shadow-inner">
                    <ShieldCheck className="w-9 h-9" />
                  </div>
                  <div className="max-w-xl mx-auto space-y-2">
                    <h3 className="text-lg font-bold text-slate-900">
                      🟢 Zona Limpia: {(TOTAL_ALIMENTADOR - CORTE_CRITICA).toLocaleString()} Clientes Honestos Protegidos
                    </h3>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      El modelo de Machine Learning analizó los {TOTAL_ALIMENTADOR.toLocaleString()} suministros del alimentador y <strong>descartó al {(((TOTAL_ALIMENTADOR - CORTE_CRITICA) / TOTAL_ALIMENTADOR) * 100).toFixed(1)}%</strong> por presentar consumos continuos, regulares y sin desviaciones significativas respecto a sus transformadores (SED).
                    </p>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-md mx-auto pt-2">
                    <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                      <p className="text-[10px] font-bold text-slate-500 uppercase">Suministros No Priorizados</p>
                      <p className="text-xl font-extrabold text-slate-950 mt-0.5">{(TOTAL_ALIMENTADOR - CORTE_CRITICA).toLocaleString()}</p>
                    </div>
                    <div className="bg-emerald-50 p-3 rounded-lg border border-emerald-200">
                      <p className="text-[10px] font-bold text-emerald-800 uppercase">Ahorro en Despacho</p>
                      <p className="text-xl font-extrabold text-emerald-900 mt-0.5">{(((TOTAL_ALIMENTADOR - CORTE_CRITICA) / TOTAL_ALIMENTADOR) * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  <div className="pt-2">
                    <button 
                      onClick={() => setSelectedTier('CRITICA')}
                      className="px-4 py-2 bg-emerald-800 text-white rounded-md text-xs font-bold hover:bg-emerald-900 transition-colors shadow-2xs"
                    >
                      Volver a la Zona Crítica (Top {CORTE_CRITICA} para Despacho)
                    </button>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
                  {/* Banner de Estado de la Selección Actual */}
                  <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-slate-800">
                        {`🔴 Zona Crítica (#1 al #${CORTE_CRITICA}) — Despacho Inmediato`}
                      </span>
                      <span className="text-slate-400">|</span>
                      <span className="text-slate-500 font-medium">Mostrando <strong>{filteredData.length}</strong> suministros</span>
                    </div>
                    {selectedSed && (
                      <span className="text-[11px] font-bold text-emerald-900 bg-emerald-100 border border-emerald-300 px-2 py-0.5 rounded">
                        Filtro SED: {selectedSed}
                      </span>
                    )}
                  </div>

                  {/* Tabla con ancho completo y proporciones generosas */}
                  <div className="w-full overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-slate-100/80 border-b border-slate-200 text-slate-700 font-bold uppercase text-[10px] tracking-wider">
                          <th className="px-3.5 py-3 text-center w-14"># Global</th>
                          <th className="px-3.5 py-3 min-w-[130px]">Suministro</th>
                          <th className="px-3.5 py-3 min-w-[120px] leading-tight">
                            <span>Subestación</span>
                            <span className="block text-[9px] text-slate-400 font-semibold">transformador del sector</span>
                          </th>
                          <th className="px-3.5 py-3 min-w-[140px] leading-tight">
                            <span>Hipótesis Vulneración</span>
                            <span className="block text-[9px] text-slate-400 font-semibold">cómo estaría robando</span>
                          </th>
                          <th className="px-3.5 py-3 text-right min-w-[120px]">Pérdida Estimada</th>
                          <th className="px-3.5 py-3 text-right min-w-[130px]">Recupero Proyectado</th>
                          <th className="px-3.5 py-3 text-center min-w-[130px] leading-tight">
                            <span>Nivel de Sospecha</span>
                            <span className="block text-[9px] text-slate-400 font-semibold">vs. fraude confirmado</span>
                          </th>
                          <th className="px-3.5 py-3 text-center min-w-[190px]">Sugerencia Operativa</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {filteredData.length === 0 ? (
                          <tr>
                            <td colSpan={8} className="px-4 py-8 text-center text-slate-500 font-medium">
                              No se encontraron suministros con los filtros actuales.
                            </td>
                          </tr>
                        ) : (
                          filteredData.map((row) => {
                            return (
                              <tr key={row.suministro} className="transition-colors hover:bg-red-50/50 bg-red-50/15">
                                <td className="px-3.5 py-2.5 text-center">
                                  <span className="inline-flex items-center justify-center w-6 h-6 rounded-full text-[11px] font-extrabold bg-red-600 text-white shadow-2xs">
                                    {row.globalRank}
                                  </span>
                                </td>
                                <td className="px-3.5 py-2.5 font-mono font-bold text-slate-950 text-xs">
                                  {row.suministro}
                                </td>
                                <td className="px-3.5 py-2.5 font-semibold text-slate-700 text-xs">
                                  <span className="px-2 py-0.5 bg-slate-100 border border-slate-200 rounded text-slate-800">
                                    {row.sed}
                                  </span>
                                </td>
                                <td className="px-3.5 py-2.5">
                                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-sm text-[10px] font-bold uppercase tracking-wide ${
                                    row.tipo === 'CLANDESTINA' 
                                      ? 'bg-slate-900 text-white border border-slate-900' 
                                      : 'bg-slate-100 text-slate-800 border border-slate-300'
                                  }`}>
                                    {row.tipo}
                                  </span>
                                </td>
                                <td className="px-3.5 py-2.5 text-right font-bold text-emerald-800 text-xs">
                                  {formatKwh(row.recupero_soles / TARIFA_KWH)}
                                </td>
                                <td className="px-3.5 py-2.5 text-right font-bold text-slate-950 text-xs">
                                  {formatCurrency(row.recupero_soles)}
                                </td>
                                <td className="px-3.5 py-2.5 text-center">
                                  <span
                                    title={`${AYUDA_NIVEL[row.nivel_sospecha] ?? ''} No es una tasa de acierto.`}
                                    className={`inline-flex items-center px-2.5 py-0.5 rounded text-[11px] font-extrabold tracking-wide border ${ESTILO_NIVEL[row.nivel_sospecha] ?? ESTILO_NIVEL['MODERADO']}`}>
                                    {row.nivel_sospecha}
                                  </span>
                                </td>
                                <td className="px-3.5 py-2.5">
                                  <div className="flex items-center justify-center gap-2">
                                    {row.tipo === 'CLANDESTINA' ? (
                                      <span
                                        title="Conexión directa a la red sin pasar por el medidor: rastrear el cable de acometida desde la red antes de abrir la caja."
                                        className="inline-flex items-center gap-1 text-[10px] font-bold tracking-tight text-amber-900 bg-amber-100 border border-amber-300 px-2 py-0.5 rounded shadow-2xs"
                                      >
                                        <AlertTriangle className="w-3 h-3 text-amber-700" />
                                        Revisar Red / Acometida
                                      </span>
                                    ) : (
                                      <span
                                        title="Patrón típico de alteración interna: revisar sellos, borneras y contrastación del medidor."
                                        className="inline-flex items-center gap-1 text-[10px] font-bold tracking-tight text-emerald-900 bg-emerald-100 border border-emerald-300 px-2 py-0.5 rounded shadow-2xs"
                                      >
                                        Revisar Medidor
                                      </span>
                                    )}
                                    {row.incertidumbre_alta && (
                                      <span
                                        title="Las dos señales del modelo (similitud a fraude confirmado y rareza frente a sus pares) discrepan más de 20 puntos. Confirmar en campo antes de concluir."
                                        className="inline-flex items-center text-[10px] font-bold text-slate-700 bg-slate-100 border border-slate-300 px-1.5 py-0.5 rounded"
                                      >
                                        ⚠ Dudoso
                                      </span>
                                    )}
                                    <button
                                      onClick={() => setSelectedRow(row)}
                                      className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-800 hover:bg-emerald-900 active:scale-95 text-white text-xs font-bold rounded-md shadow-xs hover:shadow transition-all cursor-pointer whitespace-nowrap border border-emerald-950"
                                      title="Abrir ventana de diagnóstico y ficha técnica del suministro"
                                    >
                                      <Eye className="w-3.5 h-3.5 text-white" />
                                      <span>Diagnóstico</span>
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            );
                          })
                        )}
                      </tbody>
                    </table>
                  </div>

                  {/* Footer informativo */}
                  <div className="bg-slate-50 border-t border-slate-200 px-4 py-3 text-xs text-slate-500 font-medium flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                    <span>
                      Mostrando <strong>{filteredData.length}</strong> suministros. Ordenados por retorno económico esperado.
                    </span>
                    <span className="text-[11px] text-slate-400">
                      Población total del alimentador: {resumenAlimentador.suministros.toLocaleString('es-PE')} suministros.
                    </span>
                  </div>
                </div>
              )}

              {/* Ficha operativa: cómo se opera esto mes a mes y cuánto rinde */}
              {selectedTier !== 'LIMPIA' && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="bg-white border border-slate-200 rounded-lg p-3 shadow-2xs">
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Cada cuánto se actualiza</p>
                    <p className="text-sm font-extrabold text-slate-950 mt-0.5">Mensual</p>
                    <p className="text-[11px] text-slate-600 leading-relaxed mt-1">
                      Las alertas se recalculan al cerrar cada ciclo de facturación, que es la frecuencia con la que
                      llega la lectura del medidor. Con telemetría en los transformadores podría pasar a diaria.
                    </p>
                  </div>
                  <div className="bg-white border border-slate-200 rounded-lg p-3 shadow-2xs">
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Rinde por inspección</p>
                    <p className="text-sm font-extrabold text-emerald-800 mt-0.5">
                      {formatCurrency(resumenAlimentador.recupero_soles / resumenAlimentador.inspeccionar)} por visita
                    </p>
                    <p className="text-[11px] text-slate-600 leading-relaxed mt-1">
                      {formatCurrency(resumenAlimentador.recupero_soles)} proyectados en {resumenAlimentador.inspeccionar} inspecciones,
                      en el escenario conservador. Comparar contra el costo de cuadrilla da el retorno de la operación.
                    </p>
                  </div>
                  <div className="bg-white border border-slate-200 rounded-lg p-3 shadow-2xs">
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Esfuerzo que ahorra</p>
                    <p className="text-sm font-extrabold text-slate-950 mt-0.5">
                      {resumenAlimentador.inspeccionar} de {resumenAlimentador.suministros.toLocaleString('es-PE')}
                    </p>
                    <p className="text-[11px] text-slate-600 leading-relaxed mt-1">
                      Se inspecciona el {(100 * resumenAlimentador.inspeccionar / resumenAlimentador.suministros).toFixed(1)}% de
                      la red, concentrado en {resumenAlimentador.transformadores_con_alerta} de {resumenAlimentador.transformadores} transformadores,
                      en vez de recorrerla completa a ciegas.
                    </p>
                  </div>
                </div>
              )}
            </div>
          );

          return (
            <div className="space-y-6">
              {renderHeatmap()}
              <section className="w-full">
                {renderTableSection()}
              </section>
            </div>
          );
        })()}

        {/* Modal de Explicación Técnica y Diagnóstico de Suministro */}
        {selectedRow && (
          <div className="fixed inset-0 z-50 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center p-4">
            <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full p-6 space-y-4 border border-slate-200">
              <div className="flex items-center justify-between border-b pb-3 border-slate-200">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-lg font-bold text-slate-950">{selectedRow.suministro}</h3>
                    <span className="text-xs font-bold text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                      Rank #{mockData.findIndex(d => d.suministro === selectedRow.suministro) + 1}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 font-medium">
                    Subestación {selectedRow.sed} — {selectedRow.tipo === 'CLANDESTINA'
                      ? 'posible conexión directa a la red, sin pasar por el medidor'
                      : 'posible medidor intervenido para que registre de menos'}
                  </p>
                </div>
                <button 
                  onClick={() => setSelectedRow(null)}
                  className="text-slate-400 hover:text-slate-600 text-lg font-bold px-2"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3 text-sm">
                {/* Lo que el supervisor necesita para decidir el despacho */}
                <div className="grid grid-cols-2 gap-3 bg-slate-50 p-3 rounded-lg border border-slate-200">
                  <div>
                    <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">Energía a recuperar</p>
                    <p className="text-base font-extrabold text-slate-950 leading-tight">{formatKwh(selectedRow.recupero_soles / TARIFA_KWH)}</p>
                    <p className="text-xs font-bold text-emerald-800">{formatCurrency(selectedRow.recupero_soles)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">Nivel de sospecha</p>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-extrabold border mt-0.5 ${ESTILO_NIVEL[selectedRow.nivel_sospecha] ?? ESTILO_NIVEL['MODERADO']}`}>
                      {selectedRow.nivel_sospecha}
                    </span>
                    <p className="text-[11px] text-slate-500 leading-snug mt-1">
                      {AYUDA_NIVEL[selectedRow.nivel_sospecha] ?? ''}
                    </p>
                  </div>
                </div>

                <div className="bg-emerald-50/70 border border-emerald-200 p-3.5 rounded-lg text-slate-900 space-y-1.5">
                  <div className="flex items-center gap-1.5 text-emerald-950 font-bold text-xs uppercase tracking-wider">
                    <Zap className="w-3.5 h-3.5 text-emerald-800" /> Qué se observó en este suministro
                  </div>
                  <p className="text-xs leading-relaxed text-slate-700">
                    {selectedRow.explicacion}
                  </p>
                </div>

                {selectedRow.incertidumbre_alta && (
                  <div className="bg-amber-50 border border-amber-200 p-3 rounded-lg text-amber-950 text-xs leading-relaxed">
                    <strong>⚠ Caso dudoso:</strong> las dos señales que usa el sistema no coinciden entre sí en este
                    suministro, así que la hipótesis es menos firme que en el resto de la lista. Vale la pena ir, pero
                    conviene documentar bien lo que se encuentre, salga positivo o negativo.
                  </div>
                )}
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => setSelectedRow(null)}
                  className="px-4 py-2 bg-slate-900 text-white rounded-md text-xs font-semibold hover:bg-slate-800"
                >
                  Cerrar Diagnóstico
                </button>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
