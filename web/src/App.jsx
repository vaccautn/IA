import React from "react";
import { History, Icon, Results, Upload } from "./components";
import { useVideoAnalysis } from "./useVideoAnalysis";

export function App() {
  const state = useVideoAnalysis();
  return <div className="shell">
    <aside><div className="brand"><img src="/videos/vacca_logo.png" alt="VACCA" /><span>Manejo de rodeos</span></div><div className="nav-active"><Icon /> Videos</div></aside>
    <div className="workspace"><header><span>VACCA / <strong>Videos</strong></span></header>
    <main><div className="page-heading"><div><span className="eyebrow">ANÁLISIS VISUAL</span><h1>Detección en video</h1><p>Cargá un video de tu rodeo y consultá las detecciones.</p></div><span className="page-icon"><Icon size={32} /></span></div>
    {state.error && <div className="alert" role="alert">{state.error}<button onClick={() => state.setError("")} aria-label="Cerrar aviso">×</button></div>}
    <div className="columns"><Upload {...state} /><History {...state} /></div>
    <Results {...state} />
    <footer>Las cantidades corresponden a detecciones por frame, no a animales únicos. El modelo puede omitir vacas o generar cajas duplicadas.</footer></main></div>
  </div>;
}
