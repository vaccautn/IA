import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { videoApi } from "./api";
import "./brand.css";
import "./style.css";
const labels = { pending: "En espera", processing: "Procesando", completed: "Completado", failed: "Error" };
function Icon({ type = "video", size = 22 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{type === "upload" ? <><path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" /></> : <><rect x="3" y="5" width="13" height="14" rx="2" /><path d="m16 10 5-3v10l-5-3" /></>}</svg>;
}
function App() {
  const [jobs, setJobs] = useState([]), [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [frame, setFrame] = useState(null), [index, setIndex] = useState(0), [playing, setPlaying] = useState(false);
  const input = useRef(null);
  const job = jobs.find((item) => item.id === selected);
  const count = job?.frames_analyzed ?? 0;
  useEffect(() => {
    let active = true, timer;
    async function refresh() {
      try {
        const data = await videoApi.list();
        if (active) {
          setJobs(data);
          setSelected((current) => current ?? data[0]?.id ?? null);
        }
      } catch (err) {
        if (active) setError(err.message);
      }
      if (active) timer = setTimeout(refresh, 2e3);
    }
    refresh();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    let active = true;
    setFrame(null);
    if (selected && count > 0) videoApi.frames(selected, index).then((data) => {
      if (active) setFrame(data.items[0] ?? null);
    }).catch((err) => {
      if (active) setError(err.message);
    });
    return () => {
      active = false;
    };
  }, [selected, index, count]);
  useEffect(() => {
    if (!playing || count < 2) return;
    const timer = setInterval(() => setIndex((current) => (current + 1) % count), 400);
    return () => clearInterval(timer);
  }, [playing, count]);
  function choose(item) {
    setSelected(item.id);
    setIndex(0);
    setPlaying(false);
  }
  function chooseFile(candidate) {
    setError("");
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".mp4") || candidate.size > 50 * 1024 * 1024) {
      setError("Eleg\xED un MP4 de hasta 50 MB.");
      return;
    }
    setFile(candidate);
  }
  async function upload() {
    setBusy(true);
    setError("");
    try {
      const created = await videoApi.upload(file);
      setJobs((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      choose(created);
      setFile(null);
      if (input.current) input.current.value = "";
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }
  return <div className="shell">
    <aside><div className="brand"><img src="/vacca_logo.png" alt="VACCA" /><span>Manejo de rodeos</span></div><div className="nav-active"><Icon /> Videos</div></aside>
    <div className="workspace"><header><span>VACCA / <strong>Videos</strong></span></header>
    <main><div className="page-heading"><div><span className="eyebrow">ANÁLISIS VISUAL</span><h1>Detección en video</h1><p>Cargá un video de tu rodeo y consultá las detecciones.</p></div><span className="page-icon"><Icon size={32} /></span></div>
    {error && <div className="alert" role="alert">{error}<button onClick={() => setError("")} aria-label="Cerrar aviso">×</button></div>}
    <div className="columns"><section className="card upload"><h2>Nuevo análisis</h2><p>Cargá una grabación de tus animales.</p>
      <label className="drop" onDragOver={(event) => event.preventDefault()} onDrop={(event) => {
    event.preventDefault();
    if (!busy) chooseFile(event.dataTransfer.files[0]);
  }}>
        <Icon type="upload" size={32} /><strong>{file ? file.name : "Eleg\xED o arrastr\xE1 un video"}</strong><span>MP4 · hasta 50 MB</span><input ref={input} type="file" accept="video/mp4,.mp4" disabled={busy} onChange={(event) => chooseFile(event.target.files[0])} />
      </label><div className="settings"><span>Frecuencia de análisis<strong>5 frames / segundo</strong></span><span>Máximo por análisis<strong>300 frames</strong></span></div>
      <button className="primary" disabled={!file || busy} onClick={upload}><Icon type="upload" size={18} />{busy ? "Cargando video\u2026" : "Procesar video"}</button>
      <p className="hint">Se analizan hasta unos 60 segundos. El procesamiento continúa aunque cierres esta pestaña, mientras el servicio esté encendido.</p>
    </section><section className="card history"><div className="section-title"><h2>Mis análisis</h2><span className="number">{jobs.length}</span></div><p>Retomá tus resultados cuando quieras.</p>
      <div className="job-list">{!jobs.length && <div className="empty"><Icon size={32} /><strong>Todavía no hay análisis</strong><span>Cargá tu primer video para comenzar.</span></div>}
      {jobs.map((item) => <button key={item.id} className={`job ${selected === item.id ? "selected" : ""}`} onClick={() => choose(item)}><span className="job-icon"><Icon /></span><span className="job-info"><strong>{item.name}</strong><small>{new Date(item.created_at).toLocaleString("es-AR")} · {item.frames_analyzed} frames</small></span><span className={`status ${item.status}`}>{labels[item.status]}</span></button>)}</div>
    </section></div>
    <section className="card results"><div className="section-title"><div><h2>Resultados del análisis</h2><p>{job?.name ?? "Las detecciones aparecer\xE1n ac\xE1."}</p></div>{job && count > 0 && !["pending", "processing"].includes(job.status) && <a className="secondary" href={`/api/v1/analyses/${job.id}/results`}>Descargar detecciones</a>}</div>
    {job?.status === "pending" && <div className="notice" role="status">En espera. Los videos se procesan de a uno.</div>}
    {job?.status === "processing" && <div className="notice" role="status"><span className="spinner" /> Procesando · {count} frames analizados. La carga inicial del modelo puede tardar.</div>}
    {job?.error && <div className="alert" role="alert">{job.error}</div>}
    {job?.summary?.stop_reason === "frame_limit" && <div className="notice">Resultado parcial: se alcanzó el límite de 300 frames.</div>}
    {count > 0 ? <><div className="metrics"><div><span>Frames analizados</span><strong>{count}</strong></div><div><span>Detecciones en este frame</span><strong>{frame?.detection_count ?? "\u2014"}</strong></div><div><span>Tiempo en el video</span><strong>{frame ? `${(frame.timestamp_ms / 1e3).toFixed(2)} s` : "\u2014"}</strong></div></div><div className="viewer">{frame ? <img key={frame.image_url} src={frame.image_url} alt={`Frame a ${(frame.timestamp_ms / 1e3).toFixed(2)} segundos con ${frame.detection_count} detecciones`} /> : <span>Cargando frame…</span>}</div><div className="controls"><button className="secondary" onClick={() => setPlaying(!playing)}>{playing ? "Pausar" : "Reproducir frames"}</button><input aria-label="Frame del análisis" type="range" min="0" max={Math.max(0, count - 1)} value={index} onChange={(event) => {
    setPlaying(false);
    setIndex(Number(event.target.value));
  }} /><span>{index + 1} / {count}</span></div><p className="hint">Vista de frames muestreados. La reproducción del visor no representa la velocidad original del video.</p></> : <div className="empty result-empty"><Icon size={42} /><strong>{job ? "Preparando las detecciones" : "Tu video, con m\xE1s informaci\xF3n"}</strong><span>Vas a ver cajas, cantidades y el momento de cada detección.</span></div>}
    </section><footer>Las cantidades corresponden a detecciones por frame, no a animales únicos. El modelo puede omitir vacas o generar cajas duplicadas.</footer></main></div>
  </div>;
}
createRoot(document.getElementById("root")).render(<App />);
