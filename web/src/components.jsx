import React from "react";
import { MAX_FRAMES, SAMPLE_FPS } from "./policy";

const labels = { pending: "En espera", processing: "Procesando", completed: "Completado", failed: "Error" };

export function Icon({ type = "video", size = 22 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{type === "upload" ? <><path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" /></> : <><rect x="3" y="5" width="13" height="14" rx="2" /><path d="m16 10 5-3v10l-5-3" /></>}</svg>;
}

export function Upload({ busy, file, input, chooseFile, upload }) {
  return <section className="card upload"><h2>Nuevo análisis</h2><p>Cargá una grabación de tus animales.</p>
    <label className="drop" onDragOver={(event) => event.preventDefault()} onDrop={(event) => {
      event.preventDefault();
      if (!busy) chooseFile(event.dataTransfer.files[0]);
    }}>
      <Icon type="upload" size={32} /><strong>{file ? file.name : "Eleg\xED o arrastr\xE1 un video"}</strong><span>MP4 · hasta 50 MB</span><input ref={input} type="file" accept="video/mp4,.mp4" disabled={busy} onChange={(event) => chooseFile(event.target.files[0])} />
    </label><div className="settings"><span>Frecuencia de análisis<strong>{SAMPLE_FPS} frames / segundo</strong></span><span>Máximo por análisis<strong>{MAX_FRAMES} frames</strong></span></div>
    <button className="primary" disabled={!file || busy} onClick={upload}><Icon type="upload" size={18} />{busy ? "Cargando video\u2026" : "Procesar video"}</button>
    <p className="hint">Se analizan hasta unos 60 segundos. El procesamiento continúa aunque cierres esta pestaña, mientras el servicio esté encendido.</p>
  </section>;
}

export function History({ jobs, selected, choose }) {
  return <section className="card history" aria-labelledby="video-history-title"><div className="section-title"><h2 id="video-history-title">Mis análisis</h2><span className="number">{jobs.length}</span></div><p>Retomá tus resultados cuando quieras.</p>
    <div className="job-list">{!jobs.length && <div className="empty"><Icon size={32} /><strong>Todavía no hay análisis</strong><span>Cargá tu primer video para comenzar.</span></div>}
    {jobs.map((item) => <button key={item.id} aria-current={selected === item.id ? "true" : undefined} className={`job ${selected === item.id ? "selected" : ""}`} onClick={() => choose(item)}><span className="job-icon"><Icon /></span><span className="job-info"><strong>{item.name}</strong><small>{new Date(item.created_at).toLocaleString("es-AR")} · {item.frames_analyzed} frames</small></span><span className={`status ${item.status}`}>{labels[item.status]}</span></button>)}</div>
  </section>;
}

export function Results({ job, count, frame, index, playing, setPlaying, setIndex }) {
  return <section className="card results" aria-labelledby="video-results-title"><div className="section-title"><div><h2 id="video-results-title">Resultados del análisis</h2><p>{job?.name ?? "Las detecciones aparecer\xE1n ac\xE1."}</p></div>{job && count > 0 && !["pending", "processing"].includes(job.status) && <a className="secondary" href={`/api/v1/analyses/${job.id}/results`}>Descargar detecciones</a>}</div>
    {job?.status === "pending" && <div className="notice" role="status">En espera. Los videos se procesan de a uno.</div>}
    {job?.status === "processing" && <div className="notice" role="status"><span className="spinner" /> Procesando · {count} frames analizados. La carga inicial del modelo puede tardar.</div>}
    {job?.error && <div className="alert" role="alert">{job.error}</div>}
    {job?.summary?.stop_reason === "frame_limit" && <div className="notice">Resultado parcial: se alcanzó el límite de {MAX_FRAMES} frames.</div>}
    <CandidateBCS job={job} />
    {count > 0 ? <><div className="metrics"><div><span>Frames analizados</span><strong>{count}</strong></div><div><span>Detecciones en este frame</span><strong>{frame?.detection_count ?? "\u2014"}</strong></div><div><span>Tiempo en el video</span><strong>{frame ? `${(frame.timestamp_ms / 1e3).toFixed(2)} s` : "\u2014"}</strong></div></div><div className="viewer">{frame ? <img key={frame.image_url} src={frame.image_url} alt={`Frame a ${(frame.timestamp_ms / 1e3).toFixed(2)} segundos con ${frame.detection_count} detecciones`} /> : <span>Cargando frame…</span>}</div><div className="controls"><button className="secondary" onClick={() => setPlaying(!playing)}>{playing ? "Pausar" : "Reproducir frames"}</button><input aria-label="Frame del análisis" type="range" min="0" max={Math.max(0, count - 1)} value={index} onChange={(event) => {
      setPlaying(false);
      setIndex(Number(event.target.value));
    }} /><span>{index + 1} / {count}</span></div><p className="hint">Vista de frames muestreados. La reproducción del visor no representa la velocidad original del video.</p></> : <div className="empty result-empty"><Icon size={42} /><strong>{job ? "Preparando las detecciones" : "Tu video, con m\xE1s informaci\xF3n"}</strong><span>Vas a ver cajas, cantidades y el momento de cada detección.</span></div>}
  </section>;
}

function CandidateBCS({ job }) {
  const candidate = job?.summary?.candidate, bcs = job?.summary?.bcs;
  if (!bcs || job.status !== "completed") return null;
  const explanations = {
    no_eligible_frame: "No se puede evaluar: ningún frame muestreado tiene exactamente una detección de vaca.",
    capacity_busy: "No evaluado: la capacidad BCS está ocupada. No se reintenta automáticamente.",
    capability_unavailable: "La capacidad BCS no está disponible en este servicio.",
    inference_failed: "Falló la evaluación BCS. Las detecciones del video se conservaron.",
    image_rejected: "La imagen original no cumple los límites de entrada BCS. No se recorta ni se reduce.",
    invalid_candidate: "La imagen original no cumple el contrato del candidato. No se evaluó BCS.",
    metadata_write_failed: "No se pudo guardar el resultado BCS. No hay un resultado confirmado disponible.",
    evaluation_timeout: "No evaluado: se alcanzó el tiempo máximo de espera BCS.",
    web_api_only: "BCS no evaluado: el CLI y el escritorio solo seleccionan el candidato.",
  };
  return <section className="candidate-bcs" aria-label="BCS del frame seleccionado">
    <h3>BCS del frame seleccionado</h3>
    {candidate && <><p>Frame seleccionado: {(candidate.timestamp_ms / 1000).toFixed(2)} s · índice original {candidate.frame_index}</p>
      <p>Confianza del detector: {(candidate.detector_confidence * 100).toFixed(2)} % · una detección de vaca</p>
      <img src={`/api/v1/analyses/${encodeURIComponent(job.id)}/candidate/image`} alt={`Imagen original para BCS a ${(candidate.timestamp_ms / 1000).toFixed(2)} segundos, sin anotaciones`} />
      <p className="hint">Imagen original completa para BCS, sin cajas ni recortes. Es independiente del frame actual del visor.</p></>}
    {bcs.status === "completed" ? <><strong>Categoría experimental BCS: {bcs.bcs_category} / 5</strong>
      <p>No aprobado para producción. Es una categoría ordinal del modelo, no un puntaje clínico estándar ni un diagnóstico.</p>
      <p>Procedencia: {bcs.package_id ?? "paquete no informado"} · estado del modelo: {bcs.model_status ?? "no informado"}</p></>
      : <p role="status">{explanations[bcs.reason] ?? "BCS no evaluado para este video."}</p>}
    <p className="hint">Mejor confianza individual solo entre frames muestreados elegibles (hasta {MAX_FRAMES}), no entre todos los frames del video. El detector puede omitir otros animales; una detección no garantiza una sola vaca real. No hay identificación ni tracking.</p>
  </section>;
}
