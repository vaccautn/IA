import React, { useState } from "react";
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
      <Icon type="upload" size={32} /><strong>{file ? file.name : "Eleg\xED o arrastr\xE1 un video"}</strong><span>MP4 o MOV · hasta 50 MB</span><input ref={input} type="file" accept="video/mp4,video/quicktime,.mp4,.mov" disabled={busy} onChange={(event) => chooseFile(event.target.files[0])} />
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

export function Results({ job, count, frame, index, playing, setPlaying, setIndex, tracks }) {
  const highlight = job?.highlighted_frame;
  return <section className="card results" aria-labelledby="video-results-title"><div className="section-title"><div><h2 id="video-results-title">Resultados del análisis</h2><p>{job?.name ?? "Las detecciones aparecer\xE1n ac\xE1."}</p></div>{job && count > 0 && !["pending", "processing"].includes(job.status) && <div className="download-actions"><a className="secondary" href={`/api/v1/analyses/${job.id}/frames.zip`}>Descargar frames (ZIP)</a><a className="secondary" href={`/api/v1/analyses/${job.id}/results`}>Descargar detecciones (JSONL)</a></div>}</div>
    {job?.status === "pending" && <div className="notice" role="status">En espera. Los videos se procesan de a uno.</div>}
    {job?.status === "processing" && <div className="notice" role="status"><span className="spinner" /> Procesando · {count} frames analizados. La carga inicial del modelo puede tardar.</div>}
    {job?.error && <div className="alert" role="alert">{job.error}</div>}
    {job?.summary_error && <div className="alert" role="alert">{job.summary_error}</div>}
    {job?.summary?.stop_reason === "frame_limit" && <div className="notice">Resultado parcial: se alcanzó el límite de {MAX_FRAMES} frames.</div>}
    <CandidateBCS job={job} />
    <Tracks job={job} tracks={tracks} />
    {highlight && <section className="final-summary" aria-label="Frame destacado por detecciones">
      <h2>{job.status === "failed" || job.summary?.stop_reason === "frame_limit" ? "Resumen de los frames analizados" : "Frame destacado"}</h2>
      <p>{highlight.detection_count} detecciones · confianza promedio {highlight.mean_confidence == null ? "no disponible sin detecciones" : `${(highlight.mean_confidence * 100).toFixed(1)} %`}.</p>
      <p>Frame de origen {highlight.frame_index ?? "—"} · {(highlight.timestamp_ms / 1000).toFixed(2)} s · muestra {highlight.sample_index + 1} de {count}.</p>
      <img className="highlight-image" src={highlight.image_url} alt={`Frame destacado con ${highlight.detection_count} detecciones`} />
      {highlight.detections?.length > 0 && <ul className="confidence-list">{highlight.detections.map((detection, i) => <li key={i}>Detección {i + 1}: {(detection.confidence * 100).toFixed(1)} %</li>)}</ul>}
      <a className="secondary" href={`${highlight.image_url}?download=true`}>Descargar frame destacado (JPG)</a>
      <button className="secondary" onClick={() => { setPlaying(false); setIndex(highlight.sample_index); }}>Ver frame destacado</button>
      <p className="hint">El destacado prioriza la cantidad de detecciones y luego su confianza promedio. Esto no indica animales únicos ni garantiza exactitud; es independiente de la selección de imagen original para BCS.</p>
    </section>}
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
    <p className="hint">Mejor confianza individual solo entre frames muestreados elegibles (hasta {MAX_FRAMES}), no entre todos los frames del video. El detector puede omitir otros animales; una detección no garantiza una sola vaca real. No hay identificación y esta selección no usa las trayectorias.</p>
  </section>;
}

const trackWarnings = {
  short_track_review_required: "Pocas observaciones",
  long_gap_review_required: "Ausencia larga: trayectoria fragmentada",
  other_detection_in_crop_at_best_frame: "Otra vaca dentro del recorte",
  border_touch_at_best_frame: "Animal cortado por el borde",
  small_at_best_frame: "Animal muy lejos: pocos píxeles",
  possible_fragment_review_required: "Posible fragmento de otra trayectoria",
};
const trackReasons = {
  not_tracked: "Este análisis es anterior a las trayectorias por vaca.",
  video_not_completed: "Las trayectorias solo se calculan para videos completados.",
  disabled_by_configuration: "Las trayectorias por vaca están desactivadas en este servicio.",
  timeout: "El cálculo de trayectorias superó el tiempo máximo. Las detecciones se conservaron.",
  interrupted: "El cálculo de trayectorias se interrumpió. Las detecciones se conservaron.",
  request_failed: "No se pudieron cargar las trayectorias. Volvé a seleccionar el análisis.",
};
const seconds = (ms) => (ms / 1000).toFixed(2);

export function Tracks({ job, tracks }) {
  if (job?.status !== "completed") return null;
  const items = tracks?.tracks ?? [];
  const review = items.filter((track) => track.status === "review").length;
  return <section className="tracks" aria-labelledby="video-tracks-title">
    <div className="section-title"><div><h3 id="video-tracks-title">Trayectorias por vaca</h3>
      {tracks?.status === "completed" && <p>{items.length} trayectorias · {review} para revisión{tracks.stop_reason && tracks.stop_reason !== "end_of_video" && tracks.fps ? ` · solo los primeros ${seconds(tracks.frames_read / tracks.fps * 1000)} s del video` : ""}</p>}</div></div>
    {!tracks ? <p role="status">Cargando trayectorias…</p>
      : tracks.status === "pending" ? <p role="status"><span className="spinner" /> Calculando trayectorias. Las detecciones ya están disponibles.</p>
      : tracks.status !== "completed" ? <p role="status">{trackReasons[tracks.reason] ?? "Las trayectorias no están disponibles. Las detecciones del video se conservaron."}</p>
      : !items.length ? <p role="status">No se formó ninguna trayectoria en este video.</p>
      : <ul className="track-grid">{items.map((track) => <TrackCard key={track.track_id} track={track} />)}</ul>}
    <p className="hint">Una trayectoria vale solo dentro de este video: no es una caravana ni identifica al animal. El recorte es rectangular y puede incluir otra vaca que el detector no vio. En los clips, los momentos sin detección aparecen en gris.</p>
  </section>;
}

function TrackCard({ track }) {
  const [open, setOpen] = useState(false), [whole, setWhole] = useState(false);
  const segments = track.segments?.length ?? 1;
  // A one-frame clip never becomes playable; the crop already shows that frame.
  const clip = track.clip_frames > 1 ? track.clip_url : null;
  const silhouette = track.best_masked_url && !whole;
  return <li className={`track ${track.status}`}>
    {open && clip
      ? <video src={clip} controls muted autoPlay loop playsInline aria-label={`Clip de la trayectoria ${track.track_id}`} />
      : silhouette
        ? <img src={track.best_masked_url} alt={`Silueta de la trayectoria ${track.track_id}, sin el fondo`} loading="lazy" />
        : <img src={track.best_crop_url} alt={`Mejor recorte de la trayectoria ${track.track_id}`} loading="lazy" />}
    <div className="track-head"><strong>Trayectoria {track.track_id}</strong><span className={`status ${track.status === "review" ? "review" : "completed"}`}>{track.status === "review" ? "Revisar" : "Candidata"}</span></div>
    <p>{seconds(track.first_timestamp_ms)}–{seconds(track.last_timestamp_ms)} s · {track.observations} {track.observations === 1 ? "observación" : "observaciones"}{segments > 1 ? ` · ${segments} tramos` : ""}</p>
    {track.warnings.length > 0 && <ul className="track-warnings">{track.warnings.map((warning) => <li key={warning}>{trackWarnings[warning] ?? warning}</li>)}</ul>}
    {track.possible_continuation_of != null && <p className="track-link">Puede ser el mismo animal que la trayectoria {track.possible_continuation_of}.</p>}
    {track.possible_continued_by != null && <p className="track-link">Puede continuar en la trayectoria {track.possible_continued_by}.</p>}
    <div className="track-actions">
      {clip && <button className="secondary" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "Ver recorte" : "Ver clip"}</button>}
      {track.best_masked_url && !open && <button className="secondary" aria-pressed={!whole} onClick={() => setWhole(!whole)}>{whole ? "Ver silueta" : "Ver con fondo"}</button>}
      <a className="secondary" href={track.best_original_url} target="_blank" rel="noreferrer">Frame original</a>
    </div>
  </li>;
}
