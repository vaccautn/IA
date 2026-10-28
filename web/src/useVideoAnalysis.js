import { useEffect, useRef, useState } from "react";
import { videoApi } from "./api";
import { MAX_UPLOAD_BYTES, PLAYBACK_INTERVAL_MS, POLL_INTERVAL_MS, TRACKS_POLL_INTERVAL_MS } from "./policy";
import { createRequestOwner, startPolling } from "./requests";

export function useVideoAnalysis() {
  const [jobs, setJobs] = useState([]), [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [frame, setFrame] = useState(null), [index, setIndex] = useState(0), [playing, setPlaying] = useState(false);
  const input = useRef(null);
  const listRequests = useRef(null), uploadRequests = useRef(null);
  if (!listRequests.current) listRequests.current = createRequestOwner();
  if (!uploadRequests.current) uploadRequests.current = createRequestOwner();
  const [tracks, setTracks] = useState(null);
  const job = jobs.find((item) => item.id === selected);
  const count = job?.frames_analyzed ?? 0;
  const jobStatus = job?.status;

  // Tracks are optional: a failed request never raises the page-level alert.
  // Tracking runs after the job completes, so refresh while it reports pending.
  useEffect(() => {
    const requests = createRequestOwner();
    let timer;
    setTracks(null);
    function load() {
      requests.run(
        (signal) => videoApi.tracks(selected, signal),
        (data) => {
          setTracks(data);
          if (data.status === "pending") timer = setTimeout(load, TRACKS_POLL_INTERVAL_MS);
        },
        () => setTracks({ status: "failed", reason: "request_failed", tracks: [] })
      );
    }
    if (selected && jobStatus === "completed") load();
    return () => {
      clearTimeout(timer);
      requests.cancel();
    };
  }, [selected, jobStatus]);

  useEffect(() => {
    const stop = startPolling(() => listRequests.current.run(
      (signal) => videoApi.list(signal),
      (data) => {
        setJobs(data);
        setSelected((current) => current ?? data[0]?.id ?? null);
      },
      (err) => setError(err.message)
    ), POLL_INTERVAL_MS);
    return () => {
      stop();
      listRequests.current.cancel();
      uploadRequests.current.cancel();
    };
  }, []);

  useEffect(() => {
    const requests = createRequestOwner();
    setFrame(null);
    if (selected && count > 0) requests.run(
      (signal) => videoApi.frames(selected, index, signal),
      (data) => setFrame(data.items[0] ?? null),
      (err) => setError(err.message)
    );
    return () => requests.cancel();
  }, [selected, index, count]);

  useEffect(() => {
    if (!playing || count < 2) return;
    const timer = setInterval(() => setIndex((current) => (current + 1) % count), PLAYBACK_INTERVAL_MS);
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
    if (!/\.(mp4|mov)$/i.test(candidate.name) || candidate.size > MAX_UPLOAD_BYTES) {
      setError("Eleg\xED un MP4 o MOV de hasta 50 MB.");
      return;
    }
    setFile(candidate);
  }
  async function upload() {
    setBusy(true);
    setError("");
    await uploadRequests.current.run(
      (signal) => videoApi.upload(file, signal),
      (created) => {
        listRequests.current.cancel();
        setJobs((current) => [created, ...current.filter((item) => item.id !== created.id)]);
        choose(created);
        setFile(null);
        if (input.current) input.current.value = "";
        setBusy(false);
      },
      (err) => {
        setError(err.message);
        setBusy(false);
      }
    );
  }
  return { jobs, selected, file, busy, error, frame, index, playing, input, job, count, tracks,
    setError, setIndex, setPlaying, choose, chooseFile, upload };
}
