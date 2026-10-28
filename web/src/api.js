const base = "/api/v1";
async function request(path, options) {
  const response = await fetch(`${base}${path}`, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "No se pudo completar la solicitud.");
  return data;
}
export const videoApi = {
  list: (signal) => request("/analyses", { signal }),
  upload: (file, signal) => request(`/analyses?filename=${encodeURIComponent(file.name)}`, { method: "POST", body: file, headers: { "Content-Type": file.name.toLowerCase().endsWith(".mov") ? "video/quicktime" : "video/mp4" }, signal }),
  frames: (id, offset, signal) => request(`/analyses/${id}/frames?offset=${offset}&limit=1`, { signal }),
  tracks: (id, signal) => request(`/analyses/${encodeURIComponent(id)}/tracks`, { signal })
};
