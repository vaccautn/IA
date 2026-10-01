const base = "/api/v1";
async function request(path, options) {
  const response = await fetch(`${base}${path}`, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "No se pudo completar la solicitud.");
  return data;
}
export const videoApi = {
  list: () => request("/analyses"),
  upload: (file) => request(`/analyses?filename=${encodeURIComponent(file.name)}`, { method: "POST", body: file, headers: { "Content-Type": "video/mp4" } }),
  frames: (id, offset) => request(`/analyses/${id}/frames?offset=${offset}&limit=1`)
};
