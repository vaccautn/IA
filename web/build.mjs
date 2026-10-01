import { build } from "esbuild";
import { mkdir, cp, copyFile } from "node:fs/promises";
await mkdir("dist", { recursive: true });
await build({ entryPoints: ["src/main.jsx"], bundle: true, minify: true, sourcemap: true, external: ["/fonts/*"], outdir: "dist/assets", define: { "process.env.NODE_ENV": '"production"' } });
await copyFile("index.html", "dist/index.html");
await cp("public", "dist", { recursive: true });
