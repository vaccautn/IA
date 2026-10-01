"""Desktop launcher for the existing video CLI (no additional dependencies)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


class VideoApp:
    def __init__(self, window):
        self.window = window
        self.events = queue.Queue()
        self.busy = False
        self.output = None
        window.title("VACCA — Procesar video")
        window.geometry("760x540")
        window.minsize(640, 480)
        panel = ttk.Frame(window, padding=24)
        panel.pack(fill="both", expand=True)
        ttk.Label(
            panel, text="Detección de vacas en video", font=("Segoe UI", 20)
        ).pack(anchor="w")
        ttk.Label(
            panel,
            text="Elegí un archivo y presioná Procesar video. Los resultados se guardan automáticamente.",
            wraplength=650,
        ).pack(anchor="w", pady=(8, 16))
        self.path = tk.StringVar()
        ttk.Entry(panel, textvariable=self.path).pack(fill="x")
        self.choose = ttk.Button(panel, text="1. Elegir video…", command=self.select)
        self.choose.pack(anchor="w", pady=10)
        self.full = tk.BooleanVar(value=False)
        self.check = ttk.Checkbutton(
            panel,
            text="Procesar todo el video (sin marcar: máximo 300 frames, unos 60 s)",
            variable=self.full,
        )
        self.check.pack(anchor="w")
        ttk.Label(
            panel,
            text="Se analizan 5 frames por segundo. El visor muestra hasta 120; el MP4 incluye todos los analizados.",
            wraplength=650,
        ).pack(anchor="w", pady=8)
        self.run = ttk.Button(panel, text="2. Procesar video", command=self.start)
        self.run.pack(anchor="w", pady=8)
        self.status = tk.StringVar(
            value="Listo para elegir un video. No requiere Internet."
        )
        ttk.Label(panel, textvariable=self.status, wraplength=650).pack(
            anchor="w", pady=8
        )
        self.bar = ttk.Progressbar(panel, mode="indeterminate")
        self.bar.pack(fill="x")
        self.log = tk.Text(panel, height=8, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, pady=10)
        self.open_button = ttk.Button(
            panel,
            text="3. Abrir resultados",
            command=self.open_report,
            state="disabled",
        )
        self.open_button.pack(anchor="w")
        ttk.Label(
            panel,
            text="El conteo es por frame; puede haber errores o cajas duplicadas.",
        ).pack(anchor="w", pady=8)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(100, self.poll)

    def select(self):
        path = filedialog.askopenfilename(
            title="Elegir video de vacas",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.path.set(path)

    def start(self):
        if self.busy:
            return
        source = Path(self.path.get().strip())
        if not source.is_file():
            messagebox.showerror(
                "Elegí un video", "Seleccioná un archivo de video existente."
            )
            return
        self.output = ROOT / "outputs/video" / uuid4().hex
        command = [
            str(Path(sys.executable).with_name("python.exe")),
            "-u",
            str(ROOT / "scripts/process_video.py"),
            str(source.resolve()),
            "--output",
            str(self.output),
            "--max-frames",
            "0" if self.full.get() else "300",
        ]
        self.busy = True
        for widget in (self.run, self.choose, self.check, self.open_button):
            widget.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status.set("Procesando… La carga inicial del modelo puede tardar.")
        self.bar.start()
        threading.Thread(target=self.worker, args=(command,), daemon=True).start()

    def worker(self, command):
        try:
            environment = dict(os.environ, PYTHONIOENCODING="utf-8")
            with subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ) as process:
                for line in process.stdout:
                    self.events.put(("line", line))
                code = process.wait()
            self.events.put(("done", code))
        except Exception as exc:
            self.events.put(("line", f"No se pudo iniciar: {exc}\n"))
            self.events.put(("done", 1))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    self.log.configure(state="normal")
                    self.log.insert("end", value)
                    self.log.see("end")
                    self.log.configure(state="disabled")
                else:
                    self.busy = False
                    self.bar.stop()
                    for widget in (self.run, self.choose, self.check):
                        widget.configure(state="normal")
                    exists = (self.output / "report.html").is_file()
                    self.open_button.configure(state="normal" if exists else "disabled")
                    if value == 0:
                        summary = {}
                        try:
                            summary = json.loads(
                                (self.output / "summary.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        except (OSError, ValueError):
                            pass
                        limited = summary.get("stop_reason") == "frame_limit"
                        self.status.set(
                            "Vista parcial lista: se alcanzó el límite de frames."
                            if limited
                            else "Procesamiento completo. Ya podés abrir los resultados."
                        )
                    else:
                        self.status.set(
                            "Falló el procesamiento. Revisá el detalle de arriba."
                            + (
                                " Hay resultados parciales disponibles."
                                if exists
                                else ""
                            )
                        )
        except queue.Empty:
            pass
        self.window.after(100, self.poll)

    def open_report(self):
        if self.output:
            webbrowser.open((self.output / "report.html").as_uri())

    def close(self):
        if self.busy:
            messagebox.showinfo(
                "Procesamiento en curso",
                "Esperá a que termine antes de cerrar esta ventana.",
            )
            return
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    VideoApp(root)
    root.mainloop()
