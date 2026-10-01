"""Serve the independent application and API at http://127.0.0.1:8010."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    import uvicorn
    from vacca_video.webapp import create_app

    if not (ROOT / "web/dist/index.html").is_file():
        raise SystemExit("Primero compilar la interfaz: cd web; npm ci; npm run build")
    if "--open" in sys.argv:
        import threading
        import webbrowser

        threading.Timer(2, lambda: webbrowser.open("http://127.0.0.1:8010")).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=8010)
