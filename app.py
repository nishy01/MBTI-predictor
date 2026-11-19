"""
Web server for Typology — MBTI Personality Inference Web Application.

Uses FastAPI + Uvicorn when installed, and seamlessly falls back to Python's
built-in ThreadingHTTPServer so the app runs out-of-the-box with zero extra
web framework dependencies.
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from model_engine import MBTIEngine, MBTI_PROFILES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_MODEL_PATH = BASE_DIR / "mbti_predictor.joblib"
HOST = "127.0.0.1"
PORT = 8000

engine = MBTIEngine(DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None)


def handle_predict_payload(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    posts = payload.get("posts")
    text = payload.get("text")

    entries: List[str] = []
    if isinstance(posts, list):
        entries = [str(p).strip() for p in posts if p and str(p).strip()]
    elif isinstance(text, str) and text.strip():
        entries = [text.strip()]

    if not entries:
        return (
            400,
            {"detail": "Please enter at least a few sentences of writing before running an analysis."},
        )

    combined = " ".join(entries)
    if len(combined.split()) < 8:
        return (
            400,
            {"detail": "Input is too short. Write at least 8 words (150+ words recommended for reliable results)."},
        )

    result = engine.predict(entries if len(entries) > 1 else entries[0])
    return 200, result


def handle_model_upload(filename: str, raw_bytes: bytes) -> Tuple[int, Dict[str, Any]]:
    if not filename.lower().endswith(".joblib"):
        return (
            400,
            {"detail": "Please upload a valid .joblib artifact exported from the notebook or train.py."},
        )
    if not raw_bytes:
        return 400, {"detail": "Uploaded file is empty."}

    target_path = BASE_DIR / "mbti_predictor.joblib"
    temp_path = BASE_DIR / "_uploaded_model.tmp.joblib"
    try:
        temp_path.write_bytes(raw_bytes)
        ok, message = engine.load_model(temp_path)
        if not ok:
            temp_path.unlink(missing_ok=True)
            return 400, {"detail": message}
        temp_path.replace(target_path)
        engine.model_path = target_path
        return 200, {"ok": True, "message": message, "status": engine.get_status()}
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return 500, {"detail": f"Could not save model: {exc}"}


try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


if HAS_FASTAPI:
    app = FastAPI(title="Typology - MBTI Personality Predictor", version="1.0.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    class PredictRequest(BaseModel):
        text: Optional[str] = None
        posts: Optional[List[str]] = None

    @app.get("/")
    def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def api_status():
        return engine.get_status()

    @app.get("/api/types")
    def api_types():
        return {"types": MBTI_PROFILES}

    @app.post("/api/predict")
    def api_predict(payload: PredictRequest):
        code, body = handle_predict_payload(payload.model_dump())
        if code != 200:
            raise HTTPException(status_code=code, detail=body["detail"])
        return JSONResponse(content=body)

    @app.post("/api/upload-model")
    async def api_upload_model(request: Request):
        filename = request.headers.get("X-Filename", "mbti_predictor.joblib")
        raw_bytes = await request.body()
        code, body = handle_model_upload(filename, raw_bytes)
        if code != 200:
            raise HTTPException(status_code=code, detail=body["detail"])
        return JSONResponse(content=body)


class TypologyHTTPHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP handler matching the FastAPI routes identically."""

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.address_string()}] {format % args}")

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Not Found")
            return
        content = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        content_type = mime_type or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type = f"{content_type}; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/":
            self._send_file(STATIC_DIR / "index.html")
            return
        if parsed_path == "/api/status":
            self._send_json(200, engine.get_status())
            return
        if parsed_path == "/api/types":
            self._send_json(200, {"types": MBTI_PROFILES})
            return
        if parsed_path.startswith("/static/"):
            rel = parsed_path[len("/static/"):]
            candidate = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() in candidate.parents:
                self._send_file(candidate)
                return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        if parsed_path == "/api/predict":
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self._send_json(400, {"detail": "Invalid JSON payload."})
                return
            code, body = handle_predict_payload(payload)
            self._send_json(code, body)
            return

        if parsed_path == "/api/upload-model":
            filename = self.headers.get("X-Filename", "mbti_predictor.joblib")
            code, body = handle_model_upload(filename, raw_body)
            self._send_json(code, body)
            return

        self.send_error(404, "Not Found")


def run_server() -> None:
    print("Typology - MBTI Personality Predictor")
    print(f"Active engine: {engine.get_status()['mode_label']}")
    print(f"Open in browser: http://{HOST}:{PORT}")

    if HAS_FASTAPI:
        print("Server backend: FastAPI + Uvicorn")
        uvicorn.run(app, host=HOST, port=PORT)
    else:
        print("Server backend: Python Standard Library HTTP Server")
        server = ThreadingHTTPServer((HOST, PORT), TypologyHTTPHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    run_server()
