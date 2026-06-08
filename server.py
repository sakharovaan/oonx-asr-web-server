import io
import os
import tempfile
from typing import Optional, Literal

import onnx_asr
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

MODEL_NAME = os.environ.get("ONNX_ASR_MODEL", "gigaam-v3-e2e-rnnt")
VAD_MODEL = os.environ.get("ONNX_ASR_VAD", "silero")
MAX_FILE_SIZE_MB = 2500
SUPPORTED_FORMATS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}

model = None
vad = None


def load_model():
    global model, vad
    print(f"Loading model: {MODEL_NAME}")
    model = onnx_asr.load_model(MODEL_NAME, providers=["CPUExecutionProvider"])
    print("Model loaded successfully")
    print(f"Loading VAD: {VAD_MODEL}")
    vad = onnx_asr.load_vad(VAD_MODEL)
    model = model.with_vad(vad)
    print("VAD loaded successfully")


def convert_to_wav(file_content: bytes, source_format: str) -> bytes:
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(io.BytesIO(file_content))
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        return buffer.getvalue()
    except ImportError:
        pass

    try:
        import numpy as np
        import scipy.io.wavfile as wavfile
        audio_data = np.frombuffer(file_content, dtype=np.int16)
        buffer = io.BytesIO()
        wavfile.write(buffer, 16000, audio_data)
        return buffer.getvalue()
    except ImportError:
        pass

    return file_content


def parse_multipart_form(body: bytes, boundary: bytes) -> dict:
    import re
    parts = body.split(b"--" + boundary)
    files = {}
    fields = {}

    for part in parts:
        if not part.strip() or b"Content-Disposition" not in part:
            continue

        match = re.search(rb'name="([^"]+)"(?:; filename="([^"]+)")?', part)
        if not match:
            continue

        name = match.group(1).decode("utf-8")
        filename = match.group(2)

        content_start = part.find(b"\r\n\r\n") + 4
        if content_start == 3:
            continue

        content = part[content_start:].rstrip(b"\r\n--")

        if filename:
            files[name] = (filename.decode("utf-8") if filename else "audio", content)
        else:
            fields[name] = content.decode("utf-8") if content else ""

    return {"files": files, "fields": fields}


def get_audio_format(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in SUPPORTED_FORMATS:
        return ext

    if content[:4] == b"RIFF" and b"WAVE" in content[:16]:
        return "wav"

    if content[:3] == b"ID3" or (content[0] & 0xE0 == 0xE0 and content[:3] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"]):
        return "mp3"

    return ext or "mp3"


def recognize_audio(audio_path: str, language: Optional[str] = None) -> dict:
    result = model.recognize(audio_path, channel="mean")

    if hasattr(result, "__iter__"):
        segments = []
        for segment in result:
            if hasattr(segment, "text"):
                segments.append(str(segment.text))
            elif isinstance(segment, str):
                segments.append(segment)
            else:
                segments.append(str(segment))
        text = " ".join(segments)
    else:
        if hasattr(result, "text"):
            text = str(result.text)
        else:
            text = str(result)

    return {"text": text}


def format_srt(text: str) -> str:
    return f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"


def format_vtt(text: str) -> str:
    return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n"


async def transcriptions(request: Request) -> Response:
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    boundary = None
    for part in content_type.split(";"):
        if "boundary" in part:
            boundary = part.split("=")[1].strip().encode()
            break

    if boundary:
        parsed = parse_multipart_form(body, boundary)
        files = parsed.get("files", {})
        fields = parsed.get("fields", {})
        file_name, file_content = files.get("file", (None, None))
        model_name = fields.get("model", "whisper-1")
        response_format = fields.get("response_format", "json")
        language = fields.get("language", "")
    else:
        form = await request.form()
        file_name = form.get("file")
        model_name = form.get("model", "whisper-1")
        response_format = form.get("response_format", "json")
        language = form.get("language", "")

        if hasattr(file_name, "filename"):
            file_content = await file_name.read()
            file_name = file_name.filename
        else:
            file_content = file_name
            file_name = "audio.wav"

    if not file_content:
        return JSONResponse(
            {"error": {"message": "No audio file provided", "type": "invalid_request_error"}},
            status_code=400
        )

    if len(file_content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return JSONResponse(
            {"error": {"message": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB", "type": "invalid_request_error"}},
            status_code=413
        )

    ext = get_audio_format(file_name, file_content)
    audio_format = ext if ext in SUPPORTED_FORMATS else "wav"

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_in:
            tmp_in.write(file_content)
            tmp_in_path = tmp_in.name

        try:
            wav_content = convert_to_wav(file_content, audio_format)
            if wav_content != file_content:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    tmp_wav.write(wav_content)
                    tmp_audio_path = tmp_wav.name
            else:
                tmp_audio_path = tmp_in_path

            result = recognize_audio(tmp_audio_path, language or None)
            text = result["text"]

        finally:
            os.unlink(tmp_in_path)
            if tmp_audio_path != tmp_in_path and os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

    except Exception as e:
        return JSONResponse(
            {"error": {"message": f"Recognition error: {str(e)}", "type": "invalid_request_error"}},
            status_code=500
        )

    if response_format == "text":
        return Response(content=text, media_type="text/plain")

    if response_format == "srt":
        return Response(content=format_srt(text), media_type="text/plain")

    if response_format == "vtt":
        return Response(content=format_vtt(text), media_type="text/vtt")

    if response_format == "verbose_json":
        return JSONResponse({
            "text": text,
            "duration": 0.0,
            "language": language or "ru",
            "segments": [{
                "id": 0,
                "start": "00:00:00.000",
                "end": "00:00:10.000",
                "text": text,
            }],
        })

    return JSONResponse({"text": text})


async def translations(request: Request) -> Response:
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    boundary = None
    for part in content_type.split(";"):
        if "boundary" in part:
            boundary = part.split("=")[1].strip().encode()
            break

    if boundary:
        parsed = parse_multipart_form(body, boundary)
        files = parsed.get("files", {})
        fields = parsed.get("fields", {})
        file_name, file_content = files.get("file", (None, None))
        model_name = fields.get("model", "whisper-1")
        response_format = fields.get("response_format", "json")
    else:
        form = await request.form()
        file_name = form.get("file")
        model_name = form.get("model", "whisper-1")
        response_format = form.get("response_format", "json")

        if hasattr(file_name, "filename"):
            file_content = await file_name.read()
            file_name = file_name.filename
        else:
            file_content = file_name
            file_name = "audio.wav"

    if not file_content:
        return JSONResponse(
            {"error": {"message": "No audio file provided", "type": "invalid_request_error"}},
            status_code=400
        )

    ext = get_audio_format(file_name, file_content)
    audio_format = ext if ext in SUPPORTED_FORMATS else "wav"

    try:
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_in:
            tmp_in.write(file_content)
            tmp_in_path = tmp_in.name

        try:
            wav_content = convert_to_wav(file_content, audio_format)
            if wav_content != file_content:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    tmp_wav.write(wav_content)
                    tmp_audio_path = tmp_wav.name
            else:
                tmp_audio_path = tmp_in_path

            result = recognize_audio(tmp_audio_path)
            text = result["text"]

        finally:
            os.unlink(tmp_in_path)
            if tmp_audio_path != tmp_in_path and os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

    except Exception as e:
        return JSONResponse(
            {"error": {"message": f"Translation error: {str(e)}", "type": "invalid_request_error"}},
            status_code=500
        )

    if response_format == "text":
        return Response(content=text, media_type="text/plain")

    return JSONResponse({"text": text})


async def health(request: Request) -> Response:
    return JSONResponse({"status": "ok", "model": MODEL_NAME})


async def models(request: Request) -> Response:
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "gigaam-v3-e2e-rnnt",
                "object": "model",
                "created": 1700000000,
                "owned_by": "gigachat",
                "permission": [],
                "root": "gigaam-v3-e2e-rnnt",
            }
        ]
    })


class CorsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response


async def lifespan(app):
    load_model()
    yield


routes = [
    Route("/health", health),
    Route("/v1/models", models),
    Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
    Route("/v1/audio/translations", translations, methods=["POST"]),
]

app = Starlette(routes=routes, lifespan=lifespan)
app.add_middleware(CorsMiddleware)


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="ONNX ASR HTTP Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to bind")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
