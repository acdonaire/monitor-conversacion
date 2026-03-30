"""
monitor-conversacion — Backend de orquestación
Python 3.12 + FastAPI + Uvicorn
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import base64
import wave
import struct
import io
import audioop
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("backend")

# ── Configuración ──────────────────────────────────────────────────────────────

WHISPER_API_URL = os.getenv("WHISPER_API_URL", "http://whisper:9000/asr")
QWEN_HTTP_URL   = os.getenv("QWEN_HTTP_URL",  "http://vllm-qwen-small:8001/v1/chat/completions")
QWEN_MODEL      = os.getenv("QWEN_MODEL",     "qwen")

BUFFER_WINDOW_SECONDS  = int(os.getenv("BUFFER_WINDOW_SECONDS",         "90"))
MIN_ANALYSIS_INTERVAL  = float(os.getenv("MIN_ANALYSIS_INTERVAL_SECONDS", "15"))
MAX_ANALYSIS_INTERVAL  = float(os.getenv("MAX_ANALYSIS_INTERVAL_SECONDS", "20"))

THRESHOLD_GREEN       = int(os.getenv("THRESHOLD_GREEN",       "65"))
THRESHOLD_RED         = int(os.getenv("THRESHOLD_RED",         "40"))
CONFIRMATIONS_FOR_RED = int(os.getenv("CONFIRMATIONS_FOR_RED", "2"))

SYSTEM_PROMPT = """\
Eres un sistema de análisis de tono conversacional. Tu única función es evaluar si una conversación es cordial o inaceptable.
Devuelve ÚNICAMENTE un JSON con: estado (verde, amarillo, rojo), puntuacion (0-100), tendencia (mejorando, estable, empeorando), razon y consejo.
"""

# ── Clases de Estructura de Datos ─────────────────────

@dataclass
class Fragment:
    text: str
    timestamp: float

class ContextBuffer:
    def __init__(self, window_seconds: int = BUFFER_WINDOW_SECONDS) -> None:
        self._fragments: list[Fragment] = []
        self._window = window_seconds
        self._analyzed_count: int = 0

    def add(self, text: str) -> None:
        stripped = text.strip()
        if stripped:
            self._fragments.append(Fragment(text=stripped, timestamp=time.time()))
            self._evict()

    def _evict(self) -> None:
        cutoff = time.time() - self._window
        self._fragments = [f for f in self._fragments if f.timestamp >= cutoff]
        self._analyzed_count = min(self._analyzed_count, len(self._fragments))

    def has_new_content(self) -> bool:
        self._evict()
        return len(self._fragments) > self._analyzed_count

    def mark_analyzed(self) -> None:
        self._analyzed_count = len(self._fragments)

    def is_empty(self) -> bool:
        self._evict()
        return len(self._fragments) == 0

    def build_user_message(self) -> str:
        self._evict()
        now = time.time()
        lines = [f'[hace {int(now - f.timestamp)}s] "{f.text}"' for f in self._fragments]
        return "\n".join(lines)

    def clear(self) -> None:
        self._fragments.clear()
        self._analyzed_count = 0

class SemaphoreStateMachine:
    def __init__(self) -> None:
        self.estado: str = "verde"
        self.puntuacion: int = 100
        self.tendencia: str = "estable"
        self.razon: str = ""
        self.consejo: str = ""
        self._red_confirmations: int = 0

    def update(self, result: dict) -> bool:
        try:
            puntuacion = max(0, min(100, int(result.get("puntuacion", 100))))
            tendencia  = result.get("tendencia", "estable")
            razon      = result.get("razon", "")
            consejo    = result.get("consejo", "")
            prev = self.estado
            
            if puntuacion > THRESHOLD_GREEN:
                new_estado = "verde"
                self._red_confirmations = 0
            elif puntuacion < THRESHOLD_RED:
                self._red_confirmations += 1
                new_estado = "rojo" if self._red_confirmations >= CONFIRMATIONS_FOR_RED else "amarillo"
            else:
                new_estado = "amarillo"
                self._red_confirmations = 0

            self.estado, self.puntuacion, self.tendencia = new_estado, puntuacion, tendencia
            self.razon = razon if new_estado != "verde" else ""
            self.consejo = consejo if new_estado != "verde" else ""
            return new_estado != prev
        except Exception:
            return False

class Session:
    def __init__(self) -> None:
        self.buffer = ContextBuffer()
        self.semaphore = SemaphoreStateMachine()
        self.start_time = time.time()
        self.last_analysis_time: float = 0.0
        self.event_clients: list[WebSocket] = []

    async def broadcast(self, event: dict) -> None:
        msg = json.dumps(event, ensure_ascii=False)
        for ws in list(self.event_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                with suppress(ValueError): self.event_clients.remove(ws)

    def reset(self) -> None:
        self.buffer.clear()
        self.semaphore.__init__()
        self.start_time = time.time()
        self.last_analysis_time = 0.0

# ── Análisis de Inteligencia (Qwen) ───────────────────────────────────────────

async def run_analysis(session: Session) -> None:
    if session.buffer.is_empty(): return
    user_message = session.buffer.build_user_message()
    session.buffer.mark_analyzed()
    session.last_analysis_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # --- LÍNEA DE DEBUG ---
            with wave.open("voxtral_debug.wav", "wb") as debug_file:
                debug_file.setnchannels(1)
                debug_file.setsampwidth(2)
                debug_file.setframerate(16000)
                debug_file.writeframes(audio_to_process)
# ----------------------
            resp = await client.post(QWEN_HTTP_URL, json={
                "model": QWEN_MODEL, 
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ]
            })
            resp.raise_for_status()
            
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            backticks = "`" * 3
            if raw.startswith(f"{backticks}json"): raw = raw[7:]
            elif raw.startswith(backticks): raw = raw[3:]
            if raw.endswith(backticks): raw = raw[:-3]
            
            result = json.loads(raw.strip())
            
            if session.semaphore.update(result):
                logger.info(f"[SEMÁFORO] Actualizado a {session.semaphore.estado}")
                await session.broadcast({
                    "type": "semaphore.update",
                    "estado": session.semaphore.estado,
                    "puntuacion": session.semaphore.puntuacion,
                    "tendencia": session.semaphore.tendencia,
                    "razon": session.semaphore.razon,
                    "consejo": session.semaphore.consejo,
                })
    except Exception as e:
        logger.error(f"Error en análisis Qwen: {e}")

# ── Puente Whisper (Audio a Texto) ─────────────────────────────────────────────

# ── Puente Whisper (NUEVO SISTEMA DE ARCHIVOS DIRECTOS) ────────────────────────

# --- NUEVO: Función para medir el volumen del audio ---
def get_audio_volume(pcm_bytes):
    # Calculamos cuántas muestras hay (cada una ocupa 2 bytes)
    count = len(pcm_bytes) // 2
    if count == 0: return 0
    # Desempaquetamos los bytes a números enteros
    shorts = struct.unpack(f'<{count}h', pcm_bytes)
    # Devolvemos el pico de volumen (de 0 a 32767)
    return max(abs(s) for s in shorts)

async def whisper_bridge(browser_ws: WebSocket, session: Session) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_buffer = bytearray()
        silence_chunks = 0
        has_speech = False
        
        try:
            while True:
                data = await browser_ws.receive_text()
                msg = json.loads(data)
                
                if msg.get("type") == "input_audio_buffer.append":
                    pcm_bytes = base64.b64decode(msg.get("audio", ""))
                    audio_buffer.extend(pcm_bytes)
                    
                    # 1. Medimos el volumen de este pedacito de audio
                    vol = get_audio_volume(pcm_bytes)
                    
                    if vol >= 4000: # 1000 es un buen umbral para descartar el ruido de fondo
                        has_speech = True
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                        
                    # Si llevamos casi 3 segundos de puro silencio inicial, limpiamos para no saturar memoria
                    if not has_speech and len(audio_buffer) > 48000:
                        audio_buffer.clear()
                        silence_chunks = 0
                        continue
                        
                    # 2. ¿Cuándo enviamos la frase a Whisper?
                    # - Opción A: Hubo voz, y ahora detectamos una pausa de ~1 segundo (2 chunks)
                    # - Opción B: Llevamos más de 8 segundos hablando sin parar (para no atascar la memoria)
                    is_phrase_finished = has_speech and silence_chunks >= 2
                    is_buffer_maxed = len(audio_buffer) > 128000 
                    
                    if not (is_phrase_finished or is_buffer_maxed):
                        continue # La frase no ha terminado, seguimos acumulando
                        
                    # 3. ¡Tenemos una frase completa! La empaquetamos.
                    audio_to_process = audio_buffer[:]
                    
                    # Vaciamos el cubo y reseteamos para escuchar la siguiente frase
                    audio_buffer.clear()
                    silence_chunks = 0
                    has_speech = False
                    
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, 'wb') as wav_file:
                        wav_file.setnchannels(1)       
                        wav_file.setsampwidth(2)       
                        wav_file.setframerate(16000)   
                        wav_file.writeframes(audio_to_process)
                    
                    wav_io.seek(0) 

                    try:
                        # 4. Enviamos el archivo WAV a Whisper
                        resp = await client.post(
                            WHISPER_API_URL, 
                            files={"audio_file": ("audio.wav", wav_io, "audio/wav")},
                            params={
                                "language": "es", 
                                "output": "text",
                                "vad_filter": "true",
                                "initial_prompt": "Transcripción de una conversación en español de España sobre tecnología y monitorización."
                            }
                        )
                        
                        if resp.status_code == 200:
                            transcript = resp.text.strip()
                            
                            # Filtro anti-alucinaciones básico
                            lower_t = transcript.lower()
                            bad_phrases = ["amara.org", "qué?", "suscríbete", "translated by", "daría"]
                            is_hallucination = any(bp in lower_t for bp in bad_phrases) or len(transcript) < 3

                            if transcript and not is_hallucination:
                                logger.info(f"[WHISPER] {transcript}")
                                
                                event_payload = {"type": "transcript", "text": transcript}
                                await session.broadcast(event_payload)
                                
                                try:
                                    await browser_ws.send_text(json.dumps(event_payload))
                                except Exception:
                                    pass
                                
                                session.buffer.add(transcript)
                                if (time.time() - session.last_analysis_time) >= MIN_ANALYSIS_INTERVAL:
                                    asyncio.create_task(run_analysis(session))
                    except Exception as e:
                        logger.error(f"Error comunicando con Whisper: {e}")

                elif msg.get("type") == "session.close":
                    break
        except WebSocketDisconnect:
            pass

app = FastAPI(title="Monitor Conversacion Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = Session()

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("Conexion de audio establecida")
    _session.reset()
    await whisper_bridge(websocket, _session)

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    _session.event_clients.append(websocket)
    
    await websocket.send_text(json.dumps({
        "type": "semaphore.update",
        "estado": _session.semaphore.estado,
        "puntuacion": _session.semaphore.puntuacion,
        "tendencia": _session.semaphore.tendencia,
        "razon": _session.semaphore.razon,
        "consejo": _session.semaphore.consejo,
    }))
    
    try:
        while True:
            await asyncio.sleep(20)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except Exception:
        pass
    finally:
        with suppress(ValueError): _session.event_clients.remove(websocket)

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": int(time.time() - _session.start_time)}