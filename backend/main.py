"""
monitor-conversacion — Backend de orquestación
Python 3.12 + FastAPI + Uvicorn

Responsabilidades:
  - Recibe audio PCM16 del navegador (/ws/audio) y lo reenvía a Voxtral
  - Mantiene un buffer de contexto con ventana deslizante de 90s
  - Evalúa el tono con Qwen cada ~20s (o en cada transcription.done si han pasado >15s)
  - Aplica máquina de estados con histéresis (rojo requiere 2 evaluaciones consecutivas)
  - Emite eventos al frontend (/ws/events)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("backend")

# ── Configuración ──────────────────────────────────────────────────────────────

VOXTRAL_WS_URL = os.getenv("VOXTRAL_WS_URL", "ws://vllm-voxtral:8000/v1/realtime")
QWEN_HTTP_URL  = os.getenv("QWEN_HTTP_URL",  "http://vllm-qwen:8001/v1/chat/completions")
VOXTRAL_MODEL  = os.getenv("VOXTRAL_MODEL",  "mistralai/Voxtral-Mini-4B-Realtime-2602")
QWEN_MODEL     = os.getenv("QWEN_MODEL",     "Qwen/Qwen2.5-7B-Instruct")

BUFFER_WINDOW_SECONDS  = int(os.getenv("BUFFER_WINDOW_SECONDS",        "90"))
MIN_ANALYSIS_INTERVAL  = float(os.getenv("MIN_ANALYSIS_INTERVAL_SECONDS", "15"))
MAX_ANALYSIS_INTERVAL  = float(os.getenv("MAX_ANALYSIS_INTERVAL_SECONDS", "20"))

THRESHOLD_GREEN       = int(os.getenv("THRESHOLD_GREEN",       "65"))
THRESHOLD_RED         = int(os.getenv("THRESHOLD_RED",         "40"))
CONFIRMATIONS_FOR_RED = int(os.getenv("CONFIRMATIONS_FOR_RED", "2"))

# ── Prompt del sistema ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Eres un sistema de análisis de tono conversacional. Tu única función es evaluar \
si una conversación se está desarrollando de forma cordial o si está derivando \
hacia un tono inaceptable.

IDIOMA: Analiza el fragmento en el idioma en que esté escrito. No traduzcas. \
Devuelve el campo "razon" en el mismo idioma que la conversación.

TAREA: Recibirás un fragmento de conversación reciente con marcas temporales \
relativas. Los fragmentos están ordenados cronológicamente del más antiguo al \
más reciente. Los fragmentos más recientes tienen mayor peso en tu evaluación.

---

CRITERIOS PARA TONO INACEPTABLE (rojo, puntuacion < 40):
- Insultos explícitos o implícitos dirigidos a personas presentes
- Descalificaciones personales ("eres un incompetente", "no sabes nada")
- Amenazas directas o veladas de cualquier tipo
- Tono despectivo sostenido durante más de una intervención consecutiva
- Sarcasmo agresivo o humillante dirigido a una persona
- Negaciones absolutas repetidas sobre la integridad del otro ("estás mintiendo")
- Escalada progresiva de agresividad aunque ninguna frase aislada sea grave

CRITERIOS PARA ZONA DE ADVERTENCIA (amarillo, puntuacion 40-65):
- Una frase de frustración o descalificación aislada sin escalada
- Tono tenso pero sin insulto directo
- Debate agresivo que no llega a descalificación personal
- Señales iniciales de escalada (interrupción, negación frecuente)

CRITERIOS PARA TONO ACEPTABLE (verde, puntuacion > 65):
- Desacuerdo expresado con respeto ("no estoy de acuerdo porque...")
- Debate técnico intenso sin descalificación personal
- Tono directo pero sin agresividad hacia personas
- Frustración expresada sin insulto ("esto no funciona", "no podemos seguir así")
- Fragmentos neutros o de silencio
- Una sola frase negativa aislada sin contexto de escalada

---

REGLAS CRÍTICAS:
1. Una frase aislada negativa NO es suficiente para rojo. Debe haber patrón o escalada.
2. El desacuerdo técnico enérgico NO es tono inaceptable.
3. La frustración sin insulto NO es tono inaceptable.
4. En caso de duda, devuelve verde. Es preferible un falso negativo a un falso positivo.
5. Evalúa la conversación colectiva, no a personas individuales.
6. Si el fragmento contiene silencio o es muy corto, devuelve verde con puntuacion 80.

---

FORMATO DE RESPUESTA:
Devuelve ÚNICAMENTE el siguiente JSON. Sin texto adicional. Sin markdown.
Sin explicaciones fuera del JSON. El JSON debe ser válido y parseable.

{
  "estado": "verde" | "rojo",
  "puntuacion": <entero entre 0 y 100, donde 100 es máxima cordialidad>,
  "tendencia": "mejorando" | "estable" | "empeorando",
  "razon": "<frase de máximo 15 palabras explicando el estado, en el idioma de la conversación. Vacío si estado es verde>"
}

El campo "tendencia" refleja cómo ha evolucionado el tono a lo largo del fragmento completo.
"""

# ── Buffer de contexto ─────────────────────────────────────────────────────────


@dataclass
class Fragment:
    text: str
    timestamp: float


class ContextBuffer:
    """Ventana deslizante de fragmentos de transcripción (sin persistencia)."""

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
        """Construye el texto para Qwen con marcas temporales relativas."""
        self._evict()
        now = time.time()
        lines = [
            f'[hace {int(now - f.timestamp)}s] "{f.text}"'
            for f in self._fragments
        ]
        return "\n".join(lines)

    def clear(self) -> None:
        self._fragments.clear()
        self._analyzed_count = 0


# ── Máquina de estados del semáforo ───────────────────────────────────────────


class SemaphoreStateMachine:
    """
    Estados públicos: verde | amarillo | rojo
    Histéresis:
      - verde   → requiere puntuacion > THRESHOLD_GREEN (1 evaluación)
      - amarillo → puntuacion entre THRESHOLD_RED y THRESHOLD_GREEN (inmediato)
      - rojo    → requiere puntuacion < THRESHOLD_RED durante CONFIRMATIONS_FOR_RED
                  evaluaciones consecutivas. Mientras no se confirma, se muestra amarillo.
      - rojo → verde: 1 sola evaluación con puntuacion > THRESHOLD_GREEN
    """

    def __init__(self) -> None:
        self.estado: str = "verde"
        self.puntuacion: int = 100
        self.tendencia: str = "estable"
        self.razon: str = ""
        self._red_confirmations: int = 0

    def update(self, result: dict) -> bool:
        """Aplica el resultado de Qwen. Devuelve True si el estado público cambió."""
        puntuacion = max(0, min(100, int(result.get("puntuacion", 100))))
        tendencia  = result.get("tendencia", "estable")
        razon      = result.get("razon", "")

        prev = self.estado

        if puntuacion > THRESHOLD_GREEN:
            new_estado = "verde"
            self._red_confirmations = 0
        elif puntuacion < THRESHOLD_RED:
            self._red_confirmations += 1
            # Pendiente de confirmación → se muestra como amarillo hasta confirmar
            new_estado = "rojo" if self._red_confirmations >= CONFIRMATIONS_FOR_RED else "amarillo"
        else:
            # Rango amarillo: 40–65
            new_estado = "amarillo"
            self._red_confirmations = 0

        self.estado     = new_estado
        self.puntuacion = puntuacion
        self.tendencia  = tendencia
        self.razon      = razon if new_estado == "rojo" else ""

        return new_estado != prev

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


# ── Sesión ─────────────────────────────────────────────────────────────────────


class Session:
    def __init__(self) -> None:
        self.buffer = ContextBuffer()
        self.semaphore = SemaphoreStateMachine()
        self.start_time = time.time()
        self.last_analysis_time: float = 0.0
        self.last_done_time: float = 0.0
        self.event_clients: list[WebSocket] = []

    async def broadcast(self, event: dict) -> None:
        msg = json.dumps(event, ensure_ascii=False)
        dead: list[WebSocket] = []
        for ws in list(self.event_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            with suppress(ValueError):
                self.event_clients.remove(ws)

    def reset(self) -> None:
        self.buffer.clear()
        self.semaphore.reset()
        self.start_time = time.time()
        self.last_analysis_time = 0.0
        self.last_done_time = 0.0

    def uptime(self) -> int:
        return int(time.time() - self.start_time)


# ── Análisis de tono (Qwen) ────────────────────────────────────────────────────


async def run_analysis(session: Session) -> None:
    if session.buffer.is_empty():
        return

    user_message = session.buffer.build_user_message()
    session.buffer.mark_analyzed()
    session.last_analysis_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                QWEN_HTTP_URL,
                json={
                    "model": QWEN_MODEL,
                    "temperature": 0.0,
                    "max_tokens": 150,
                    "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            result = json.loads(raw)

    except Exception as exc:
        logger.error("Qwen analysis failed: %s", exc)
        await session.broadcast({
            "type": "system.error",
            "message": f"Error en análisis de tono: {exc}",
            "recoverable": True,
        })
        return

    changed = session.semaphore.update(result)
    logger.info(
        "Analysis: estado=%s puntuacion=%d tendencia=%s%s",
        session.semaphore.estado,
        session.semaphore.puntuacion,
        session.semaphore.tendencia,
        " [CHANGED]" if changed else "",
    )

    if changed:
        await session.broadcast({
            "type": "semaphore.update",
            "estado":     session.semaphore.estado,
            "puntuacion": session.semaphore.puntuacion,
            "tendencia":  session.semaphore.tendencia,
            "razon":      session.semaphore.razon,
        })


# ── Scheduler de análisis — Condición B (tiempo máximo) ───────────────────────


async def analysis_scheduler(session: Session) -> None:
    """Dispara análisis si han pasado MAX_ANALYSIS_INTERVAL sin hacerlo y hay contenido nuevo."""
    while True:
        await asyncio.sleep(1.0)
        if (
            session.buffer.has_new_content()
            and (time.time() - session.last_analysis_time) >= MAX_ANALYSIS_INTERVAL
        ):
            await run_analysis(session)


# ── Bridge Voxtral (por sesión de audio) ──────────────────────────────────────


async def voxtral_bridge(browser_ws: WebSocket, session: Session) -> None:
    """
    Conecta con Voxtral y hace de puente entre el navegador y el modelo ASR.
    Dura mientras el WebSocket de audio del navegador esté activo.
    """
    try:
        async with websockets.connect(
            VOXTRAL_WS_URL,
            ping_interval=20,
            ping_timeout=20,
        ) as voxtral:
            logger.info("[VOXTRAL] Conectado: %s", VOXTRAL_WS_URL)

            # ── Inicialización ────────────────────────────────────────────────
            first = json.loads(await voxtral.recv())
            logger.info("[VOXTRAL] ← %s", json.dumps(first))
            if first.get("type") != "session.created":
                logger.warning("[VOXTRAL] Esperaba session.created, recibido: %s", first.get("type"))

            session_update = {"type": "session.update", "model": VOXTRAL_MODEL}
            logger.info("[VOXTRAL] → %s", json.dumps(session_update))
            await voxtral.send(json.dumps(session_update))

            # Espera confirmación de session.update antes de enviar audio
            upd_ack = json.loads(await voxtral.recv())
            logger.info("[VOXTRAL] ← %s", json.dumps(upd_ack))

            commit_msg = {"type": "input_audio_buffer.commit"}
            logger.info("[VOXTRAL] → %s", json.dumps(commit_msg))
            await voxtral.send(json.dumps(commit_msg))

            # Acumula deltas de la frase en curso para reconstruir el texto completo
            sentence_parts: list[str] = []
            audio_chunks_sent = 0

            # ── Task: recibe mensajes de Voxtral ──────────────────────────────
            async def recv_voxtral() -> None:
                async for raw in voxtral:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("[VOXTRAL] ← mensaje no-JSON: %r", raw[:120])
                        continue

                    msg_type = data.get("type", "?")

                    # Log completo de cada mensaje (truncamos campos de audio/delta largos)
                    log_data = {k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v)
                                for k, v in data.items()}
                    logger.info("[VOXTRAL] ← %s", json.dumps(log_data, ensure_ascii=False))

                    if msg_type == "transcription.delta":
                        delta = data.get("delta", "").strip()
                        if delta:
                            sentence_parts.append(delta)
                            session.buffer.add(delta)
                            await session.broadcast({
                                "type": "transcript.delta",
                                "timestamp_ms": int(time.time() * 1000),
                            })

                    elif msg_type == "transcription.done":
                        full_text = (
                            data.get("text", "").strip()
                            or " ".join(sentence_parts).strip()
                        )
                        logger.info("[VOXTRAL] transcription.done — texto: %r (partes: %d)",
                                    full_text, len(sentence_parts))
                        sentence_parts.clear()
                        session.last_done_time = time.time()
                        if full_text:
                            await session.broadcast({
                                "type": "transcript.done",
                                "text": full_text,
                            })
                        if (time.time() - session.last_analysis_time) >= MIN_ANALYSIS_INTERVAL:
                            asyncio.create_task(run_analysis(session))

                    elif msg_type == "error":
                        logger.error("[VOXTRAL] Error del servidor: %s", json.dumps(data))

            recv_task = asyncio.create_task(recv_voxtral())

            # ── Bucle principal: reenvía audio navegador → Voxtral ────────────
            try:
                while True:
                    data = await browser_ws.receive_text()
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")

                    if msg_type == "input_audio_buffer.append":
                        audio_chunks_sent += 1
                        if audio_chunks_sent <= 3 or audio_chunks_sent % 50 == 0:
                            logger.info("[VOXTRAL] → chunk de audio #%d (b64 len=%d)",
                                        audio_chunks_sent, len(msg.get("audio", "")))
                        await voxtral.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": msg.get("audio", ""),
                        }))

                    elif msg_type == "session.close":
                        logger.info("[VOXTRAL] Señal de cierre recibida del navegador")
                        break

            except WebSocketDisconnect:
                logger.info("[VOXTRAL] WebSocket de audio desconectado")
            finally:
                recv_task.cancel()
                logger.info("[VOXTRAL] Enviando commit final (final=true)")
                with suppress(Exception):
                    await voxtral.send(json.dumps({
                        "type": "input_audio_buffer.commit",
                        "final": True,
                    }))

    except Exception as exc:
        logger.error("Error en bridge Voxtral: %s", exc)
        await session.broadcast({
            "type": "system.error",
            "message": "Conexión con Voxtral perdida",
            "recoverable": True,
        })


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="monitor-conversacion — backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = Session()
_scheduler_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup() -> None:
    global _scheduler_task
    _scheduler_task = asyncio.create_task(analysis_scheduler(_session))
    logger.info("Backend iniciado. Scheduler de análisis activo.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _scheduler_task:
        _scheduler_task.cancel()


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    """Recibe audio PCM16 en base64 del navegador y lo reenvía a Voxtral."""
    await websocket.accept()
    logger.info("WebSocket /ws/audio conectado")

    # Nueva sesión: limpiar estado anterior
    _session.reset()

    await _session.broadcast({
        "type": "system.status",
        "connected": True,
        "session_seconds": 0,
    })

    # El bridge bloquea hasta que el navegador se desconecte
    await voxtral_bridge(websocket, _session)

    # Limpieza post-sesión
    _session.buffer.clear()
    logger.info("Sesión de audio finalizada")


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Emite eventos de transcripción y semáforo al frontend."""
    await websocket.accept()
    _session.event_clients.append(websocket)
    logger.info("WebSocket /ws/events conectado (%d clientes)", len(_session.event_clients))

    # Envía estado actual del semáforo al conectar
    await websocket.send_text(json.dumps({
        "type": "semaphore.update",
        "estado":     _session.semaphore.estado,
        "puntuacion": _session.semaphore.puntuacion,
        "tendencia":  _session.semaphore.tendencia,
        "razon":      _session.semaphore.razon,
    }))

    try:
        # Mantiene la conexión viva con pings de estado periódicos
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({
                "type": "system.status",
                "connected": True,
                "session_seconds": _session.uptime(),
            }))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        with suppress(ValueError):
            _session.event_clients.remove(websocket)
        logger.info("WebSocket /ws/events desconectado (%d clientes)", len(_session.event_clients))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "uptime_seconds": _session.uptime()}


@app.get("/status")
async def status() -> dict:
    return {
        "session_seconds": _session.uptime(),
        "event_clients":   len(_session.event_clients),
        "buffer_empty":    _session.buffer.is_empty(),
        "semaphore": {
            "estado":     _session.semaphore.estado,
            "puntuacion": _session.semaphore.puntuacion,
            "tendencia":  _session.semaphore.tendencia,
            "razon":      _session.semaphore.razon,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info")
