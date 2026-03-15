# monitor-conversacion

> Una iniciativa de [ColoqIALab](https://coloquialab.es)

## Introducción

**monitor-conversacion** es un sistema de monitorización de conversaciones en tiempo real desarrollado como herramienta educativa por [ColoqIALab](https://coloquialab.es).

Su propósito es demostrar en talleres y workshops cómo combinar modelos de IA open source para construir aplicaciones prácticas:

- **Transcripción en tiempo real** con [Voxtral](https://mistral.ai/news/voxtral)
- **Análisis de tono conversacional** con [Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Visualización** en un dashboard web con semáforo visual (verde / amarillo / rojo)

El sistema detecta si una conversación deriva hacia un tono inapropiado y lo muestra de forma inmediata en el dashboard. Todo funciona **on-premise**, sin APIs externas, desplegado con Docker sobre GPU.

## Arquitectura

```
Audio (micrófono) → WebSocket → [vllm-voxtral] → chunks de texto
                                                        ↓
                                              [vllm-qwen] evaluador
                                                        ↓
                                     { estado: verde|amarillo|rojo, razon: "..." }
                                                        ↓
                                         [dashboard] web en tiempo real
```

![Arquitectura del sistema](docs/arquitectura_monitor_conversacion.svg)

## Cómo funciona el análisis

**Sistema de puntuación:** Qwen evalúa cada fragmento de conversación con una puntuación de 0 a 100 (100 = máxima cordialidad) cada 15–20 segundos.

**Estados del semáforo:**

| Estado | Puntuación | Significado |
|---|---|---|
| 🟢 Verde | > 65 | Tono cordial o neutro |
| 🟡 Amarillo | 40–65 | Tensión o frustración sin insulto directo |
| 🔴 Rojo | < 40 | Insultos, descalificaciones o amenazas |

**Histéresis:** el estado rojo requiere 2 evaluaciones consecutivas por debajo de 40 para activarse, evitando falsas alarmas por una frase aislada. La recuperación a verde ocurre con una sola evaluación por encima de 65.

**Buffer de contexto:** se mantiene una ventana deslizante de 90 segundos de conversación con marcas temporales relativas. Esto permite al modelo detectar tendencias y patrones de escalada, no solo frases aisladas.

**Criterios de evaluación:**
- **Verde** — desacuerdo técnico respetuoso, debate intenso sin descalificación personal, frustración expresada sin insulto
- **Amarillo** — tensión sostenida, tono agresivo sin insulto directo, señales iniciales de escalada
- **Rojo** — insultos explícitos o implícitos, descalificaciones personales, amenazas directas o veladas, escalada progresiva de agresividad

**Consejos de mediación:** cuando la conversación deriva hacia amarillo o rojo, el sistema genera un consejo concreto dirigido al grupo en segunda persona del plural para ayudar a reconducir la situación (ej: _"Os propongo hacer una pausa y retomar el punto de desacuerdo con calma"_).

**Privacidad:** ningún fragmento de audio ni texto se almacena en disco. Todo opera en memoria durante la sesión y se destruye al terminar.

## Cómo funciona el frontend

El dashboard es una página web estática que se abre en cualquier navegador moderno, sin necesidad de instalar nada en el cliente.

Al pulsar **Iniciar sesión**, el navegador solicita permiso para usar el micrófono. Una vez concedido, el audio se captura con la **Web Audio API** a 16 kHz, mono, en formato PCM16 — el mismo formato que espera Voxtral — y se envía en tiempo real al backend mediante un **WebSocket** en la ruta `/ws/audio`.

Simultáneamente, el dashboard mantiene abierto un segundo WebSocket en `/ws/events` para recibir eventos del backend:

- **`transcript.delta`** — el texto transcrito llega fragmento a fragmento y se añade al panel de transcripción en tiempo real, sin esperar a que termine la frase.
- **`semaphore.update`** — el semáforo cambia de color con una transición animada. Si la conversación está en amarillo o rojo, se muestran también la razón del estado y un consejo de mediación.

Todo el análisis ocurre en el servidor. El frontend es un visualizador puro: no evalúa el tono, no interpreta el audio, no toma decisiones. Solo muestra lo que el backend le envía.

El dashboard funciona en cualquier dispositivo con navegador: ordenador, tablet o móvil.

## Requisitos

- Docker + Docker Compose con soporte NVIDIA (`nvidia-container-toolkit`)
- GPU NVIDIA (probado en A100 80GB)
- Cuenta en HuggingFace con acceso a los modelos

## Configuración

```bash
cp .env.example .env
# Editar .env y poner tu HF_TOKEN
```

## Despliegue

```bash
# Construir e iniciar todos los servicios
docker compose up --build -d

# Ver logs
docker compose logs -f

# Verificar salud de los servicios
curl http://localhost:8000/health   # Voxtral
curl http://localhost:8001/health   # Qwen
```

El dashboard estará disponible en: **http://localhost**

## Servicios

| Servicio | Puerto | Modelo |
|---|---|---|
| vllm-voxtral | 8000 | mistralai/Voxtral-Mini-4B-Realtime-2602 |
| vllm-qwen | 8001 | Qwen/Qwen2.5-7B-Instruct |
| dashboard | 80 | nginx (HTML/JS) |

## Uso de VRAM estimado

- Voxtral-Mini-4B: ~10 GB
- Qwen2.5-7B: ~15 GB
- **Total: ~25 GB de 80 GB disponibles**

## Prueba individual de servicios

```bash
# Solo Voxtral
docker compose up vllm-voxtral

# Solo Qwen
docker compose up vllm-qwen

# Probar evaluador manualmente
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Hola, ¿cómo estás?"}
    ]
  }'
```

## Estructura del proyecto

```
monitor-conversacion/
├── docker-compose.yml
├── .env.example
├── README.md
├── SPECS.md
├── especificaciones-sistema-semaforo.md
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── dashboard/
│   ├── Dockerfile
│   ├── index.html
│   └── nginx.conf
├── vllm-voxtral/
│   └── Dockerfile
├── vllm-qwen/
│   └── Dockerfile
└── docs/
    └── arquitectura_monitor_conversacion.svg
```
