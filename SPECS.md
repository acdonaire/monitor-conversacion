# monitor-conversacion — Especificaciones del Proyecto

## Descripción general

Sistema de monitorización de conversaciones en tiempo real compuesto por tres servicios independientes orquestados con Docker Compose, desplegados en una instancia GPU en Verda Cloud (1x A100 80GB).

El sistema transcribe audio en tiempo real, analiza el tono y contenido de la conversación, y muestra el estado en un dashboard web con semáforo visual.

---

## Arquitectura

```
Audio (micrófono) → WebSocket → [vllm-voxtral] → chunks de texto
                                                         ↓
                                               [vllm-qwen] evaluador
                                                         ↓
                                        { estado: verde|amarillo|rojo, razon: "..." }
                                                         ↓
                                            [dashboard] web en tiempo real
                                         ┌──────────────────────────────┐
                                         │  Transcripción  │  Semáforo  │
                                         └──────────────────────────────┘
```

---

## Estructura de carpetas

```
monitor-conversacion/
├── docker-compose.yml
├── .env.example
├── README.md
├── vllm-voxtral/
│   └── Dockerfile
├── vllm-qwen/
│   └── Dockerfile
└── dashboard/
    ├── Dockerfile
    └── index.html
```

---

## Servicios

### 1. vllm-voxtral — Transcripción en tiempo real

- **Modelo:** `mistralai/Voxtral-Mini-4B-Realtime-2602`
- **Base image:** `vllm/vllm-openai:latest`
- **Puerto:** `8000`
- **Dependencias pip:** `soxr`, `librosa`, `soundfile`, `mistral-common>=1.9.0`
- **Variables de entorno:**
  - `VLLM_DISABLE_COMPILE_CACHE=1`
  - `HF_HOME=/root/.cache/huggingface`
- **Comando vLLM:**
  ```
  vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602
    --host 0.0.0.0
    --port 8000
    --compilation_config '{"cudagraph_mode": "PIECEWISE"}'
    --max-model-len 32768
  ```
- **Volumen para caché del modelo:** `/root/.cache/huggingface` → volumen Docker persistente
- **Healthcheck:** `curl -f http://localhost:8000/health`

#### Dockerfile existente (vllm-voxtral/Dockerfile):
```dockerfile
FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir \
    soxr \
    librosa \
    soundfile \
    "mistral-common>=1.9.0"

ENV VLLM_DISABLE_COMPILE_CACHE=1
ENV HF_HOME=/root/.cache/huggingface

ARG HF_TOKEN
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
    fi && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Voxtral-Mini-4B-Realtime-2602')" || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["vllm", "serve", \
     "mistralai/Voxtral-Mini-4B-Realtime-2602", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--compilation_config", "{\"cudagraph_mode\": \"PIECEWISE\"}", \
     "--max-model-len", "32768"]
```

---

### 2. vllm-qwen — Evaluador de conversación

- **Modelo:** `Qwen/Qwen2.5-7B-Instruct`
- **Base image:** `vllm/vllm-openai:latest`
- **Puerto:** `8001`
- **Variables de entorno:**
  - `HF_HOME=/root/.cache/huggingface`
- **Comando vLLM:**
  ```
  vllm serve Qwen/Qwen2.5-7B-Instruct
    --host 0.0.0.0
    --port 8001
    --max-model-len 32768
  ```
- **Volumen para caché del modelo:** `/root/.cache/huggingface` → mismo volumen compartido con voxtral (ahorra disco)
- **Healthcheck:** `curl -f http://localhost:8001/health`

#### Comportamiento esperado del evaluador:

Recibe como entrada el texto acumulado de la conversación (últimos N chunks) y devuelve **siempre** un JSON con esta estructura:

```json
{
  "estado": "verde" | "amarillo" | "rojo",
  "razon": "Explicación breve en una frase"
}
```

**Criterios de clasificación:**
- 🟢 `verde` → conversación normal, tono adecuado, sin palabras inapropiadas
- 🟡 `amarillo` → tono elevado, lenguaje límite, posible tensión, frases ambiguas
- 🔴 `rojo` → insultos, amenazas, lenguaje claramente inapropiado, conversación que debe detenerse

El prompt del sistema se definirá en la siguiente fase del proyecto.

---

### 3. dashboard — Interfaz web

- **Tecnología:** HTML + JavaScript puro (sin frameworks), servido con nginx
- **Puerto:** `80`
- **Layout:**
  - Columna izquierda (60%): transcripción en tiempo real, texto que va apareciendo
  - Columna derecha (40%): semáforo visual (círculo grande verde/amarillo/rojo) + razón del estado
- **Comportamiento:**
  - Conecta por WebSocket a `vllm-voxtral` para recibir chunks de transcripción
  - Cada N segundos (configurable, por defecto 5s) envía el texto acumulado al evaluador (`vllm-qwen`) via REST
  - Actualiza el semáforo según la respuesta del evaluador
  - El semáforo empieza en verde
  - Las transiciones de color son animadas (fade suave)

---

## docker-compose.yml — Estructura esperada

```yaml
version: '3.8'

services:
  vllm-voxtral:
    build: ./vllm-voxtral
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  vllm-qwen:
    build: ./vllm-qwen
    ports:
      - "8001:8001"
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  dashboard:
    build: ./dashboard
    ports:
      - "80:80"
    depends_on:
      - vllm-voxtral
      - vllm-qwen

volumes:
  hf_cache:
```

---

## .env.example

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Infraestructura — Verda Cloud

- **Instancia:** 1x A100 80GB
- **SO:** Ubuntu 22.04
- **Disco recomendado:** 150GB
- **VRAM estimada:**
  - Voxtral-Mini-4B: ~10GB
  - Qwen2.5-7B: ~15GB
  - Total: ~25GB de 80GB disponibles
- **Acceso:** SSH con clave ed25519

---

## Fases de implementación

1. ✅ Definir arquitectura y servicios
2. ⬜ Crear Dockerfile para vllm-qwen
3. ⬜ Crear docker-compose.yml completo
4. ⬜ Definir prompt del sistema para el evaluador
5. ⬜ Crear dashboard (HTML/JS/CSS)
6. ⬜ Crear Dockerfile para dashboard (nginx)
7. ⬜ Crear README.md con instrucciones de despliegue
8. ⬜ Desplegar en Verda y probar

---

## Notas técnicas

- Los dos modelos comparten el mismo volumen de caché de HuggingFace para evitar descargas duplicadas
- Voxtral y Qwen comparten la misma GPU A100; Voxtral tiene prioridad de carga
- El dashboard no tiene backend propio — consume directamente las APIs de los dos servicios vLLM
- El evaluador se llama con la API compatible OpenAI (`/v1/chat/completions`) que expone vLLM
- Se recomienda probar cada servicio individualmente antes de levantar el compose completo
