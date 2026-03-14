# monitor-conversacion

Sistema de monitorización de conversaciones en tiempo real. Transcribe audio con Voxtral, evalúa el tono con Qwen y muestra el estado en un dashboard web con semáforo visual.

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
├── vllm-voxtral/
│   └── Dockerfile
├── vllm-qwen/
│   └── Dockerfile
└── dashboard/
    ├── Dockerfile
    └── index.html
```
