# World Compiler

World Compiler is an initial integrated AI platform scaffold that combines:

- NLP understanding through a BERT-inspired interface with lightweight heuristics
- Computer vision analysis through an OpenCV-backed adapter
- Dictionary and lexical lookup through a configurable provider wrapper with local fallback behavior
- A coordinating orchestration layer for multimodal composition
- A safety and empathy policy layer for supportive interactions

The repository's original deterministic world-generation pipeline is still present, and this v1 scaffold is added alongside it under `app/`.

## Architecture

```text
                +----------------------+
                |      FastAPI API     |
                |    app/main.py       |
                +----------+-----------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
 +-------------+   +---------------+   +--------------+
 | NLP Module   |   | Vision Module |   | Knowledge    |
 | tokenization |   | OpenCV        |   | provider API |
 | intent/sent. |   | size/edges    |   | + fallback   |
 +------+------+   +-------+-------+   +------+-------+
        \                  |                  /
         \                 |                 /
          +----------------+----------------+
                           |
                           v
                 +----------------------+
                 | Policy / Safety      |
                 | empathy + escalation |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Orchestrator         |
                 | trace + aggregation  |
                 +----------------------+
```

## Project structure

```text
app/
  api/routes/
  core/
  modules/
    nlp/
    vision/
    knowledge/
    policy/
  schemas/
  main.py
tests/
Dockerfile
requirements.txt
.github/workflows/ci.yml
```

## Setup

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Or run in Docker:

```bash
docker build -t world-compiler .
docker run -p 8000:8000 world-compiler
```

## Environment variables

- `WORLD_COMPILER_APP_NAME`
- `WORLD_COMPILER_APP_VERSION`
- `WORLD_COMPILER_DICTIONARY_API_URL`
- `WORLD_COMPILER_DICTIONARY_TIMEOUT`
- `WORLD_COMPILER_DICTIONARY_MAX_RETRIES`
- `WORLD_COMPILER_VISION_TIMEOUT`
- `WORLD_COMPILER_ALLOW_REMOTE_VISION_URLS`
- `WORLD_COMPILER_UNCERTAINTY_DISCLAIMER`

## API

### Health

```bash
curl http://localhost:8000/health
```

### Chat analyze

```bash
curl -X POST http://localhost:8000/v1/chat/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"I feel overwhelmed and need help planning my next step.","tone":"supportive"}'
```

### Vision analyze

```bash
curl -X POST http://localhost:8000/v1/vision/analyze \
  -F "file=@sample.png"
```

Or with a URL:

```bash
curl -X POST http://localhost:8000/v1/vision/analyze \
  -F "image_url=https://example.com/sample.png"
```

### Dictionary lookup

```bash
curl "http://localhost:8000/v1/knowledge/define?term=world"
```

### Compose

```bash
curl -X POST http://localhost:8000/v1/compose \
  -H "Content-Type: application/json" \
  -d '{"text":"Please help define world.","knowledge_terms":["world"],"tone":"supportive"}'
```

## Module notes

### NLP module

- Tokenization and preprocessing utilities are implemented in `app/modules/nlp/service.py`
- The classifier interface is intentionally lightweight and pluggable
- The current implementation is heuristic-based and prepared for future transformer adapters

### Vision module

- Uses OpenCV when available
- Supports upload bytes or safe `http/https` image URLs
- Computes image size, edge density, and a dominant-color approximation
- Gracefully degrades when OpenCV is unavailable

### Knowledge module

- `DictionaryProvider` abstraction with HTTP implementation
- Retries and timeout handling built into the provider
- Response normalization across provider shapes
- Local fallback definitions preserve the API contract on network failure

### Policy module

- Detects supportiveness level needs
- Flags high-risk terms for safe escalation messaging
- Supports `supportive`, `neutral`, and `concise` tone modes
- Adds uncertainty disclaimers when confidence is low

## Testing

Run all tests:

```bash
python -m pytest tests/ -v --tb=short
```

Focused platform tests:

```bash
python -m pytest tests/test_app_platform.py -v --tb=short
```

## CI

`.github/workflows/ci.yml` installs dependencies, compiles the Python sources, and runs the pytest suite on push and pull request events.

## Roadmap

- Replace heuristic NLP classification with a real transformer/BERT adapter
- Add stronger CV models for object and face detection beyond the v1 placeholder
- Add caching and richer provider federation for lexical knowledge
- Add streaming orchestration and structured observability
- Expand policy coverage with richer risk detection and human handoff integrations
