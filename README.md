# AlphaDetect

**Modern end-to-end human-pose detection platform powered by [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).**  
AlphaDetect offers a production-ready CLI, an asynchronous FastAPI backend and a sleek Next.js/Tailwind frontend that work together to transform videos or image sequences into rich pose-estimation data and visualisations.

---

## 1  Project Overview
AlphaDetect was built to help researchers, engineers and creators extract accurate multi-person, whole-body joint positions from media with minimal effort.  It wraps the state-of-the-art AlphaPose engine in a highly-extensible micro-service architecture, delivering:

* **Batch & real-time inference** from the command line or REST/WebSocket API  
* **Interactive web UI** for project management, uploads and result exploration  
* **Scalable deployment** via Docker/Kubernetes with optional GPU workers  
* **Rich artefacts** — structured JSON, raw frames and key-point overlays

---

## 2  Key Features
| Area | Highlights |
|------|------------|
| Detection | Whole-body (136 kp) / COCO (17 kp) pose estimation, tracking & optional 3-D |
| Inputs | Local videos, image folders or remote URLs |
| Outputs | Pose JSON · `frames_*` raw dumps · `overlay_*` annotated frames |
| API | Async FastAPI, Swagger & WebSocket log/-progress streaming |
| UI | Next.js 14 (App Router), Tailwind CSS, dark/light, drag-and-drop uploads |
| DevOps | Loguru logging, SQLModel ORM, pre-commit, 90 %+ test coverage |

---

## 3  System Architecture

```
┌────────────┐  HTTPS/WS  ┌─────────────┐  asyncio/pipe  ┌─────────────┐
│  Frontend  │◀──────────▶│   FastAPI    │◀──────────────▶│ detect.py   │
│  Next.js   │            │    API       │   stdio/json   │ (AlphaPose) │
└────────────┘            └─────────────┘                └─────────────┘
               ▲                                       │
               └──────────────  outputs/  ─────────────┘
```

All artefacts live under `outputs/<taskId>/` for easy browsing and download.

---

## 4  Quick-Start (Local)

```bash
git clone https://github.com/your-org/alphadetect.git
cd alphadetect

# ①  Python env & deps  (powered by ultra-fast “uv”)
uv venv                                # creates .venv using the Python in .python-version
uv pip install -e ".[dev]"            # CLI + server (CPU-only)
#   └─  add ,gpu  extra for CUDA builds:  uv pip install -e ".[dev,gpu]"

# ② Front-end deps
npm install --prefix frontend

# ②.5 Install AlphaPose core (requires NumPy already installed)
uv pip install "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose"

# ④ Download AlphaPose weights (≈200 MB)
make download-models                  # or see docs/INSTALL.md

# ⑤ Run services
uvicorn server.app:app --reload       # http://localhost:8000
npm run dev --prefix frontend         # http://localhost:3000
```

---

## 5  Installation

| Layer | Cmd |
|-------|-----|
| Core (local) | `uv pip install -e .` |
| GPU build    | `uv pip install -e .[gpu]` *(CUDA 11.8 wheels)* |
| Frontend | `npm install --prefix frontend` |
| Docker | `docker compose up` *(Postgres + Redis + API + UI)* |

See [`docs/INSTALL.md`](docs/INSTALL.md) for OS-specific instructions, optional Postgres configuration and container profiles (CPU, GPU, prod).

---

## 6  Usage Examples

### CLI  

```bash
# Video
python cli/detect.py --video demo.mp4 --backend ultralytics

# Image folder with explicit output file
python cli/detect.py --image-dir samples/frames --output outputs/pose_run1.json --backend ultralytics
```

### API  

```bash
# Create task
curl -F file=@demo.mp4 http://localhost:8000/tasks

# Poll status
curl http://localhost:8000/tasks/<id>
```

Swagger / Redoc automatically available at `http://localhost:8000/docs`.

### Frontend  

1. Open `http://localhost:3000`  
2. **Upload** media via drag-and-drop  
3. Watch live progress and explore results in the gallery  

---

## 7  API Documentation

* **Swagger / OpenAPI**: `GET /docs`  
* **WebSocket**: `ws://<host>/ws/tasks/{taskId}` for live logs & status  
* Python client examples in [`docs/API.md`](docs/API.md) *(WIP)*

---

## 8  Frontend Highlights

* Responsive App-Router layout with breadcrumbs & dark mode  
* Project dashboard, task monitor, overlay gallery & stats widgets  
* Tool-tips, onboarding quick-start and progressive-disclosure settings  
* Built with Tailwind, Bootstrap Icons, Headless UI & SWR hooks  

See full design rationale in [`frontend/docs/DESIGN.md`](frontend/docs/DESIGN.md).

---

## 9  Development Setup

```bash
# Lint, format, type-check
make format lint typecheck

# Run the full test suite
make test                # pytest + coverage

# Live-reload back-end & UI
make run
```

Containerised workflow:

```bash
docker compose up        # CPU dev stack
docker compose --profile gpu up  # GPU workers

# Re-create exact Python env in CI  
uv pip sync requirements.txt   #  requirements.txt is **auto-generated** lock-file
```

---

## 10  Contributing

We 💙 contributions!  Please:

1. Fork & create feature branch (`feat/my-feature`)  
2. Ensure `make test` passes and add tests for new behaviour  
3. Follow [Conventional Commits](https://www.conventionalcommits.org/)  
4. Open a PR – the CI will run linting, typing and coverage gates

Read `docs/CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for details.

---

## 11  License

AlphaDetect is licensed under the **MIT License** – see [`LICENSE`](LICENSE).  
Note: AlphaPose models are released for **non-commercial research**; commercial usage may require separate permission from the original authors.

---

## 12  Acknowledgments

* **AlphaPose Team** – for the cutting-edge pose engine powering this project  
* **FastAPI**, **Next.js**, **Tailwind CSS** – for their exceptional open-source tooling  

> Made with passion by the AlphaDetect engineering team.  
> To contact the team, please send email to: Theodore Mui <theodoremui@gmail.com>
