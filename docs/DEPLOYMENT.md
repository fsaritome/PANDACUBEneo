# Deploying to ai01 (GPU server)

The pipeline was migrated from a local Windows CPU-only machine to `ai01`, a
Linux server with 2× NVIDIA RTX 3090 GPUs, to make the dual-engine
(tesseract+paddleocr) OCR strategy fast enough to be practical (see
[BENCHMARKS.md](BENCHMARKS.md)).

## ai01 specs

- Hostname: `ai01` (reachable via `ssh ai01`, key-based auth already set up)
- 2× NVIDIA GeForce RTX 3090 (24,576 MiB each), driver 580.173.02
- AMD Ryzen Threadripper PRO 5955WX (16 cores / 32 threads)
- OS: Ubuntu 22.04 (jammy)
- Project path on server: `~/patent_ocr`
- Non-root user (`install`); sudo requires a password (not stored/known by
  tooling — must be typed interactively by a human when needed)

## System dependencies (installed via apt, requires sudo)

```bash
sudo apt-get update && sudo apt-get install -y \
    ghostscript libgl1 libglib2.0-0
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

Note: `tesseract-ocr` is no longer required — the pipeline uses PaddleOCR-VL
via Docker vLLM server (see below). Ghostscript is still required by OCRmyPDF.

`python3.11` is required (not the system default `python3.10.12`) because
`ocrmypdf>=17` (used locally, and required for the plugin hookspec this
project's `ocrmypdf_plugin.py` targets) needs Python >=3.11. Installing
`ocrmypdf` on Python 3.10 silently resolves to `16.13.0`, whose
`get_ocr_engine` hookspec doesn't accept the `options` argument our plugin
passes — this fails at pipeline startup with
`PluginValidationError: Argument(s) {'options'} are declared in the hookimpl
but can not be found in the hookspec`, not at install time, so it's easy to
miss until you actually run something.

## Python environment

```bash
cd ~/patent_ocr
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e '.[paddle]'   # pulls in paddleocr + CPU paddlepaddle by default
pip install pytest           # dev-only, not a runtime dependency
```

### Swapping in GPU-enabled PaddlePaddle

`pip install -e '.[paddle]'` installs the plain (CPU-only) `paddlepaddle`
wheel from PyPI. Replace it with the CUDA build matching the driver:

```bash
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

CUDA 12.6 was chosen because driver 580.173.02 supports it comfortably (no
system CUDA toolkit is installed on ai01 — the pip wheel bundles its own CUDA
12.6 runtime libraries, so no `nvcc`/toolkit install is needed).

Verify:

```bash
python -c "import paddle; print(paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
# Expected: True 2
```

## Config changes for ai01

In `config.yaml`:

```yaml
engine:
  engine_options:
    paddleocr: { use_gpu: true }   # false on the local CPU-only Windows machine
```

Runtime directories (`input/`, `output/`, `qc/`, `failed/`, `state/`) are
gitignored/excluded from the deployment tarball and must be created manually
on first deploy:

```bash
mkdir -p ~/patent_ocr/{input,output,qc,failed,state}
```

## GPU contention with vLLM

ai01 also runs a production vLLM instance (`vllm serve
cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4 --tensor-parallel-size=2 ...`, PID owned by
`root`) that normally occupies ~23.6GB/24GB VRAM on **each** GPU
(`--gpu-memory-utilization=0.88`). This leaves only a few hundred MB free per
GPU — not enough for PaddleOCR to load its detection/orientation/recognition
models, even with `FLAGS_initial_gpu_memory_in_mb` capped low. Attempting to
run PaddleOCR on GPU while vLLM is up fails with:

```
OSError: (External) CUDA error(2), out of memory.
```

**PaddleOCR and full-VRAM vLLM cannot run at the same time on this box.**
vLLM's presence is not constant, though — it has been observed idle/stopped
at times (0% utilization, only a few hundred MB used per GPU). Always check
current state before assuming contention:

```bash
ssh ai01 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv'
```

Options, in order of simplicity:
1. Stop vLLM while running OCR batches, restart it afterward (what was done
   for the benchmarks in [BENCHMARKS.md](BENCHMARKS.md)). vLLM must be
   stopped/restarted by a human with sudo — it runs as `root` and other
   systems may depend on it (it's also this project's own
   `fallback.base_url` LLM vision endpoint).
2. Reduce vLLM's `--gpu-memory-utilization` to free a fixed amount of VRAM
   for PaddleOCR permanently (not yet done/tested).
3. Run PaddleOCR on CPU on ai01 instead of GPU — still ~3x faster than the
   local Windows CPU (Threadripper vs local hardware) with zero GPU
   contention, at the cost of losing the ~10x GPU speedup (see
   [BENCHMARKS.md](BENCHMARKS.md)).

## Deployment tarball

The initial code transfer used a tarball excluding local/generated state:

```bash
tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='state' --exclude='input' --exclude='output' --exclude='qc' \
    --exclude='failed' --exclude='*.egg-info' --exclude='PDF STUFF' \
    --exclude='full_batch_dualengine.log' \
    -czf patent_ocr_deploy.tar.gz .
scp patent_ocr_deploy.tar.gz ai01:~/patent_ocr/
ssh ai01 "cd ~/patent_ocr && tar xzf patent_ocr_deploy.tar.gz && rm patent_ocr_deploy.tar.gz"
```

For subsequent code changes, individual files can be `scp`'d directly rather
than re-packaging the whole tree, e.g.:

```bash
scp patent_ocr/ocr/paddleocr_engine.py ai01:~/patent_ocr/patent_ocr/ocr/paddleocr_engine.py
```

## Verifying the deployment

```bash
ssh ai01 "cd ~/patent_ocr && source .venv/bin/activate && python -m pytest tests -q"
# Expected: 13 passed
```

## Detached background processes on ai01

Plain `nohup cmd &` over SSH is **not** sufficient on ai01 — the process
gets killed as soon as the SSH session that started it closes (observed with
`uvicorn`: `/tmp/api.log` showed a clean "Shutting down"/"Application
shutdown complete" immediately after the SSH command returned, despite
`nohup`). This looks like session-cleanup behavior killing lingering child
processes on logout.

**Fix**: fully detach with `setsid` (new session, no controlling terminal)
plus `disown` and explicit fd redirection:

```bash
ssh ai01 'cd ~/patent_ocr && source .venv/bin/activate && \
  setsid nohup <command> < /dev/null > /tmp/some.log 2>&1 & disown; sleep 1; echo started'
```

Verified by checking `ps aux` for the PID in a completely separate,
subsequent SSH connection — the process stays alive (`?` in the TTY column)
instead of dying when the launching session ends. Use this pattern for any
long-running service on ai01 (the dashboard API, ad-hoc sweeps you want to
survive a dropped connection, etc.).

**Orphaned subprocess gotcha**: if you need to abort a running sweep,
`pkill -f "patent_ocr.cli"` only kills the Python parent/worker processes —
it does **not** kill `tesseract` subprocess children they've already
spawned, which become orphaned and keep consuming CPU. Separately run
`pkill -9 -f "^tesseract "` (or the equivalent for other external engine
binaries) to fully clean up before starting a new run.

## PaddleOCR-VL Docker vLLM server

The primary OCR engine is PaddleOCR-VL-1.6, served via the official
PaddleOCR Docker genai-vllm-server image. This is required; direct
PaddlePaddle local inference is unstable (static-graph errors in production).

### One-time setup

**1. Add the `install` user to the `docker` group (requires interactive sudo):**

```bash
ssh ai01
sudo usermod -aG docker install
exit
# Reconnect — new SSH session is required for the group change to take effect
```

**2. Pull and start the vLLM server:**

```bash
docker run -d --rm --gpus all --network host --name paddleocr-vllm \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
  paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B \
    --host 0.0.0.0 --port 8118 --backend vllm
```

The image is ~13GB; first pull takes several minutes. Model load takes ~60s.
Verify it's ready:
```bash
curl http://localhost:8118/v1/models
```

**3. Install the genai client plugin in the pipeline venv:**

```bash
cd ~/patent_ocr
.venv/bin/pip install 'openai>=1.63'
```

This is the only client-side dependency — the `openai` package provides the
HTTP client that `paddlex`'s genai-client engine uses to call the vLLM server.

### Running the pipeline with vLLM backend

Set the following in `config.yaml`:

```yaml
engine:
  primary: paddleocr_vl
  engine_options:
    paddleocr_vl:
      use_gpu: true
      vl_rec_backend: "vllm-server"
      vl_rec_server_url: "http://localhost:8118/v1"
```

Then run as normal:
```bash
.venv/bin/python -m patent_ocr.cli --config config.yaml sweep
```

### Architecture notes

- Layout analysis (PP-DocLayoutV3) runs locally in the pipeline venv on GPU
- VLM recognition (PaddleOCR-VL-1.6-0.9B) runs in the Docker container via vLLM
- The container uses ~10GB VRAM on GPU 0; GPU 1 remains free
- `operates_on_full_page = True` on `PaddleOCRVLEngine` — the pipeline feeds
  whole page images (not pre-cropped regions) to this engine, matching
  PaddleOCR-VL's expected input; word-level results are distributed back into
  layout regions by bbox overlap afterwards
- OCRmyPDF still owns PDF/A sandwich composition; only text production changes

### Docker container management

```bash
docker ps                    # check if running
docker stop paddleocr-vllm   # graceful stop
docker logs paddleocr-vllm   # view server logs
```

The container uses `--rm` so it auto-removes on stop. Restart with the same
`docker run` command above. The vLLM server holds the model in GPU memory as
long as the container is running.

## Admin dashboard (FastAPI + React)

A live-monitoring/history dashboard lives in `patent_ocr/api/` (backend) and
`dashboard/` (frontend, built locally with Node/Vite since Node isn't
installed on ai01).

**Backend** — installed via the `api` extra:

```bash
pip install -e '.[api]'   # fastapi + uvicorn[standard]
```

Run (detached, per the pattern above):

```bash
ssh ai01 'cd ~/patent_ocr && source .venv/bin/activate && \
  PATENT_OCR_CONFIG=config.yaml setsid nohup uvicorn patent_ocr.api.app:app \
  --host 0.0.0.0 --port 8000 < /dev/null > /tmp/api.log 2>&1 & disown'
```

Env vars: `PATENT_OCR_CONFIG` (path to config.yaml, defaults to the CWD's
`config.yaml` via `load_config(None)`), `PATENT_OCR_DASHBOARD_DIST`
(defaults to `dashboard_dist`, relative to CWD).

**Frontend** — built locally (Windows dev machine) and shipped as static
files, since building the React app requires Node.js which isn't installed
on ai01:

```powershell
cd dashboard
npm install
npm run build           # -> dashboard/dist/
scp -r dist/. ai01:~/patent_ocr/dashboard_dist/
```

The FastAPI app mounts `dashboard_dist/` as static files at `/` (after all
`/api/*` routes, so they aren't shadowed) — nothing runs locally at
runtime, the local Node/Vite step only produces the static bundle that gets
copied to ai01. Access at `http://ai01:8000/`.

## Known Windows-vs-Linux code differences

None required — the codebase was audited for hardcoded Windows-specific
logic (`os.name`, `sys.platform`, `win32`, path separators, etc.) before
migrating and had none. `enable_mkldnn` in `patent_ocr/ocr/paddleocr_engine.py`
is now conditional on `use_gpu` (only passed/disabled on CPU, since mkldnn is
a CPU-only oneDNN optimization flag that doesn't apply on GPU and previously
crashed the CPU PIR executor when left at its default of `True`).
