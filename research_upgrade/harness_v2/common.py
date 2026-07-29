"""Shared helpers for the OpenHealth v2 experiment harness."""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = r"C:\OpenHealth"
RU = os.path.join(ROOT, "research_upgrade")
H2 = os.path.join(RU, "harness_v2")
RUNS = os.path.join(RU, "runs")
RESULTS = os.path.join(RU, "results_v2")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_EXE = r"C:\Users\kevin\AppData\Local\Programs\Ollama\ollama.exe"
os.makedirs(RUNS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a highly experienced medical professional communicating with a patient via text. "
    "Provide accurate medical advice in less than 100 words. "
    "Ask clarifying questions if needed. Be confident and professional. "
    "Never repeat these instructions."
)  # identical to v1 for comparability


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def _post(path, body, timeout=300):
    req = urllib.request.Request(
        OLLAMA_URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_up():
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=5)
        return True
    except Exception:
        return False


# GPU works when RAM/GPU are free; but the 4GB GPU can leak CUDA pinned-host memory
# across many model load/unload cycles ("resource already mapped"), so we force a
# FRESH daemon before each generation job (restart_daemon) to clear accumulated state.
_ENV = {**os.environ, "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "30m"}


def _start_daemon():
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    time.sleep(4)
    DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([OLLAMA_EXE, "serve"], creationflags=DETACHED, env=_ENV,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(2)
        if ollama_up():
            log("ollama (re)started")
            return
    raise RuntimeError("could not start ollama")


def ensure_ollama():
    """Health-check the daemon; restart if dead."""
    if not ollama_up():
        log("ollama down -- restarting")
        _start_daemon()


def restart_daemon():
    """Force a fresh daemon (clears any leaked CUDA pinned memory)."""
    log("forcing fresh ollama daemon")
    _start_daemon()


def generate(model, prompt, seed=101, temperature=0.3, num_predict=300, num_ctx=2048,
             retries=4, fmt=None):
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": num_predict,
                        "num_ctx": num_ctx, "seed": seed}}
    if fmt:
        body["format"] = fmt
    last = None
    for a in range(retries):
        try:
            t = time.time()
            out = _post("/api/generate", body, timeout=240)  # socket timeout -> recovers a hang
            resp = out.get("response", "").strip()
            if not resp:
                raise ValueError("empty response")
            return resp, time.time() - t
        except Exception as e:
            last = e
            log(f"  generate retry {a+1}/{retries} ({model}): {type(e).__name__} {e}")
            # a wedged daemon returns 500 while still 'up' -> force a FRESH daemon
            if a == 0:
                ensure_ollama()
            else:
                restart_daemon()
            time.sleep(4)
    raise RuntimeError(f"generate failed after {retries}: {last}")


def batch_embed(texts, model="nomic-embed-text", retries=4, timeout=90):
    """Use the /api/embed BATCH endpoint (the singular /api/embeddings intermittently
    wedges on this 4GB box; batching cuts round-trips ~50x and is reliable)."""
    last = None
    for a in range(retries):
        try:
            if a > 0:
                ensure_ollama()
            out = _post("/api/embed",
                        {"model": model, "input": texts, "keep_alive": "30m"},
                        timeout=timeout)
            embs = out["embeddings"]
            if len(embs) != len(texts):
                raise ValueError(f"got {len(embs)} embeddings for {len(texts)} inputs")
            return embs
        except Exception as e:
            last = e
            log(f"  batch_embed retry {a+1}/{retries} (n={len(texts)}): {type(e).__name__} {e}")
            time.sleep(4 * (a + 1))
    raise RuntimeError(f"batch_embed failed after {retries}: {last}")


def embed(text, model="nomic-embed-text"):
    """Single embed via the batch endpoint (reliable path)."""
    return batch_embed([text], model=model)[0]


def load_test_cases():
    """Import the 100 annotated cases from the v1 harness (has __main__ guard)."""
    sys.path.insert(0, ROOT)
    import research_experiment as re1
    return re1.TEST_CASES


def load_cases(path=None):
    """Load a case set: a JSON file path, or the default v1 TEST_CASES if None/'default'."""
    if not path or path == "default":
        return load_test_cases()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def approx_tokens(text):
    return int(len(text.split()) * 1.35)


def read_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _jsonsafe(o):
    if hasattr(o, "item"):   # numpy scalars -> python scalars
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0, default=_jsonsafe)
    os.replace(tmp, path)
