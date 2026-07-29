"""The multi-day job queue. Detached, resume-safe, fail-soft.
Each job runs once (guarded by runs/<id>.done). A failed job writes runs/<id>.failed
and the queue CONTINUES to the next job. runs/STOP halts gracefully between jobs.
Everything is logged to runs/queue.log. Launch detached via launch_queue.ps1 so it
outlives the Claude Code session.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RUNS, ensure_ollama, restart_daemon, log as _log

QLOG = os.path.join(RUNS, "queue.log")
PY = sys.executable
OLLAMA = r"C:\Users\kevin\AppData\Local\Programs\Ollama\ollama.exe"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(QLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


CLEAN = r"C:\OpenHealth\research_upgrade\results_v2\clean_benchmark.json"
FT = "llama3,openhealth-doctor"

# (id, kind, payload).  kind: "py" -> [script,args...] ; "pull" -> model
# Order = scientific priority: clean-benchmark grid FIRST (the defensible primary),
# then mitigation, original grid, generalization pair, externals, extra seeds.
JOBS = [
    ("R0_corpus_build",       "py",   ["corpus_build.py"]),  # skipped (marker exists)
    ("R1_ctx_clean",          "py",   ["precompute_contexts.py", "--cases", CLEAN, "--tag", "clean"]),
    ("R1_ctx_orig",           "py",   ["precompute_contexts.py", "--cases", "default", "--tag", "orig"]),
    ("R2_grid_clean_s101",    "py",   ["gen_grid.py", "--models", FT, "--seed", "101", "--tag", "grid_clean_s101", "--cases", CLEAN, "--contexts", "contexts_cache_clean.json"]),
    ("A1_mitigation_clean",   "py",   ["prompt_mitigation.py", "--cases", CLEAN, "--contexts", "contexts_cache_clean.json", "--tag", "mitigation_clean"]),
    ("PULL_llama2",           "pull", "llama2"),
    ("PULL_medllama2",        "pull", "medllama2"),
    ("G1_pair2_clean_s101",   "py",   ["gen_grid.py", "--models", "llama2,medllama2", "--seed", "101", "--tag", "grid_pair2_clean_s101", "--cases", CLEAN, "--contexts", "contexts_cache_clean.json"]),
    ("R2_grid_orig_s101",     "py",   ["gen_grid.py", "--models", FT, "--seed", "101", "--tag", "grid_orig_s101", "--cases", "default", "--contexts", "contexts_cache_orig.json"]),
    ("B2_healthbench",        "py",   ["healthbench_run.py"]),
    ("B1_mcq_bench",          "py",   ["mcq_bench.py"]),
    ("R2_grid_clean_s102",    "py",   ["gen_grid.py", "--models", FT, "--seed", "102", "--tag", "grid_clean_s102", "--cases", CLEAN, "--contexts", "contexts_cache_clean.json"]),
    ("R2_grid_clean_s103",    "py",   ["gen_grid.py", "--models", FT, "--seed", "103", "--tag", "grid_clean_s103", "--cases", CLEAN, "--contexts", "contexts_cache_clean.json"]),
    ("PULL_mistral",          "pull", "mistral:7b-instruct"),
    # --- medical fine-tune PANEL (strengthens §4.11 vs the n=1 critique): additional
    # independently-trained medical fine-tunes run through the clean benchmark, to show
    # the keyword-judge validity collapse tracks the training corpus (only ChatDoctor-
    # lineage gamed) — a diagnostic validated across a panel, not a single case study.
    # --- IRR FIRST: local multi-judge (Qwen+Mistral) completes §4.6 WITHOUT Claude
    # (majority_irr combines these with the already-done Claude judgments). Emergency +
    # no-retrieval subset (144 cells) keeps it tractable (~72 min) on the 4GB GPU.
    ("J_local_clean",         "py",   ["judge_local.py", "clean_s101", "--emergency", "--cond=none"]),
    # --- then the medical-fine-tune PANEL generation (§4.11); its Claude judging waits
    # for the session-limit reset, but the generation is local and can proceed now.
    ("PULL_meditron",         "pull", "meditron:7b"),
    ("P1_panel_meditron",     "py",   ["gen_grid.py", "--models", "meditron:7b", "--seed", "101", "--tag", "panel_meditron", "--cases", CLEAN, "--contexts", "contexts_cache_clean.json"]),
    ("J_local_pair2",         "py",   ["judge_local.py", "pair2_clean_s101", "--emergency", "--cond=none"]),
]
# NOTE: edits here take effect on the next (re)launch; .done jobs are skipped.


def run_py(args):
    p = subprocess.run([PY] + args, cwd=HERE,
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.stdout:
        log("  | " + p.stdout.strip().replace("\n", "\n  | ")[-4000:])
    if p.returncode != 0:
        log("  ! stderr: " + (p.stderr or "")[-1500:])
    return p.returncode == 0


def run_pull(model):
    p = subprocess.run([OLLAMA, "pull", model], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log(f"  pull {model} rc={p.returncode} {(p.stderr or p.stdout or '')[-300:]}")
    return p.returncode == 0


def main():
    log("==== QUEUE START ==== jobs=" + ",".join(j[0] for j in JOBS))
    for jid, kind, payload in JOBS:
        if os.path.exists(os.path.join(RUNS, "STOP")):
            log("STOP file present -- halting."); return
        done = os.path.join(RUNS, jid + ".done")
        if os.path.exists(done):
            log(f"skip {jid} (done)"); continue
        # script existence check for py jobs (fail-soft if not yet implemented)
        if kind == "py" and not os.path.exists(os.path.join(HERE, payload[0])):
            log(f"SKIP {jid}: {payload[0]} not implemented yet"); continue
        log(f">>> START {jid} ({kind} {payload})")
        t0 = time.time()
        try:
            if kind == "py" and os.path.exists(os.path.join(HERE, payload[0])):
                restart_daemon()   # fresh daemon per generation job -> no leaked pinned mem
            else:
                ensure_ollama()
            ok = run_pull(payload) if kind == "pull" else run_py(payload)
        except Exception as e:
            log(f"  EXC {jid}: {e}"); ok = False
        dt = (time.time() - t0) / 60
        if ok:
            open(done, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"<<< DONE {jid} in {dt:.1f} min")
        else:
            open(os.path.join(RUNS, jid + ".failed"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"<<< FAILED {jid} in {dt:.1f} min -- continuing")
    log("==== QUEUE COMPLETE ====")
    # Signal completion only when every critical (non-external) py job has succeeded,
    # so the watchdog keeps relaunching to retry transiently-failed grids but does not
    # loop forever on external data-download jobs (B1/B2).
    critical = [j[0] for j in JOBS if j[1] == "py" and not j[0].startswith(("B1", "B2"))]
    if all(os.path.exists(os.path.join(RUNS, c + ".done")) for c in critical):
        open(os.path.join(RUNS, "ALL_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        log("ALL critical grids done -> wrote ALL_DONE (watchdog will stop)")


if __name__ == "__main__":
    main()
