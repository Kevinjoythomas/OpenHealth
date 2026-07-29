"""Watchdog: keep the resume-safe queue alive over a long unattended run.
Single-instance check; relaunches run_queue.py detached (breakaway) if it is not
running. Meant to be invoked repeatedly (e.g. by a Windows Scheduled Task every
few minutes) and/or to run as its own detached loop. Honors runs/STOP."""
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
PY = r"C:\Users\kevin\anaconda3\python.exe"
DETACHED = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED|NEW_GROUP|BREAKAWAY_FROM_JOB


def _count(pattern):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-WmiObject Win32_Process | Where-Object {{ $_.Name -eq 'python.exe' -and $_.CommandLine -match '{pattern}' }} | Measure-Object).Count"],
        capture_output=True, text=True)
    try:
        return int((r.stdout or "0").strip() or "0")
    except Exception:
        return 0


def wlog(msg):
    with open(os.path.join(RUNS, "watchdog.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def relaunch_if_dead():
    if os.path.exists(os.path.join(RUNS, "STOP")):
        return "STOP present"
    if os.path.exists(os.path.join(RUNS, "ALL_DONE")):
        return "ALL_DONE"
    if _count(r"run_queue\.py") > 0:
        return "alive"
    try:
        subprocess.Popen([PY, os.path.join(HERE, "run_queue.py")], cwd=HERE,
                         creationflags=DETACHED, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
        wlog("queue was DEAD -> relaunched")
        return "relaunched"
    except OSError:
        subprocess.Popen([PY, os.path.join(HERE, "run_queue.py")], cwd=HERE,
                         creationflags=0x00000008 | 0x00000200,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wlog("queue was DEAD -> relaunched (no breakaway)")
        return "relaunched-nobreak"


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        print(relaunch_if_dead())
    else:  # loop mode
        while not os.path.exists(os.path.join(RUNS, "STOP")):
            relaunch_if_dead()
            time.sleep(90)
