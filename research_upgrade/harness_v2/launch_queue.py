"""Launch run_queue.py as a fully-detached process that survives this session.
Uses CREATE_BREAKAWAY_FROM_JOB to escape any parent job object (falls back if
the job disallows breakaway)."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\kevin\anaconda3\python.exe"
DETACHED = 0x00000008        # DETACHED_PROCESS
NEWGROUP = 0x00000200        # CREATE_NEW_PROCESS_GROUP
BREAKAWAY = 0x01000000       # CREATE_BREAKAWAY_FROM_JOB


def spawn(flags):
    return subprocess.Popen(
        [PY, os.path.join(HERE, "run_queue.py")], cwd=HERE,
        creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, close_fds=True)


try:
    p = spawn(DETACHED | NEWGROUP | BREAKAWAY)
    print(f"launched detached+breakaway pid={p.pid}")
except OSError as e:
    p = spawn(DETACHED | NEWGROUP)
    print(f"breakaway unavailable ({e}); launched detached pid={p.pid}")
