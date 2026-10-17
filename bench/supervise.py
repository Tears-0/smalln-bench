"""Run resumable runners one after another, restarting on failure or stall.

Every runner skips blocks already written, so a restart costs at most one block.

  - stall watchdog: if the child prints nothing for --stall minutes it is killed and
    restarted (a GPU driver reset on 2026-10-14 03:52, nvlddmkm event 153, left both
    CUDA processes hung without exiting; nothing restarted them);
  - keep-awake: while running, the process asks Windows not to idle-sleep
    (SetThreadExecutionState; a per-process request, not a system setting).

    python -m bench.supervise --tries 8 --stall 60 -- -m bench.run --runner tabpfn --out main
    python -m bench.supervise --tries 8 --stall 60 -- STEP_A_ARGS ;; STEP_B_ARGS   (sequential steps)
"""
import os
import subprocess
import sys
import threading
import time


def keep_awake():
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def log(msg):
    print(f"[supervise {time.strftime('%F %T')}] {msg}", flush=True)


def run_step(args, tries, stall_min):
    for attempt in range(1, tries + 1):
        log(f"attempt {attempt}/{tries}: {' '.join(args)}")
        p = subprocess.Popen([sys.executable, "-u", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
        last = [time.time()]

        def relay():
            for line in p.stdout:
                last[0] = time.time()
                if "warn" not in line.lower():
                    sys.stdout.write(line); sys.stdout.flush()

        th = threading.Thread(target=relay, daemon=True); th.start()
        stalled = False
        while p.poll() is None:
            time.sleep(30)
            if time.time() - last[0] > stall_min * 60:
                stalled = True
                log(f"no output for {stall_min} min -- killing pid {p.pid} (stall)")
                p.kill()
                break
        p.wait(); th.join(timeout=10)
        log(f"exit code {p.returncode}{' (stalled)' if stalled else ''}")
        if p.returncode == 0 and not stalled:
            return True
        time.sleep(30)
    return False


if __name__ == "__main__":
    argv = sys.argv[1:]
    tries, stall = 5, 60
    while argv and argv[0] in ("--tries", "--stall"):
        if argv[0] == "--tries":
            tries = int(argv[1])
        else:
            stall = float(argv[1])
        argv = argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    steps, cur = [], []
    for a in argv:
        if a == ";;":
            steps.append(cur); cur = []
        else:
            cur.append(a)
    steps.append(cur)
    keep_awake()
    for i, step in enumerate(steps, 1):
        log(f"step {i}/{len(steps)}")
        if not run_step(step, tries, stall):
            log(f"step {i} failed after {tries} attempts -- stopping")
            sys.exit(1)
    log("all steps finished")
