"""Gate the full run on the leakage suite (spec §8: tests must pass before any full run).

1. wait until the B1 permuted-label cells used by tests 1 and 1b exist
   (cpu and tabpfn at N = 12, 20; fusion at N = 20);
2. run the leakage suite (tests 1, 1b; tests 2-4 passed earlier, rerun here too);
3. only if it passes, launch the full run: cpu, tabpfn and fusion runners, each
   supervised and resumable. Blocks already written (the leakage cells) are skipped.

    python -m bench.orchestrate
Writes results/logs/orchestrate.log and results/logs/LEAKAGE_{PASSED,FAILED}.
"""
import os
import subprocess
import sys
import time

from .paths import RESULTS, ROOT, RUNS

LOGS = RESULTS / "logs"
EXPECTED = ([RUNS / "main" / "cpu" / f"B1_N{N}_perm_r{b:03d}.parquet" for N in (12, 20) for b in range(0, 100, 10)]
            + [RUNS / "main" / "tabpfn" / f"B1_N{N}_perm_r{b:03d}.parquet" for N in (12, 20) for b in range(0, 100, 10)]
            + [RUNS / "main" / "fus" / f"B1_N20_perm_r{b:03d}.parquet" for b in (0, 50)])

FULL = [
    ("main_cpu", ["-m", "bench.supervise", "--tries", "8", "--", "-m", "bench.run", "--runner", "cpu",
                  "--out", "main", "--workers", "16"]),
    ("main_tabpfn", ["-m", "bench.supervise", "--tries", "8", "--", "-m", "bench.run", "--runner", "tabpfn",
                     "--out", "main"]),
    ("main_fus", ["-m", "bench.supervise", "--tries", "8", "--", "-m", "bench.fusion", "--out", "main",
                  "--block", "50"]),
]


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "orchestrate.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    log(f"waiting for {len(EXPECTED)} leakage-cell files")
    while True:
        missing = [p for p in EXPECTED if not p.exists()]
        if not missing:
            break
        log(f"{len(EXPECTED) - len(missing)}/{len(EXPECTED)} present")
        time.sleep(300)
    log("leakage cells complete; running tests/test_no_leakage.py")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_no_leakage.py", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    (LOGS / "leakage_tests.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        (LOGS / "LEAKAGE_FAILED").write_text(r.stdout[-4000:], encoding="utf-8")
        log("LEAKAGE SUITE FAILED -- full run NOT launched; see leakage_tests.log")
        sys.exit(1)
    (LOGS / "LEAKAGE_PASSED").write_text(r.stdout[-2000:], encoding="utf-8")
    log("leakage suite passed; launching the full run")
    flags = 0x00004000 | 0x00000008 if os.name == "nt" else 0   # BELOW_NORMAL_PRIORITY_CLASS | DETACHED_PROCESS
    for name, args in FULL:
        out = open(LOGS / f"{name}.out.log", "a", encoding="utf-8")
        p = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                             creationflags=flags)
        log(f"started {name} pid {p.pid}")


if __name__ == "__main__":
    main()
