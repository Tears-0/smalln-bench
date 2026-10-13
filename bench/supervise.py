"""Restart a resumable runner until it finishes (every runner skips finished blocks).

    python -m bench.supervise --tries 5 -- -m bench.run --runner tabpfn --out main
"""
import subprocess
import sys
import time

if __name__ == "__main__":
    args = sys.argv[1:]
    tries = 5
    if args[:1] == ["--tries"]:
        tries, args = int(args[1]), args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    for attempt in range(1, tries + 1):
        print(f"[supervise {time.strftime('%F %T')}] attempt {attempt}/{tries}: {' '.join(args)}", flush=True)
        code = subprocess.call([sys.executable, *args])
        print(f"[supervise {time.strftime('%F %T')}] exit code {code}", flush=True)
        if code == 0:
            sys.exit(0)
        time.sleep(30)
    sys.exit(1)
