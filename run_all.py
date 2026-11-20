import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "preprocessing.py",
    "modelling.py",
    "classification.py",
    "derived_statistics.py",
    "make_figures.py",
]

start_at = 1
if "--from" in sys.argv:
    start_at = int(sys.argv[sys.argv.index("--from") + 1])

for number, script in enumerate(STEPS, start=1):
    if number < start_at:
        continue
    print(f"\n{'=' * 70}\nSTEP {number}: {script}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"\n{script} failed (exit code {result.returncode}). Stopping.")
    print(f"\n[{script} finished in {time.time() - t0:.0f}s]", flush=True)

print("\nAll steps finished. Results are in the outputs/ folder.")
