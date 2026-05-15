"""nb001 — noise sweep at fixed hidden width.

Holds H=16 fixed and sweeps --noise across {0.05, 0.1, 0.2, 0.35, 0.5}.
Writes plots + summary.json to src/docs/public/notebooks/nb001.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "lib" / "cli.py"
OUT = ROOT / "docs" / "public" / "notebooks" / "nb001"
OUT.mkdir(parents=True, exist_ok=True)

NOISES = [0.05, 0.10, 0.20, 0.35, 0.50]
HIDDEN = 16
EPOCHS = 2000
LR = 0.05
SEED = 0

results = []
for n in NOISES:
    prefix = f"n{int(n*100):03d}_"
    print(f"\n=== noise={n} ===")
    proc = subprocess.run(
        [sys.executable, str(CLI),
         "--noise", str(n),
         "--hidden", str(HIDDEN),
         "--epochs", str(EPOCHS),
         "--lr", str(LR),
         "--seed", str(SEED),
         "--out", str(OUT),
         "--prefix", prefix],
        capture_output=True, text=True,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    final_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("final loss=")][0]
    loss = float(final_line.split("loss=")[1].split()[0])
    acc = float(final_line.split("acc=")[1])
    results.append({"noise": n, "loss": loss, "acc": acc})

summary = {
    "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR, "seed": SEED,
    "noises": NOISES, "results": results,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nsummary:")
print(json.dumps(summary, indent=2))
print(f"\nartifacts in {OUT}")
