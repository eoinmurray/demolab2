"""nb000 — hidden-width sweep for the tiny MLP.

Orchestrates src/lib/cli.py across hidden widths {1, 2, 4, 8, 16, 32}
and writes plots + a summary.json to src/docs/public/notebooks/nb000.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "lib" / "cli.py"
OUT = ROOT / "docs" / "public" / "notebooks" / "nb000"
OUT.mkdir(parents=True, exist_ok=True)

WIDTHS = [1, 2, 4, 8, 16, 32]
EPOCHS = 2000
LR = 0.05
SEED = 0

results = []
for h in WIDTHS:
    prefix = f"h{h:02d}_"
    print(f"\n=== hidden={h} ===")
    proc = subprocess.run(
        [sys.executable, str(CLI),
         "--hidden", str(h),
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
    results.append({"hidden": h, "loss": loss, "acc": acc})

summary = {
    "epochs": EPOCHS, "lr": LR, "seed": SEED,
    "widths": WIDTHS, "results": results,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nsummary:")
print(json.dumps(summary, indent=2))
print(f"\nartifacts in {OUT}")
