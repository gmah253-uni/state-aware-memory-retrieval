"""
One-command reproduction of every number and figure in the report.

    python scripts/run_all.py            # full run: ~4 min on a laptop CPU
    python scripts/run_all.py --quick    # ~10 s smoke run in quick_run/ (results/ untouched)

Steps: src.experiments (main grid, beta sweep, pool-size sweep, summary
statistics, A-vs-B tests) -> src.plots (fig1-fig5) -> src.analysis (paired tests,
retrieval categories) -> src.render --figure (fig0, the environment figure). Run the property tests separately with
`python -m pytest`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(module: str, *args: str, cwd: Path) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd), f"(cwd={cwd.relative_to(ROOT) if cwd != ROOT else '.'})", flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small smoke run, written to quick_run/")
    args = ap.parse_args()

    if args.quick:
        cwd = ROOT / "quick_run"
        cwd.mkdir(exist_ok=True)
        run("src.experiments", "--quick", cwd=cwd)
        run("src.plots", cwd=cwd)
        run("src.analysis", cwd=cwd)
    else:
        cwd = ROOT
        run("src.experiments", cwd=cwd)
        run("src.plots", cwd=cwd)
        run("src.analysis", cwd=cwd)
        run("src.render", "--figure", cwd=cwd)
    print(f"\ndone: results in {cwd / 'results'}, figures in {cwd / 'figures'}")


if __name__ == "__main__":
    main()
