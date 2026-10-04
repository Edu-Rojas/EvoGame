"""Corre la simulación sin gráficos y muestra cómo evolucionan las especies.

Uso:
    python scripts/headless.py                    # 3000 ticks, config por defecto
    python scripts/headless.py --ticks 10000 --every 1000 --seed 7
    python scripts/headless.py --config config/mi_prueba.toml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from especies.config import load_config  # noqa: E402
from especies.genes import Gene
from especies.metrics import summarize
from especies.state import create_world
from especies.step import step

SHORT = {Gene.SIZE: "tam", Gene.ACCELERATION: "acel", Gene.AGGRESSION: "agr",
         Gene.DIET: "die", Gene.CAMOUFLAGE: "cam", Gene.DETECTION: "det",
         Gene.MATING: "apa"}


def print_report(w, ms_per_tick: float) -> None:
    total = int(w.alive.sum())
    print(f"\n--- tick {w.tick} | vivos {total} | nacidos {w.births_total} | "
          f"muertos {w.deaths_total} | {ms_per_tick:.2f} ms/tick ---")
    header = f"{'especie':<11}{'n':>6}{'gen':>5}{'energ':>7}  " + " ".join(f"{v:>5}" for v in SHORT.values())
    print(header)
    for s in summarize(w):
        genes = " ".join(f"{s.mean_genes[g.key]:5.2f}" for g in SHORT)
        print(f"{s.name:<11}{s.count:>6}{s.max_generation:>5}{s.mean_energy_frac:>7.0%}  {genes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    w = create_world(cfg, seed=args.seed)
    print(f"semilla: {w.seed}  (repite exactamente esta corrida con --seed {w.seed})")
    print_report(w, 0.0)

    t0 = time.perf_counter()
    last = t0
    for i in range(1, args.ticks + 1):
        step(w)
        if i % args.every == 0:
            now = time.perf_counter()
            print_report(w, (now - last) * 1000 / args.every)
            last = now
        if not w.alive.any():
            print(f"\nExtinción total en el tick {w.tick}.")
            break
    print(f"\nTotal: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
