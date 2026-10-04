"""Corre la simulación sin gráficos y muestra cómo evolucionan las especies.

Uso:
    especies-headless                         # 3000 ticks, config por defecto
    especies-headless --ticks 10000 --every 1000 --seed 7
    especies-headless --config mi_prueba.toml
"""
from __future__ import annotations

import argparse
import time

from .config import load_config
from .genes import Gene
from .metrics import summarize
from .state import World, create_world
from .step import step

SHORT = {Gene.SIZE: "tam", Gene.ACCELERATION: "acel", Gene.AGGRESSION: "agr",
         Gene.DIET: "die", Gene.CAMOUFLAGE: "cam", Gene.DETECTION: "det",
         Gene.MATING: "apa"}


def print_report(w: World, ms_per_tick: float) -> None:
    total = int(w.alive.sum())
    print(f"\n--- tick {w.tick} | vivos {total} | nacidos {w.births_total} | "
          f"muertos {w.deaths_total} | {ms_per_tick:.2f} ms/tick ---")
    header = f"{'especie':<11}{'n':>6}{'gen':>5}{'energ':>7}  " + " ".join(f"{v:>5}" for v in SHORT.values())
    print(header)
    for s in summarize(w):
        if s.count == 0:
            print(f"{s.name:<11}{0:>6}  extinta")
            continue
        genes = " ".join(f"{s.mean_genes[g.key]:5.2f}" for g in SHORT)
        print(f"{s.name:<11}{s.count:>6}{s.max_generation:>5}{s.mean_energy_frac:>7.0%}  {genes}")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"debe ser un entero >= 1 (recibí {value})")
    return value


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="especies-headless", description="Simulación sin gráficos.")
    ap.add_argument("--config", default=None, help="TOML de config (por defecto, el del paquete)")
    ap.add_argument("--ticks", type=positive_int, default=3000)
    ap.add_argument("--every", type=positive_int, default=500, help="cada cuántos ticks imprimir")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

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
