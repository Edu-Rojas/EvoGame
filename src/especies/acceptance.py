"""Corrida de aceptación: ¿cuántas especies sobreviven a largo plazo?

Corre varias semillas en paralelo (un proceso por semilla, cada una determinista) y
escribe una tabla en Markdown: especies vivas, tick de cada extinción y causa principal
de muerte por especie. Pensado para dejarlo corriendo de noche.

Uso:
    especies-acceptance --ticks 50000 --seeds 1 2 3 4 5 --min-alive 3 --out resultado.md
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .cli import positive_int
from .config import load_config
from .state import DeathCause, create_world
from .step import step

CAUSE_LABELS = {DeathCause.STARVATION: "hambre", DeathCause.OLD_AGE: "vejez",
                DeathCause.PREDATION: "cazado"}


@dataclass(frozen=True)
class SeedResult:
    seed: int
    ticks_run: int
    seconds: float
    alive: dict[str, int]              # especie -> población final
    extinct_at: dict[str, int]         # especie -> tick de extinción (solo extintas)
    main_cause: dict[str, str]         # especie -> causa de muerte más frecuente
    deaths: dict[str, dict[str, int]]  # especie -> causa -> muertes


def run_seed(seed: int, ticks: int, config: str | None = None) -> SeedResult:
    cfg = load_config(config)
    w = create_world(cfg, seed=seed)
    names = [s.name for s in cfg.species]
    t0 = time.perf_counter()
    for _ in range(ticks):
        step(w)
        if w.tick % 1000 == 0 and (w.extinct_at < 0).sum() <= 1:
            break                                  # queda una o ninguna: ya está decidido
    alive = {n: int((w.alive & (w.species == i)).sum()) for i, n in enumerate(names)}
    deaths = {n: {CAUSE_LABELS[c]: int(w.deaths_by_cause[i, c]) for c in DeathCause}
              for i, n in enumerate(names)}
    return SeedResult(
        seed=seed, ticks_run=w.tick, seconds=time.perf_counter() - t0, alive=alive,
        extinct_at={n: int(w.extinct_at[i]) for i, n in enumerate(names) if w.extinct_at[i] >= 0},
        main_cause={n: max(d, key=d.__getitem__) if sum(d.values()) else "-"
                    for n, d in deaths.items()},
        deaths=deaths,
    )


def report(results: list[SeedResult], ticks: int, min_alive: int) -> str:
    names = list(results[0].alive)
    lines = [
        f"Ticks objetivo: {ticks:,}. Criterio: al menos {min_alive} especies vivas.".replace(",", "."),
        "",
        "| Semilla | Vivas | " + " | ".join(names) + " | Ticks | Tiempo |",
        "|---|---|" + "---|" * len(names) + "---|---|",
    ]
    passed = 0
    for r in results:
        n_alive = sum(v > 0 for v in r.alive.values())
        ok = n_alive >= min_alive and r.ticks_run >= ticks
        passed += ok
        cells = []
        for n in names:
            if r.alive[n] > 0:
                cells.append(f"{r.alive[n]} vivos")
            else:
                cells.append(f"extinta t={r.extinct_at.get(n, '?')} ({r.main_cause[n]})")
        lines.append(f"| {r.seed} | {n_alive}{' ✓' if ok else ''} | " + " | ".join(cells)
                     + f" | {r.ticks_run} | {r.seconds:.0f} s |")
    lines += ["", f"**Semillas que cumplen: {passed} de {len(results)}.**", "",
              "Muertes por causa (todas las semillas):", ""]
    causes = list(CAUSE_LABELS.values())
    lines.append("| Especie | " + " | ".join(causes) + " |")
    lines.append("|---|" + "---|" * len(causes))
    for n in names:
        tot = [sum(r.deaths[n][c] for r in results) for c in causes]
        lines.append(f"| {n} | " + " | ".join(str(t) for t in tot) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="especies-acceptance", description="Corrida de aceptación.")
    ap.add_argument("--ticks", type=positive_int, default=50_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--min-alive", type=positive_int, default=3)
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=positive_int, default=None, help="procesos en paralelo")
    ap.add_argument("--out", default=None, help="archivo Markdown donde guardar la tabla")
    args = ap.parse_args(argv)

    jobs = args.jobs or len(args.seeds)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_seed, s, args.ticks, args.config) for s in args.seeds]
        results = [f.result() for f in futures]
    text = report(results, args.ticks, args.min_alive)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
