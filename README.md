# Guerra de especies

An evolutionary ecosystem game. Each player designs a species by distributing points
across genes, and the creatures eat, mate with inheritance and mutation, hunt, age and
die in a shared world. Nobody scripts who wins: it emerges from the costs and benefits
of each gene.

**Current stage: 1c (in progress).** Herbivores and carnivores in a world with biomes:
hunting, fleeing, fights, rotting meat and tall leaves only large creatures can reach.
Stage 1c adds camouflage, hiding and stalking, and sprint fatigue. A development viewer
renders the world in pixel art with procedural animation.

> In-game text, species files and config keys are in Spanish, because the players are.

## Requirements

- Python 3.11 or newer (uses `tomllib`)

## Installation

```powershell
cd guerra-de-especies
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[viewer,dev]"
```

On Linux or macOS, use `source .venv/bin/activate` instead of the third line. For the
git hooks (secret scanning, ruff, personal-data guard), also run
`pip install pre-commit` and `pre-commit install`.

## Usage

| What | Command |
|---|---|
| Watch the world live | `python -m viewer` |
| Another seed / faster | `python -m viewer --seed 7 --tps 40` |
| Headless simulation | `especies-headless --ticks 5000 --every 500` |
| Acceptance run (seeds in parallel) | `especies-acceptance --ticks 50000 --out results.md` |
| Tests (all / fast only) | `pytest` / `pytest -m "not slow"` |
| Lint and type check | `ruff check .` and `mypy` |

Viewer controls:

| Key | Action |
|---|---|
| Mouse wheel | Zoom toward the cursor (up close you can see legs, claws and gills) |
| Right click + drag, or WASD | Move the camera |
| Left click | Select a creature: card with genes, instincts, parents, energy and detection radius |
| F | Camera follows the selected creature |
| P | Pixel size (2, 3 or 4) |
| Space | Pause |
| ↑ / ↓ | More or fewer ticks per second |
| Esc | Quit |

## Tuning the balance

Every balance number lives in `src/especies/data/default.toml`, which ships inside the
package. To experiment, copy it, edit it and run with `--config my_test.toml`. A
misspelled key or an out-of-range value fails to load with an error naming the section
and the key.

## Design highlights

- **Structure of arrays.** Creatures are columns of NumPy arrays with a fixed capacity
  and an `alive` mask, so each system updates every creature in one vectorized
  operation. A slot (where a creature lives in memory) is separate from its `uid` (who
  it is), so references survive slot reuse.
- **Systems in a fixed order.** A tick runs perception, decision, movement, combat,
  feeding, reproduction, metabolism and death as pure functions over the world state.
- **Utility-based decisions.** Each action gets a score from a base instinct times a
  heritable instinct; the highest score wins.
- **Deterministic.** All randomness comes from a single seeded generator: the same seed
  produces the same run.
- **Strict config.** TOML loaded into frozen dataclasses; unknown keys and invalid
  values are rejected.
- **Simulation separate from rendering.** The simulation knows nothing about graphics;
  the procedural animation is plain geometry that any renderer can draw.

The reasoning behind each decision is recorded in [docs/adr](docs/adr).

## Project layout

```
src/especies/        pure simulation (unaware that graphics exist)
  config.py          TOML -> immutable dataclasses; validates keys, ranges and budget
  data/default.toml  every balance number
  cli.py             especies-headless: per-species reports without graphics
  geometry.py        toroidal world: wrapping and shortest path
  genes.py           genes, uniform crossover, mutation, stochastic rounding
  state.py           World: creatures as array columns + derived traits
  terrain.py         biomes; grass, tall leaves and meat per cell
  perception.py      food, mate, prey and threat (per-species KD-trees)
  decision.py        utility per action = base instinct x heritable instinct
  physics.py         movement, fleeing toward cover, philopatry, separation, fatigue
  combat.py          bites, retaliation, regeneration
  metabolism.py      feeding, energy cost (Kleiber + eyes + movement + biome), death and meat
  disease.py         density-dependent disease
  reproduction.py    mating, litter size, inheritance
  step.py            order of the systems in a tick
  metrics.py         per-species summaries and creature card (raw data)
  acceptance.py      especies-acceptance: long multi-seed runs
viewer/              pygame viewer (read-only access to the state)
  animation.py       procedural NumPy rig: spine, IK legs, claws, gills
  palette.py         ink, paper and per-species colors
  render.py          pixel art: dithered terrain with hand-drawn decor, creatures with LOD
  app.py             camera, input, interpolation between ticks and HUD
tests/               pytest, one file per module
docs/adr/            architecture decision records
```

## Status

- **Active genes:** Size, Acceleration, Aggression, Diet, Detection and Mating.
  Camouflage is being activated in stage 1c.
- **Predators** hunt prey of other species up to `max_prey_ratio` times their own size
  (size refuge), get sated, give up on long chases and leave meat behind.
- **Tall forest leaves** give large creatures a niche of their own.
- **Coexistence:** without stabilizing mechanisms, the fastest-breeding species drives
  the rest extinct within a generation. The mechanisms that prevent it are covered in
  [tests/test_stabilizers.py](tests/test_stabilizers.py). At the end of stage 1b, at
  least 3 species survived 50,000 ticks in 4 of 5 seeds.

### Roadmap

| Stage | Content |
|---|---|
| 1c | Camouflage, hiding and stalking, sprint fatigue. Goal: 3+ species alive at 200,000 ticks |
| 2 | One species file per player, species creation wizard, rankings and events |
| 3 | Discord integration: major events and a daily summary |
| 4 | FastAPI + PostgreSQL server and a PixiJS web client |

## More

- [CHANGELOG.md](CHANGELOG.md): changes per version
- [SECURITY.md](SECURITY.md): how to report a vulnerability
