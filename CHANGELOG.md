# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `especies-headless` console script and `python -m viewer` entry points.
- Strict range validation for every config section, plus cross-section checks
  (capacity fits all founders, world size is a whole number of terrain cells).
- `[genes] inactive`: genes with no effect yet do not mutate.
- `CreatureInfo`: typed, raw creature data ready for an API (`dataclasses.asdict`).
- Pixel-art viewer: low-resolution canvas, ink-outlined lizard-like creatures with
  clawed feet, gills and shadows, dithered pastel terrain with hand-drawn decor,
  three levels of detail and switchable pixel size (`P`).
- Test suite split per module with a `make_world` fixture (73 tests); slow
  evolution runs are marked `slow`.
- ruff (including security rules) and mypy configuration.
- CI: lint, type check, tests on Linux and Windows (Python 3.11 and 3.13),
  secret scanning and dependency audit.
- Architecture decision records in `docs/adr/`.

### Changed
- The default config ships inside the package (`src/especies/data/default.toml`)
  and is loaded with `importlib.resources`.
- Remaining balance numbers moved to the config (litter size now follows
  `max_litter`); geometric factors became named constants.
- Torus helpers live in a dependency-free `geometry` module.
- `Action` is an `IntEnum`.

### Fixed
- Unknown sections or species keys in the config were silently ignored, and
  invalid values (such as `think_interval = 0`) were accepted.
- Genes stuck at the minimum drifted upward through clipped mutation.
- At full capacity, mating couples paid the energy cost without offspring.
- `spawn` could place a creature exactly on the world edge, crashing the KD-tree.
- Two bodies on the same point drifted together instead of separating.
- The viewer kept running 50 ticks per frame after lowering the speed.
- Newborns in recycled slots were interpolated from the previous occupant, and
  pausing jumped the world one tick back.
- Extinct species printed rows of `nan` in the headless report.

### Security
- `.gitignore` covers environment files, private keys and local overrides.

## [0.1.0] - 2026-10-04

### Added
- Stage 1a: herbivores in a toroidal world with biomes and regrowing grass,
  sexual reproduction with uniform crossover and mutation, utility-based
  decisions and a development viewer with procedural animation.
