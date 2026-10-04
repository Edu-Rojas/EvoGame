from especies.acceptance import report, run_seed
from especies.state import DeathCause, create_world
from especies.step import run


def test_run_seed_reports_every_species():
    r = run_seed(seed=1, ticks=200)
    assert r.ticks_run == 200
    assert set(r.alive) == set(r.deaths) == set(r.main_cause)


def test_report_counts_passing_seeds():
    r = run_seed(seed=1, ticks=50)
    text = report([r], ticks=50, min_alive=1)
    assert "Semillas que cumplen: 1 de 1" in text


def test_deaths_are_counted_by_cause(cfg):
    from dataclasses import replace
    mixed = replace(cfg, sim=replace(cfg.sim, founder_spawn="uniform"))   # todos mezclados
    w = create_world(mixed, seed=2)
    run(w, 400)
    assert w.deaths_by_cause.sum() == w.deaths_total
    hunted = dict(zip([s.name for s in cfg.species],
                      w.deaths_by_cause[:, DeathCause.PREDATION], strict=True))
    assert hunted["Conejos"] > 0
    assert hunted["Elefantes"] == 0          # refugio por tamaño: demasiado grandes
    assert hunted["Lobos"] == 0              # nadie caza a su propia especie
