import pytest

from especies.cli import main


def test_headless_runs_and_reports(capsys):
    main(["--ticks", "20", "--every", "10", "--seed", "1"])
    out = capsys.readouterr().out
    assert "semilla: 1" in out and "--- tick 20" in out


@pytest.mark.parametrize("flag", ["--every", "--ticks"])
def test_rejects_non_positive_counts(flag):
    with pytest.raises(SystemExit):
        main([flag, "0"])
