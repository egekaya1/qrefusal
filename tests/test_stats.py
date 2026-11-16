import pytest

from qrefusal.stats import clopper_pearson, holm, mcnemar_exact, paired


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 0 vs 6 discordant: 2 * 0.5**6
    assert mcnemar_exact(0, 6) == pytest.approx(2 * 0.5**6)
    assert mcnemar_exact(2, 8) == pytest.approx(0.109375)


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(5, 100)
    assert lo == pytest.approx(0.01643, abs=1e-4)
    assert hi == pytest.approx(0.11283, abs=1e-4)
    assert clopper_pearson(0, 50)[0] == 0.0
    assert clopper_pearson(50, 50)[1] == 1.0


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_paired():
    a = {1: True, 2: False, 3: False, 4: True}
    b = {1: True, 2: True, 3: True, 4: False, 5: True}
    r = paired(a, b)
    assert r["n"] == 4 and r["only_a"] == 1 and r["only_b"] == 2
