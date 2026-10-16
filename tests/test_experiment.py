import numpy as np
import pytest

from recstack.experiment import assign, binary_result, cuped, sample_size_per_arm, srm


def test_assignment_srm_and_binary_inference():
    assert assign("user-1") == assign("user-1")
    assert srm(900, 100)["flag_0_001"]
    assert not srm(500, 500)["flag_0_001"]
    result = binary_result([0] * 90 + [1] * 10, [0] * 80 + [1] * 20, bootstrap=200)
    assert result["difference"] == pytest.approx(0.1)
    assert result["bootstrap_ci95"][0] < result["difference"] < result["bootstrap_ci95"][1]


def test_power_and_cuped_on_correlated_fixture():
    n = sample_size_per_arm(0.2, 0.01)
    assert 20_000 < n < 30_000
    rng = np.random.default_rng(7)
    pre = rng.normal(size=1000)
    y = pre + rng.normal(0, 0.3, size=1000)
    adjusted, theta = cuped(y.tolist(), pre.tolist())
    assert 0.7 < theta < 1.3
    assert np.var(adjusted) < 0.2 * np.var(y)
