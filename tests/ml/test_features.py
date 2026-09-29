"""Synthetic geometry is test-only; never an experiment data source."""
import pytest

np = pytest.importorskip("numpy")
from ml.features import angle, frame_features, resample, sequence_features, summary_features


def upright():
    p = np.full((33, 3), .9)
    for left, right, y in ((11, 12, .2), (23, 24, .4), (25, 26, .6), (27, 28, .8)):
        p[left] = [.4, y, .95]
        p[right] = [.6, y, .95]
    return p


def test_known_geometry():
    assert angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) == pytest.approx(90)
    assert np.isnan(angle(np.zeros(2), np.zeros(2), np.ones(2)))
    np.testing.assert_allclose(frame_features(upright()), [180, 180, 180, 180, 0, 1])


@pytest.mark.parametrize("fault", ["visibility", "nan", "ankle_overlap", "outside"])
def test_invalid_frame(fault):
    p = upright()
    if fault == "visibility":
        p[25, 2] = .1
    elif fault == "nan":
        p[11, 0] = np.nan
    elif fault == "ankle_overlap":
        p[27] = p[28]
    else:
        p[11, 0] = -1
    assert np.isnan(frame_features(p)).all()


def test_resampling_preserves_time_and_endpoints():
    x = np.array([[0.] * 6, [1.] * 6, [4.] * 6])
    actual = resample(x, 5, np.array([0., 1., 4.]))
    np.testing.assert_allclose(actual, np.repeat(np.arange(5)[:, None], 6, axis=1))
    assert summary_features(actual).shape == (24,)
    with pytest.raises(ValueError):
        resample(x, 5, np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        sequence_features(np.repeat(upright()[None], 7, axis=0))
    assert sequence_features(np.repeat(upright()[None], 8, axis=0)).shape == (32, 6)
