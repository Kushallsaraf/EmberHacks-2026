"""Synthetic hip-height signals are test-only; never an experiment data source."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from ml.segment_reps import detect_reps, smooth_height


def squats(reps=5, fps=15., period=2., hold=.6):
    """Standing hold, then `reps` cosine dips in hip height, then a hold."""
    stand = np.full(int(hold * fps), .3)
    dip = .3 - .075 * (1 - np.cos(np.linspace(0, 2 * np.pi, int(period * fps), endpoint=False)))
    return np.concatenate([stand, np.tile(dip, reps), stand])


def test_one_range_per_rep_covering_the_bottom():
    h = squats()
    reps = detect_reps(h, 15.)
    assert len(reps) == 5
    bottoms = [int(np.argmin(h[a:b])) + a for a, b in reps]
    for (start, end), bottom in zip(reps, bottoms):
        assert start < bottom < end and end - start >= 8
    assert all(a[1] <= b[0] for a, b in zip(reps, reps[1:]))


def test_boundaries_include_standing_frames():
    h = squats()
    for start, end in detect_reps(h, 15.):
        assert h[start] > .27 and h[end - 1] > .27


def test_sparse_reps_are_dropped_not_interpolated():
    h = squats()
    first = detect_reps(h, 15.)[0]
    h[first[0]:first[1]][::2] = np.nan
    h[first[0]:first[1]][1::4] = np.nan
    assert first not in detect_reps(h, 15., min_valid=.35)
    assert len(detect_reps(h, 15., min_valid=0)) == 5


def test_flat_or_missing_signals():
    assert detect_reps(np.full(90, .3), 15.) == []
    with pytest.raises(ValueError):
        smooth_height(np.full(90, np.nan))
