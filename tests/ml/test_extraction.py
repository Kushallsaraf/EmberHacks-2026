"""Empty-data guards and synthetic blank-video rejection, never research data."""
import pytest

np = pytest.importorskip("numpy")
from ml.extract_features import read_manifest, run


def test_empty_clips_create_no_archive(tmp_path):
    output = tmp_path / "features.npz"
    with pytest.raises(ValueError, match="No real clips"):
        run(tmp_path / "clips", tmp_path / "manifest.csv", output)
    assert not output.exists()


def test_manifest_requires_provenance(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("clip,subject_id,source_url,license,label_source\ngood/a.mp4,,,,\n")
    with pytest.raises(ValueError, match="nonempty"):
        read_manifest(path)


def test_missing_training_archive_creates_no_results(tmp_path):
    pytest.importorskip("torch")
    from ml.train_form_classifier import load_data
    with pytest.raises(ValueError, match="Add real clips"):
        load_data(tmp_path / "missing.npz")
    assert not list(tmp_path.iterdir())


def test_real_mediapipe_api_on_blank_frame():
    mp = pytest.importorskip("mediapipe")
    assert mp.__version__ == "0.10.21"
    assert hasattr(mp, "solutions")
    with mp.solutions.pose.Pose(model_complexity=1) as pose:
        result = pose.process(np.zeros((128, 128, 3), dtype=np.uint8))
    assert result.pose_landmarks is None


def test_blank_video_rejected_without_an_archive(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from ml.extract_features import extract_clip
    path = tmp_path / "TEST-ONLY-BLANK.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 128))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="Only 0 valid frames"):
        extract_clip(path, 32)
    assert not list(tmp_path.glob("*.npz"))
