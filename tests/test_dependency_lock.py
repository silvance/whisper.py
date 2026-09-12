"""What a bundle is allowed to be built from.

The failure these hold shut is quiet by construction: the release workflow
installs ranges, so rebuilding the same commit months later can produce software
nobody field-tested, and nothing in the bundle would say so.
"""

import pytest

from whispr.asset_lock import (
    digest_for,
    empty,
    mismatch,
    record_file,
    record_revision,
    revision_for,
    summary,
)
from whispr.dependencies import (
    TRACKED_PACKAGES,
    canonical,
    drift,
    format_lock,
    parse_lock,
)

# -- The dependency lock ---------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("sherpa_onnx", "sherpa-onnx"),
        ("sherpa-onnx", "sherpa-onnx"),
        ("pyannote.audio", "pyannote-audio"),
        ("  Pillow  ", "pillow"),
        ("huggingface_hub", "huggingface-hub"),
    ],
)
def test_one_package_has_one_name(raw, expected):
    """Two spellings of one package must not read as two unpinned packages."""
    assert canonical(raw) == expected


def test_a_lock_round_trips():
    pins = {"ctranslate2": "4.8.2", "numpy": "1.26.4"}
    assert parse_lock(format_lock(pins, "# header\n")) == pins


def test_comments_and_blank_lines_are_not_pins():
    text = "# a comment\n\nnumpy==1.26.4  # trailing\n\n"
    assert parse_lock(text) == {"numpy": "1.26.4"}


def test_a_matching_install_is_not_drift():
    pins = {"ctranslate2": "4.8.2"}
    assert drift(pins, {"ctranslate2": "4.8.2"}) == []


def test_a_different_version_is_drift():
    problems = drift({"ctranslate2": "4.8.2"}, {"ctranslate2": "4.9.0"})
    assert len(problems) == 1
    assert "locked 4.8.2" in problems[0] and "installed 4.9.0" in problems[0]


def test_a_locked_package_that_is_not_installed_is_not_drift():
    """A transcribe-only bundle installs none of the optional engines."""
    pins = {"argostranslate": "1.11.0", "pytesseract": "0.3.13"}
    assert drift(pins, {}) == []


def test_a_tracked_package_missing_from_the_lock_is_drift():
    """The version nobody chose - the whole point of the exercise."""
    problems = drift({}, {"ctranslate2": "4.9.0"}, tracked=("ctranslate2",))
    assert len(problems) == 1
    assert "not in the lock" in problems[0]


def test_an_untracked_package_missing_from_the_lock_is_not_drift():
    """Locking every transitive package on every platform is a different job."""
    assert drift({}, {"some-transitive-thing": "1.0"}, tracked=("ctranslate2",)) == []


def test_the_engines_that_decide_the_answer_are_all_tracked():
    """If one of these can move under a rebuild, the bundle is not the bundle."""
    for name in ("faster-whisper", "ctranslate2", "sherpa-onnx", "pyannote.audio"):
        assert canonical(name) in {canonical(n) for n in TRACKED_PACKAGES}


def test_the_shipped_lock_covers_every_tracked_package():
    """A tracked package absent from the committed lock would fail every build."""
    from pathlib import Path

    lockfile = Path(__file__).resolve().parent.parent / "packaging" / "lockfile.txt"
    pins = parse_lock(lockfile.read_text(encoding="utf-8"))
    missing = [n for n in TRACKED_PACKAGES if canonical(n) not in pins]
    assert not missing, f"not pinned in packaging/lockfile.txt: {missing}"


# -- The asset lock --------------------------------------------------------


def test_nothing_is_pinned_before_the_first_build():
    lock = empty()
    assert revision_for(lock, "Systran/faster-whisper-medium") is None
    assert digest_for(lock, "https://example/model.onnx") is None


def test_a_recorded_revision_comes_back():
    lock = empty()
    assert record_revision(lock, "Systran/faster-whisper-medium", "abc123")
    assert revision_for(lock, "Systran/faster-whisper-medium") == "abc123"
    # Recording the same thing twice is not a change, so a build that resolved
    # what was already pinned does not rewrite the file.
    assert not record_revision(lock, "Systran/faster-whisper-medium", "abc123")


def test_a_recorded_digest_comes_back():
    lock = empty()
    assert record_file(lock, "https://example/model.onnx", "deadbeef", 12)
    assert digest_for(lock, "https://example/model.onnx") == "deadbeef"


def test_an_unpinned_asset_passes_because_there_is_nothing_to_check_yet():
    """The first build after a freeze records; it does not refuse."""
    assert mismatch(None, "deadbeef", "a model") is None


def test_matching_weights_pass():
    assert mismatch("deadbeef", "deadbeef", "a model") is None


def test_different_weights_are_refused_and_say_why():
    complaint = mismatch("deadbeef", "cafebabe", "segmentation.onnx")
    assert complaint is not None
    assert "segmentation.onnx" in complaint
    assert "deadbeef" in complaint and "cafebabe" in complaint
    assert "tested" in complaint


def test_a_summary_names_everything_pinned():
    lock = empty()
    record_revision(lock, "pyannote/segmentation-3.0", "abc123")
    record_file(lock, "https://example/model.onnx", "deadbeefcafebabe", 12)
    lines = summary(lock)
    assert any("pyannote/segmentation-3.0 @ abc123" == line for line in lines)
    assert any("https://example/model.onnx" in line for line in lines)


# -- Extending the lock from a real build ----------------------------------
#
# The lock was seeded from a resolution that could be done here. Some of it
# could not be: pyannote 3.1.1 brings its own tree, and resolving it needs the
# build machine. So a build records everything it installed, and the lock is
# extended from that - which means a pip freeze has to read as a lock.


def test_a_pip_freeze_reads_as_a_lock():
    """Same shape, so one can be poured into the other with no conversion."""
    freeze = "lightning==2.1.4\nasteroid-filterbanks==0.4.0\n-e git+ssh://x#egg=y\n"
    assert parse_lock(freeze) == {
        "lightning": "2.1.4",
        "asteroid-filterbanks": "0.4.0",
    }


def test_extending_a_lock_keeps_what_was_already_pinned():
    """Adding pyannote's tree must not quietly move faster-whisper."""
    existing = {"faster-whisper": "1.2.1"}
    from_build = {"lightning": "2.1.4", "faster-whisper": "1.2.1"}
    merged = dict(existing)
    for name in sorted(set(existing) | set(from_build)):
        if name in from_build:
            merged[name] = from_build[name]
    assert merged["faster-whisper"] == "1.2.1"
    assert merged["lightning"] == "2.1.4"


def test_an_unpinned_dependency_tree_is_not_reported_as_drift():
    """Until it is pinned it cannot fail a build - which is why it is a gap."""
    assert drift({"faster-whisper": "1.2.1"}, {"lightning": "2.1.4"}) == []
