import whispr.profiles as profiles
from whispr.profiles import (
    Profile,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from whispr.voiceprints import Voiceprint


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "profiles_dir", lambda: tmp_path / "profiles")


def test_save_load_round_trip(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    prof = Profile(name="Op Nightfall", settings={"model": "small.en", "diarize": True})
    prof.voiceprint_for("Alpha").add([1.0, 0.0])
    prof.voiceprint_for("Bravo").add([0.0, 1.0])
    save_profile(prof)

    loaded = load_profile("Op Nightfall")
    assert loaded is not None
    assert loaded.name == "Op Nightfall"
    assert loaded.settings["model"] == "small.en"
    assert set(loaded.voiceprints) == {"Alpha", "Bravo"}
    assert loaded.voiceprints["Alpha"].vectors == [[1.0, 0.0]]


def test_list_profiles_sorted(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Zulu"))
    save_profile(Profile(name="Alpha"))
    assert list_profiles() == ["Alpha", "Zulu"]


def test_delete_profile(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Temp"))
    assert list_profiles() == ["Temp"]
    delete_profile("Temp")
    assert list_profiles() == []


def test_load_missing_profile_is_none(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert load_profile("nope") is None


def test_voiceprint_for_creates_and_reuses(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    prof = Profile(name="Op")
    first = prof.voiceprint_for("Alpha")
    again = prof.voiceprint_for("Alpha")
    assert first is again
    assert isinstance(first, Voiceprint)


def test_names_with_odd_characters_persist(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    save_profile(Profile(name="Op: Night/Fall #2"))
    # The display name is preserved even though the filename is slugged.
    assert list_profiles() == ["Op: Night/Fall #2"]
    assert load_profile("Op: Night/Fall #2") is not None
