import pytest

import whispr.profiles as profiles
from whispr.profiles import (
    Profile,
    delete_profile,
    export_profile,
    list_profiles,
    load_profile,
    read_profile_file,
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


def test_export_then_read_round_trip(tmp_path):
    prof = Profile(name="Op Falcon", settings={"model": "medium.en"})
    prof.voiceprint_for("Alpha").add([1.0, 0.0])
    out = tmp_path / "falcon.whispr-profile.json"
    export_profile(prof, out)

    loaded = read_profile_file(out)
    assert loaded.name == "Op Falcon"
    assert loaded.settings["model"] == "medium.en"
    assert loaded.voiceprints["Alpha"].vectors == [[1.0, 0.0]]


def test_import_lands_in_profiles_dir(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    src = tmp_path / "shared.whispr-profile.json"
    export_profile(Profile(name="Shared", settings={"engine": "sherpa"}), src)
    # Simulate the GUI import: read the file, then save into this machine's dir.
    imported = read_profile_file(src)
    save_profile(imported)
    assert list_profiles() == ["Shared"]
    assert load_profile("Shared").settings["engine"] == "sherpa"


def test_read_profile_file_rejects_non_profile(tmp_path):
    bad = tmp_path / "notaprofile.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="isn't a Whispers profile"):
        read_profile_file(bad)


def test_read_profile_file_rejects_garbage(tmp_path):
    bad = tmp_path / "garbage.json"
    bad.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ValueError):
        read_profile_file(bad)
