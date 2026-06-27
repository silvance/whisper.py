from whispr import settings
from whispr.settings import load_settings, save_settings


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "settings.json"
    monkeypatch.setattr(settings, "settings_path", lambda: path)
    save_settings({"transcribe": {"model": "small", "diarize": True}})
    assert path.exists()
    assert load_settings() == {"transcribe": {"model": "small", "diarize": True}}


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "settings_path", lambda: tmp_path / "nope.json")
    assert load_settings() == {}


def test_load_corrupt_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(settings, "settings_path", lambda: path)
    assert load_settings() == {}


def test_load_non_dict_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(settings, "settings_path", lambda: path)
    assert load_settings() == {}


def test_save_is_silent_on_unwritable_path(tmp_path, monkeypatch):
    # A path whose parent can't be created shouldn't raise.
    bad = tmp_path / "file"
    bad.write_text("x", encoding="utf-8")  # now bad/ can't be a directory
    monkeypatch.setattr(settings, "settings_path", lambda: bad / "settings.json")
    save_settings({"a": 1})  # must not raise
