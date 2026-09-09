import json

import pytest

from whispr import app, bundle_check
from whispr.bundle_check import (
    INVENTORY_NAME,
    build_inventory,
    load_inventory,
    verify,
)


def _bundle(tmp_path):
    """A stand-in for a built one-dir bundle: an executable and its libraries."""
    root = tmp_path / "whispr"
    (root / "_internal").mkdir(parents=True)
    (root / "whispr.exe").write_bytes(b"MZ" + b"\x00" * 60)
    (root / "_internal" / "_socket.pyd").write_bytes(b"socket-extension")
    (root / "_internal" / "python311.dll").write_bytes(b"python-runtime")
    (root / "_internal" / "tcl8.6").mkdir()
    (root / "_internal" / "tcl8.6" / "init.tcl").write_text("# tcl", encoding="utf-8")
    return root


def _write_inventory(root):
    (root / INVENTORY_NAME).write_text(
        json.dumps(build_inventory(root)), encoding="utf-8"
    )
    return root


# -- recording what was shipped --------------------------------------------


def test_inventory_covers_every_file_but_itself(tmp_path):
    root = _bundle(tmp_path)
    inventory = build_inventory(root)
    paths = {entry["path"] for entry in inventory["files"]}
    assert paths == {
        "whispr.exe",
        "_internal/_socket.pyd",
        "_internal/python311.dll",
        "_internal/tcl8.6/init.tcl",
    }
    # A file cannot record its own digest.
    assert INVENTORY_NAME not in paths
    assert inventory["file_count"] == 4
    assert inventory["total_bytes"] > 0


def test_paths_are_recorded_platform_independently(tmp_path):
    """A bundle is built on one machine and checked on another."""
    inventory = build_inventory(_bundle(tmp_path))
    assert all("\\\\" not in entry["path"] for entry in inventory["files"])


# -- the failures this exists to name --------------------------------------


def test_an_intact_copy_passes(tmp_path):
    root = _write_inventory(_bundle(tmp_path))
    result = verify(root)
    assert result.intact
    assert result.usable
    assert result.checked == 4
    assert "matches the build" in "\n".join(result.summary_lines())


def test_a_missing_library_is_named(tmp_path):
    """The reported symptom: an extension present at build, gone on the desktop."""
    root = _write_inventory(_bundle(tmp_path))
    (root / "_internal" / "_socket.pyd").unlink()
    result = verify(root)
    assert not result.intact
    assert result.missing == ["_internal/_socket.pyd"]
    assert "_internal/_socket.pyd" in "\n".join(result.summary_lines())


def test_a_missing_data_directory_is_named(tmp_path):
    """The second symptom, from the same copy: Tcl's own files gone."""
    root = _write_inventory(_bundle(tmp_path))
    (root / "_internal" / "tcl8.6" / "init.tcl").unlink()
    result = verify(root)
    assert result.missing == ["_internal/tcl8.6/init.tcl"]


def test_a_truncated_file_is_caught_by_size_before_hashing(tmp_path):
    """A short file is the signature of a copy that stopped early."""
    root = _write_inventory(_bundle(tmp_path))
    (root / "_internal" / "python311.dll").write_bytes(b"pyth")
    result = verify(root)
    assert result.altered == ["_internal/python311.dll"]


def test_an_altered_file_of_the_same_size_is_still_caught(tmp_path):
    root = _write_inventory(_bundle(tmp_path))
    (root / "_internal" / "python311.dll").write_bytes(b"PYTHON-RUNTIME")
    result = verify(root)
    assert result.altered == ["_internal/python311.dll"]


def test_several_problems_are_all_reported_not_just_the_first(tmp_path):
    """The point of the check: a list, not whichever file happened to be first."""
    root = _write_inventory(_bundle(tmp_path))
    (root / "_internal" / "_socket.pyd").unlink()
    (root / "_internal" / "tcl8.6" / "init.tcl").unlink()
    (root / "_internal" / "python311.dll").write_bytes(b"short")
    result = verify(root)
    assert sorted(result.missing) == [
        "_internal/_socket.pyd",
        "_internal/tcl8.6/init.tcl",
    ]
    assert result.altered == ["_internal/python311.dll"]


def test_extra_files_are_reported_but_do_not_fail_the_check(tmp_path):
    """An operator may keep their own files beside the application."""
    root = _write_inventory(_bundle(tmp_path))
    (root / "notes.txt").write_text("case 12", encoding="utf-8")
    result = verify(root)
    assert result.intact
    assert result.extra == ["notes.txt"]


# -- when the check itself cannot run --------------------------------------


def test_no_inventory_is_reported_as_uncheckable_not_as_intact(tmp_path):
    """A copy that cannot be checked has not passed."""
    result = verify(_bundle(tmp_path))
    assert not result.usable
    assert "nothing to check this copy against" in result.reason


def test_a_newer_inventory_schema_is_refused(tmp_path):
    path = tmp_path / INVENTORY_NAME
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="newer version"):
        load_inventory(path)


def test_running_from_source_says_there_is_nothing_to_check():
    result = verify()
    assert not result.usable
    assert "not an installed bundle" in result.reason


# -- the command an operator actually runs ---------------------------------


def test_verify_exit_codes_separate_broken_from_uncheckable(tmp_path, monkeypatch):
    root = _write_inventory(_bundle(tmp_path))
    monkeypatch.setattr(bundle_check, "install_root", lambda: root)

    with pytest.raises(SystemExit) as ok:
        app.main(["--verify"])
    assert ok.value.code == 0

    (root / "_internal" / "_socket.pyd").unlink()
    with pytest.raises(SystemExit) as broken:
        app.main(["--verify"])
    assert broken.value.code == 1

    (root / INVENTORY_NAME).unlink()
    with pytest.raises(SystemExit) as uncheckable:
        app.main(["--verify"])
    # Not 1: "could not be checked" is not "checked and found broken".
    assert uncheckable.value.code == 2
