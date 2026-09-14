"""A bundled model cache that is complete and still unusable.

The failure these hold shut reached the field. Pinning every model to an exact
commit made the build reproducible and, as a side effect nobody saw, took away
the ``refs/main`` file that says which commit ``main`` is. Every weight was in
the bundle. pyannote asked for ``main``, offline, got nothing, and told an
analyst with no network to check their internet connection.

The build would not have caught it either: the check at the time was that the
cache directory existed.
"""

from whispr import hub_cache

COMMIT = "84fd25912480287da0247647c3d2b4853cb3ee5d"
REPO = "pyannote/speaker-diarization-3.1"


def make_snapshot(root, repo=REPO, commit=COMMIT, *, ref=True):
    """A repository in the cache, with or without the reference to its commit."""
    folder = hub_cache.repo_folder(root, repo)
    snapshot = folder / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.yaml").write_text("version: 3.1.0\n", encoding="utf-8")
    if ref:
        hub_cache.write_ref(root, repo, commit)
    return snapshot


def test_the_folder_name_matches_the_hub_layout():
    """Not ours to choose - huggingface_hub looks for exactly this."""
    assert hub_cache.repo_folder("/c", REPO).name == (
        "models--pyannote--speaker-diarization-3.1"
    )


def test_a_cache_with_the_reference_resolves(tmp_path):
    snapshot = make_snapshot(tmp_path)
    assert hub_cache.resolve(tmp_path, REPO) == snapshot


def test_a_pinned_cache_without_the_reference_does_not(tmp_path):
    """The regression, in one line: the weights are there and unreachable."""
    make_snapshot(tmp_path, ref=False)
    assert hub_cache.resolve(tmp_path, REPO) is None


def test_writing_the_reference_makes_it_resolve(tmp_path):
    snapshot = make_snapshot(tmp_path, ref=False)
    hub_cache.write_ref(tmp_path, REPO, COMMIT)
    assert hub_cache.resolve(tmp_path, REPO) == snapshot


def test_a_reference_pointing_at_a_commit_that_is_not_there_resolves_to_nothing(
    tmp_path,
):
    make_snapshot(tmp_path)
    hub_cache.write_ref(tmp_path, REPO, "0" * 40)
    assert hub_cache.resolve(tmp_path, REPO) is None


def test_a_blank_reference_resolves_to_nothing(tmp_path):
    make_snapshot(tmp_path, ref=False)
    hub_cache.ref_path(tmp_path, REPO).parent.mkdir(parents=True, exist_ok=True)
    hub_cache.ref_path(tmp_path, REPO).write_text("  \n", encoding="utf-8")
    assert hub_cache.resolve(tmp_path, REPO) is None


# -- What the build and the report ask ------------------------------------


def test_a_good_cache_has_nothing_to_report(tmp_path):
    for repo in hub_cache.PYANNOTE_REPOS:
        make_snapshot(tmp_path, repo)
    assert hub_cache.unreadable(tmp_path, hub_cache.PYANNOTE_REPOS) == []


def test_a_missing_repository_is_named(tmp_path):
    make_snapshot(tmp_path)
    problems = hub_cache.unreadable(tmp_path, hub_cache.PYANNOTE_REPOS)
    assert len(problems) == 2
    assert any("segmentation-3.0" in p and "not in the cache" in p for p in problems)


def test_the_missing_reference_is_reported_as_what_it_is(tmp_path):
    """ "Not in the cache" would send somebody looking for absent files."""
    make_snapshot(tmp_path, ref=False)
    (problem,) = hub_cache.unreadable(tmp_path, [REPO])
    assert "refs/main" in problem
    assert "even though they are here" in problem


def test_a_reference_to_a_missing_commit_names_the_commit(tmp_path):
    make_snapshot(tmp_path)
    hub_cache.write_ref(tmp_path, REPO, "0" * 40)
    (problem,) = hub_cache.unreadable(tmp_path, [REPO])
    assert "0" * 40 in problem and "missing" in problem


def test_an_empty_snapshot_is_not_a_usable_one(tmp_path):
    snapshot = hub_cache.repo_folder(tmp_path, REPO) / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    hub_cache.write_ref(tmp_path, REPO, COMMIT)
    (problem,) = hub_cache.unreadable(tmp_path, [REPO])
    assert "empty" in problem


# -- Putting back what pinning took away -----------------------------------


def test_a_pinned_cache_is_repaired_from_what_is_on_disk(tmp_path):
    """One snapshot means one candidate; this restates it, it does not guess."""
    snapshot = make_snapshot(tmp_path, ref=False)
    assert hub_cache.repair(tmp_path, [REPO]) == [REPO]
    assert hub_cache.resolve(tmp_path, REPO) == snapshot
    assert hub_cache.unreadable(tmp_path, [REPO]) == []


def test_a_cache_that_is_already_readable_is_left_alone(tmp_path):
    make_snapshot(tmp_path)
    assert hub_cache.repair(tmp_path, [REPO]) == []


def test_two_snapshots_are_not_guessed_between(tmp_path):
    """Choosing which commit 'main' means is exactly the guess to refuse."""
    make_snapshot(tmp_path, ref=False)
    make_snapshot(tmp_path, commit="b" * 40, ref=False)
    assert hub_cache.repair(tmp_path, [REPO]) == []
    assert hub_cache.resolve(tmp_path, REPO) is None


def test_an_empty_snapshot_is_not_repaired_to(tmp_path):
    (hub_cache.repo_folder(tmp_path, REPO) / "snapshots" / COMMIT).mkdir(parents=True)
    assert hub_cache.repair(tmp_path, [REPO]) == []


def test_a_repository_that_is_not_there_is_not_invented(tmp_path):
    assert hub_cache.repair(tmp_path, [REPO]) == []
    assert not hub_cache.repo_folder(tmp_path, REPO).exists()


def test_a_stale_reference_is_repaired_to_the_snapshot_that_exists(tmp_path):
    snapshot = make_snapshot(tmp_path)
    hub_cache.write_ref(tmp_path, REPO, "0" * 40)
    assert hub_cache.repair(tmp_path, [REPO]) == [REPO]
    assert hub_cache.resolve(tmp_path, REPO) == snapshot


def test_a_read_only_cache_is_reported_rather_than_forced(tmp_path, monkeypatch):
    """An installation directory may not be writable; that is not a crash.

    Simulated rather than chmod-ed: the build and test machines may be root,
    where a read-only directory is not read-only at all.
    """
    make_snapshot(tmp_path, ref=False)

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(hub_cache, "write_ref", refuse)
    assert hub_cache.repair(tmp_path, [REPO]) == []
    assert hub_cache.unreadable(tmp_path, [REPO])


def test_the_repository_list_is_the_one_the_pipeline_needs(tmp_path):
    """The build fills these and the report checks these - one list, not two."""
    assert hub_cache.PYANNOTE_REPOS[0] == "pyannote/speaker-diarization-3.1"
    assert len(hub_cache.PYANNOTE_REPOS) == 3


def test_every_repository_the_pipeline_needs_is_pinned_in_the_shipped_lock():
    """Pinning is what broke the references; leaving one unpinned would hide it.

    A repository fetched without a pin gets a ``refs/main`` for free, which is
    exactly the case the build no longer exercises. Keeping all three pinned
    keeps the build honest about the layout it ships.
    """
    import json
    from pathlib import Path

    lock = json.loads(
        (
            Path(__file__).resolve().parent.parent / "packaging" / "assets.lock.json"
        ).read_text(encoding="utf-8")
    )
    pinned = lock.get("huggingface", {})
    for repo in hub_cache.PYANNOTE_REPOS:
        assert len(pinned.get(repo, "")) == 40, f"{repo} is not pinned"
