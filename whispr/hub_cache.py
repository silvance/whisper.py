"""Reading a bundled Hugging Face cache offline, and what makes one readable.

A Hugging Face cache is not just files on disk. A repository folder holds the
weights under ``snapshots/<commit>``, and a *reference* under ``refs/<name>``
saying which commit a name like ``main`` points at. Offline, a caller that asks
for ``main`` - which is what pyannote asks for - can only be answered from that
reference. Without it the cache is complete and unusable at the same time.

That is not hypothetical. Pinning the model revisions gave every repository an
exact commit and, because ``snapshot_download`` writes a reference only when the
revision it was given is a *name* rather than a commit, took the references
away. The build looked right, the bundle looked right, and diarization died on
an operator's machine with a message about checking their internet connection.

So the layout is written down here: what a readable cache looks like, checked
where the cache is made and again where the application reports what it can do.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

# What pyannote (and anything else that does not name a revision) asks for.
DEFAULT_REF = "main"

# The repositories pyannote.audio 3.1.1's speaker-diarization-3.1 pipeline needs:
# the pipeline itself and the two models it loads. Kept here rather than in the
# build script so the side that fills the cache and the side that checks it
# cannot come to disagree about what belongs in it.
PYANNOTE_REPOS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
)


def repo_folder(cache_root: Path, repo_id: str) -> Path:
    """The cache folder for ``repo_id`` - ``models--org--name`` beside its peers."""
    return Path(cache_root) / ("models--" + repo_id.replace("/", "--"))


def ref_path(cache_root: Path, repo_id: str, ref: str = DEFAULT_REF) -> Path:
    """Where the commit that ``ref`` points at is recorded."""
    return repo_folder(cache_root, repo_id) / "refs" / ref


def write_ref(
    cache_root: Path, repo_id: str, commit: str, ref: str = DEFAULT_REF
) -> Path:
    """Point ``ref`` at ``commit``, so an offline caller asking by name is answered.

    A download pinned to a commit leaves no reference behind, because the commit
    *is* the revision. Writing one puts the cache in the state an unpinned
    download of that same commit would have left it in: the weights pinned, and
    still reachable by the name everything asks for.
    """
    path = ref_path(cache_root, repo_id, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commit, encoding="utf-8")
    return path


def resolve(cache_root: Path, repo_id: str, ref: str = DEFAULT_REF) -> "Optional[Path]":
    """The snapshot ``ref`` points at, or None if this cache cannot answer offline."""
    path = ref_path(cache_root, repo_id, ref)
    try:
        commit = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not commit:
        return None
    snapshot = repo_folder(cache_root, repo_id) / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def unreadable(
    cache_root: Path, repo_ids: "Sequence[str]", ref: str = DEFAULT_REF
) -> "List[str]":
    """Which of ``repo_ids`` this cache could not serve offline, and why.

    Empty means every one of them resolves to a snapshot that is really there.
    """
    problems: List[str] = []
    for repo_id in repo_ids:
        folder = repo_folder(cache_root, repo_id)
        if not folder.is_dir():
            problems.append(f"{repo_id}: not in the cache")
            continue
        reference = ref_path(cache_root, repo_id, ref)
        if not reference.is_file():
            problems.append(
                f"{repo_id}: no refs/{ref}, so an offline load cannot find the "
                "weights even though they are here"
            )
            continue
        snapshot = resolve(cache_root, repo_id, ref)
        if snapshot is None:
            commit = reference.read_text(encoding="utf-8").strip() or "(empty)"
            problems.append(
                f"{repo_id}: refs/{ref} points at {commit}, which is missing"
            )
            continue
        if not any(snapshot.iterdir()):
            problems.append(f"{repo_id}: the snapshot is empty")
    return problems


def snapshots(cache_root: Path, repo_id: str) -> "List[Path]":
    """Every commit snapshot present for ``repo_id``, in name order."""
    folder = repo_folder(cache_root, repo_id) / "snapshots"
    try:
        return sorted(p for p in folder.iterdir() if p.is_dir())
    except OSError:
        return []


def repair(
    cache_root: Path, repo_ids: "Sequence[str]", ref: str = DEFAULT_REF
) -> "List[str]":
    """Give a pinned cache back the reference it needs, where that is unambiguous.

    A cache built by pinning every revision holds exactly one snapshot per
    repository and no reference to it. There is then only one commit the name
    could mean, so writing it is a restatement of what is already on disk
    rather than a guess - and it turns a bundle that cannot diarize into one
    that can, without a network and without a rebuild.

    Two or more snapshots, and it stays out of the way: picking between commits
    is exactly the kind of guess this application must not make. Failures are
    swallowed (an installation directory may be read-only); the caller finds
    out from :func:`unreadable`, which is the honest answer either way.

    Returns the repositories it repaired.
    """
    fixed: List[str] = []
    for repo_id in repo_ids:
        if resolve(cache_root, repo_id, ref) is not None:
            continue
        present = [p for p in snapshots(cache_root, repo_id) if any(p.iterdir())]
        if len(present) != 1:
            continue
        try:
            write_ref(cache_root, repo_id, present[0].name, ref)
        except OSError:
            continue
        fixed.append(repo_id)
    return fixed
