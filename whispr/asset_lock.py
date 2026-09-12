"""Which model weights a bundle is allowed to contain, and how to tell.

A Hugging Face repository id names a moving target: ``snapshot_download`` with
no revision fetches whatever is at the head of that repo today. The dependency
lock stops the *code* drifting under a rebuild; this stops the *weights*. Both
matter for the same reason - the bundle on the removable media is the only copy
anyone will ever run, and there is no update channel behind it to correct a
model nobody chose.

It covers the plain HTTPS downloads too, and there it does a second job. A
model fetched over a URL with no revision and no digest is taken on trust from
whoever is serving it that day. Recording the digest of what was fetched, and
refusing anything else afterwards, turns that into something checkable.

The lock is written by a real build rather than by hand: the first build after
a freeze records what it resolved, and every build after that is held to it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Bumped only if the file's shape changes in a way older tooling cannot read.
LOCK_VERSION = 1

NOTE = (
    "Written by packaging/fetch_assets.py during a build, and enforced on every "
    "build afterwards. Commit the file produced by the build you tested; a later "
    "build that would fetch different weights then fails instead of shipping."
)


def empty() -> "Dict[str, Any]":
    """A lock with nothing pinned yet - the state before the first build."""
    return {"version": LOCK_VERSION, "note": NOTE, "huggingface": {}, "files": {}}


def revision_for(lock: "Dict[str, Any]", repo_id: str) -> "Optional[str]":
    """The commit this repository is pinned to, or None if it is not pinned."""
    value = (lock.get("huggingface") or {}).get(repo_id)
    return str(value) if value else None


def digest_for(lock: "Dict[str, Any]", url: str) -> "Optional[str]":
    """The SHA-256 this URL is pinned to, or None if it is not pinned."""
    entry = (lock.get("files") or {}).get(url)
    if isinstance(entry, dict):
        value = entry.get("sha256")
        return str(value) if value else None
    return str(entry) if entry else None


def record_revision(lock: "Dict[str, Any]", repo_id: str, revision: str) -> bool:
    """Pin a repository. Returns True if this changed the lock."""
    repos = lock.setdefault("huggingface", {})
    if repos.get(repo_id) == revision:
        return False
    repos[repo_id] = revision
    return True


def record_file(
    lock: "Dict[str, Any]", url: str, sha256: str, size: "Optional[int]" = None
) -> bool:
    """Pin a downloaded file by digest. Returns True if this changed the lock."""
    files = lock.setdefault("files", {})
    entry: Dict[str, Any] = {"sha256": sha256}
    if size is not None:
        entry["size"] = size
    if files.get(url) == entry:
        return False
    files[url] = entry
    return True


def mismatch(
    expected: "Optional[str]", actual: "Optional[str]", what: str
) -> "Optional[str]":
    """The complaint to raise when a fetched asset is not the pinned one.

    None when there is nothing to complain about: nothing pinned yet (the first
    build after a freeze), or the two agree.
    """
    if not expected or not actual or expected == actual:
        return None
    return (
        f"{what} is pinned to {expected} but the download is {actual}. "
        "This bundle would not contain the weights that were tested. Either "
        "restore the pinned asset, or re-pin deliberately on a branch and test "
        "the bundle it produces."
    )


def summary(lock: "Dict[str, Any]") -> "List[str]":
    """One line per pinned asset, for the build log."""
    lines = [
        f"{repo} @ {rev}"
        for repo, rev in sorted((lock.get("huggingface") or {}).items())
    ]
    for url, entry in sorted((lock.get("files") or {}).items()):
        digest = entry.get("sha256", "?") if isinstance(entry, dict) else str(entry)
        lines.append(f"{url} sha256:{digest[:12]}")
    return lines
