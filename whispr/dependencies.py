"""Which dependency versions a bundle is allowed to have, and how to tell.

The build manifest records what a build *was*. This is what a build *may be*:
``packaging/lockfile.txt`` pins every dependency whose version can change how
the application behaves, and a build whose installed versions drift from it
fails rather than shipping.

Without that, the release workflow installs ranges - ``ctranslate2>=4.8,<5`` and
the like - so rebuilding the same commit in six months can produce software
nobody ever field-tested, with nothing in the bundle to say so. For an
air-gapped deployment that matters more than usual: the copy on the removable
media is the only one anyone will ever run, and there is no update channel
behind it.

The comparison lives here rather than in the build script so it can be tested
without a build.

The lock covers the whole installed set, including ``pyannote.audio``'s own
dependency tree, which can only be resolved by a real build rather than from a
checkout. Every release run uploads what it installed as ``pip-freeze.txt`` so
the lock can be brought back in step with the build that was tested - see
``packaging/check_lock.py --from-freeze``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Production dependencies whose versions materially affect behaviour, and so
# must be pinned rather than merely recorded. Transcription and diarization
# first, then packaging and the optional engines.
TRACKED_PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "sherpa-onnx",
    "onnxruntime",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "numpy",
    "ttkbootstrap",
    "python-docx",
    "argostranslate",
    "pytesseract",
    "pypdfium2",
    "pillow",
    "pyinstaller",
    "huggingface-hub",
    "tokenizers",
    "av",
)


def canonical(name: str) -> str:
    """PEP 503 normalised name, so ``sherpa_onnx`` and ``sherpa-onnx`` are one."""
    out: List[str] = []
    separator = False
    for char in name.strip().lower():
        if char in "-_.":
            if not separator:
                out.append("-")
            separator = True
        else:
            out.append(char)
            separator = False
    return "".join(out)


def parse_lock(text: str) -> "Dict[str, str]":
    """Read ``name==version`` lines from a constraints file, ignoring the rest."""
    pins: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, _, version = line.partition("==")
        pins[canonical(name)] = version.strip()
    return pins


def split_local(version: str) -> "Tuple[str, str]":
    """A version as (public, local): ``2.2.2+cpu`` -> ``("2.2.2", "cpu")``.

    The bit after the ``+`` is PEP 440's local version label. PyTorch uses it to
    say which build of a version this is - ``+cpu`` against ``+cu121`` - so it
    names a variant, not a version.
    """
    public, _, local = version.partition("+")
    return public, local


def same_version(pinned: str, installed: str) -> bool:
    """Whether an installed version satisfies a pin, as pip reads ``==``.

    ``torch==2.2.2`` in a constraints file is satisfied by ``2.2.2+cpu`` - pip
    ignores the local label unless the specifier names one - so a checker that
    compares the strings disagrees with the tool it is checking. That is exactly
    what failed the first real build this lock ran against: the install was
    correct and the comparison was not.

    A pin that *does* name a local label means it: ``torch==2.2.2+cpu`` is not
    satisfied by a CUDA build of the same version, which for this project is a
    distinction worth being able to draw.
    """
    pinned_public, pinned_local = split_local(pinned)
    installed_public, installed_local = split_local(installed)
    if pinned_public != installed_public:
        return False
    return not pinned_local or pinned_local == installed_local


def variants(pins: "Dict[str, str]", installed: "Dict[str, str]") -> "List[str]":
    """Packages installed as a build variant of what was pinned.

    Not drift - the pin is satisfied - but worth printing, because "torch 2.2.2"
    and "torch 2.2.2 built for CUDA" are different software and the lock as
    written does not tell them apart.
    """
    notes: List[str] = []
    for name, pinned in sorted(pins.items()):
        actual = installed.get(name)
        if actual is None or not same_version(pinned, actual):
            continue
        _, pinned_local = split_local(pinned)
        _, actual_local = split_local(actual)
        if actual_local and not pinned_local:
            notes.append(f"{name}: {pinned} installed as {actual}")
    return notes


def format_lock(pins: "Dict[str, str]", header: str = "") -> str:
    """The lock file's text: a header, then one pin per line, sorted."""
    body = "".join(f"{name}=={version}\n" for name, version in sorted(pins.items()))
    return (header + "\n" if header else "") + body


def drift(
    pins: "Dict[str, str]",
    installed: "Dict[str, str]",
    tracked: "Optional[Sequence[str]]" = None,
) -> "List[str]":
    """Every way the installed set fails to be the locked set, in plain words.

    A locked package that is not installed is not drift: the lock covers the
    optional engines too, and a transcribe-only bundle installs none of them.
    An installed package that is *tracked* but absent from the lock is drift -
    that is a version nobody chose, which is the whole failure being prevented.
    """
    names: Iterable[str] = TRACKED_PACKAGES if tracked is None else tracked
    problems: List[str] = []
    for name, pinned in sorted(pins.items()):
        actual = installed.get(name)
        if actual is not None and not same_version(pinned, actual):
            problems.append(f"{name}: locked {pinned}, installed {actual}")
    for raw in names:
        name = canonical(raw)
        if name in installed and name not in pins:
            problems.append(
                f"{name}: installed {installed[name]} but not in the lock "
                "(a version nobody chose)"
            )
    return problems
