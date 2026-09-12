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

What the lock does *not* yet cover: ``pyannote.audio`` brings its own dependency
tree, and resolving that needs the build machine rather than a checkout. Those
packages can still move under a rebuild of a diarizing bundle. Every release run
uploads what it installed as ``pip-freeze.txt`` so the lock can be extended from
the build that was tested - see ``packaging/check_lock.py --from-freeze``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

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
        if actual is not None and actual != pinned:
            problems.append(f"{name}: locked {pinned}, installed {actual}")
    for raw in names:
        name = canonical(raw)
        if name in installed and name not in pins:
            problems.append(
                f"{name}: installed {installed[name]} but not in the lock "
                "(a version nobody chose)"
            )
    return problems
