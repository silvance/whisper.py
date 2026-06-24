"""W.H.I.S.P.R. - a desktop GUI front-end for Whisper transcription.

The transcription backend (:mod:`whispr.transcription`) is importable without the
optional GUI/transcription dependencies; those are only needed to actually run a
transcription or launch the GUI. Install them with ``pip install 'silvance-whisper[gui]'``.
"""

from .transcription import (
    AUDIO_EXTENSIONS,
    MODEL_SIZES,
    VIDEO_EXTENSIONS,
    Segment,
    TranscriptionResult,
    convert_to_wav,
    is_supported_media,
    is_video,
    transcribe_audio,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "MODEL_SIZES",
    "VIDEO_EXTENSIONS",
    "Segment",
    "TranscriptionResult",
    "convert_to_wav",
    "is_supported_media",
    "is_video",
    "transcribe_audio",
]
