"""W.H.I.S.P.R. - a desktop GUI front-end for Whisper transcription.

The transcription backend (:mod:`whispr.transcription`) is importable without the
optional GUI/transcription dependencies; those are only needed to actually run a
transcription or launch the GUI. Install them with ``pip install 'silvance-whisper[gui]'``.
"""

from .diarization import SpeakerSegment, assign_speakers, diarize
from .resources import (
    bundled_diarization_models,
    bundled_models,
    find_bundled_ffmpeg,
    find_ffmpeg,
)
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
    "SpeakerSegment",
    "TranscriptionResult",
    "assign_speakers",
    "bundled_diarization_models",
    "bundled_models",
    "convert_to_wav",
    "diarize",
    "find_bundled_ffmpeg",
    "find_ffmpeg",
    "is_supported_media",
    "is_video",
    "transcribe_audio",
]
