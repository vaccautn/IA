"""File-video prototype, independent of HTTP, BCS and domain persistence."""

from .pipeline import Frame, SamplingConfig, process_frames

__all__ = ["Frame", "SamplingConfig", "process_frames"]
