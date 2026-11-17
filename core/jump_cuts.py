"""Speech-aware trim ranges and subtitle timing for jump-cut rendering."""
from __future__ import annotations

from core.models import Word


def speech_ranges(words: list[Word], duration: float, silence_threshold: float = 1.1,
                 padding: float = 0.12) -> list[tuple[float, float]]:
    if not words:
        return [(0.0, duration)]
    ranges: list[tuple[float, float]] = []
    start = max(0.0, words[0].start - padding)
    last_end = words[0].end
    for word in words[1:]:
        if word.start - last_end > silence_threshold:
            ranges.append((start, min(duration, last_end + padding)))
            start = max(0.0, word.start - padding)
        last_end = max(last_end, word.end)
    ranges.append((start, min(duration, last_end + padding)))
    return [(start, end) for start, end in ranges if end > start]


def compress_words(words: list[Word], ranges: list[tuple[float, float]]) -> list[Word]:
    result: list[Word] = []
    removed = 0.0
    for start, end in ranges:
        for word in words:
            if start <= word.start and word.end <= end:
                result.append(word.model_copy(update={
                    "start": max(0.0, word.start - removed - start),
                    "end": max(0.0, word.end - removed - start),
                }))
        removed += end - start
    return result
