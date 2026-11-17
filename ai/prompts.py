"""Prompt templates."""

CLIP_SYSTEM = (
    "You are an expert short-form video editor (YouTube Shorts, Reels, TikTok). "
    "You find self-contained, high-retention moments in long-form transcripts. "
    "Be critical and honest with scores; most segments are NOT great. "
    "Never invent facts or quotes."
)

CLIP_PROMPT = """Below is a transcript excerpt with timestamps in seconds, one sentence per line as [start-end] text.

Find up to {n} distinct clip candidates, each between {min_d:.0f} and {max_d:.0f} seconds long.

Rules:
- "start" must equal the start time of a sentence and "end" the end time of a sentence, copied from the transcript.
- Each clip must make sense on its own: setup/question + payoff. Do not begin mid-thought, with unexplained pronouns ("this", "that", "they"), or end before the main point lands.
- Prefer insight, surprise, strong stories, humor, emotion, practical advice, question-then-compelling-answer.
- Avoid filler-heavy or repetitive sections. Candidates must not overlap each other.
- Score each factor from 0 to 10: hook (strength of the first sentence), value, story (narrative structure), emotion (emotional/entertainment value), clarity (works standalone), retention, payoff (strong ending).
- Titles/captions must be honest: do not mislead or fabricate claims. Hashtags without '#' are fine.

Transcript:
{transcript}
"""
