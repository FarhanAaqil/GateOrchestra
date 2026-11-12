"""
assistant/smalltalk.py
======================
Recognise greetings, thanks and similar one-liners.

These need no safety screen, no probe and no gate: there is nothing to verify, and a
reply is only worth a few words. Skipping the screen and the probe turns an 18-second
"hi" into a near-instant answer and saves their tokens. The match is deliberately
strict (the whole message must be one of these phrases), so a real question is never
mistaken for small talk.
"""

from __future__ import annotations

import re

_PHRASES = (
    r"(?:hi|hello|hey|hiya|howdy|yo|sup)(?: there| everyone| all| team)?",
    r"good (?:morning|afternoon|evening|night)",
    r"thanks?(?: you)?(?: (?:a lot|so much|very much|a ton))?",
    r"thx|ty|cheers|much appreciated",
    r"ok(?:ay)?|alright|cool|great|nice|awesome|perfect|got it|sounds good|makes sense|sure|fine|yes|no|yep|nope",
    r"bye|goodbye|good bye|see you(?: later)?|see ya|cya|take care",
    r"how are you(?: doing| today)?|how'?s it going|what'?s up|whats up",
    r"who are you|what are you|what can you do|what do you do",
)
_SMALL_TALK = re.compile(r"^(?:" + "|".join(_PHRASES) + r")$")
MAX_LENGTH = 40


def _normalise(text: str) -> str:
    """Lower-case, drop punctuation and emoji, keep apostrophes, collapse spaces."""
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return " ".join(cleaned.split())


def is_small_talk(text: str) -> bool:
    """True when the whole message is a greeting, thanks, acknowledgement or goodbye."""
    if not text or len(text) > MAX_LENGTH:
        return False
    return bool(_SMALL_TALK.match(_normalise(text)))
