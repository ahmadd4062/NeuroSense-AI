import re


RED_FLAG_PATTERNS = [
    (r"\bchest pain\b", "chest pain"),

    (r"\b(can'?t|cannot|unable to)\s+breathe\b", "difficulty breathing"),
    (r"\bdifficulty breathing\b", "difficulty breathing"),
    (r"\btrouble breathing\b", "difficulty breathing"),
    (r"\bshortness of breath\b", "shortness of breath"),

    (r"\bpassed out\b", "loss of consciousness"),
    (r"\bfainted\b", "fainting"),
    (r"\bunconscious\b", "loss of consciousness"),

    (r"\bface\b.{0,40}\bdroop(?:ing|ed|s)?\b", "facial drooping"),
    (r"\bfacial\b.{0,30}\bdroop(?:ing|ed|s)?\b", "facial drooping"),
    (r"\bdroop(?:ing|ed|s)?\b.{0,30}\bface\b", "facial drooping"),

    (r"\b(can'?t|cannot|unable to)\s+speak\b", "difficulty speaking"),
    (r"\bslurred speech\b", "slurred speech"),
    (
        r"\bspeech\b.{0,30}\bslurr(?:ed|ing)?\b",
        "slurred speech"
    ),

    (r"\bsudden weakness\b", "sudden weakness"),
    (r"\bone[- ]sided weakness\b", "one-sided weakness"),
    (
        r"\bweakness\s+on\s+(one|the)\s+side\b",
        "one-sided weakness"
    ),

    (r"\bsudden numbness\b", "sudden numbness"),
    (
        r"\bnumbness\s+on\s+(one|the)\s+side\b",
        "one-sided numbness"
    ),

    (r"\bsevere sudden headache\b", "severe sudden headache"),
    (r"\bsudden severe headache\b", "severe sudden headache"),
    (r"\bworst headache\b", "severe headache"),

    (r"\bseizure\b", "seizure"),
    (r"\bhaving a seizure\b", "seizure"),
]


def check_emergency(message):
    if not isinstance(message, str):
        return {
            "is_emergency": False,
            "detected_flags": [],
        }

    text = message.lower().strip()

    detected_flags = []

    for pattern, flag in RED_FLAG_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if flag not in detected_flags:
                detected_flags.append(flag)

    return {
        "is_emergency": bool(detected_flags),
        "detected_flags": detected_flags,
    }


def get_emergency_response():
    return (
        "These symptoms may require urgent medical attention. "
        "Please contact your local emergency medical services or go to "
        "the nearest emergency department now. If possible, ask someone "
        "nearby to help you get medical attention immediately."
    )