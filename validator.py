"""
Post-hoc boundary validator for the A1 tutor agent.

Checks an LLM reply against every rule and forbidden item in A1spec.json.
This is regex/heuristic matching, not a grammar parser — it catches the
patterns that were proven to leak through prompting alone (see
../report.txt), but a "valid" result is not proof the reply is perfect
A1 English, and a "flagged" result is not proof of a genuine grammar error.

Every forbidden-list check iterates the spec's own arrays directly
(spec["forbidden"]["modals"], etc.) rather than hardcoding a copy of the
list here, so editing A1spec.json automatically changes what gets checked.
"""

import re


IDIOMS_AND_PHRASALS = [
    "hit the books", "raining cats and dogs", "under the weather", "kick the bucket",
    "break a leg", "call it a day", "piece of cake", "break the ice", "give me a hand",
    "hang out", "figure out", "look forward to", "get along", "call off", "put off",
    "run into", "come across", "bring up", "turn down", "give up", "look up to",
    "hit the road", "once in a while", "on the same page", "in a nutshell",
]

STRUCTURE_PATTERNS = {
    "passive voice": r"\b(is|are|was|were|be|been)\s+\w+ed\s+by\b",
    "conditionals (all types)": r"\bif\b[^.!?]*\b(would|could|might|will)\b",
    "relative clauses (who/which/that as subordinator)": r"\b(the|a|an)\s+\w+\s+(who|which|that)\s+\w+",
    "reported speech": r"\b(said|told|asked)\s+(that\s+)?(he|she|they|it|i|we)\s+(was|were|had|would|is|are)\b",
    "question tags": r",\s*(don't|doesn't|isn't|aren't|didn't|won't|can't|wasn't|weren't)\s+(you|he|she|it|they|we)\s*\?",
    "subordinate clauses beyond 'because'": r"\b(since|as|when|while|after|before)\b\s+\w+.*[,.]",
    "superlatives": r"\bthe\s+(most|best|worst|biggest|easiest|hardest)\b|\b\w+est\b",
    "comparatives": r"\b(more|less|better|worse)\s+(\w+\s+)?than\b|\b\w+er\s+than\b",
}

TENSE_PATTERNS = {
    "past perfect": r"\bhad\s+(been|gone|done|seen|eaten|taken|written|made|said|\w+ed)\b",
    "past perfect continuous": r"\bhad\s+been\s+\w+ing\b",
    "future perfect": r"\bwill\s+have\s+\w+(ed|en)\b",
    "present perfect continuous": r"\b(have|has|'ve|'s)\s+been\s+\w+ing\b",
    "present perfect": r"\b(have|has|'ve)\s+(\w+ed|been|gone|done|seen|eaten|taken|written|made|said|bought|read)\b",
}

CHARACTER_BREAK_PATTERN = r"\b(spec|specification|system prompt|these rules|my rules|my instructions|cefr|a1 level|a1spec)\b"


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _word_count(sentence):
    return len(re.findall(r"[A-Za-z']+", sentence))


def _check_structural_limits(reply, limits):
    reasons = []
    sentences = _split_sentences(reply)

    max_sentences = limits["max_sentences_per_reply"]
    if len(sentences) > max_sentences:
        reasons.append(f"structural_limits.max_sentences_per_reply: {len(sentences)} sentences (limit {max_sentences})")

    max_words = limits["max_words_per_sentence"]
    for s in sentences:
        n = _word_count(s)
        if n > max_words:
            reasons.append(f"structural_limits.max_words_per_sentence: {n}-word sentence (limit {max_words}): {s[:60]!r}")

    allowed = set(c.lower() for c in limits["allowed_coordinators"])
    disallowed_coordinators = {"or", "yet", "for", "nor"} - allowed
    for s in sentences:
        for coord in disallowed_coordinators:
            if re.search(rf",\s*{re.escape(coord)}\b|\b{re.escape(coord)}\b,", s.lower()):
                reasons.append(f"structural_limits.allowed_coordinators: used '{coord}' to join clauses: {s[:60]!r}")

    return reasons


def _check_forbidden_tenses(lower, forbidden_tenses):
    reasons = []
    for tense in forbidden_tenses:
        pattern = TENSE_PATTERNS.get(tense)
        if pattern and re.search(pattern, lower):
            reasons.append(f"forbidden.tenses: possible '{tense}'")
    return reasons


def _check_forbidden_structures(lower, forbidden_structures):
    reasons = []
    for structure in forbidden_structures:
        pattern = STRUCTURE_PATTERNS.get(structure)
        if pattern and re.search(pattern, lower):
            reasons.append(f"forbidden.structures: possible '{structure}'")
    return reasons


def _check_forbidden_word_list(lower, words, spec_key):
    reasons = []
    for word in words:
        if re.search(rf"\b{re.escape(word.lower())}\b", lower):
            reasons.append(f"{spec_key}: used '{word}'")
    return reasons


def _check_lexical(lower):
    reasons = []
    for idiom in IDIOMS_AND_PHRASALS:
        if idiom in lower:
            reasons.append(f"forbidden.lexical: idiom/phrasal verb '{idiom}'")
    return reasons


def _check_correction_policy(reply, lower, correction_policy):
    reasons = []
    for word in correction_policy["never_use"]:
        if re.search(rf"\b{re.escape(word.lower())}\b", lower):
            reasons.append(f"correction_policy.never_use: used '{word}'")

    for line in reply.split("\n"):
        stripped = line.strip().lower()
        if "we say:" in stripped and not stripped.startswith("we say:"):
            reasons.append(f"correction_policy.pattern: 'We say:' not on its own line: {line.strip()[:60]!r}")

    return reasons


def _check_character_break(lower):
    if re.search(CHARACTER_BREAK_PATTERN, lower):
        return ["character break: reply refers to its own rules/spec/instructions"]
    return []


def validate(reply, user_input, spec):
    """
    Checks `reply` against every section of `spec` (A1spec.json's structure).
    Returns (is_valid, reasons) — reasons is a list of human-readable
    strings, empty iff is_valid is True.
    """
    reasons = []
    lower = reply.lower()
    forbidden = spec["forbidden"]

    reasons += _check_structural_limits(reply, spec["structural_limits"])
    reasons += _check_forbidden_tenses(lower, forbidden["tenses"])
    reasons += _check_forbidden_structures(lower, forbidden["structures"])
    reasons += _check_forbidden_word_list(lower, forbidden["modals"], "forbidden.modals")
    reasons += _check_forbidden_word_list(lower, forbidden["connectors"], "forbidden.connectors")
    reasons += _check_lexical(lower)
    reasons += _check_correction_policy(reply, lower, spec["correction_policy"])
    reasons += _check_character_break(lower)

    return (len(reasons) == 0, reasons)
