# src/obfuscation.py
# Text obfuscation functions simulating common hate speech evasion tactics

import random

LEET_MAP = {'a':'4', 'e':'3', 'i':'1', 'o':'0', 's':'5', 't':'7', 'g':'9', 'b':'8'}


def leet_speak(text, rate=0.4):
    """
    Replace letters with leet equivalents at given rate.
    e.g. "hate" -> "h4t3"
    """
    result = []
    for ch in text:
        if ch.lower() in LEET_MAP and random.random() < rate:
            result.append(LEET_MAP[ch.lower()])
        else:
            result.append(ch)
    return "".join(result)


def insert_punctuation(text, rate=0.3):
    """
    Insert punctuation between characters of random words.
    e.g. "hate" -> "h.a.t.e"
    """
    words, result = text.split(), []
    for word in words:
        if len(word) > 3 and random.random() < rate:
            punct = random.choice([".", "-", "_", "*"])
            word  = punct.join(list(word))
        result.append(word)
    return " ".join(result)


def char_repeat(text, rate=0.3):
    """
    Randomly repeat characters within words.
    e.g. "hate" -> "haate"
    """
    words, result = text.split(), []
    for word in words:
        if len(word) > 3 and random.random() < rate:
            idx  = random.randint(1, len(word) - 2)
            n    = random.randint(2, 4)
            word = word[:idx] + word[idx] * n + word[idx+1:]
        result.append(word)
    return " ".join(result)


def combined_obfuscation(text):
    """Apply all three obfuscation strategies together."""
    text = leet_speak(text,         rate=0.3)
    text = insert_punctuation(text, rate=0.2)
    text = char_repeat(text,        rate=0.2)
    return text


# Registry for easy iteration
OBFUSCATION_FNS = {
    "leet_speak":  leet_speak,
    "punctuation": insert_punctuation,
    "char_repeat": char_repeat,
    "combined":    combined_obfuscation,
}
