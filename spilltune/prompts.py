"""Built-in test prompts. A mix of task types on purpose: guess-ahead helpers are accepted far more often on
predictable text (math steps, code) than on open writing, so a math-only test overstates their speedup."""

VERSION = 1

PROMPTS = [
    ("math", "A bakery sells muffins for $3 each and cookies for $1.25 each. On Monday it sold 48 muffins and "
             "three times as many cookies. On Tuesday sales of both fell by a quarter. How much money did the "
             "bakery take over the two days? Work through it step by step."),
    ("code", "Write a Python function that merges overlapping intervals in a list of (start, end) pairs and "
             "returns them sorted. Include a docstring and three example calls with their expected output."),
    ("explain", "Explain to a curious teenager why the sky is blue during the day but red and orange at sunset. "
                "Use plain words and one everyday comparison."),
    ("writing", "Write the opening of a short story about a lighthouse keeper who finds a message in a bottle "
                "that is addressed to her, written in her own handwriting."),
]


def pick(n):
    return PROMPTS[:max(1, min(n, len(PROMPTS)))]
