"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "proud",
    "hopeful",
    "grateful",
    "blessed",
    "hyped",
    "winning",
    "fire",
    "lit",
    "vibing",
    "thrilled",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "exhausted",
    "frustrated",
    "dread",
    "miserable",
    "overwhelmed",
    "anxious",
    "broken",
    "disappointed",
    "sick",
    "rough",
]

# Emoji sentiment map used by MoodAnalyzer
EMOJI_SCORES = {
    "😊": 1, "😄": 1, "😁": 1, "🥰": 2, "❤️": 2, "💪": 1,
    "🙌": 1, "🎉": 1, "😂": 1, "🔥": 1, "✨": 1,
    "😢": -1, "😭": -2, "😡": -2, "😤": -1, "💀": -1,
    "😩": -1, "😰": -1, "🥲": -1, "😒": -1, ":)": 1, ":(": -1,
}

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # Slang
    "lowkey stressed but no cap I'm proud of what I built",
    "this assignment is lowkey fire ngl",
    "vibing so hard rn, life is good",
    # Emojis
    "just got the job offer 😄🎉 so hyped!!",
    "failed my exam 😭 worst day ever",
    "I'm fine 🙂",
    # Sarcasm (hard for rule-based)
    "oh great, another Monday, I absolutely love this",
    "wow I just love being stuck in traffic for 2 hours",
    # Mixed feelings
    "exhausted but really grateful for everyone who showed up",
    "I hate that I love this show so much",
    # Ambiguous / neutral-ish
    "just woke up and made coffee",
    "meeting at 3pm today",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # lowkey stressed but proud
    "positive",  # lowkey fire ngl
    "positive",  # vibing so hard
    "positive",  # got job offer 😄🎉
    "negative",  # failed exam 😭
    "neutral",   # I'm fine 🙂  (ambiguous — could be mixed)
    "negative",  # sarcasm: love another Monday
    "negative",  # sarcasm: love being stuck in traffic
    "mixed",     # exhausted but grateful
    "mixed",     # hate that I love this show
    "neutral",   # just woke up, made coffee
    "neutral",   # meeting at 3pm
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
