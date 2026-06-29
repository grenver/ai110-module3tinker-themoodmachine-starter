# Model Card: Mood Machine

This model card covers two versions of the Mood Machine mood classifier:

1. A **rule-based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

---

## 1. Model Overview

**Model type:** Both models were built and compared.

**Intended purpose:** Classify short text posts (social media style, 5–20 words) into one of four mood categories: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**

- *Rule-based:* Tokenizes text, strips punctuation, then scores each token by matching it against hand-curated positive and negative word lists. A negation handler flips the polarity of the next sentiment word after words like "not" or "never." Emoji tokens are scored via a separate lookup table. The final numeric score is mapped to a label using thresholds (score ≥ 2 → positive, ≤ −2 → negative; scores of ±1 check for mixed signals).
- *ML model:* Converts each post into a bag-of-words vector using `CountVectorizer`, then trains a `LogisticRegression` classifier to predict labels from those vectors. No hand-written rules — the model learns associations directly from the labeled examples.

---

## 2. Data

**Dataset description:** The dataset contains 18 labeled posts in `SAMPLE_POSTS`. The original 6 starter posts were expanded with 12 additional examples covering slang, emojis, sarcasm, and mixed feelings.

**Labeling process:** Labels were assigned by reading each post carefully and choosing the label that best described the dominant emotional tone. Posts like `"I'm fine 🙂"` were genuinely ambiguous — the 🙂 emoji is often used sarcastically, but `neutral` was chosen as the safer default. Posts that contained clearly both positive and negative signals (e.g., "exhausted but really grateful") were labeled `mixed`.

**Important characteristics of the dataset:**

- Contains slang: "lowkey," "no cap," "fire," "ngl," "vibing"
- Contains emoji sentiment signals: 😄, 🎉, 😭, 🙂
- Includes sarcasm: two posts labeled `negative` despite containing positive words like "love" and "great"
- Includes mixed-emotion posts that explicitly pair opposing signals
- Contains short neutral posts with no sentiment words

**Possible issues:**

- Dataset is very small (18 examples). The ML model memorizes it rather than generalizing.
- Sarcasm posts are labeled `negative` but the surface words are positive — this is an irresolvable ambiguity for any model that only reads tokens.
- The dataset skews toward English-speaking internet slang and may not represent other dialects or languages.
- Label distribution is roughly balanced but `mixed` is underrepresented (5 out of 18).

---

## 3. How the Rule-Based Model Works

**Scoring rules:**

- Each token is matched against `POSITIVE_WORDS` (+1) and `NEGATIVE_WORDS` (−1).
- Negation words (`not`, `never`, `no`, `can't`, `don't`, etc.) flip the polarity of the next sentiment word encountered.
- Emoji and emoticon tokens are scored via `EMOJI_SCORES` in `dataset.py` (e.g., 😄 → +1, 😭 → −2).
- Label thresholds: score ≥ 2 → `positive`; score ≤ −2 → `negative`; score of +1 with negative tokens present → `mixed`; score of −1 with positive tokens present → `mixed`; score = 0 → `neutral`.

**Strengths:**

- Transparent and fully explainable — you can trace every decision to a specific word and rule.
- Handles basic negation correctly: "I am not happy" scores −1 (happy is negated) and is predicted `mixed` (close to negative with a positive word present), which is a reasonable approximation.
- Emoji signals work correctly: "just got the job offer 😄🎉" → `positive`.
- Does not overfit — word lists generalize across unseen sentences.

**Weaknesses:**

- Cannot detect sarcasm. "I absolutely love being stuck in traffic" is predicted `positive` because `love` scores +1 and nothing counters it.
- Vocabulary gaps cause misses. "exhausted but really grateful" → `neutral` because `exhausted` is in the negative list (−1) but `grateful` is not in the positive list, yielding a net score of −1 with no positive counter → `negative` is not triggered; the mixed threshold also isn't met.
- "Feeling tired but kind of hopeful" → `neutral` because `tired` is in the negative list (−1) but `hopeful` is not in the positive list at all. Score = −1, no positive hit → predicted `negative`, not `mixed`.
- Slang like `lowkey`, `vibing`, `no cap` has no effect unless explicitly added to word lists.
- The negation window is only one token wide — "I was not at all happy" would still apply the negation only to "at."

**Rule-based accuracy on the 18-post dataset: 0.61 (11/18 correct)**

---

## 4. How the ML Model Works

**Features used:** Bag-of-words representation using `CountVectorizer`. Each post is converted into a vector of word counts over the full vocabulary of the training set.

**Training data:** Trained directly on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:** The model achieved 100% accuracy on the training data. This is expected — with only 18 examples and no held-out test set, logistic regression memorizes the dataset rather than learning generalizable patterns. Adding more examples and evaluating on a separate test set would give a more honest accuracy figure.

When new examples were added, accuracy changed noticeably because the model's word-to-label associations shifted. A single added post can flip a word's learned polarity.

**Strengths and weaknesses:**

- *Strengths:* Learns patterns automatically without hand-written rules. Correctly handled all 18 posts including the two sarcasm examples — but only because those exact words appeared in labeled examples and the model memorized them.
- *Weaknesses:* Severely overfit to training data. The sarcasm posts work because "traffic" appeared in a negative-labeled post, not because the model "understands" sarcasm. The ML model is highly sensitive to the labels you assign — mislabeling a few posts will silently corrupt the classifier. It also can't handle out-of-vocabulary words at test time.

**ML accuracy on the 18-post dataset: 1.00 (training accuracy only — not a reliable measure of generalization)**

---

## 5. Evaluation

**How models were evaluated:** Both models were run on all 18 labeled posts in `SAMPLE_POSTS`. Accuracy was computed as the fraction of predictions that exactly matched `TRUE_LABELS`.

**Examples of correct predictions (rule-based):**

| Post | Predicted | True | Why it worked |
|---|---|---|---|
| "I love this class so much" | positive | positive | `love` scores +1, no counters, score = 1 → positive |
| "Today was a terrible day" | negative | negative | `terrible` scores −1, no counters → negative |
| "just got the job offer 😄🎉 so hyped!!" | positive | positive | 😄 (+1), 🎉 (+1), `hyped` (+1) → score = 3 → positive |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why it failed |
|---|---|---|---|
| "oh great, another Monday, I absolutely love this" | positive | negative | Sarcasm — `love` scores +1, nothing negative in the list → false positive |
| "Feeling tired but kind of hopeful" | neutral | mixed | `tired` scores −1 but `hopeful` is not in `POSITIVE_WORDS` → score = −1, no positive hit → misses `mixed` |
| "exhausted but really grateful for everyone who showed up" | neutral | mixed | `exhausted` scores −1 but `grateful` is not in `POSITIVE_WORDS` → same vocabulary gap |

**ML vs rule-based differences:** The ML model correctly predicted both sarcasm posts (`negative`) and all `mixed` posts. However, its success on sarcasm is incidental — it memorized those specific posts. The rule-based model's failures were consistent and predictable (vocabulary gaps, sarcasm blindness), while the ML model's failures would only appear on unseen data.

---

## 6. Limitations

1. **Tiny dataset.** 18 examples is far too few for meaningful generalization. Both models are unreliable on sentences not closely resembling the training examples.
2. **No sarcasm detection.** The rule-based model cannot distinguish "I love traffic" (sarcastic) from "I love this song" (genuine). Sarcasm requires world knowledge or very large context windows.
3. **Vocabulary gaps in rule-based model.** Words like `hopeful`, `grateful`, `overwhelmed`, `proud` are missing from the starter word lists and are invisible to the scorer unless added.
4. **Negation window is narrow.** Only one token of negation context is tracked. Multi-word negations ("not at all happy") are handled incorrectly.
5. **ML model evaluates only on training data.** The 100% accuracy figure is meaningless as a generalization measure. A proper evaluation would require a held-out test set.
6. **No handling of intensity.** "I am devastated" and "I am slightly annoyed" get the same score (both just −1 from "sad"/"annoyed" if those words existed).
7. **Emojis are hard to tokenize.** Multi-character emoji sequences and skin-tone modifiers may not be split correctly by the simple `split()` tokenizer.

---

## 7. Ethical Considerations

- **Misclassifying distress signals.** A message like "I'm fine 🙂" could represent masking serious distress. A mood classifier confidently labeling it `neutral` could suppress a flag that a human reviewer would catch.
- **Dialect and slang bias.** The word lists and training data reflect one variety of English internet slang. Expressions common in other dialects, languages, or age groups may be systematically misclassified or ignored entirely.
- **Label subjectivity.** Human labelers disagree on ambiguous posts. The "true" labels in this dataset reflect one person's judgment. Deploying a model trained on subjective labels encodes those assumptions silently.
- **Privacy.** Analyzing personal messages for mood without user consent is a significant privacy risk, even with a simple keyword-based model.
- **Automation risk.** In any high-stakes context (mental health apps, content moderation), these models are far too brittle for autonomous decision-making. They should be treated as rough filters, not final judgments.

---

## 8. Ideas for Improvement

- **More data:** At minimum 200–500 labeled examples across all four classes would give the ML model a meaningful training signal.
- **Held-out test set:** Split data into train/test before evaluation to get honest accuracy numbers.
- **Expand vocabulary:** Add `hopeful`, `grateful`, `proud`, `overwhelmed`, `devastated`, and other common sentiment words to the rule-based lists.
- **TF-IDF instead of CountVectorizer:** Downweights common words and gives more signal to distinctive terms.
- **Wider negation window:** Track a "negation is active" flag for 2–3 tokens after a negation word instead of just 1.
- **Emoji normalization:** Map common emoji to sentiment tags before tokenizing so they work consistently.
- **Sarcasm signals:** Add common sarcasm markers ("oh great", "I absolutely love", "just what I needed") as negative-weighted phrases in the rule-based model.
- **Pre-trained embeddings or a small transformer:** A model fine-tuned on sentiment data (e.g., DistilBERT on SST-2) would handle sarcasm, slang, and context far better than either approach here.
