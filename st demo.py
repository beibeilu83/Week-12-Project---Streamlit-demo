import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# -----------------------------
# Helper functions
# -----------------------------

def preprocess(text: str) -> str:
    """Basic normalization: lowercase, remove bracketed stage directions, keep letters and spaces."""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)  # remove stage directions like [Enter HAMLET]
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def sentence_split(text: str) -> List[str]:
    # Lightweight sentence splitter (avoids heavy dependencies)
    sentences = re.split(r"(?<=[\.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    return sentences if sentences else [text]


def compute_style_features(text: str) -> np.ndarray:
    """Return a small vector of style features.
    Features:
      0: average sentence length (words)
      1: type-token ratio (unique/total tokens)
      2: archaic function-word ratio
      3: bigram-to-unigram ratio (as a proxy for phrase density)
    """
    raw = text
    clean = preprocess(text)

    tokens = clean.split()
    n_tokens = len(tokens) or 1

    # Avg sentence length
    sentences = sentence_split(raw)
    sent_lengths = [word_count(s) for s in sentences] or [n_tokens]
    avg_sent_len = sum(sent_lengths) / max(1, len(sent_lengths))

    # Type-token ratio
    ttr = len(set(tokens)) / n_tokens

    # Archaic/function words common in Shakespeare
    archaic = {
        "thee","thou","thy","thine","hath","hast","doth","dost","art","ere",
        "nay","ay","oft","whence","wherefore","whither","hither","o","tis","twas",
        "ye","thyself","thyself","wert","prithee","marry","anon","sirrah","zounds"
    }
    archaic_ratio = sum(1 for w in tokens if w in archaic) / n_tokens

    # Bigram-to-unigram ratio
    bigrams = sum(1 for i in range(len(tokens) - 1) if tokens[i] and tokens[i+1])
    bigram_ratio = bigrams / n_tokens

    return np.array([avg_sent_len, ttr, archaic_ratio, bigram_ratio], dtype=float)


def style_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Turn Euclidean distance into a bounded [0,1] similarity.
    We scale features to reduce dominance, then convert distance -> similarity.
    """
    # Simple scaling factors chosen empirically for balance
    scales = np.array([30.0, 0.5, 0.02, 0.6])
    a = vec_a / scales
    b = vec_b / scales
    dist = np.linalg.norm(a - b)
    # Map distance to similarity with 1/(1+dist); clamp [0,1]
    return float(1.0 / (1.0 + dist))


# -----------------------------
# Reference Shakespeare corpus (public domain excerpts)
# Compact concatenation of several well-known passages.
# -----------------------------
SHAKESPEARE_REF = "\n".join([
    # Hamlet (Act 3, Scene 1)
    "To be, or not to be: that is the question: Whether 'tis nobler in the mind to suffer",
    "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles",
    "And by opposing end them. To die—to sleep, No more; and by a sleep to say we end",
    "The heart-ache and the thousand natural shocks That flesh is heir to. 'Tis a consummation",
    "Devoutly to be wish'd. To die, to sleep— To sleep—perchance to dream: ay, there's the rub:",

    # Macbeth (Act 2, Scene 1; Act 5, Scene 5)
    "Is this a dagger which I see before me, The handle toward my hand? Come, let me clutch thee.",
    "I have thee not, and yet I see thee still. Art thou not, fatal vision, sensible",
    "To feeling as to sight? or art thou but A dagger of the mind, a false creation",
    "Proceeding from the heat-oppressed brain?", 
    "Tomorrow, and tomorrow, and tomorrow, Creeps in this petty pace from day to day",
    "To the last syllable of recorded time; And all our yesterdays have lighted fools",
    "The way to dusty death. Out, out, brief candle! Life's but a walking shadow, a poor player",
    "That struts and frets his hour upon the stage And then is heard no more. It is a tale",
    "Told by an idiot, full of sound and fury, Signifying nothing.",

    # Romeo and Juliet (Act 2, Scene 2)
    "But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun.",
    "Arise, fair sun, and kill the envious moon, Who is already sick and pale with grief",
    "That thou her maid art far more fair than she:",

    # Julius Caesar (Act 3, Scene 2)
    "Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him.",
    "The evil that men do lives after them; The good is oft interrèd with their bones.",

    # King Lear (Act 3, Scene 2)
    "Blow, winds, and crack your cheeks! rage! blow! You cataracts and hurricanoes, spout",
    "Till you have drench'd our steeples, drown'd the cocks!"
])

# Precompute Shakespeare style features once
REF_STYLE = compute_style_features(SHAKESPEARE_REF)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Shakespeare Similarity Analyzer", page_icon="🎭")

st.title("🎭 Test Your Shakespeare Precentage!")
st.write(
    "Paste up to **500 words** of English prose. We'll estimate how stylistically similar it is to **William Shakespeare** "
    "using a combination of lexical/phrase similarity and a few lightweight style indicators."
)

# Input area
sample = (
    "O gentle moon, attend my musing thought; for in thy silver lamp I spy the shape of time, "
    "which like a thief doth pick the pocket of our days."
)

user_text = st.text_area(
    label="Your text (≤ 500 words)",
    value="",
    height=220,
    placeholder=sample,
)

# Word limit check
words = word_count(user_text)
limit = 500
if words > limit:
    st.warning(f"Your input is {words} words. Please trim to {limit} words or fewer.")

analyze = st.button("Analyze Similarity")

if analyze:
    if not user_text.strip():
        st.error("Please enter some text to analyze.")
        st.stop()
    if words > limit:
        st.stop()

    # Compute TF–IDF similarity (unigram + bigram)
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform([SHAKESPEARE_REF, user_text])
    tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0])

    # Compute style similarity
    user_style = compute_style_features(user_text)
    styl_sim = style_similarity(REF_STYLE, user_style)

    # Weighted blend (tweakable)
    final_score = 0.7 * tfidf_sim + 0.3 * styl_sim

    # Display results
    st.subheader("Results")
    st.metric(label="Overall Shakespeare Similarity", value=f"{final_score*100:.1f}%")

    with st.expander("See breakdown"):
        st.write(
            {
                "Lexical/phrase similarity (TF–IDF cosine)": round(tfidf_sim, 4),
                "Style similarity": round(styl_sim, 4),
                "Blend weights": {"lexical": 0.7, "style": 0.3},
                "Vectorizer vocabulary size": len(vectorizer.vocabulary_),
            }
        )
        st.write("Style features (your text vs. reference):")
        labels = [
            "Avg sentence length (words)",
            "Type–token ratio",
            "Archaic-function word ratio",
            "Bigram/Unigram ratio",
        ]
        col1, col2 = st.columns(2)
        with col1:
            st.write({lbl: float(val) for lbl, val in zip(labels, user_style)})
        with col2:
            st.write({lbl: float(val) for lbl, val in zip(labels, REF_STYLE)})

    # Short interpretation
    if final_score >= 0.60:
        verdict = "High resemblance — your passage echoes Shakespeare strongly."
    elif final_score >= 0.35:
        verdict = "Moderate resemblance — some Shakespearean elements present."
    else:
        verdict = "Low resemblance — stylistically distant from Shakespeare."
    st.info(verdict)

    st.caption(
        "This tool offers an *approximate* signal based on small reference passages and lightweight features; "
        "it does not detect authorship."
    )

st.markdown("""
---
**How it works (quickly):** We compare your text with a compact reference built from several famous Shakespeare passages. 
First, we look at word and bigram usage via TF–IDF cosine similarity. Next, we compare a few style indicators (sentence length, lexical diversity, archaic function words, and phrase density). A weighted blend gives the final score.

**Run locally:**
```bash
pip install streamlit scikit-learn numpy
streamlit run shakespeare_similarity_app.py
```
""")
