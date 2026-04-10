"""
utils.py — Helper functions and custom domain lexicon
Sentiment Analysis — Product Review NLP Engine
Author: Nirdesh Patel
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────────
#  CUSTOM DOMAIN LEXICON (1,200 terms subset)
# ─────────────────────────────────────────────
POSITIVE_TERMS = [
    "excellent", "outstanding", "superb", "fantastic", "brilliant", "amazing",
    "wonderful", "perfect", "love", "loved", "great", "awesome", "best",
    "incredible", "impressive", "solid", "reliable", "durable", "sturdy",
    "fast", "quick", "smooth", "easy", "simple", "comfortable", "soft",
    "bright", "clear", "sharp", "powerful", "efficient", "lightweight",
    "portable", "compact", "stylish", "sleek", "elegant", "beautiful",
    "recommend", "recommended", "satisfied", "happy", "pleased", "delighted",
    "exceeded", "exceeded expectations", "worth", "value", "bargain",
    "affordable", "premium", "high quality", "top notch", "five star",
    "must buy", "must have", "game changer", "life saver", "10/10",
]

NEGATIVE_TERMS = [
    "terrible", "horrible", "awful", "dreadful", "disgusting", "pathetic",
    "worst", "useless", "broken", "defective", "faulty", "cheap", "flimsy",
    "fragile", "slow", "laggy", "buggy", "glitchy", "disappointing",
    "disappointed", "waste", "garbage", "trash", "junk", "overpriced",
    "scam", "fraud", "fake", "poor quality", "bad quality", "not working",
    "stopped working", "dead on arrival", "doa", "returned", "refund",
    "never again", "avoid", "do not buy", "zero stars", "one star",
    "regret", "regretted", "unhappy", "frustrated", "angry", "annoyed",
]

NEGATION_WORDS = [
    "not", "no", "never", "neither", "nor", "without", "lack", "lacking",
    "cannot", "can't", "won't", "doesn't", "didn't", "isn't", "wasn't",
    "hardly", "barely", "scarcely",
]

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) - set(NEGATION_WORDS)


def clean_text(text: str) -> str:
    """Full text cleaning pipeline."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Expand contractions (basic)
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    }
    for pattern, replacement in contractions.items():
        text = text.replace(pattern, replacement)
    # Remove punctuation (keep apostrophes for negation)
    text = re.sub(r"[^\w\s']", ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_stem(text: str) -> str:
    """Tokenize, remove stopwords, and apply stemming."""
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def preprocess(text: str) -> str:
    """Full preprocessing: clean → tokenize → stem."""
    return tokenize_and_stem(clean_text(text))


def get_lexicon_score(text: str) -> dict:
    """
    Score text using the custom domain lexicon.
    Returns dict with pos_count, neg_count, net_score.
    """
    text_lower = text.lower()
    pos = sum(1 for term in POSITIVE_TERMS if term in text_lower)
    neg = sum(1 for term in NEGATIVE_TERMS if term in text_lower)
    # Handle negation: if negation word precedes positive term, flip
    for neg_word in NEGATION_WORDS:
        pattern = re.compile(rf'\b{neg_word}\b.{{0,20}}(' + '|'.join(POSITIVE_TERMS[:20]) + r')', re.IGNORECASE)
        matches = len(pattern.findall(text_lower))
        pos = max(0, pos - matches)
        neg += matches
    return {"pos_count": pos, "neg_count": neg, "net_score": pos - neg}


def rating_to_label(rating: float) -> str:
    """Convert star rating to sentiment label for training."""
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"
