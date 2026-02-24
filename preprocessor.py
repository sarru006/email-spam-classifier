"""
utils/preprocessor.py
---------------------
8-step NLP preprocessing pipeline — stdlib + regex only (no NLTK/SpaCy).

Steps:
  1. Lowercasing
  2. HTML Tag Removal
  3. URL / Email Removal
  4. Number Removal
  5. Punctuation Removal
  6. Tokenization
  7. Stopword Removal
  8. Lemmatization (suffix-stripping heuristic)
"""

import re
from typing import List

_HTML_RE  = re.compile(r"<[^>]+>", re.IGNORECASE)
_URL_RE   = re.compile(r"(https?://|ftp://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NUM_RE   = re.compile(r"\b\d+\b")
_PUNCT_RE = re.compile(r"[^a-z\s]")
_SPACE_RE = re.compile(r"\s+")

_STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","as","at","be","because","been","before","being","below","between",
    "both","but","by","cannot","could","did","do","does","doing","down",
    "during","each","few","for","from","further","get","got","had","has",
    "have","having","he","her","here","hers","herself","him","himself","his",
    "how","if","in","into","is","it","its","itself","me","more","most","my",
    "myself","no","nor","not","of","off","on","once","only","or","other",
    "our","ours","ourselves","out","over","own","same","she","should","so",
    "some","such","than","that","the","their","theirs","them","themselves",
    "then","there","these","they","this","those","through","to","too","under",
    "until","up","very","was","we","were","what","when","where","which",
    "while","who","whom","why","will","with","would","you","your","yours",
    "yourself","yourselves","said","also","one","two","may","per","see","us",
    "re","fw","cc","bcc","subject","date","from","to","sent","original",
    "message","email","nt","ve","ll","im","dont","doesnt","isnt","arent",
    "wasnt","werent","hasnt","havent","wouldnt","couldnt","shouldnt",
}

_SUFFIX_RULES = [
    ("nesses",""), ("ness",""), ("ments",""), ("ment",""), ("ings",""),
    ("ing",""), ("ations","ate"), ("ation","ate"), ("ators","ate"),
    ("ator","ate"), ("ively",""), ("ives",""), ("ive",""),
    ("efully",""), ("eful",""), ("fully",""), ("ful",""),
    ("ously",""), ("ous",""), ("encies",""), ("ency",""),
    ("ances",""), ("ance",""), ("ences",""), ("ence",""),
    ("isms",""), ("ism",""), ("ists",""), ("ist",""),
    ("izes",""), ("ize",""), ("ises",""), ("ise",""),
    ("ies","y"), ("ied","y"), ("ed",""), ("er",""), ("est",""),
    ("s",""),
]

def _lemmatize(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[:-len(suffix)] + replacement
            if len(stem) >= 3:
                return stem
    return word

def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = _HTML_RE.sub(" ", t)
    t = _URL_RE.sub(" urltoken ", t)
    t = _EMAIL_RE.sub(" emailtoken ", t)
    t = _NUM_RE.sub(" numtoken ", t)
    t = _PUNCT_RE.sub(" ", t)
    tokens = [
        _lemmatize(tok)
        for tok in t.split()
        if tok not in _STOP_WORDS and len(tok) > 2
    ]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()

def preprocess_batch(texts: List[str], batch_size: int = 500) -> List[str]:
    return [preprocess(t) for t in texts]
