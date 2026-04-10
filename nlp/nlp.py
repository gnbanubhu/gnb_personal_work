"""
nlp/nlp.py
-----------
Comprehensive NLP (Natural Language Processing) programs covering:

  1.  Text Tokenization              — word and sentence tokenization (NLTK)
  2.  Stopwords Removal              — filter common words
  3.  Stemming                       — reduce words to root form (PorterStemmer)
  4.  Lemmatization                  — reduce words to dictionary form (WordNetLemmatizer)
  5.  Part-of-Speech (POS) Tagging   — label each word with its grammatical role
  6.  Named Entity Recognition (NER) — identify persons, orgs, locations (NLTK + spaCy)
  7.  Sentiment Analysis             — VADER sentiment scoring
  8.  TF-IDF Vectorization           — term frequency–inverse document frequency
  9.  Cosine Similarity              — measure text similarity
  10. N-Grams                        — bigrams and trigrams
  11. Word Frequency Analysis        — most common words
  12. Text Cleaning                  — lowercasing, punctuation removal, regex

Libraries: nltk, spacy, scikit-learn
"""

import re
import string
from collections import Counter

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
from nltk.tokenize           import word_tokenize, sent_tokenize
from nltk.corpus             import stopwords
from nltk.stem               import PorterStemmer, SnowballStemmer
from nltk.stem               import WordNetLemmatizer
from nltk                    import pos_tag, ne_chunk
from nltk.util               import ngrams
from nltk.sentiment.vader    import SentimentIntensityAnalyzer

# ── spaCy ─────────────────────────────────────────────────────────────────────
import spacy

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE CORPUS
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_TEXT = (
    "Natural Language Processing (NLP) is a subfield of linguistics, computer science, "
    "and artificial intelligence concerned with the interactions between computers and human language. "
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California. "
    "The company released the iPhone in 2007, which revolutionized the smartphone industry. "
    "Google and Microsoft are also major players in the technology sector."
)

SENTENCES = [
    "I absolutely love this product! It is amazing and works perfectly.",
    "This is the worst experience I have ever had. Terrible service.",
    "The package arrived on time. Delivery was okay.",
    "Fantastic quality and great value for money. Highly recommend!",
    "It's broken and the customer support was useless. Very disappointed.",
]

DOCUMENTS = [
    "Machine learning algorithms learn from data to make predictions.",
    "Deep learning uses neural networks with many layers for complex tasks.",
    "Natural language processing enables computers to understand human text.",
    "Computer vision allows machines to interpret and understand images.",
    "Reinforcement learning trains agents by rewarding desired behaviors.",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════

section("1. TEXT TOKENIZATION")

text = "Hello! This is an NLP demo. We will explore many exciting concepts."

# Sentence tokenization
sentences = sent_tokenize(text)
subsection("Sentence Tokenization")
for i, s in enumerate(sentences, 1):
    print(f"    Sentence {i}: {s}")

# Word tokenization
words = word_tokenize(text)
subsection("Word Tokenization")
print(f"    Tokens  : {words}")
print(f"    Count   : {len(words)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. STOPWORDS REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

section("2. STOPWORDS REMOVAL")

stop_words = set(stopwords.words("english"))
raw_tokens = word_tokenize(SAMPLE_TEXT.lower())
filtered   = [w for w in raw_tokens if w.isalpha() and w not in stop_words]

print(f"    Original token count : {len(raw_tokens)}")
print(f"    After removal        : {len(filtered)}")
print(f"    Sample stopwords     : {sorted(list(stop_words))[:10]}")
print(f"    Filtered tokens      : {filtered[:15]} ...")


# ══════════════════════════════════════════════════════════════════════════════
# 3. STEMMING
# ══════════════════════════════════════════════════════════════════════════════

section("3. STEMMING")

porter   = PorterStemmer()
snowball = SnowballStemmer("english")

words_to_stem = ["running", "runs", "runner", "easily", "fairly",
                 "caring", "cared", "generously", "happiness", "studies"]

print(f"\n    {'Word':<16} {'Porter':<16} {'Snowball':<16}")
print(f"    {'─'*16} {'─'*16} {'─'*16}")
for w in words_to_stem:
    print(f"    {w:<16} {porter.stem(w):<16} {snowball.stem(w):<16}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEMMATIZATION
# ══════════════════════════════════════════════════════════════════════════════

section("4. LEMMATIZATION (WordNet)")

lemmatizer = WordNetLemmatizer()

# pos: 'n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb
words_pos = [
    ("running",  "v"), ("ran",       "v"), ("better",    "a"),
    ("studies",  "v"), ("wolves",    "n"), ("caring",    "v"),
    ("happier",  "a"), ("children",  "n"), ("flies",     "v"),
    ("greatest", "a"),
]

print(f"\n    {'Word':<16} {'POS':<8} {'Lemma':<16}")
print(f"    {'─'*16} {'─'*8} {'─'*16}")
for word, pos in words_pos:
    lemma = lemmatizer.lemmatize(word, pos=pos)
    print(f"    {word:<16} {pos:<8} {lemma:<16}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PART-OF-SPEECH (POS) TAGGING
# ══════════════════════════════════════════════════════════════════════════════

section("5. PART-OF-SPEECH (POS) TAGGING")

pos_text  = "The quick brown fox jumps over the lazy dog near the river bank."
pos_tokens = word_tokenize(pos_text)
pos_tags   = pos_tag(pos_tokens)

# Tag legend
TAG_MAP = {
    "DT": "Determiner", "JJ": "Adjective", "NN": "Noun (singular)",
    "NNS": "Noun (plural)", "VBZ": "Verb (3rd person)", "IN": "Preposition",
    "NNP": "Proper noun", "RB": "Adverb", "VBG": "Verb (gerund)",
    "CD": "Cardinal number", "PRP": "Personal pronoun", "VBD": "Verb (past)",
}

print(f"\n    {'Token':<18} {'Tag':<8} {'Description':<25}")
print(f"    {'─'*18} {'─'*8} {'─'*25}")
for token, tag in pos_tags:
    desc = TAG_MAP.get(tag, tag)
    print(f"    {token:<18} {tag:<8} {desc:<25}")

# ── spaCy POS tagging ─────────────────────────────────────────────────────────
subsection("spaCy POS Tagging")

nlp_spacy = spacy.load("en_core_web_sm")
doc = nlp_spacy(pos_text)

print(f"    {'Token':<18} {'POS':<10} {'Dep':<12} {'Fine-grained tag'}")
print(f"    {'─'*18} {'─'*10} {'─'*12} {'─'*20}")
for token in doc:
    print(f"    {token.text:<18} {token.pos_:<10} {token.dep_:<12} {token.tag_}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. NAMED ENTITY RECOGNITION (NER)
# ══════════════════════════════════════════════════════════════════════════════

section("6. NAMED ENTITY RECOGNITION (NER)")

# ── NLTK NER ──────────────────────────────────────────────────────────────────
subsection("NLTK NER")

ner_tokens = word_tokenize(SAMPLE_TEXT)
ner_pos    = pos_tag(ner_tokens)
ner_tree   = ne_chunk(ner_pos)

print("    Entities found (NLTK):")
for subtree in ner_tree:
    if hasattr(subtree, "label"):
        entity = " ".join(w for w, t in subtree.leaves())
        print(f"      [{subtree.label():12s}]  {entity}")

# ── spaCy NER ─────────────────────────────────────────────────────────────────
subsection("spaCy NER")

doc_ner = nlp_spacy(SAMPLE_TEXT)
print("    Entities found (spaCy):")
print(f"    {'Entity':<30} {'Label':<12} {'Description'}")
print(f"    {'─'*30} {'─'*12} {'─'*30}")
for ent in doc_ner.ents:
    print(f"    {ent.text:<30} {ent.label_:<12} {spacy.explain(ent.label_)}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SENTIMENT ANALYSIS (VADER)
# ══════════════════════════════════════════════════════════════════════════════

section("7. SENTIMENT ANALYSIS  (VADER)")

sia = SentimentIntensityAnalyzer()

print(f"\n    {'Sentence':<55} {'NEG':>6} {'NEU':>6} {'POS':>6} {'Compound':>9} {'Label'}")
print(f"    {'─'*55} {'─'*6} {'─'*6} {'─'*6} {'─'*9} {'─'*10}")

for sent in SENTENCES:
    scores = sia.polarity_scores(sent)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    preview = sent[:52] + "..." if len(sent) > 52 else sent
    print(
        f"    {preview:<55} {scores['neg']:>6.3f} {scores['neu']:>6.3f} "
        f"{scores['pos']:>6.3f} {compound:>9.4f} {label}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. TF-IDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════════

section("8. TF-IDF VECTORIZATION")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(DOCUMENTS)
feature_names = vectorizer.get_feature_names_out()

print(f"\n    Corpus size      : {len(DOCUMENTS)} documents")
print(f"    Vocabulary size  : {len(feature_names)} terms")
print(f"    TF-IDF matrix    : {tfidf_matrix.shape}")

subsection("Top TF-IDF Terms per Document")
import numpy as np

for idx, doc in enumerate(DOCUMENTS):
    row    = tfidf_matrix[idx].toarray()[0]
    top_n  = np.argsort(row)[::-1][:5]
    terms  = [(feature_names[i], round(row[i], 4)) for i in top_n if row[i] > 0]
    print(f"    Doc {idx+1}: {doc[:45]:<45}")
    print(f"          Top terms: {terms}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

section("9. COSINE SIMILARITY")

query = "How do neural networks learn from data?"
corpus_with_query = DOCUMENTS + [query]

vec   = TfidfVectorizer(stop_words="english")
mat   = vec.fit_transform(corpus_with_query)
sims  = cosine_similarity(mat[-1], mat[:-1]).flatten()

print(f"\n    Query  : \"{query}\"")
print(f"\n    {'Document':<55} {'Similarity':>10}")
print(f"    {'─'*55} {'─'*10}")

ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
for rank, (i, score) in enumerate(ranked, 1):
    preview = DOCUMENTS[i][:52] + "..." if len(DOCUMENTS[i]) > 52 else DOCUMENTS[i]
    print(f"    {rank}. {preview:<53} {score:>10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. N-GRAMS
# ══════════════════════════════════════════════════════════════════════════════

section("10. N-GRAMS")

ngram_text  = "data engineering involves building pipelines that process transform and load data"
ngram_tokens = ngram_text.split()

bigrams  = list(ngrams(ngram_tokens, 2))
trigrams = list(ngrams(ngram_tokens, 3))

subsection("Bigrams")
for bg in bigrams:
    print(f"    {bg}")

subsection("Trigrams")
for tg in trigrams:
    print(f"    {tg}")

# Most frequent bigrams from the full corpus
subsection("Most Frequent Bigrams in SAMPLE_TEXT")
corpus_tokens = [
    w for w in word_tokenize(SAMPLE_TEXT.lower())
    if w.isalpha() and w not in stopwords.words("english")
]
corpus_bigrams = list(ngrams(corpus_tokens, 2))
bigram_freq    = Counter(corpus_bigrams).most_common(8)
for bg, freq in bigram_freq:
    print(f"    {str(bg):<35}  freq: {freq}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. WORD FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

section("11. WORD FREQUENCY ANALYSIS")

freq_tokens = [
    w for w in word_tokenize(SAMPLE_TEXT.lower())
    if w.isalpha() and w not in stop_words
]

freq_dist = Counter(freq_tokens)
top_words = freq_dist.most_common(10)

print(f"\n    Total tokens (cleaned) : {len(freq_tokens)}")
print(f"    Unique words           : {len(freq_dist)}")
print(f"\n    {'Rank':<6} {'Word':<20} {'Count':<8} {'Bar'}")
print(f"    {'─'*6} {'─'*20} {'─'*8} {'─'*25}")
for rank, (word, count) in enumerate(top_words, 1):
    bar = "█" * count
    print(f"    {rank:<6} {word:<20} {count:<8} {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

section("12. TEXT CLEANING")

dirty_text = (
    "  HELLO!!! This is a <b>sample</b> text with URLs: https://example.com "
    "and email: user@domain.com. Numbers like 12345 & special chars #@$% "
    "should be cleaned.   Extra   spaces   too!!  "
)

def clean_text(text: str) -> str:
    text = text.lower()                                 # lowercase
    text = re.sub(r"<[^>]+>",        "",  text)        # remove HTML tags
    text = re.sub(r"http\S+|www\S+", "",  text)        # remove URLs
    text = re.sub(r"\S+@\S+",        "",  text)        # remove emails
    text = re.sub(r"\d+",            "",  text)        # remove numbers
    text = re.sub(r"[^\w\s]",        " ", text)        # remove punctuation
    text = re.sub(r"\s+",            " ", text).strip()# normalize whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

cleaned = clean_text(dirty_text)
print(f"\n    Original : {dirty_text.strip()}")
print(f"\n    Cleaned  : {cleaned}")

# ── spaCy linguistic pipeline summary ────────────────────────────────────────
subsection("spaCy Linguistic Pipeline on Cleaned Text")

doc_clean = nlp_spacy("Hello sample text URLs emails numbers special characters cleaned extra spaces")
print(f"    {'Token':<20} {'Lemma':<20} {'POS':<10} {'Is Stop'}")
print(f"    {'─'*20} {'─'*20} {'─'*10} {'─'*8}")
for token in doc_clean:
    print(f"    {token.text:<20} {token.lemma_:<20} {token.pos_:<10} {token.is_stop}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

section("SUMMARY")
print("""
    NLP Programs Demonstrated:
    ─────────────────────────────────────────────────────────
     1.  Tokenization          Word & sentence splitting (NLTK)
     2.  Stopwords Removal     Filter common non-informative words
     3.  Stemming              PorterStemmer & SnowballStemmer
     4.  Lemmatization         WordNetLemmatizer with POS hints
     5.  POS Tagging           NLTK pos_tag + spaCy dependency parse
     6.  Named Entity Rec.     NLTK ne_chunk + spaCy NER pipeline
     7.  Sentiment Analysis    VADER compound scoring
     8.  TF-IDF                Term frequency–inverse document frequency
     9.  Cosine Similarity     Query-document relevance ranking
    10.  N-Grams               Bigrams, trigrams, frequency analysis
    11.  Word Frequency        Counter-based most-common analysis
    12.  Text Cleaning         Regex pipeline for noisy real-world text
    ─────────────────────────────────────────────────────────
    Libraries: nltk · spacy · scikit-learn · numpy
""")
