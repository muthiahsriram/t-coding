"""BM25 keyword search over the chunked corpus.

Hand-rolled rather than pulled from ``rank_bm25`` for two reasons. It is forty
lines of well-specified arithmetic, and more importantly the domain needs *term
upweighting* - "ABSD", "CPFIS" and "Full Retirement Sum" have to dominate a
query they appear in, and bolting that onto a library that scores every term
equally is more work than owning the scorer.

This is also the fallback path: if the workspace serves no embedding endpoint,
this is the whole retrieval layer rather than half of it.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from app.rag.documents import Chunk

K1 = 1.5  # term-frequency saturation
B = 0.75  # length normalisation

_TOKEN = re.compile(r"[^\W_]+(?:[.\-'][^\W_]+)*", re.UNICODE)

# CJK is written without spaces, so a whole clause matches as one token and
# matches nothing. Worse, the Latin-only pattern this replaced dropped CJK
# entirely, which made every Chinese document look a handful of tokens long --
# BM25 length normalisation then scored them enormously on any Latin term they
# happened to contain, and the Chinese SRS page outranked the English one on an
# English query. Character bigrams are the standard fix and need no segmenter.
_CJK_RANGES = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_CJK = re.compile(f"[{_CJK_RANGES}]")
_CJK_SPLIT = re.compile(f"[{_CJK_RANGES}]+|[^{_CJK_RANGES}]+")


def _expand(token: str) -> list[str]:
    """Split a token into searchable terms, bigramming any CJK runs.

    CJK is unspaced, so a Latin acronym embedded in a Chinese sentence arrives
    glued to it as one token. Bigramming the whole thing would shred the
    acronym into "sr", "rs" and lose the only term the query and the document
    reliably share. Runs are separated first, then only the CJK ones bigram.
    """
    if not _CJK.search(token):
        return [token]

    terms: list[str] = []
    for run in _CJK_SPLIT.findall(token):
        if _CJK.search(run):
            terms.extend(
                [run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)]
            )
        else:
            terms.append(run)
    return terms


# Words that carry no discriminating power in a corpus that is *entirely* about
# personal finance. "investment" appears in almost every chunk; keeping it
# costs recall precision and nothing else.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
is are was were be been being do does did have has had i me my we our you your it its
what which who how when where why can could should would may might will shall
""".split())

# Terms whose presence in a query is close to decisive. A query mentioning ABSD
# is about ABSD; ordinary TF-IDF weighting treats it as one term among many and
# lets a longer chunk with more incidental matches outrank the one document
# that actually defines it.
DOMAIN_TERMS: dict[str, float] = {
    "absd": 3.0, "bsd": 3.0, "cpfis": 3.0, "srs": 2.5, "cpf": 2.0,
    "medisave": 2.5, "hdb": 2.0, "iras": 2.0, "mas": 2.0,
    "tdsr": 3.0, "msr": 3.0, "ltv": 2.5,
    "frs": 2.5, "brs": 2.5, "ers": 2.5, "bhs": 2.5,
    "drift": 2.0, "rebalancing": 2.0, "concentration": 2.0, "duration": 2.0,
    "withholding": 2.5, "relief": 2.0, "penalty": 2.0, "cap": 1.5,
    "supersede": 2.0, "superseded": 2.0, "withdrawn": 2.0,
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        for token in _expand(match.group(0).lower()):
            if token not in STOPWORDS:
                tokens.append(token)
    return tokens


class BM25Index:
    """Okapi BM25 with per-term query weighting."""

    def __init__(self, chunks: tuple[Chunk, ...]):
        self.chunks = chunks
        # The title is repeated into the indexed text: a chunk from halfway
        # down a document otherwise has no lexical trace of what it is about.
        documents = [tokenize(f"{c.meta.title} {c.text}") for c in chunks]

        self.lengths = np.array([len(d) for d in documents], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if len(documents) else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(documents):
            for term, count in Counter(tokens).items():
                self.postings[term].append((index, count))

        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def scores(self, query: str, *, boosts: dict[str, float] | None = None) -> np.ndarray:
        """BM25 score of every chunk against ``query``."""
        weights = {**DOMAIN_TERMS, **(boosts or {})}
        scores = np.zeros(len(self.chunks), dtype=np.float32)

        # Length normalisation, precomputed once per query rather than per term.
        denominator_length = K1 * (
            1 - B + B * self.lengths / (self.average_length or 1.0)
        )

        for term, query_count in Counter(tokenize(query)).items():
            posting = self.postings.get(term)
            if not posting:
                continue
            weight = weights.get(term, 1.0) * query_count
            idf = self.idf[term]
            indices = np.fromiter((i for i, _ in posting), dtype=np.int64, count=len(posting))
            counts = np.fromiter((c for _, c in posting), dtype=np.float32, count=len(posting))
            scores[indices] += weight * idf * (
                counts * (K1 + 1) / (counts + denominator_length[indices])
            )

        return scores
