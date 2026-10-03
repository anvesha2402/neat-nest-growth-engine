"""Retrieval over the Neat Nest knowledge base (brand book, PRD, KPI rules, GTM playbook).

Each "## [ID] Title" section of a markdown file in knowledge/ is one chunk with
a stable source ID. TF-IDF (unigrams + bigrams) with the title repeated three
times gives short definition questions a fair chance; cosine similarity ranks
chunks and only those above a minimum score are returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import KNOWLEDGE_DIR

HEAD = re.compile(r"^## \[(?P<id>[A-Z0-9_]+)\] (?P<title>.+)$", re.M)


TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(text: str) -> str:
    """Lower-case and strip simple plurals so 'colours' matches 'colour'."""
    words = TOKEN.findall(text.lower())
    return " ".join(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def _terms(text: str) -> list[str]:
    return [w for w in _normalise(text).split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    doc: str


def load_chunks(folder: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(folder.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        heads = list(HEAD.finditer(body))
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
            text = body[h.end():end].strip()
            chunks.append(Chunk(h["id"], h["title"].strip(), text, path.stem))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate knowledge IDs")
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, min_score: float = 0.08):
        self.chunks = chunks or load_chunks()
        self.min_score = min_score
        corpus = [f"{c.title} {c.title} {c.title} {c.text}" for c in self.chunks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=sorted({_normalise(w) for w in ENGLISH_STOP_WORDS}),
                                   sublinear_tf=True,
                                   preprocessor=_normalise)
        self.matrix = self.vec.fit_transform(corpus)
        self.by_id = {c.id: c for c in self.chunks}
        self.title_terms = [set(_terms(c.title)) for c in self.chunks]

    def search(self, query: str, k: int = 2, title_bonus: float = 0.15) -> list[tuple[Chunk, float]]:
        """Cosine similarity plus a small bonus for query terms that appear in a chunk title."""
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        q = set(_terms(query))
        if q:
            sims = sims + title_bonus * np.array([len(q & t) / len(q) for t in self.title_terms])
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] >= self.min_score]
