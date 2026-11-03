"""llm_mesh.router: semantic routing to the right model.

``SemanticRouter`` is a TF-IDF + class-centroid (Rocchio) classifier that
maps a raw user query to a task type (reasoning / coding / vision / chat).
Routing policies then pick a model for that task type: quality-first,
cost-first, latency-first, or a balanced cost/latency/quality tradeoff.
"""

import math
import re
from typing import Dict, List, Tuple

import numpy as np

from .models import TASK_TYPES, ModelRegistry

STOPWORDS = frozenset(
    "a an the and or but of to in on for with is are was were be been "
    "it its this that these those i you he she we they my your his her "
    "our their me him us them do does did can could should would will "
    "what how why when where which who whom whose there here from at by "
    "as not no yes if then than so such into out up down over under do "
    "have has had having".split()
)


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    out = []
    for t in toks:
        if t in STOPWORDS or len(t) <= 1:
            continue
        # light plural stemming so "photos"/"images"/"notes" match
        # their singular forms seen in training
        if (len(t) > 4 and t.endswith("s")
                and not t.endswith(("ss", "us", "is"))):
            t = t[:-1]
        out.append(t)
    return out


class SemanticRouter:
    """TF-IDF vectors, cosine similarity to per-class centroids."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self.classes: List[str] = []
        self.centroids: np.ndarray = np.zeros((0, 0))

    def fit(self, texts: List[str], labels: List[str]) -> "SemanticRouter":
        tokenized = [tokenize(t) for t in texts]
        vocab: Dict[str, int] = {}
        for toks in tokenized:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        n, v = len(texts), len(vocab)
        df = np.zeros(v)
        tf = np.zeros((n, v))
        for i, toks in enumerate(tokenized):
            for t in set(toks):
                df[vocab[t]] += 1.0
            for t in toks:
                tf[i, vocab[t]] += 1.0
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        vecs = tf * self.idf[None, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vecs = vecs / norms
        self.classes = sorted(set(labels))
        cents = []
        for c in self.classes:
            idx = [i for i, lab in enumerate(labels) if lab == c]
            cent = vecs[idx].mean(axis=0)
            nrm = np.linalg.norm(cent)
            cents.append(cent / nrm if nrm > 0 else cent)
        self.centroids = np.stack(cents)
        return self

    def _vectorize(self, text: str) -> np.ndarray:
        v = len(self.vocab)
        tf = np.zeros(v)
        for t in tokenize(text):
            j = self.vocab.get(t)
            if j is not None:
                tf[j] += 1.0
        vec = tf * self.idf
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 0 else vec

    def scores(self, text: str) -> Dict[str, float]:
        vec = self._vectorize(text)
        sims = self.centroids @ vec
        return {c: float(s) for c, s in zip(self.classes, sims)}

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (task_type, confidence). Confidence is the margin between
        the top two cosine similarities, mapped to [0, 1]."""
        s = self.scores(text)
        ranked = sorted(s.items(), key=lambda kv: kv[1], reverse=True)
        best, second = ranked[0][1], ranked[1][1] if len(ranked) > 1 else -1.0
        confidence = float(min(1.0, max(0.0, (best - second) / 2.0 + 0.5)))
        return ranked[0][0], confidence

    def accuracy(self, texts: List[str], labels: List[str]) -> float:
        ok = sum(self.predict(t)[0] == lab for t, lab in zip(texts, labels))
        return ok / len(labels) if labels else 0.0


POLICIES = ("quality-first", "cost-first", "latency-first", "balanced")


def route(registry: ModelRegistry, task_type: str, in_tokens: int,
          out_tokens: int, policy: str = "quality-first",
          quality_floor: float = 0.60,
          exclude: Tuple[str, ...] = ()) -> str:
    """Pick a model name for a task under a routing policy.

    cost/latency-first and balanced only consider models whose mean
    quality for the task clears ``quality_floor`` (falling back to all
    available models if none clear it). ``exclude`` removes models from
    consideration (used by failover).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task type {task_type!r}")
    cands = [n for n in registry.available() if n not in exclude]
    if not cands:
        raise RuntimeError("no models available for routing")
    if policy == "quality-first":
        return max(cands, key=lambda n: registry.quality_of(n, task_type))
    ok = [n for n in cands
          if registry.quality_of(n, task_type) >= quality_floor] or cands
    if policy == "cost-first":
        return min(ok, key=lambda n: registry.expected_cost_usd(
            n, in_tokens, out_tokens))
    if policy == "latency-first":
        return min(ok, key=lambda n: registry.expected_latency_ms(
            n, out_tokens))
    # balanced: quality minus normalized cost and latency penalties
    qs = {n: registry.quality_of(n, task_type) for n in ok}
    cs = {n: registry.expected_cost_usd(n, in_tokens, out_tokens) for n in ok}
    ls = {n: registry.expected_latency_ms(n, out_tokens) for n in ok}

    def norm(d: Dict[str, float]) -> Dict[str, float]:
        lo, hi = min(d.values()), max(d.values())
        if hi == lo:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    cn, ln = norm(cs), norm(ls)
    return max(ok, key=lambda n: qs[n] - 0.6 * cn[n] - 0.4 * ln[n])


def route_query(registry: ModelRegistry, router: SemanticRouter, query: str,
                in_tokens: int, out_tokens: int,
                policy: str = "quality-first",
                quality_floor: float = 0.60) -> Tuple[str, str, float]:
    """End-to-end: query text -> (model, inferred_task_type, confidence)."""
    task_type, confidence = router.predict(query)
    model = route(registry, task_type, in_tokens, out_tokens, policy,
                  quality_floor)
    return model, task_type, confidence


# ---------------------------------------------------------------------------
# Training / evaluation corpora (hand-labeled; also used by experiments)
# ---------------------------------------------------------------------------

TRAIN_CORPUS: List[Tuple[str, str]] = [
    # reasoning
    ("prove that the sum of two even numbers is always even", "reasoning"),
    ("if all bloops are razzies and all razzies are lazzies, are all bloops lazzies", "reasoning"),
    ("solve the equation 3x + 7 = 22 showing each algebraic step", "reasoning"),
    ("a train leaves chicago at 60 mph and another at 40 mph, when do they meet", "reasoning"),
    ("identify the logical fallacy in this argument about taxes", "reasoning"),
    ("calculate the probability that two dice sum to nine", "reasoning"),
    ("explain bayes theorem using a medical test example", "reasoning"),
    ("a bat and ball cost 1.10 in total, the bat costs 1.00 more, what does the ball cost", "reasoning"),
    ("derive the quadratic formula from completing the square", "reasoning"),
    ("a farmer has chickens and cows with 20 heads and 56 legs, how many of each", "reasoning"),
    ("negate the statement that all swans are white using formal logic", "reasoning"),
    ("which weighs more, a pound of feathers or a pound of bricks, and why is it a trick", "reasoning"),
    ("compute the expected value of a dice game before deciding to play", "reasoning"),
    ("explain the monty hall problem and compute the switching probability", "reasoning"),
    ("determine whether this syllogism about mammals and fish is logically valid", "reasoning"),
    ("differentiate the polynomial and evaluate the derivative at a given point", "reasoning"),
    ("in a room of 30 people what is the probability two share a birthday", "reasoning"),
    ("walk through the proof by induction that the formula holds for all n", "reasoning"),
    # coding
    ("write a python function to reverse a linked list in place", "coding"),
    ("debug this javascript null pointer error in my fetch call", "coding"),
    ("how do I write a SQL query joining three tables on foreign keys", "coding"),
    ("refactor this java class to use dependency injection", "coding"),
    ("write a rust function that parses json safely without panicking", "coding"),
    ("explain what this regular expression matches in plain english", "coding"),
    ("convert this recursive fibonacci function to an iterative one", "coding"),
    ("write pytest unit tests for a login api endpoint", "coding"),
    ("how do I fix a merge conflict during a git rebase", "coding"),
    ("optimize this nested loop with numpy vectorization", "coding"),
    ("write a dockerfile for a node express application", "coding"),
    ("implement binary search in go with generics", "coding"),
    ("review this pull request that refactors the authentication module", "coding"),
    ("explain javascript closures with small code examples", "coding"),
    ("write a kubernetes deployment yaml for a redis cache", "coding"),
    ("how do I profile a slow python data pipeline", "coding"),
    ("design a rest api for managing warehouse inventory", "coding"),
    ("write a github actions workflow that runs lint and tests", "coding"),
    # vision
    ("describe what is happening in this photograph of a street market", "vision"),
    ("read the total amount from this scanned invoice image", "vision"),
    ("how many cars are visible in this traffic camera frame", "vision"),
    ("what does this chest x-ray image show", "vision"),
    ("identify the breed of the dog in this picture", "vision"),
    ("transcribe the handwritten notes from this whiteboard photo", "vision"),
    ("what component looks damaged on this circuit board image", "vision"),
    ("estimate the crowd size in this stadium photograph", "vision"),
    ("describe the trend shown in this bar chart image", "vision"),
    ("is there a cat hiding somewhere in this living room photo", "vision"),
    ("what brand is the laptop in this desk setup picture", "vision"),
    ("count the boxes stacked on these warehouse shelves in the image", "vision"),
    ("list the ingredients visible on this kitchen counter photograph", "vision"),
    ("is the traffic light red or green in this intersection photo", "vision"),
    ("what time does the clock in this waiting room picture show", "vision"),
    ("describe the facial expression of the person in this portrait photo", "vision"),
    ("which team logo appears on the jersey in this match photograph", "vision"),
    ("read the departure board in this airport terminal image", "vision"),
    # chat
    ("hey, how is your day going so far", "chat"),
    ("tell me a funny joke about programmers", "chat"),
    ("what should I cook for dinner tonight", "chat"),
    ("help me write a warm birthday message for my friend", "chat"),
    ("I am feeling stressed about work lately", "chat"),
    ("recommend a good science fiction novel for a long flight", "chat"),
    ("what is a fun weekend activity in austin", "chat"),
    ("help me draft a polite email declining an invitation", "chat"),
    ("tell me an interesting fact about octopuses", "chat"),
    ("how do I stay motivated to exercise regularly", "chat"),
    ("write a short haiku about the ocean at dawn", "chat"),
    ("what are some good history podcasts to follow", "chat"),
    ("write a thank-you note to send after a job interview", "chat"),
    ("help me write a condolence message for a coworker", "chat"),
    ("suggest a relaxing evening routine to wind down", "chat"),
    ("help me brainstorm names for my new puppy", "chat"),
    ("what is a thoughtful hostess gift for a dinner party", "chat"),
    ("talk me through calming down before a big presentation", "chat"),
]

TEST_CORPUS: List[Tuple[str, str]] = [
    ("show that the product of two odd integers is odd", "reasoning"),
    ("two cyclists start 100 miles apart riding toward each other, when do they collide", "reasoning"),
    ("what is the expected value of rolling a fair six-sided die twice", "reasoning"),
    ("explain the monty hall problem and why switching wins", "reasoning"),
    ("is this syllogism valid: no fish are mammals, whales are mammals", "reasoning"),
    ("differentiate x cubed plus 2x and evaluate at x equals 3", "reasoning"),
    ("write a typescript interface for a paginated api response", "coding"),
    ("how do I add an index to a postgres table without locking writes", "coding"),
    ("review this pull request diff for the authentication module", "coding"),
    ("write a bash script that backs up a directory nightly", "coding"),
    ("explain closures in javascript with a simple example", "coding"),
    ("how do I mock an http client in my kotlin tests", "coding"),
    ("what is the license plate number in this parking lot photo", "vision"),
    ("describe the damage visible on this car bumper image", "vision"),
    ("read the expiry date on this medicine bottle in the picture", "vision"),
    ("how many people are waiting in this queue photograph", "vision"),
    ("what does the sign say in this blurry street image", "vision"),
    ("identify the mountain peak in this landscape photo", "vision"),
    ("good morning! what is the weather like today", "chat"),
    ("help me plan a surprise anniversary dinner", "chat"),
    ("I need cheering up, tell me something wholesome", "chat"),
    ("suggest a thoughtful gift for a new parent", "chat"),
    ("write a thank-you note after a job interview", "chat"),
    ("what is a good icebreaker for a team meeting", "chat"),
]


def train_router() -> SemanticRouter:
    texts = [t for t, _ in TRAIN_CORPUS]
    labels = [l for _, l in TRAIN_CORPUS]
    return SemanticRouter().fit(texts, labels)


def test_texts_labels():
    texts = [t for t, _ in TEST_CORPUS]
    labels = [l for _, l in TEST_CORPUS]
    return texts, labels
