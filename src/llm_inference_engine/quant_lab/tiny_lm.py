"""Minimal character-level transformer LM in numpy (CPU), with Adam training.

Architecture: token + learned positional embeddings, ``n_layers`` decoder
blocks (pre-norm attention with causal masking + pre-norm MLP with GELU),
final layer norm, tied-less linear head. All math is float32 numpy; the
forward/backward pass is dtype-agnostic so gradient checks can run in float64.

Linear weights use the ``(out_features, in_features)`` layout
(``y = x @ W.T + b``), matching :mod:`quant_lab.schemes`.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    seq_len: int = 64

    @property
    def d_head(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


LINEAR_SUFFIXES = ("wq", "wk", "wv", "wo", "w1", "w2", "head_w")


def is_linear_weight(name):
    """True for matmul weight matrices (the tensors quantization targets)."""
    return name.split(".")[-1] in LINEAR_SUFFIXES


def init_params(rng, cfg):
    """Initialize parameters (dict of float32 arrays)."""
    C, F, V, T = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.seq_len
    p = {}
    p["tok_emb"] = rng.normal(0, 0.02, (V, C)).astype(np.float32)
    p["pos_emb"] = rng.normal(0, 0.02, (T, C)).astype(np.float32)
    out_scale = 1.0 / math.sqrt(2.0 * cfg.n_layers)  # residual-branch scaling
    for i in range(cfg.n_layers):
        k = f"l{i}."
        p[k + "ln1_g"] = np.ones(C, dtype=np.float32)
        p[k + "ln1_b"] = np.zeros(C, dtype=np.float32)
        for nm in ("wq", "wk", "wv"):
            p[k + nm] = rng.normal(0, 0.02, (C, C)).astype(np.float32)
        p[k + "wo"] = (rng.normal(0, 0.02, (C, C)) * out_scale).astype(np.float32)
        p[k + "ln2_g"] = np.ones(C, dtype=np.float32)
        p[k + "ln2_b"] = np.zeros(C, dtype=np.float32)
        p[k + "w1"] = rng.normal(0, 0.02, (F, C)).astype(np.float32)
        p[k + "b1"] = np.zeros(F, dtype=np.float32)
        p[k + "w2"] = (rng.normal(0, 0.02, (C, F)) * out_scale).astype(np.float32)
        p[k + "b2"] = np.zeros(C, dtype=np.float32)
    p["lnf_g"] = np.ones(C, dtype=np.float32)
    p["lnf_b"] = np.zeros(C, dtype=np.float32)
    p["head_w"] = rng.normal(0, 0.02, (V, C)).astype(np.float32)
    p["head_b"] = np.zeros(V, dtype=np.float32)
    return p


def count_params(params):
    return int(sum(v.size for v in params.values()))


# ----------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------

def _ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(axis=-1, keepdims=True)
    xhat = xc / np.sqrt(var + eps)
    return g * xhat + b, (x, xhat, mu, var, g, eps)


def _ln_backward(dy, cache):
    x, xhat, mu, var, g, eps = cache
    C = x.shape[-1]
    dxhat = dy * g
    inv = 1.0 / np.sqrt(var + eps)
    inv3 = inv * inv * inv
    dvar = np.sum(dxhat * (x - mu) * -0.5 * inv3, axis=-1, keepdims=True)
    dmu = (np.sum(dxhat * -inv, axis=-1, keepdims=True)
           + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True))
    dx = dxhat * inv + dvar * 2.0 * (x - mu) / C + dmu / C
    dg = np.sum(dy * xhat, axis=(0, 1))
    db = np.sum(dy, axis=(0, 1))
    return dx, dg, db


_GELU_C = math.sqrt(2.0 / math.pi)


def _gelu(x):
    x2 = x * x
    a = _GELU_C * (x + 0.044715 * x2 * x)
    t = np.tanh(a)
    return 0.5 * x * (1.0 + t), (x, t)


def _gelu_backward(dy, cache):
    x, t = cache
    sech2 = 1.0 - t * t
    da = _GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * sech2 * da)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def forward(params, cfg, ids):
    """Forward pass. Returns (logits (B,T,V), cache)."""
    B, T = ids.shape
    if T > cfg.seq_len:
        raise ValueError(f"sequence length {T} > configured {cfg.seq_len}")
    C, H, Dh = cfg.d_model, cfg.n_heads, cfg.d_head
    cache = {"ids_shape": (B, T), "ids": ids}
    h = params["tok_emb"][ids] + params["pos_emb"][:T][None, :, :]
    for i in range(cfg.n_layers):
        k = f"l{i}."
        hn, ln_cache = _ln_forward(h, params[k + "ln1_g"], params[k + "ln1_b"])
        q = (hn @ params[k + "wq"].T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        kh = (hn @ params[k + "wk"].T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        vh = (hn @ params[k + "wv"].T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        scores = (q @ kh.transpose(0, 1, 3, 2)) / math.sqrt(Dh)
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal[None, None], -1e4, scores)
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        prob = e / e.sum(axis=-1, keepdims=True)
        att = (prob @ vh).transpose(0, 2, 1, 3).reshape(B, T, C)
        o = att @ params[k + "wo"].T
        h_res = h
        h = h + o
        h_post_attn = h
        hn2, ln2_cache = _ln_forward(h, params[k + "ln2_g"], params[k + "ln2_b"])
        z1 = hn2 @ params[k + "w1"].T + params[k + "b1"]
        f, gelu_cache = _gelu(z1)
        m_out = f @ params[k + "w2"].T + params[k + "b2"]
        h = h + m_out
        cache[i] = dict(hn=hn, ln_cache=ln_cache, qh=q, kh=kh, vh=vh,
                        prob=prob, att=att, h_res=h_res, h_post_attn=h_post_attn,
                        hn2=hn2, ln2_cache=ln2_cache, f=f,
                        gelu_cache=gelu_cache)
    h_pre = h
    hf, lnf_cache = _ln_forward(h, params["lnf_g"], params["lnf_b"])
    logits = hf @ params["head_w"].T + params["head_b"]
    cache["h_pre"] = h_pre
    cache["hf"] = hf
    cache["lnf_cache"] = lnf_cache
    return logits, cache


def ce_loss_and_grad(logits, targets):
    """Mean cross-entropy loss and d(loss)/d(logits)."""
    B, T = targets.shape
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(axis=-1, keepdims=True)
    nll = -np.log(probs[np.arange(B)[:, None], np.arange(T), targets] + 1e-12)
    loss = float(nll.mean())
    dlogits = probs
    dlogits[np.arange(B)[:, None], np.arange(T), targets] -= 1.0
    dlogits = dlogits / (B * T)
    return loss, dlogits


def backward(params, cfg, cache, dlogits):
    """Backprop. Returns dict of gradients matching ``params`` keys."""
    B, T = cache["ids_shape"]
    C, H, Dh = cfg.d_model, cfg.n_heads, cfg.d_head
    grads = {}
    hf = cache["hf"]
    grads["head_w"] = np.einsum("btv,btc->vc", dlogits, hf)
    grads["head_b"] = dlogits.sum(axis=(0, 1))
    dhf = dlogits @ params["head_w"]
    dh, g, b_ = _ln_backward(dhf, cache["lnf_cache"])
    grads["lnf_g"], grads["lnf_b"] = g, b_
    for i in reversed(range(cfg.n_layers)):
        c = cache[i]
        k = f"l{i}."
        # MLP block: h_out = h_post_attn + m_out
        dm = dh
        grads[k + "w2"] = np.einsum("btc,btf->cf", dm, c["f"])
        grads[k + "b2"] = dm.sum(axis=(0, 1))
        df = dm @ params[k + "w2"]
        dz1 = _gelu_backward(df, c["gelu_cache"])
        grads[k + "w1"] = np.einsum("btf,btc->fc", dz1, c["hn2"])
        grads[k + "b1"] = dz1.sum(axis=(0, 1))
        dhn2 = dz1 @ params[k + "w1"]
        d_hpost, g2, b2_ = _ln_backward(dhn2, c["ln2_cache"])
        grads[k + "ln2_g"], grads[k + "ln2_b"] = g2, b2_
        d_hpost = d_hpost + dh  # residual pass-through
        # Attention block: h_post_attn = h_res + o
        do = d_hpost
        grads[k + "wo"] = np.einsum("btc,btd->cd", do, c["att"])
        datt = (do @ params[k + "wo"]).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        vh, kh, prob = c["vh"], c["kh"], c["prob"]
        dprob = np.einsum("bhtd,bhsd->bhts", datt, vh)
        dvh = np.einsum("bhts,bhtd->bhsd", prob, datt)
        dscores = prob * (dprob - (dprob * prob).sum(axis=-1, keepdims=True))
        dscores = dscores / math.sqrt(Dh)
        dqh = np.einsum("bhts,bhsd->bhtd", dscores, kh)
        dkh = np.einsum("bhts,bhtd->bhsd", dscores, c["qh"])
        dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, C)
        hn = c["hn"]
        grads[k + "wq"] = np.einsum("btc,btd->cd", dq, hn)
        grads[k + "wk"] = np.einsum("btc,btd->cd", dk, hn)
        grads[k + "wv"] = np.einsum("btc,btd->cd", dv, hn)
        dhn = dq @ params[k + "wq"] + dk @ params[k + "wk"] + dv @ params[k + "wv"]
        dhn_in, g1, b1_ = _ln_backward(dhn, c["ln_cache"])
        grads[k + "ln1_g"], grads[k + "ln1_b"] = g1, b1_
        dh = d_hpost + dhn_in  # attention residual pass-through
    # Embeddings.
    ids = cache["ids"]
    d_tok = np.zeros_like(params["tok_emb"])
    np.add.at(d_tok, ids, dh)
    grads["tok_emb"] = d_tok
    d_pos = np.zeros_like(params["pos_emb"])
    d_pos[:T] = dh.sum(axis=0)
    grads["pos_emb"] = d_pos
    return grads


class Adam:
    """Adam optimizer over a parameter dict."""

    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = {kk: np.zeros_like(vv) for kk, vv in params.items()}
        self.v = {kk: np.zeros_like(vv) for kk, vv in params.items()}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for kk in params:
            g = grads[kk]
            self.m[kk] = self.beta1 * self.m[kk] + (1.0 - self.beta1) * g
            self.v[kk] = self.beta2 * self.v[kk] + (1.0 - self.beta2) * g * g
            mh = self.m[kk] / bc1
            vh = self.v[kk] / bc2
            params[kk] = params[kk] - self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------------------
# data: small synthetic repetitive corpus
# ----------------------------------------------------------------------------

_BASE_SENTENCES = [
    "the quick brown fox jumps over the lazy dog near the river",
    "a slow red boat drifts down the wide river at dawn",
    "the old mill stands quiet beside the cold stream",
    "bright birds sing in the tall green trees each spring",
    "the farmer walks the long field road every morning",
    "soft rain falls on the dark roof through the night",
    "the small child reads a worn book by warm fire light",
    "wild deer run swift across the open frosty meadow",
    "the baker sells fresh bread from a small wood cart",
    "a thin grey mist hangs low over the still lake",
    "the shepherd counts his restless sheep at dusk",
    "cold winds bend the bare trees on the high hill",
    "the miner digs deep in the dark narrow shaft",
    "a brave knight rides out past the old stone wall",
]

_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz .,!?;:'\"\n()-")


def build_corpus(seed=41, target_chars=12000):
    """Build a deterministic repetitive character corpus (~12 KB).

    Base sentences are resampled with ~15% word substitution, so the text is
    highly repetitive (learnable by a tiny model) but not trivially periodic.
    """
    rng = np.random.default_rng(seed)
    words = sorted({w for s in _BASE_SENTENCES for w in s.split()})
    chunks = []
    total = 0
    while total < target_chars:
        n_sent = int(rng.integers(3, 7))
        sents = []
        for _ in range(n_sent):
            s = _BASE_SENTENCES[int(rng.integers(len(_BASE_SENTENCES)))]
            ws = s.split()
            ws = [words[int(rng.integers(len(words)))] if rng.random() < 0.15 else w
                  for w in ws]
            sents.append(" ".join(ws))
        chunk = ". ".join(sents) + ".\n"
        chunks.append(chunk)
        total += len(chunk)
    text = "".join(chunks).lower()
    text = "".join(ch if ch in _ALLOWED_CHARS else " " for ch in text)
    return text


def encode_corpus(text):
    """Return (ids, char_to_idx, idx_to_char); asserts vocab fits in 64."""
    vocab = sorted(set(text))
    if len(vocab) > 64:
        raise ValueError(f"vocab size {len(vocab)} exceeds 64-char budget")
    c2i = {ch: i for i, ch in enumerate(vocab)}
    ids = np.array([c2i[ch] for ch in text], dtype=np.int64)
    return ids, c2i, vocab


# ----------------------------------------------------------------------------
# training / evaluation
# ----------------------------------------------------------------------------

def _batch_windows(rng, ids, T, batch_size, max_start):
    starts = rng.integers(0, max_start + 1, size=batch_size)
    xb = np.stack([ids[s:s + T] for s in starts])
    yb = np.stack([ids[s + 1:s + T + 1] for s in starts])
    return xb, yb


def train(cfg, train_ids, n_steps=2000, batch_size=16, lr=3e-3, seed=41,
          log_every=500, grad_clip=1.0):
    """Train with Adam on next-character prediction. Returns (params, history)."""
    rng = np.random.default_rng(seed)
    params = init_params(rng, cfg)
    opt = Adam(params, lr=lr)
    T = cfg.seq_len
    max_start = len(train_ids) - T - 1
    history = []
    for step in range(1, n_steps + 1):
        xb, yb = _batch_windows(rng, train_ids, T, batch_size, max_start)
        logits, cache = forward(params, cfg, xb)
        loss, dlogits = ce_loss_and_grad(logits, yb)
        grads = backward(params, cfg, cache, dlogits)
        if grad_clip:
            norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            if norm > grad_clip:
                grads = {kk: g * (grad_clip / norm) for kk, g in grads.items()}
        opt.step(params, grads)
        history.append(loss)
        if step % log_every == 0 or step == 1:
            print(f"[train] step {step}/{n_steps} loss={loss:.4f}", flush=True)
    return params, history


def evaluate_perplexity(params, cfg, eval_ids, batch_size=64):
    """Mean-per-token perplexity over non-overlapping windows of eval_ids."""
    T = cfg.seq_len
    n = (len(eval_ids) - 1) // T
    total_nll = 0.0
    total_tok = 0
    for b in range(0, n, batch_size):
        nb = min(batch_size, n - b)
        xb = np.stack([eval_ids[(b + i) * T:(b + i) * T + T] for i in range(nb)])
        yb = np.stack([eval_ids[(b + i) * T + 1:(b + i) * T + T + 1]
                       for i in range(nb)])
        logits, _ = forward(params, cfg, xb)
        B = xb.shape[0]
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        probs = e / e.sum(axis=-1, keepdims=True)
        nll = -np.log(probs[np.arange(B)[:, None], np.arange(T), yb] + 1e-12)
        total_nll += float(nll.sum())
        total_tok += int(nll.size)
    return float(np.exp(total_nll / total_tok))
