"""GPU model loading (BGE-M3 bi-encoder + bge-reranker-v2-m3 cross-encoder), fp16 on CUDA."""
from __future__ import annotations

from functools import lru_cache

from ..config import CFG


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_kwargs():
    import torch
    import transformers
    if device() == "cpu":
        return {}
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"  # renamed in v5
    return {key: torch.float16}


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(CFG.embed_model, device=device(), model_kwargs=_dtype_kwargs())
    m.max_seq_length = 1024
    return m


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CFG.rerank_model, device=device(), max_length=1024, model_kwargs=_dtype_kwargs())


def embed(texts: list[str], batch_size: int = 32, show_progress: bool = False):
    import numpy as np
    return np.asarray(embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                        show_progress_bar=show_progress), dtype="float32")


def rerank_scores(query: str, texts: list[str], batch_size: int = 16):
    import numpy as np
    if not texts:
        return np.zeros(0, dtype="float32")
    return np.asarray(reranker().predict([(query, t) for t in texts], batch_size=batch_size,
                                         show_progress_bar=False), dtype="float32")


def free_gpu():
    import gc
    import torch
    embedder.cache_clear(); reranker.cache_clear(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
