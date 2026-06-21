"""Redis-backed exact and semantic cache for RAG answers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

try:
    import redis
except ImportError:  # pragma: no cover - Redis is an optional runtime service
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    """Serializable answer and retrieval metadata returned by the cache."""

    answer: str
    sources: list[dict[str, Any]]
    grounding_score: float
    grounding_confidence: str = "LOW"
    chunks_total: int = 0
    chunks_dropped: int = 0
    warning: str | None = None
    created_at: float = 0.0
    cache_type: str = "exact"

    @property
    def age_seconds(self) -> int:
        return max(0, int(time.time() - self.created_at))


class SemanticCache:
    """Two-level cache using exact hashes and cosine similarity in Redis."""

    EXACT_PREFIX = "rag:exact:"
    SEMANTIC_PREFIX = "rag:semantic:"
    SEMANTIC_INDEX_KEY = "rag:semantic:index"

    def __init__(
        self,
        redis_url: str,
        embeddings_fn: Callable[[str], list[float]] | Any,
        *,
        ttl_seconds: int = 86_400,
        similarity_threshold: float = 0.92,
        client: Any | None = None,
    ) -> None:
        self.embeddings_fn = embeddings_fn
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.client = client
        if self.client is None and redis_url and redis is not None:
            self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def question_hash(question: str) -> str:
        return hashlib.sha256(question.encode("utf-8")).hexdigest()

    def get(self, question: str) -> CachedAnswer | None:
        """Return an exact or semantically similar answer; fail open on errors."""
        if self.client is None:
            return None
        digest = self.question_hash(question)
        try:
            exact = self.client.get(f"{self.EXACT_PREFIX}{digest}")
            if exact:
                return self._decode_answer(exact, "exact")

            query_embedding = self._embed(question)
            best_score = self.similarity_threshold
            best_answer: CachedAnswer | None = None
            best_payload: dict[str, Any] | None = None
            for candidate_hash in list(self.client.smembers(self.SEMANTIC_INDEX_KEY)):
                raw = self.client.get(f"{self.SEMANTIC_PREFIX}{candidate_hash}")
                if not raw:
                    self.client.srem(self.SEMANTIC_INDEX_KEY, candidate_hash)
                    continue
                candidate = json.loads(raw)
                score = self._cosine_similarity(query_embedding, candidate["embedding"])
                if score > best_score:
                    best_score = score
                    best_payload = candidate["answer"]
                    best_answer = self._answer_from_dict(best_payload, "semantic")
            if best_answer is not None and best_payload is not None:
                # Promote the paraphrase without extending the original answer's TTL.
                remaining_ttl = max(
                    1, self.ttl_seconds - int(time.time() - best_answer.created_at)
                )
                self.client.setex(
                    f"{self.EXACT_PREFIX}{digest}",
                    remaining_ttl,
                    json.dumps(best_payload, default=str),
                )
                return best_answer
        except Exception as exc:
            logger.warning("RAG cache lookup failed; continuing without cache: %s", exc)
        return None

    def set(self, question: str, answer: CachedAnswer) -> None:
        """Store an answer in both exact and semantic cache levels."""
        if self.client is None:
            return
        digest = self.question_hash(question)
        answer.created_at = answer.created_at or time.time()
        payload = asdict(answer)
        payload["cache_type"] = "exact"
        try:
            embedding = self._embed(question)
            pipe = self.client.pipeline()
            pipe.setex(
                f"{self.EXACT_PREFIX}{digest}",
                self.ttl_seconds,
                json.dumps(payload, default=str),
            )
            pipe.setex(
                f"{self.SEMANTIC_PREFIX}{digest}",
                self.ttl_seconds,
                json.dumps({"embedding": embedding, "answer": payload}, default=str),
            )
            pipe.sadd(self.SEMANTIC_INDEX_KEY, digest)
            pipe.execute()
        except Exception as exc:
            logger.warning("RAG cache write failed; answer was not cached: %s", exc)

    def invalidate(self, question_hash: str) -> int:
        """Invalidate exact and semantic entries for one SHA-256 question hash."""
        if self.client is None:
            return 0
        try:
            pipe = self.client.pipeline()
            pipe.delete(f"{self.EXACT_PREFIX}{question_hash}")
            pipe.delete(f"{self.SEMANTIC_PREFIX}{question_hash}")
            pipe.srem(self.SEMANTIC_INDEX_KEY, question_hash)
            results = pipe.execute()
            return int(results[0]) + int(results[1])
        except Exception as exc:
            logger.warning("RAG cache invalidation failed: %s", exc)
            return 0

    def invalidate_all(self) -> int:
        """Invalidate all RAG cache entries without touching unrelated Redis keys."""
        if self.client is None:
            return 0
        try:
            hashes = list(self.client.smembers(self.SEMANTIC_INDEX_KEY))
            keys = [f"{self.SEMANTIC_PREFIX}{value}" for value in hashes]
            # Exact keys include aliases created by semantic hits, so scan this namespace.
            keys.extend(self.client.scan_iter(match=f"{self.EXACT_PREFIX}*"))
            deleted = int(self.client.delete(*keys)) if keys else 0
            deleted += int(self.client.delete(self.SEMANTIC_INDEX_KEY))
            return deleted
        except Exception as exc:
            logger.warning("RAG cache invalidation failed: %s", exc)
            return 0

    def _embed(self, question: str) -> list[float]:
        fn = self.embeddings_fn
        if callable(fn):
            value = fn(question)
        elif hasattr(fn, "embed_query"):
            value = fn.embed_query(question)
        else:
            value = fn.embed_documents([question])[0]
        return [float(item) for item in value]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
            sum(b * b for b in right)
        )
        return dot / denominator if denominator else 0.0

    def _decode_answer(self, raw: str, cache_type: str) -> CachedAnswer:
        return self._answer_from_dict(json.loads(raw), cache_type)

    @staticmethod
    def _answer_from_dict(payload: dict[str, Any], cache_type: str) -> CachedAnswer:
        fields = CachedAnswer.__dataclass_fields__
        clean = {key: value for key, value in payload.items() if key in fields}
        clean["cache_type"] = cache_type
        return CachedAnswer(**clean)
