import hashlib
import json
from typing import Any


def content_hash(value: Any) -> str:
    """Deterministic hash of a leaf value.

    Tries canonical JSON (sorted keys) first; falls back to repr()
    for non-serializable objects.  Objects that define
    ``__content_hash__() -> str`` get that called directly.
    """
    if hasattr(value, "__content_hash__"):
        return value.__content_hash__()
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        canonical = repr(value)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def node_key(name: str, input_hashes: dict[str, str]) -> str:
    """Merkle-style cache key for a node given its resolved input hashes."""
    parts = [name]
    for k in sorted(input_hashes):
        parts.append("%s=%s" % (k, input_hashes[k]))
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
