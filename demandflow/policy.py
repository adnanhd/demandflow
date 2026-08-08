from __future__ import annotations

from typing import Any, Callable


class CachePolicy:
    """How aggressively demand() trusts cached results.

    Modes:
        ACCEPT -- use cached if available, compute only on miss.
        CHECK  -- always recompute, compare against cached, warn on divergence.
        FRESH  -- recompute the demanded target, allow upstream cache hits.
        FORCE  -- recompute everything, ignore all caches.

    Parameters beyond mode:
        tolerance   -- absolute tolerance for numeric divergence checks (CHECK mode).
        comparator  -- custom (cached, fresh) -> bool equality function.
    """

    ACCEPT = "accept"
    CHECK = "check"
    FRESH = "fresh"
    FORCE = "force"

    def __init__(
        self,
        mode: str = "accept",
        tolerance: float = 0.0,
        comparator: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        self.mode = mode
        self.tolerance = tolerance
        self.comparator = comparator

    def diverged(self, cached: Any, fresh: Any) -> bool:
        if self.comparator is not None:
            return not self.comparator(cached, fresh)
        if isinstance(cached, (int, float)) and isinstance(fresh, (int, float)):
            return abs(cached - fresh) > self.tolerance
        return cached != fresh

    def __repr__(self) -> str:
        parts = ["CachePolicy(%r" % self.mode]
        if self.tolerance:
            parts.append(", tolerance=%g" % self.tolerance)
        if self.comparator is not None:
            parts.append(", comparator=...")
        return "".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.mode == other
        if not isinstance(other, CachePolicy):
            return NotImplemented
        return (self.mode, self.tolerance) == (other.mode, other.tolerance)

    def __hash__(self) -> int:
        return hash((self.mode, self.tolerance))


class ResourcePolicy:
    """Execution resource declaration for a node.

    A resource has a name (arbitrary tag), a concurrency mode, and an
    optional capacity.  Nodes sharing the same resource name are
    scheduled together according to the mode.

    EXCLUSIVE -- only one node on this resource at a time (sequential).
    SHARED    -- up to ``capacity`` nodes run concurrently (thread pool).
    """

    EXCLUSIVE = "exclusive"
    SHARED = "shared"

    def __init__(
        self,
        name: str = "default",
        mode: str = "exclusive",
        capacity: int = 1,
    ) -> None:
        self.name = name
        self.mode = mode
        self.capacity = capacity

    def __repr__(self) -> str:
        return "ResourcePolicy(%r, mode=%r, capacity=%d)" % (
            self.name, self.mode, self.capacity,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourcePolicy):
            return NotImplemented
        return (self.name, self.mode, self.capacity) == (
            other.name, other.mode, other.capacity,
        )

    def __hash__(self) -> int:
        return hash((self.name, self.mode, self.capacity))
