import enum


class CachePolicy(str, enum.Enum):
    """How aggressively demand() trusts cached results.

    ACCEPT  -- use cached if available, compute only on miss.
    CHECK   -- always recompute, compare against cached, warn on divergence.
    FRESH   -- recompute the demanded target, allow upstream cache hits.
    FORCE   -- recompute everything, ignore all caches.
    """

    ACCEPT = "accept"
    CHECK = "check"
    FRESH = "fresh"
    FORCE = "force"
