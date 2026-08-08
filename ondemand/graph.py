import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ondemand.hasher import content_hash, node_key
from ondemand.memo import MemoStore
from ondemand.policy import CachePolicy

logger = logging.getLogger(__name__)


class CycleError(Exception):
    pass


@dataclass
class ComputeNode:
    """A named pure function in the dependency graph."""

    fn: Callable
    name: str
    params: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.params:
            self.params = list(inspect.signature(self.fn).parameters.keys())

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class Graph:
    """Demand-driven computation graph with content-addressed memoization."""

    def __init__(self, memo: MemoStore | None = None) -> None:
        self._nodes: dict[str, ComputeNode] = {}
        self._memo = memo or MemoStore()

    @property
    def memo(self) -> MemoStore:
        return self._memo

    @property
    def nodes(self) -> dict[str, ComputeNode]:
        return dict(self._nodes)

    def node(
        self, fn: Callable | None = None, *, name: str | None = None
    ) -> Callable:
        """Register a function as a computation node."""
        def decorator(f: Callable) -> Callable:
            n = ComputeNode(fn=f, name=name or f.__name__)
            self._nodes[n.name] = n
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def demand(
        self,
        target: str,
        policy: CachePolicy = CachePolicy.ACCEPT,
        **kwargs: Any,
    ) -> Any:
        """Demand a node's output, resolving dependencies lazily."""
        if target not in self._nodes:
            raise KeyError("unknown node: %r" % target)

        order = self._topo_sort(target)
        results: dict[str, Any] = {}
        hashes: dict[str, str] = {}

        for node_name in order:
            node = self._nodes[node_name]
            node_kwargs, input_hashes = self._gather_inputs(
                node, results, hashes, kwargs
            )
            key = node_key(node_name, input_hashes)
            hashes[node_name] = key

            is_target = node_name == target
            hit, cached = self._memo.get(key)
            compute = self._should_compute(policy, is_target, hit)

            if compute:
                fresh = node(**node_kwargs)
                self._memo.put(key, fresh)
                if policy == CachePolicy.CHECK and hit and fresh != cached:
                    logger.warning(
                        "CHECK divergence on %r: cached=%r fresh=%r",
                        node_name,
                        cached,
                        fresh,
                    )
                results[node_name] = fresh
            else:
                results[node_name] = cached

        return results[target]

    def _gather_inputs(
        self,
        node: ComputeNode,
        results: dict[str, Any],
        hashes: dict[str, str],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        node_kwargs: dict[str, Any] = {}
        input_hashes: dict[str, str] = {}
        for param in node.params:
            if param in results:
                node_kwargs[param] = results[param]
                input_hashes[param] = hashes[param]
            elif param in kwargs:
                node_kwargs[param] = kwargs[param]
                input_hashes[param] = content_hash(kwargs[param])
            else:
                raise TypeError(
                    "node %r requires input %r: not a registered node "
                    "and not provided in kwargs" % (node.name, param)
                )
        return node_kwargs, input_hashes

    def _topo_sort(self, target: str) -> list[str]:
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name not in self._nodes:
                return
            if name in in_stack:
                raise CycleError("cycle detected involving %r" % name)
            if name in visited:
                return
            in_stack.add(name)
            visited.add(name)
            for param in self._nodes[name].params:
                visit(param)
            in_stack.discard(name)
            order.append(name)

        visit(target)
        return order

    @staticmethod
    def _should_compute(
        policy: CachePolicy, is_target: bool, has_cached: bool
    ) -> bool:
        if policy == CachePolicy.ACCEPT:
            return not has_cached
        if policy == CachePolicy.FRESH:
            return is_target or not has_cached
        return True
