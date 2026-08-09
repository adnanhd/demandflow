"""Lazy, demand-driven pub/sub dataflow -- the unifying fabric kernel.

The one mechanism behind BOTH the compute graph (data -> model -> loss -> ...)
and the observability channels (``run.metrics.<Name>`` / reducers / handlers):
a set of TOPICS with producers and subscribers, executed **by demand**.

Three properties, together:

* **Barrier join** -- a node consumes several input topics and fires only once
  ALL are fresh for the current round (``D`` waits for ``B`` *and* ``C``).
* **Demand-driven laziness** -- a node runs only if its output is DEMANDED, and a
  source is pulled only if its output is demanded. "Demanded" propagates BACKWARD
  from the sinks: a topic is demanded iff it has a subscriber, or it feeds a node
  whose output is demanded. So if nothing consumes ``D``, the whole ``A->B->D`` /
  ``C->D`` chain -- including pulling the source ``A`` -- never runs.
* **Dynamic** -- demand is read live from the current subscribers, so
  ``subscribe`` switches a dormant chain on (the next flow() starts pulling its
  sources) and ``unsubscribe`` switches it off. Nothing is privileged: the
  training spine runs only because something (torchestrator, streaming the test
  result) subscribes to its sink.

Extended with the Lingua Franca pattern (ports + triggers):

* **Triggers** -- a node can declare event triggers in addition to data inputs.
  The node fires only when all data inputs are fresh AND at least one trigger
  has fired since the last execution.  Nodes without triggers fire on data
  readiness alone (original behavior).  Nodes with ONLY triggers (no data
  inputs) fire once per trigger.

* **Memo** -- a node can declare a memo function that computes a cache key from
  its inputs.  If the key is already cached, the node skips compute and emits
  the cached output.  This is the content-addressed cache layer (chain
  checkpoints, eval memoization).

* **Recurrence** -- a loop node whose output feeds back as its own input for N
  iterations.  ``recurrence("train", inputs, outs, fn, iterations=150)``
  replaces ``for epoch in range(150)`` in the runner.  Each iteration can be
  memo-skipped independently.

This module is the pure kernel -- no torch, no threads -- so the scheduling model
can be tested in isolation.
"""

from __future__ import annotations

from abc import ABC

import inspect
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Set, get_type_hints

from demandflow.annotations import Demand as DemandMarker, Flow as FlowMarker
from demandflow.hasher import content_hash, node_key
from demandflow.memo import MemoStore
from demandflow.policy import CachePolicy, ResourcePolicy
from demandflow.qos import QoS

logger = logging.getLogger(__name__)

__all__ = [
    "DemandGraph",
    "Node",
    "NodeInterceptor",
    "Source",
    "DemandHandler",
    "Recurrence",
    "map_output",
]


class NodeInterceptor(ABC):
    """Hook into node/source execution for observability.

    ``before_node(name, inputs)`` is called before a node computes.
    ``after_node(name, inputs, duration, error)`` is called after.
    Same pattern for sources via ``before_source``/``after_source``.
    """

    def before_node(self, name: str, inputs: Dict[str, Any]) -> None:
        pass

    def after_node(
        self, name: str, inputs: Dict[str, Any], duration: float, error: Optional[Exception]
    ) -> None:
        pass

    def before_source(self, name: str) -> None:
        pass

    def after_source(
        self, name: str, duration: float, error: Optional[Exception]
    ) -> None:
        pass


def _extract_demands(fn: Callable) -> Dict[str, DemandMarker]:
    """Extract {param_name: DemandMarker} from Annotated hints."""
    globalns = getattr(fn, "__globals__", {})
    try:
        hints = get_type_hints(fn, globalns=globalns, include_extras=True)
    except Exception:
        hints = {}
    result: Dict[str, DemandMarker] = {}
    for param, hint in hints.items():
        if param == "return":
            continue
        args = getattr(hint, "__metadata__", ())
        for meta in args:
            if isinstance(meta, DemandMarker):
                result[param] = meta
                break
    return result


class Relation:
    """A parametric node template that generates demand chains.

    Given a key (e.g. a config object or epoch number), resolves
    dependencies recursively via ``Demand(resolver)`` annotations,
    hitting base cases or memo cache along the way.
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        deps: Dict[str, DemandMarker],
        memo: MemoStore,
    ) -> None:
        self.name = name
        self.fn = fn
        self.deps = deps
        self._bases: Dict[Any, Any] = {}
        self._memo = memo
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        self._key_param = params[0] if params else None

    def base(self, key: Any, value: Any) -> None:
        """Register a base case: when ``key`` is demanded, return ``value``."""
        self._bases[key] = value

    def resolve(self, key: Any, policy: CachePolicy | None = None) -> Any:
        """Recursively resolve the relation for ``key``."""
        if policy is None:
            policy = CachePolicy()
        cache_key = node_key(self.name, {self.name: content_hash(key)})

        if key in self._bases:
            return self._bases[key]

        hit, cached = self._memo.get(cache_key)
        if hit and policy.mode == CachePolicy.ACCEPT:
            return cached

        kwargs = {}
        for param, demand in self.deps.items():
            dep_key = demand.resolve(key)
            if dep_key in self._bases:
                kwargs[param] = self._bases[dep_key]
            else:
                kwargs[param] = self.resolve(dep_key, policy)

        if self._key_param is not None:
            kwargs[self._key_param] = key
        result = self.fn(**kwargs)
        self._memo.put(cache_key, result)
        return result


def map_output(outs: Sequence[str], output: Any) -> Dict[str, Any]:
    """Map a callback's return onto its out topics (the ORCHESTRA convention, so a
    node/source ported from ``engine.dataflow`` behaves identically): a tuple/list
    zips positionally, a dict maps by key, a scalar lands on the first out. A node
    with NO outs is a terminal sink -- it produces nothing, only a side effect."""
    if not outs:
        return {}
    if isinstance(output, (tuple, list)):
        return dict(zip(outs, output))
    if isinstance(output, dict):
        return {key: output[key] for key in outs if key in output}
    return {outs[0]: output}


def _as_outs(out: "str | None", outs: "Sequence[str] | None") -> List[str]:
    """Normalize the single-``out`` / multi-``outs`` spelling to a list."""
    if outs is not None:
        return list(outs)
    return [out] if out is not None else []


@dataclass
class Source:
    """A demand-pulled emission source: ``pull()`` yields the next value (``next()``
    semantics, ``StopIteration`` at end-of-stream), mapped onto ``outs`` via
    :func:`map_output`. Pulled on a flow() ONLY when some out is demanded."""

    name: str
    outs: List[str]
    pull: Callable[[], Any]


@dataclass
class Node:
    """A barrier node with optional triggers and memoization.

    Fires when every input topic is fresh AND (if triggers are declared) at least
    one trigger has fired since the last execution.  Publishes ``fn(**inputs)``
    mapped onto ``outs``.  Empty ``outs`` is a terminal SINK.

    triggers : topic names that act as event signals (epoch_end, etc).  The node
               accumulates trigger fires but does NOT consume their values as
               fn kwargs -- triggers are pure timing signals.
    memo     : ``(kwargs) -> cache_key``.  If the cache contains key, skip fn
               and emit the cached output.
    """

    name: str
    inputs: Dict[str, str]  # {param_name: topic}
    outs: List[str]
    fn: Callable[..., Any]
    triggers: List[str] = field(default_factory=list)
    memo: "Callable[..., Any] | None" = None
    resource: ResourcePolicy = field(default_factory=ResourcePolicy)
    _fresh: Set[str] = field(default_factory=set, repr=False)
    _triggered: bool = field(default=False, repr=False)

    @property
    def input_topics(self) -> Set[str]:
        return set(self.inputs.values())

    @property
    def trigger_topics(self) -> Set[str]:
        return set(self.triggers)

    @property
    def all_input_topics(self) -> Set[str]:
        return self.input_topics | self.trigger_topics

    @property
    def demand_topics(self) -> List[str]:
        """Topics whose demand keeps this node alive: its real outs, or -- for a
        terminal sink -- a private ``sink:<name>`` topic the runner subscribes to
        in order to demand the side effect."""
        return self.outs or ["sink:%s" % self.name]

    def is_ready(self) -> bool:
        """True when the node should fire: all data inputs fresh, and (if
        triggers are declared) at least one trigger has fired."""
        if not self._fresh >= self.input_topics:
            return False
        if self.triggers and not self._triggered:
            return False
        return True


@dataclass
class Recurrence:
    """A loop node: runs ``fn`` for ``iterations`` rounds, feeding each round's
    output back as the next round's input.

    feedback : maps output topic names back to input param names for the next
               iteration.  e.g. ``{"trained_model": "model"}`` means the output
               topic ``trained_model`` becomes input param ``model`` on the next
               round.
    memo     : ``(iteration, kwargs) -> cache_key``.  If cached, skip that
               iteration and emit cached output.
    on_iteration : optional callback fired after each iteration with
                   ``(iteration, output)`` -- for lifecycle events.
    """

    name: str
    inputs: Dict[str, str]  # {param_name: topic} -- initial inputs
    outs: List[str]
    fn: Callable[..., Any]
    iterations: int
    feedback: Dict[str, str]  # {out_topic: input_param}
    memo: "Callable[..., Any] | None" = None
    memo_load: "Callable[[Any], Any] | None" = None
    memo_save: "Callable[[Any, Any], None] | None" = None
    on_iteration: "Callable[[int, Any], None] | None" = None


@dataclass
class DemandHandler:
    """Handle returned by :meth:`DemandGraph.subscribe`; pass to ``unsubscribe``."""

    topic: str
    callback: Callable[[Any], None]


class DemandGraph:
    """A demand-driven pub/sub dataflow graph (see module docstring)."""

    def __init__(
        self,
        memo: MemoStore | None = None,
        interceptors: list[NodeInterceptor] | None = None,
    ) -> None:
        self._sources: List[Source] = []
        self._nodes: List[Node] = []
        self._recurrences: List[Recurrence] = []
        self._consumers: Dict[str, List[Node]] = {}  # topic -> nodes reading it
        self._producer: Dict[str, Any] = {}          # topic -> Source|Node|Recurrence
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}
        self._values: Dict[str, Any] = {}
        self._schemas: Dict[str, type] = {}
        self._qos: Dict[str, "QoS"] = {}
        self._relations: Dict[str, "Relation"] = {}
        self._memo = memo or MemoStore()
        #: instrumentation counters (per name): how often each source was pulled
        #: and each node fired -- the observable proof of laziness in tests.
        self.pulls: Dict[str, int] = {}
        self.fires: Dict[str, int] = {}
        self._demand_policy: CachePolicy = CachePolicy()
        self._demand_input_hashes: Dict[str, str] = {}
        self._resource_locks: Dict[str, threading.Lock] = {}
        self._resource_pools: Dict[str, "queue.Queue[None]"] = {}
        self._interceptors: list[NodeInterceptor] = interceptors or []

    # ---- wiring ---------------------------------------------------------- #
    def source(
        self,
        name: str,
        pull: Callable[[], Any],
        out: "str | None" = None,
        outs: "Sequence[str] | None" = None,
    ) -> Source:
        src = Source(name, _as_outs(out, outs), pull)
        for topic in src.outs:
            self._register_producer(topic, src)
        self._sources.append(src)
        self.pulls.setdefault(name, 0)
        return src

    def node(
        self,
        name: str,
        inputs: Dict[str, str],
        fn: Callable[..., Any],
        out: "str | None" = None,
        outs: "Sequence[str] | None" = None,
        triggers: "Sequence[str] | None" = None,
        memo: "Callable[..., Any] | None" = None,
        resource: ResourcePolicy | None = None,
    ) -> Node:
        trigger_list = list(triggers) if triggers else []
        n = Node(name, dict(inputs), _as_outs(out, outs), fn,
                 triggers=trigger_list, memo=memo,
                 resource=resource or ResourcePolicy())
        for topic in n.outs:
            self._register_producer(topic, n)
        self._nodes.append(n)
        for topic in n.input_topics:
            self._consumers.setdefault(topic, []).append(n)
        for topic in n.trigger_topics:
            self._consumers.setdefault(topic, []).append(n)
        self.fires.setdefault(name, 0)
        return n

    def recurrence(
        self,
        name: str,
        inputs: Dict[str, str],
        fn: Callable[..., Any],
        iterations: int,
        feedback: Dict[str, str],
        out: "str | None" = None,
        outs: "Sequence[str] | None" = None,
        memo: "Callable[..., Any] | None" = None,
        memo_load: "Callable[[Any], Any] | None" = None,
        memo_save: "Callable[[Any, Any], None] | None" = None,
        on_iteration: "Callable[[int, Any], None] | None" = None,
    ) -> Recurrence:
        out_list = _as_outs(out, outs)
        r = Recurrence(
            name, dict(inputs), out_list, fn, iterations,
            dict(feedback), memo, memo_load, memo_save, on_iteration,
        )
        for topic in out_list:
            self._register_producer(topic, r)
        self._recurrences.append(r)
        self.fires.setdefault(name, 0)
        return r

    @property
    def memo(self) -> MemoStore:
        return self._memo

    def compute(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
        out: str | None = None,
        outs: Sequence[str] | None = None,
        triggers: Sequence[str] | None = None,
        resource: ResourcePolicy | None = None,
    ) -> Callable:
        """Decorator: register a function as a node, inferring inputs from
        its parameter names.  Each param whose name matches a registered
        producer topic becomes a data input; the rest raise at demand time
        unless provided as kwargs."""
        def decorator(f: Callable) -> Callable:
            node_name = name or f.__name__
            params = list(inspect.signature(f).parameters.keys())
            inputs = {p: p for p in params}
            out_list = _as_outs(out, outs)
            if not out_list:
                out_list = [node_name]
            self.node(node_name, inputs, f, outs=out_list,
                      triggers=triggers, resource=resource)
            return f
        if fn is not None:
            return decorator(fn)
        return decorator

    def demand(
        self,
        target: str,
        policy: CachePolicy | None = None,
        **kwargs: Any,
    ) -> Any:
        """One-shot resolve: topologically walk from target back to kwargs,
        compute or cache each node per policy.  Graph.demand() on the
        unified fabric."""
        if policy is None:
            policy = CachePolicy()
        order = self._topo_sort(target)
        results: Dict[str, Any] = {}
        hashes: Dict[str, str] = {}

        for node in order:
            node_kwargs, input_hashes = self._gather_demand_inputs(
                node, results, hashes, kwargs,
            )
            key = node_key(node.name, input_hashes)
            hashes[node.name] = key
            for topic in node.outs:
                hashes[topic] = key

            hit, cached = self._memo.get(key)
            compute = self._should_compute(policy, hit)

            if compute:
                result = node.fn(**node_kwargs)
                self._memo.put(key, result)
                if policy.mode == CachePolicy.CHECK and hit:
                    if policy.diverged(cached, result):
                        logger.warning(
                            "CHECK divergence on %r: cached=%r fresh=%r",
                            node.name, cached, result,
                        )
                results[node.name] = result
            else:
                results[node.name] = cached

            for topic in node.outs:
                results[topic] = results[node.name]
                self._values[topic] = results[node.name]

        return results[order[-1].name]

    def _topo_sort(self, target: str) -> List[Node]:
        node_map = {n.name: n for n in self._nodes}
        topic_to_node: Dict[str, str] = {}
        for n in self._nodes:
            for t in n.outs:
                topic_to_node[t] = n.name

        def resolve(name_or_topic: str) -> str | None:
            if name_or_topic in node_map:
                return name_or_topic
            return topic_to_node.get(name_or_topic)

        root = resolve(target)
        if root is None:
            raise KeyError("unknown node or topic: %r" % target)

        visited: Set[str] = set()
        in_stack: Set[str] = set()
        order: List[Node] = []

        def visit(name: str) -> None:
            if name not in node_map:
                return
            if name in in_stack:
                raise ValueError("cycle detected involving %r" % name)
            if name in visited:
                return
            in_stack.add(name)
            visited.add(name)
            for topic in node_map[name].inputs.values():
                resolved = resolve(topic)
                if resolved is not None:
                    visit(resolved)
            in_stack.discard(name)
            order.append(node_map[name])

        visit(root)
        return order

    def _gather_demand_inputs(
        self,
        node: Node,
        results: Dict[str, Any],
        hashes: Dict[str, str],
        kwargs: Dict[str, Any],
    ) -> "tuple[Dict[str, Any], Dict[str, str]]":
        node_kwargs: Dict[str, Any] = {}
        input_hashes: Dict[str, str] = {}
        for param, topic in node.inputs.items():
            if topic in results:
                node_kwargs[param] = results[topic]
                input_hashes[param] = hashes[topic]
            elif param in kwargs:
                node_kwargs[param] = kwargs[param]
                input_hashes[param] = content_hash(kwargs[param])
            else:
                raise TypeError(
                    "node %r requires input %r: not a registered node "
                    "and not provided in kwargs" % (node.name, param)
                )
        return node_kwargs, input_hashes

    def inject(self, topic: str, value: Any) -> None:
        """Push a value onto a topic without a source/node -- for injecting
        lifecycle events (epoch boundaries, signals) into the dataflow."""
        self._publish(topic, value)

    def _register_producer(self, topic: str, producer: Any) -> None:
        if topic in self._producer:
            raise ValueError(
                "topic %r already produced by %r"
                % (topic, self._producer[topic].name)
            )
        self._producer[topic] = producer

    # ---- demand (dynamic subscription) ----------------------------------- #
    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> DemandHandler:
        """Register a consumer of ``topic``. Creates demand: the next
        :meth:`flow` will pull whatever sources feed it, transitively."""
        self._subs.setdefault(topic, []).append(callback)
        return DemandHandler(topic, callback)

    def unsubscribe(self, hdlr: DemandHandler) -> None:
        """Drop a subscriber. If it was the last demand on a chain, subsequent
        flows stop pulling that chain's sources."""
        subs = self._subs.get(hdlr.topic)
        if subs and hdlr.callback in subs:
            subs.remove(hdlr.callback)
            if not subs:
                del self._subs[hdlr.topic]

    def is_demanded(self, topic: str, _seen: Set[str] | None = None) -> bool:
        """True iff ``topic`` has a subscriber, or feeds (transitively) a node
        that is demanded. Backward reachability from the sinks -- the
        generalisation of ``channel_has_listeners`` to the whole DAG."""
        seen = _seen if _seen is not None else set()
        if topic in seen:
            return False
        seen.add(topic)
        if self._subs.get(topic):
            return True
        for node in self._consumers.get(topic, []):
            if self._node_demanded(node, seen):
                return True
        for rec in self._recurrences:
            if topic in rec.inputs.values():
                if any(self.is_demanded(t, seen) for t in rec.outs):
                    return True
        return False

    def _node_demanded(self, node: Node, seen: Set[str]) -> bool:
        """A node is demanded iff any topic it keeps alive is demanded -- its real
        outs, or the ``sink:<name>`` topic for a terminal side-effect node."""
        return any(self.is_demanded(topic, seen) for topic in node.demand_topics)

    # ---- static projection (what EXPORT emits) --------------------------- #
    def demanded_nodes(self) -> List[Node]:
        """The nodes on a chain that reaches a subscriber -- the ONLY nodes a
        demand-driven ``export`` should emit."""
        return [n for n in self._nodes if self._node_demanded(n, set())]

    def demanded_sources(self) -> List[Source]:
        """The sources whose output feeds something demanded."""
        return [
            s for s in self._sources
            if any(self.is_demanded(topic) for topic in s.outs)
        ]

    def demanded_recurrences(self) -> List[Recurrence]:
        """The recurrences whose output feeds something demanded."""
        return [
            r for r in self._recurrences
            if any(self.is_demanded(topic) for topic in r.outs)
        ]

    def demanded_topics(self) -> Set[str]:
        """Every topic that is demanded (has a subscriber or feeds one)."""
        topics: Set[str] = set(self._producer)
        for node in self._nodes:
            topics.update(node.input_topics)
        return {t for t in topics if self.is_demanded(t)}

    # ---- execution ------------------------------------------------------- #
    def flow(self) -> bool:
        """One streaming round: pull every DEMANDED source, cascade through
        demanded barrier nodes, deliver to subscribers. Returns True if any
        source emitted (False = fully dormant: nothing demanded)."""
        for node in self._nodes:
            if not node.triggers:
                node._fresh.clear()
        emitted = False
        for src in self._sources:
            if not any(self.is_demanded(topic) for topic in src.outs):
                continue
            for ic in self._interceptors:
                ic.before_source(src.name)
            t0 = time.monotonic()
            pull_err: Optional[Exception] = None
            try:
                value = src.pull()
            except StopIteration:
                dt = time.monotonic() - t0
                for ic in self._interceptors:
                    ic.after_source(src.name, dt, None)
                continue
            except Exception as e:
                dt = time.monotonic() - t0
                pull_err = e
                for ic in self._interceptors:
                    ic.after_source(src.name, dt, pull_err)
                raise
            dt = time.monotonic() - t0
            for ic in self._interceptors:
                ic.after_source(src.name, dt, None)
            self.pulls[src.name] += 1
            emitted = True
            for topic, mapped in map_output(src.outs, value).items():
                self._publish(topic, mapped)
        return emitted

    def run_recurrences(self) -> None:
        """Execute all demanded recurrences. Each runs its iterations,
        feeding output back as input, with optional memo skip."""
        for rec in self._recurrences:
            if not any(self.is_demanded(topic) for topic in rec.outs):
                continue
            kwargs = {}
            for param, topic in rec.inputs.items():
                if topic in self._values:
                    kwargs[param] = self._values[topic]
            for i in range(rec.iterations):
                cache_key = rec.memo(i, kwargs) if rec.memo else None
                if cache_key is not None and rec.memo_load is not None:
                    cached = rec.memo_load(cache_key)
                    if cached is not None:
                        output = cached
                        if rec.on_iteration:
                            rec.on_iteration(i, output)
                        mapped = map_output(rec.outs, output)
                        for param_out, param_in in rec.feedback.items():
                            if param_out in mapped:
                                kwargs[param_in] = mapped[param_out]
                        self.fires[rec.name] += 1
                        continue
                output = rec.fn(**kwargs)
                self.fires[rec.name] += 1
                if cache_key is not None and rec.memo_save is not None:
                    rec.memo_save(cache_key, output)
                if rec.on_iteration:
                    rec.on_iteration(i, output)
                mapped = map_output(rec.outs, output)
                for param_out, param_in in rec.feedback.items():
                    if param_out in mapped:
                        kwargs[param_in] = mapped[param_out]
            for topic, value in map_output(rec.outs, output).items():
                self._publish(topic, value)

    def topic(
        self,
        name: str,
        *,
        schema: "type | None" = None,
        qos: "QoS | None" = None,
    ) -> None:
        """Declare a topic with optional schema and QoS."""
        if schema is not None:
            self._schemas[name] = schema
        if qos is not None:
            self._qos[name] = qos

    def topic_schema(self, topic: str, schema: type) -> None:
        """Attach a pydantic schema to a topic for publish-time validation.

        Prefer :meth:`topic` for new code.
        """
        self._schemas[topic] = schema

    def serialize_topic(self, topic: str) -> dict:
        """Validate current topic value and return as serializable dict."""
        schema = self._schemas.get(topic)
        if schema is None:
            raise TypeError(f"no schema registered for topic {topic!r}")
        value = self._values[topic]
        if isinstance(value, dict):
            return schema.model_validate(value).model_dump()
        return schema.model_validate(value, from_attributes=True).model_dump()

    def _publish(self, topic: str, value: Any) -> None:
        schema = self._schemas.get(topic)
        if schema is not None:
            if isinstance(value, dict):
                schema.model_validate(value)
            else:
                schema.model_validate(value, from_attributes=True)
        self._values[topic] = value
        for callback in list(self._subs.get(topic, [])):
            callback(value)
        for node in self._consumers.get(topic, []):
            if not self._node_demanded(node, set()):
                continue
            if topic in node.trigger_topics:
                node._triggered = True
                if node.is_ready():
                    self._fire_node(node)
                continue
            node._fresh.add(topic)
            if node.is_ready():
                self._fire_node(node)

    def _compute_node_key(self, node: Node) -> str:
        input_hashes: Dict[str, str] = {}
        for param, topic in node.inputs.items():
            if topic in self._values:
                if param in self._demand_input_hashes:
                    input_hashes[param] = self._demand_input_hashes[param]
                else:
                    input_hashes[param] = content_hash(self._values[topic])
            else:
                input_hashes[param] = "missing"
        return node_key(node.name, input_hashes)

    def _ensure_resource(self, resource: ResourcePolicy) -> None:
        if resource.name not in self._resource_locks:
            self._resource_locks[resource.name] = threading.Lock()
        if resource.mode == ResourcePolicy.SHARED:
            if resource.name not in self._resource_pools:
                q: queue.Queue[None] = queue.Queue()
                for _ in range(resource.capacity):
                    q.put(None)
                self._resource_pools[resource.name] = q

    def _fire_node(self, node: Node) -> None:
        node._fresh.clear()
        node._triggered = False
        kwargs = {p: self._values[t] for p, t in node.inputs.items()}
        policy = self._demand_policy

        for ic in self._interceptors:
            ic.before_node(node.name, kwargs)
        t0 = time.monotonic()
        node_err: Optional[Exception] = None

        is_sink = not node.outs
        is_trigger_only = node.triggers and not node.inputs

        try:
            self._fire_node_inner(node, kwargs, policy, is_sink, is_trigger_only)
        except Exception as e:
            node_err = e
            raise
        finally:
            dt = time.monotonic() - t0
            for ic in self._interceptors:
                ic.after_node(node.name, kwargs, dt, node_err)

    def _fire_node_inner(
        self, node: Node, kwargs: Dict[str, Any],
        policy: CachePolicy, is_sink: bool, is_trigger_only: bool,
    ) -> None:
        if is_sink or is_trigger_only:
            output = node.fn(**kwargs)
        else:
            if node.memo is not None:
                cache_key = node.memo(**kwargs)
            else:
                cache_key = self._compute_node_key(node)

            hit, cached = self._memo.get(cache_key)
            compute = self._should_compute(policy, hit)

            if compute:
                resource = node.resource
                self._ensure_resource(resource)
                if resource.mode == ResourcePolicy.SHARED:
                    output = self._fire_shared(node, kwargs, resource)
                else:
                    output = self._fire_exclusive(node, kwargs, resource)
                self._memo.put(cache_key, output)
                if policy.mode == CachePolicy.CHECK and hit:
                    if policy.diverged(cached, output):
                        logger.warning(
                            "CHECK divergence on %r: cached=%r fresh=%r",
                            node.name, cached, output,
                        )
            else:
                output = cached

        self.fires[node.name] += 1
        for out_topic, mapped in map_output(node.outs, output).items():
            self._publish(out_topic, mapped)

    def _fire_exclusive(
        self, node: Node, kwargs: Dict[str, Any], resource: ResourcePolicy,
    ) -> Any:
        lock = self._resource_locks[resource.name]
        with lock:
            return node.fn(**kwargs)

    def _fire_shared(
        self, node: Node, kwargs: Dict[str, Any], resource: ResourcePolicy,
    ) -> Any:
        pool = self._resource_pools[resource.name]
        pool.get()
        try:
            return node.fn(**kwargs)
        finally:
            pool.put(None)

    @staticmethod
    def _should_compute(policy: CachePolicy, has_cached: bool) -> bool:
        if policy.mode == CachePolicy.ACCEPT:
            return not has_cached
        if policy.mode == CachePolicy.FRESH:
            return True
        return True

    # ---- parametric relations ----------------------------------------------- #
    def relation(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
    ) -> Callable:
        """Decorator: register a parametric relation.

        The decorated function's annotations declare dependencies via
        ``Annotated[T, Demand(...)]`` and output via return annotation
        ``Annotated[T, Flow("name")]``.

        Usage::

            @graph.relation
            def train(
                config,
                prev: Annotated[Model, Demand(lambda c: c.parent)],
                data: Annotated[Dataset, Demand("cifar10")],
            ):
                ...

            graph.base("train", base_config, fresh_model)
            model = graph.demand_relation("train", final_config)
        """
        def decorator(f: Callable) -> Relation:
            rel_name = name or f.__name__
            deps = _extract_demands(f)
            rel = Relation(rel_name, f, deps, self._memo)
            self._relations[rel_name] = rel
            return rel
        if fn is not None:
            return decorator(fn)
        return decorator

    def demand_relation(
        self,
        name: str,
        key: Any,
        policy: CachePolicy | None = None,
    ) -> Any:
        """Resolve a parametric relation for the given key.

        Walks the demand chain backwards until base cases are hit,
        then computes forward, memoizing each step.
        """
        rel = self._relations[name]
        return rel.resolve(key, policy or CachePolicy())
