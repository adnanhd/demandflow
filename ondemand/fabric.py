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
  ``subscribe`` switches a dormant chain on (the next tick starts pulling its
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
  checkpoints, eval memoisation).

* **Recurrence** -- a loop node whose output feeds back as its own input for N
  iterations.  ``recurrence("train", inputs, outs, fn, iterations=150)``
  replaces ``for epoch in range(150)`` in the runner.  Each iteration can be
  memo-skipped independently.

This module is the pure kernel -- no torch, no threads -- so the scheduling model
can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Set

__all__ = [
    "LazyFabric",
    "Node",
    "Source",
    "DemandHandler",
    "Recurrence",
    "map_output",
]


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
    """Normalise the single-``out`` / multi-``outs`` spelling to a list."""
    if outs is not None:
        return list(outs)
    return [out] if out is not None else []


@dataclass
class Source:
    """A demand-pulled emission source: ``pull()`` yields the next value (``next()``
    semantics, ``StopIteration`` at end-of-stream), mapped onto ``outs`` via
    :func:`map_output`. Pulled on a tick ONLY when some out is demanded."""

    name: str
    outs: List[str]
    pull: Callable[[], Any]


@dataclass
class Node:
    """A barrier node with optional triggers and memoisation.

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
    """Handle returned by :meth:`LazyFabric.subscribe`; pass to ``unsubscribe``."""

    topic: str
    callback: Callable[[Any], None]


class LazyFabric:
    """A demand-driven pub/sub dataflow graph (see module docstring)."""

    def __init__(self) -> None:
        self._sources: List[Source] = []
        self._nodes: List[Node] = []
        self._recurrences: List[Recurrence] = []
        self._consumers: Dict[str, List[Node]] = {}  # topic -> nodes reading it
        self._producer: Dict[str, Any] = {}          # topic -> Source|Node|Recurrence
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}
        self._values: Dict[str, Any] = {}
        self._memo_cache: Dict[Any, Any] = {}
        #: instrumentation counters (per name): how often each source was pulled
        #: and each node fired -- the observable proof of laziness in tests.
        self.pulls: Dict[str, int] = {}
        self.fires: Dict[str, int] = {}

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
    ) -> Node:
        trigger_list = list(triggers) if triggers else []
        n = Node(name, dict(inputs), _as_outs(out, outs), fn,
                 triggers=trigger_list, memo=memo)
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
        :meth:`tick` will pull whatever sources feed it, transitively."""
        self._subs.setdefault(topic, []).append(callback)
        return DemandHandler(topic, callback)

    def unsubscribe(self, hdlr: DemandHandler) -> None:
        """Drop a subscriber. If it was the last demand on a chain, subsequent
        ticks stop pulling that chain's sources."""
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
    def tick(self) -> bool:
        """One orchestra round: pull every DEMANDED source, cascade through
        demanded barrier nodes, deliver to subscribers. Returns True if any
        source emitted (False = fully dormant: nothing demanded)."""
        for node in self._nodes:
            if not node.triggers:
                node._fresh.clear()
        emitted = False
        for src in self._sources:
            if not any(self.is_demanded(topic) for topic in src.outs):
                continue
            try:
                value = src.pull()
            except StopIteration:
                continue
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

    def _publish(self, topic: str, value: Any) -> None:
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

    def _fire_node(self, node: Node) -> None:
        node._fresh.clear()
        node._triggered = False
        kwargs = {p: self._values[t] for p, t in node.inputs.items()}
        if node.memo is not None:
            cache_key = node.memo(**kwargs)
            if cache_key in self._memo_cache:
                output = self._memo_cache[cache_key]
                self.fires[node.name] += 1
                for out_topic, mapped in map_output(node.outs, output).items():
                    self._publish(out_topic, mapped)
                return
            output = node.fn(**kwargs)
            self._memo_cache[cache_key] = output
        else:
            output = node.fn(**kwargs)
        self.fires[node.name] += 1
        for out_topic, mapped in map_output(node.outs, output).items():
            self._publish(out_topic, mapped)
