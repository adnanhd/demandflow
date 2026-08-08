from __future__ import annotations

import pickle
import threading
from typing import Any, Callable

import zenoh

from demandflow.qos import DEFAULT_QOS, QoS


class Context:
    """Global communication context backed by a zenoh session.

    All nodes, topics, services, and actions live within a context.
    A default context is created lazily on first use.
    """

    def __init__(self, config: zenoh.Config | None = None) -> None:
        self._session = zenoh.open(config or zenoh.Config())
        self._publishers: dict[str, zenoh.Publisher] = {}
        self._subscribers: list[zenoh.Subscriber] = []
        self._queryables: list[zenoh.Queryable] = []
        self._lock = threading.Lock()

    @property
    def session(self) -> zenoh.Session:
        return self._session

    def publish(self, topic: str, data: Any) -> None:
        payload = pickle.dumps(data)
        with self._lock:
            if topic not in self._publishers:
                self._publishers[topic] = self._session.declare_publisher(topic)
        self._publishers[topic].put(payload)

    def subscribe(
        self, topic: str, callback: Callable[[Any], None], qos: QoS = DEFAULT_QOS
    ) -> zenoh.Subscriber:
        def _on_sample(sample: zenoh.Sample) -> None:
            data = pickle.loads(bytes(sample.payload))
            callback(data)

        sub = self._session.declare_subscriber(topic, _on_sample)
        self._subscribers.append(sub)
        return sub

    def declare_queryable(
        self, key: str, handler: Callable[[Any], Any]
    ) -> zenoh.Queryable:
        def _on_query(query: zenoh.Query) -> None:
            request = pickle.loads(bytes(query.payload)) if query.payload else None
            result = handler(request)
            query.reply(key, pickle.dumps(result))

        q = self._session.declare_queryable(key, _on_query)
        self._queryables.append(q)
        return q

    def query(self, key: str, request: Any = None, timeout_s: float = 5.0) -> Any:
        payload = pickle.dumps(request) if request is not None else None
        replies = self._session.get(
            key,
            payload=payload,
            timeout=timeout_s,
        )
        for reply in replies:
            if reply.ok is not None:
                return pickle.loads(bytes(reply.ok.payload))
        raise TimeoutError("no reply for service %r" % key)

    def close(self) -> None:
        for sub in self._subscribers:
            sub.undeclare()
        for q in self._queryables:
            q.undeclare()
        for pub in self._publishers.values():
            pub.undeclare()
        self._session.close()
        self._publishers.clear()
        self._subscribers.clear()
        self._queryables.clear()


_default_context: Context | None = None
_default_lock = threading.Lock()


def get_default_context() -> Context:
    global _default_context
    if _default_context is not None:
        return _default_context
    with _default_lock:
        if _default_context is not None:
            return _default_context
        _default_context = Context()
        return _default_context


def shutdown_default_context() -> None:
    global _default_context
    if _default_context is not None:
        _default_context.close()
        _default_context = None
