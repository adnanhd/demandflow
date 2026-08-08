from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable

from ondemand.context import Context
from ondemand.qos import DEFAULT_QOS, QoS


class Publisher:
    """Publishes messages to a named topic via zenoh."""

    def __init__(self, context: Context, topic: str, qos: QoS = DEFAULT_QOS) -> None:
        self._context = context
        self._topic = topic
        self._qos = qos

    @property
    def topic(self) -> str:
        return self._topic

    def publish(self, data: Any) -> None:
        self._context.publish(self._topic, data)


class Subscriber:
    """Subscribes to a named topic, delivering via callback or polling."""

    def __init__(
        self,
        context: Context,
        topic: str,
        callback: Callable[[Any], None] | None = None,
        qos: QoS = DEFAULT_QOS,
    ) -> None:
        self._topic = topic
        self._qos = qos
        self._callback = callback
        self._inbox: deque[Any] = deque(maxlen=qos.history if qos.history > 0 else None)
        self._lock = threading.Lock()

        def _on_message(data: Any) -> None:
            if self._callback is not None:
                self._callback(data)
            else:
                with self._lock:
                    self._inbox.append(data)

        self._zenoh_sub = context.subscribe(topic, _on_message, qos)

    @property
    def topic(self) -> str:
        return self._topic

    def take(self) -> Any:
        with self._lock:
            return self._inbox.popleft()

    def take_all(self) -> list[Any]:
        with self._lock:
            items = list(self._inbox)
            self._inbox.clear()
            return items

    def has_data(self) -> bool:
        with self._lock:
            return len(self._inbox) > 0
