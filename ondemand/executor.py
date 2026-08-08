from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ondemand.node import LifecycleNode, LifecycleState, Node

logger = logging.getLogger(__name__)


class Executor:
    """Base executor that drives node timers and subscription polling."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._running = False

    def add_node(self, node: Node) -> None:
        self._nodes.append(node)

    def remove_node(self, node: Node) -> None:
        self._nodes.remove(node)

    def spin(self) -> None:
        self._running = True
        try:
            while self._running:
                self.spin_once(timeout_s=0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def spin_once(self, timeout_s: float = 0.0) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        self._running = False


class SingleThreadedExecutor(Executor):
    """Processes all callbacks sequentially in the calling thread."""

    def spin_once(self, timeout_s: float = 0.0) -> None:
        fired = False
        for node in self._nodes:
            for timer in node.timers:
                if timer.ready():
                    timer.fire()
                    fired = True
            for sub in node.subscriptions:
                while sub.has_data():
                    msg = sub.take()
                    fired = True
        if not fired and timeout_s > 0:
            time.sleep(timeout_s)


class MultiThreadedExecutor(Executor):
    """Dispatches callbacks to a thread pool."""

    def __init__(self, num_threads: int | None = None) -> None:
        super().__init__()
        self._num_threads = num_threads or min(8, (os.cpu_count() or 2))

    def spin(self) -> None:
        self._running = True
        with ThreadPoolExecutor(self._num_threads) as pool:
            try:
                while self._running:
                    futures = []
                    for node in self._nodes:
                        for timer in node.timers:
                            if timer.ready():
                                timer.fire()
                                futures.append(pool.submit(timer.fire))
                        for sub in node.subscriptions:
                            while sub.has_data():
                                msg = sub.take()
                    if not futures:
                        time.sleep(0.01)
                    for f in futures:
                        f.result()
            except KeyboardInterrupt:
                pass
            finally:
                self._running = False

    def spin_once(self, timeout_s: float = 0.0) -> None:
        for node in self._nodes:
            for timer in node.timers:
                if timer.ready():
                    timer.fire()
        if timeout_s > 0:
            time.sleep(timeout_s)


def _node_process_entry(
    node_factory: Callable[[], Node],
    ready_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
) -> None:
    """Entry point for a node running in its own process."""
    node = node_factory()
    if isinstance(node, LifecycleNode):
        node.configure()
        node.activate()
    ready_event.set()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    while not stop_event.is_set():
        executor.spin_once(timeout_s=0.05)
    if isinstance(node, LifecycleNode):
        node.shutdown()
    node.destroy()


class MultiProcessExecutor:
    """Runs each node in its own process.

    Nodes communicate via zenoh's network transport — each process
    opens its own zenoh session and discovers peers automatically.
    Node factories (callables returning a Node) are required because
    nodes must be constructed in the child process.
    """

    def __init__(self) -> None:
        self._factories: list[Callable[[], Node]] = []
        self._processes: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.Event] = []
        self._stop_event = multiprocessing.Event()

    def add_node_factory(self, factory: Callable[[], Node]) -> None:
        self._factories.append(factory)

    def spin(self) -> None:
        self._stop_event.clear()
        for factory in self._factories:
            ready = multiprocessing.Event()
            self._ready_events.append(ready)
            p = multiprocessing.Process(
                target=_node_process_entry,
                args=(factory, ready, self._stop_event),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        for ready in self._ready_events:
            ready.wait(timeout=30)

        try:
            while any(p.is_alive() for p in self._processes):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._processes.clear()
        self._ready_events.clear()
