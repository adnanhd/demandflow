from __future__ import annotations

import enum
import inspect
import logging
from typing import Any, Callable

from demandflow.action import ActionClient, ActionServer
from demandflow.context import Context, get_default_context
from demandflow.qos import DEFAULT_QOS, QoS
from demandflow.service import ServiceClient, ServiceServer
from demandflow.timer import Timer
from demandflow.topic import Publisher, Subscriber

logger = logging.getLogger(__name__)


class Node:
    """Base computation node with communication primitives.

    A node owns publishers, subscriptions, services, actions, and timers.
    The executor drives timer and subscription callbacks.
    """

    def __init__(self, name: str, context: Context | None = None) -> None:
        self.name = name
        self._context = context or get_default_context()
        self._publishers: list[Publisher] = []
        self._subscriptions: list[Subscriber] = []
        self._services: list[ServiceServer] = []
        self._service_clients: list[ServiceClient] = []
        self._action_servers: list[ActionServer] = []
        self._action_clients: list[ActionClient] = []
        self._timers: list[Timer] = []

    @property
    def context(self) -> Context:
        return self._context

    @property
    def timers(self) -> list[Timer]:
        return list(self._timers)

    @property
    def subscriptions(self) -> list[Subscriber]:
        return list(self._subscriptions)

    def create_publisher(self, topic: str, qos: QoS = DEFAULT_QOS) -> Publisher:
        pub = Publisher(self._context, topic, qos)
        self._publishers.append(pub)
        return pub

    def create_subscription(
        self,
        topic: str,
        callback: Callable[[Any], None] | None = None,
        qos: QoS = DEFAULT_QOS,
    ) -> Subscriber:
        sub = Subscriber(self._context, topic, callback, qos)
        self._subscriptions.append(sub)
        return sub

    def create_service(
        self, name: str, handler: Callable[[Any], Any]
    ) -> ServiceServer:
        srv = ServiceServer(self._context, name, handler)
        self._services.append(srv)
        return srv

    def create_service_client(
        self, name: str, timeout_s: float = 5.0
    ) -> ServiceClient:
        client = ServiceClient(self._context, name, timeout_s)
        self._service_clients.append(client)
        return client

    def create_action_server(
        self, name: str, execute_fn: Callable
    ) -> ActionServer:
        srv = ActionServer(self._context, name, execute_fn)
        self._action_servers.append(srv)
        return srv

    def create_action_client(self, name: str) -> ActionClient:
        client = ActionClient(self._context, name)
        self._action_clients.append(client)
        return client

    def create_timer(
        self, period: float, callback: Callable[[], None]
    ) -> Timer:
        t = Timer(period, callback)
        self._timers.append(t)
        return t

    def destroy(self) -> None:
        self._publishers.clear()
        self._subscriptions.clear()
        self._services.clear()
        self._service_clients.clear()
        self._action_servers.clear()
        self._action_clients.clear()
        self._timers.clear()


class LifecycleState(str, enum.Enum):
    """Managed lifecycle state of a :class:`LifecycleNode`."""

    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"


class TransitionError(Exception):
    """Raised when a lifecycle transition is not valid from the current state."""


class LifecycleNode(Node):
    """Node with a managed lifecycle state machine.

    States: unconfigured -> inactive -> active -> finalized
    Transitions call on_configure / on_activate / on_deactivate /
    on_cleanup / on_shutdown which subclasses override.
    """

    _TRANSITIONS: dict[tuple[LifecycleState, str], LifecycleState] = {
        (LifecycleState.UNCONFIGURED, "configure"): LifecycleState.INACTIVE,
        (LifecycleState.INACTIVE, "activate"): LifecycleState.ACTIVE,
        (LifecycleState.ACTIVE, "deactivate"): LifecycleState.INACTIVE,
        (LifecycleState.INACTIVE, "cleanup"): LifecycleState.UNCONFIGURED,
        (LifecycleState.UNCONFIGURED, "shutdown"): LifecycleState.FINALIZED,
        (LifecycleState.INACTIVE, "shutdown"): LifecycleState.FINALIZED,
        (LifecycleState.ACTIVE, "shutdown"): LifecycleState.FINALIZED,
    }

    def __init__(self, name: str, context: Context | None = None) -> None:
        super().__init__(name, context)
        self._state = LifecycleState.UNCONFIGURED

    @property
    def state(self) -> LifecycleState:
        return self._state

    def _transition(self, action: str) -> None:
        key = (self._state, action)
        target = self._TRANSITIONS.get(key)
        if target is None:
            raise TransitionError(
                "cannot %s from state %s" % (action, self._state.value)
            )
        callback = {
            "configure": self.on_configure,
            "activate": self.on_activate,
            "deactivate": self.on_deactivate,
            "cleanup": self.on_cleanup,
            "shutdown": self.on_shutdown,
        }[action]
        logger.info("%s: %s -> %s", self.name, self._state.value, target.value)
        callback()
        self._state = target

    def configure(self) -> None:
        self._transition("configure")

    def activate(self) -> None:
        self._transition("activate")

    def deactivate(self) -> None:
        self._transition("deactivate")

    def cleanup(self) -> None:
        self._transition("cleanup")

    def shutdown(self) -> None:
        self._transition("shutdown")

    def on_configure(self) -> None:
        pass

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        pass

    def on_cleanup(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass
