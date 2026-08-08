from __future__ import annotations

from typing import Any, Callable

from ondemand.context import Context


class ServiceServer:
    """Request/response service backed by zenoh queryables."""

    def __init__(
        self, context: Context, name: str, handler: Callable[[Any], Any]
    ) -> None:
        self._name = name
        self._handler = handler
        self._queryable = context.declare_queryable(
            "service/%s" % name, handler
        )

    @property
    def name(self) -> str:
        return self._name


class ServiceClient:
    """Client side of a request/response service."""

    def __init__(
        self, context: Context, name: str, timeout_s: float = 5.0
    ) -> None:
        self._context = context
        self._name = name
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self._name

    def call(self, request: Any = None) -> Any:
        return self._context.query(
            "service/%s" % self._name, request, self._timeout_s
        )
