from __future__ import annotations

import enum
import threading
import uuid
from typing import Any, Callable

from demandflow.context import Context


class ActionStatus(str, enum.Enum):
    """Lifecycle status of an action goal."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    CANCELED = "canceled"


class GoalHandle:
    """Server-side handle for a running action goal."""

    def __init__(self, goal_id: str, goal_data: Any, context: Context, name: str) -> None:
        self.goal_id = goal_id
        self.data = goal_data
        self.status = ActionStatus.RUNNING
        self._context = context
        self._name = name
        self._cancel_requested = False

    def publish_feedback(self, feedback: Any) -> None:
        self._context.publish(
            "action/%s/feedback/%s" % (self._name, self.goal_id),
            feedback,
        )

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def _request_cancel(self) -> None:
        self._cancel_requested = True


class ActionServer:
    """Long-running action with feedback, backed by zenoh.

    The execute callback receives a GoalHandle and returns the result.
    It runs in a separate thread so it can block without stalling
    the executor.
    """

    def __init__(
        self,
        context: Context,
        name: str,
        execute_fn: Callable[[GoalHandle], Any],
    ) -> None:
        self._context = context
        self._name = name
        self._execute_fn = execute_fn
        self._active_goals: dict[str, GoalHandle] = {}
        self._lock = threading.Lock()

        self._goal_queryable = context.declare_queryable(
            "action/%s/goal" % name, self._on_goal
        )
        self._cancel_queryable = context.declare_queryable(
            "action/%s/cancel" % name, self._on_cancel
        )

    @property
    def name(self) -> str:
        return self._name

    def _on_goal(self, goal_data: Any) -> dict:
        goal_id = uuid.uuid4().hex[:12]
        handle = GoalHandle(goal_id, goal_data, self._context, self._name)
        with self._lock:
            self._active_goals[goal_id] = handle

        thread = threading.Thread(
            target=self._run_goal, args=(handle,), daemon=True
        )
        thread.start()
        return {"goal_id": goal_id, "accepted": True}

    def _on_cancel(self, cancel_data: Any) -> dict:
        goal_id = cancel_data.get("goal_id") if isinstance(cancel_data, dict) else None
        if goal_id is None:
            return {"canceled": False}
        with self._lock:
            handle = self._active_goals.get(goal_id)
        if handle is None:
            return {"canceled": False}
        handle._request_cancel()
        return {"canceled": True}

    def _run_goal(self, handle: GoalHandle) -> None:
        try:
            result = self._execute_fn(handle)
            handle.status = ActionStatus.SUCCEEDED
        except Exception as exc:
            result = {"error": str(exc)}
            handle.status = ActionStatus.ABORTED

        self._context.publish(
            "action/%s/result/%s" % (self._name, handle.goal_id),
            {"status": handle.status.value, "result": result},
        )
        with self._lock:
            self._active_goals.pop(handle.goal_id, None)


class ActionClient:
    """Client side of a long-running action."""

    def __init__(self, context: Context, name: str) -> None:
        self._context = context
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def send_goal(
        self,
        goal_data: Any,
        on_feedback: Callable[[Any], None] | None = None,
        on_result: Callable[[dict], None] | None = None,
    ) -> str:
        response = self._context.query("action/%s/goal" % self._name, goal_data)
        goal_id = response["goal_id"]

        if on_feedback is not None:
            self._context.subscribe(
                "action/%s/feedback/%s" % (self._name, goal_id),
                on_feedback,
            )
        if on_result is not None:
            self._context.subscribe(
                "action/%s/result/%s" % (self._name, goal_id),
                on_result,
            )
        return goal_id

    def cancel(self, goal_id: str) -> bool:
        response = self._context.query(
            "action/%s/cancel" % self._name, {"goal_id": goal_id}
        )
        return response.get("canceled", False)
