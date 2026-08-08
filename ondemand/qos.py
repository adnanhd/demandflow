from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QoS:
    """Quality-of-service profile for a topic or endpoint.

    reliability
        ``"reliable"``: every message is delivered.
        ``"best_effort"``: fire-and-forget, drops allowed.
    history
        How many past messages a new subscription receives on connect.
        0 means keep all.
    deadline
        Maximum seconds between expected messages.  ``None`` = no deadline.
    """

    reliability: Literal["reliable", "best_effort"] = "reliable"
    history: int = 1
    deadline: float | None = None

    def compatible_with(self, other: QoS) -> bool:
        if self.reliability == "best_effort" and other.reliability == "reliable":
            return False
        return True


DEFAULT_QOS = QoS()
SENSOR_QOS = QoS(reliability="best_effort", history=5)
SERVICE_QOS = QoS(reliability="reliable", history=0)
