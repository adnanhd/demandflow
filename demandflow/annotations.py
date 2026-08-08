"""Annotation markers for demandflow parametric demand chains.

Used with ``typing.Annotated`` to declare dependencies and outputs
on relation functions::

    @graph.relation
    def train(
        config,
        prev: Annotated[Model, Demand(lambda c: c.parent)],
        data: Annotated[Dataset, Demand("cifar10")],
    ) -> Annotated[Model, Flow("train")]:
        ...

``Demand`` declares a dependency -- either a static topic (string) or
a parametric resolver (callable that maps the relation key to a
dependency key).

``Flow`` declares what the relation produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union


@dataclass(frozen=True)
class Demand:
    """Declare a dependency on a topic or parametric relation.

    - ``Demand("cifar10")``: static topic, always resolves to "cifar10"
    - ``Demand(lambda c: c.parent)``: parametric, resolves relative to
      the current relation key
    """
    resolver: Union[str, Callable[[Any], Any]]

    def resolve(self, key: Any) -> Any:
        if callable(self.resolver):
            return self.resolver(key)
        return self.resolver


@dataclass(frozen=True)
class Flow:
    """Declare the output topic of a relation."""
    name: str
