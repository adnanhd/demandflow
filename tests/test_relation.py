"""Tests for parametric demand relations."""

from __future__ import annotations

from typing import Annotated

from demandflow.annotations import Demand, Flow
from demandflow.demand import DemandGraph


class TestRelation:
    def test_fibonacci(self):
        graph = DemandGraph()

        @graph.relation
        def fib(
            n,
            a: Annotated[int, Demand(lambda n: n - 1)],
            b: Annotated[int, Demand(lambda n: n - 2)],
        ):
            return a + b

        fib.base(0, 0)
        fib.base(1, 1)

        assert graph.demand_relation("fib", 0) == 0
        assert graph.demand_relation("fib", 1) == 1
        assert graph.demand_relation("fib", 2) == 1
        assert graph.demand_relation("fib", 5) == 5
        assert graph.demand_relation("fib", 10) == 55

    def test_config_based_chain(self):
        class Config:
            def __init__(self, epoch, lr, parent=None):
                self.epoch = epoch
                self.lr = lr
                self.parent = parent

            def __hash__(self):
                return hash((self.epoch, self.lr))

            def __eq__(self, other):
                return self.epoch == other.epoch and self.lr == other.lr

        base_cfg = Config(0, 0.01)
        cfg1 = Config(1, 0.01, parent=base_cfg)
        cfg2 = Config(2, 0.01, parent=cfg1)
        cfg3 = Config(3, 0.01, parent=cfg2)

        graph = DemandGraph()
        calls = []

        @graph.relation
        def train(
            config,
            prev: Annotated[str, Demand(lambda c: c.parent)],
        ):
            calls.append(config.epoch)
            return f"model_epoch{config.epoch}_lr{config.lr}"

        train.base(base_cfg, "fresh_model")

        result = graph.demand_relation("train", cfg3)
        assert result == "model_epoch3_lr0.01"
        assert calls == [1, 2, 3]  # computed in order

    def test_memo_skips_recompute(self):
        graph = DemandGraph()
        call_count = [0]

        @graph.relation
        def compute(
            n,
            prev: Annotated[int, Demand(lambda n: n - 1)],
        ):
            call_count[0] += 1
            return prev + 1

        compute.base(0, 0)

        assert graph.demand_relation("compute", 5) == 5
        first_count = call_count[0]

        # second demand should hit memo
        assert graph.demand_relation("compute", 5) == 5
        assert call_count[0] == first_count

    def test_different_configs_different_chains(self):
        class Cfg:
            def __init__(self, epoch, lr, parent=None):
                self.epoch = epoch
                self.lr = lr
                self.parent = parent

            def __hash__(self):
                return hash((self.epoch, self.lr))

            def __eq__(self, other):
                return self.epoch == other.epoch and self.lr == other.lr

        base = Cfg(0, 0.01)
        a1 = Cfg(1, 0.01, parent=base)
        b1 = Cfg(1, 0.001, parent=base)

        graph = DemandGraph()

        @graph.relation
        def train(
            config,
            prev: Annotated[str, Demand(lambda c: c.parent)],
        ):
            return f"e{config.epoch}_lr{config.lr}"

        train.base(base, "fresh")

        result_a = graph.demand_relation("train", a1)
        result_b = graph.demand_relation("train", b1)
        assert result_a == "e1_lr0.01"
        assert result_b == "e1_lr0.001"
        assert result_a != result_b
