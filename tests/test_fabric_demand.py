import logging

import pytest

from demandflow.demand import DemandGraph
from demandflow.policy import CachePolicy, ResourcePolicy


class TestComputeDecorator:
    def test_single_node(self):
        f = DemandGraph()

        @f.compute
        def double(x):
            return x * 2

        assert f.demand("double", x=5) == 10

    def test_chain(self):
        f = DemandGraph()

        @f.compute
        def add_one(x):
            return x + 1

        @f.compute
        def double(add_one):
            return add_one * 2

        assert f.demand("double", x=3) == 8

    def test_diamond(self):
        f = DemandGraph()

        @f.compute
        def source(x):
            return x

        @f.compute
        def left(source):
            return source + 10

        @f.compute
        def right(source):
            return source * 2

        @f.compute
        def merge(left, right):
            return left + right

        assert f.demand("merge", x=5) == 25

    def test_custom_name(self):
        f = DemandGraph()

        @f.compute(name="my_node")
        def some_function(x):
            return x + 1

        assert f.demand("my_node", x=5) == 6


class TestFabricMemoization:
    def test_cache_hit_skips_compute(self):
        f = DemandGraph()
        call_count = 0

        @f.compute
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        f.demand("expensive", x=4)
        assert call_count == 1
        f.demand("expensive", x=4)
        assert call_count == 1

    def test_different_input_recomputes(self):
        f = DemandGraph()
        call_count = 0

        @f.compute
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        f.demand("expensive", x=4)
        f.demand("expensive", x=5)
        assert call_count == 2

    def test_upstream_change_invalidates_downstream(self):
        f = DemandGraph()
        calls = []

        @f.compute
        def step_a(x):
            calls.append("a")
            return x + 1

        @f.compute
        def step_b(step_a):
            calls.append("b")
            return step_a * 2

        f.demand("step_b", x=1)
        assert calls == ["a", "b"]

        calls.clear()
        f.demand("step_b", x=1)
        assert calls == []

        calls.clear()
        f.demand("step_b", x=2)
        assert calls == ["a", "b"]


class TestFabricCachePolicy:
    def test_accept_uses_cache(self):
        f = DemandGraph()
        call_count = 0

        @f.compute
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x

        f.demand("compute", policy=CachePolicy(CachePolicy.ACCEPT), x=1)
        f.demand("compute", policy=CachePolicy(CachePolicy.ACCEPT), x=1)
        assert call_count == 1

    def test_check_recomputes_and_warns_on_divergence(self, caplog):
        f = DemandGraph()
        counter = 0

        @f.compute
        def impure(x):
            nonlocal counter
            counter += 1
            return x + counter

        f.demand("impure", policy=CachePolicy(CachePolicy.ACCEPT), x=0)

        with caplog.at_level(logging.WARNING):
            result = f.demand("impure", policy=CachePolicy(CachePolicy.CHECK), x=0)

        assert result == 2
        assert "CHECK divergence" in caplog.text

    def test_force_recomputes_everything(self):
        f = DemandGraph()
        calls = []

        @f.compute
        def upstream(x):
            calls.append("upstream")
            return x

        @f.compute
        def target(upstream):
            calls.append("target")
            return upstream + 1

        f.demand("target", x=1)
        calls.clear()

        f.demand("target", policy=CachePolicy(CachePolicy.FORCE), x=1)
        assert calls == ["upstream", "target"]


    def test_check_with_tolerance_suppresses_warning(self, caplog):
        f = DemandGraph()
        counter = 0

        @f.compute
        def noisy(x):
            nonlocal counter
            counter += 1
            return 1.0 + counter * 1e-8

        f.demand("noisy", x=0)
        with caplog.at_level(logging.WARNING):
            f.demand("noisy", policy=CachePolicy(CachePolicy.CHECK, tolerance=1e-6), x=0)
        assert "divergence" not in caplog.text

    def test_check_with_custom_comparator(self, caplog):
        f = DemandGraph()
        counter = 0

        @f.compute
        def varying(x):
            nonlocal counter
            counter += 1
            return {"value": x, "noise": counter}

        f.demand("varying", x=1)
        always_equal = CachePolicy(
            CachePolicy.CHECK,
            comparator=lambda a, b: a["value"] == b["value"],
        )
        with caplog.at_level(logging.WARNING):
            f.demand("varying", policy=always_equal, x=1)
        assert "divergence" not in caplog.text


class TestFabricContentHash:
    def test_dict_input_order_independent(self):
        f = DemandGraph()
        call_count = 0

        @f.compute
        def fn(config):
            nonlocal call_count
            call_count += 1
            return sum(config.values())

        f.demand("fn", config={"a": 1, "b": 2})
        f.demand("fn", config={"b": 2, "a": 1})
        assert call_count == 1

    def test_custom_content_hash(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

            def __content_hash__(self):
                return "fixed_hash"

        f = DemandGraph()
        call_count = 0

        @f.compute
        def fn(obj):
            nonlocal call_count
            call_count += 1
            return obj.val

        f.demand("fn", obj=MyObj(1))
        f.demand("fn", obj=MyObj(999))
        assert call_count == 1


class TestResourcePolicy:
    def test_exclusive_runs_sequentially(self):
        f = DemandGraph()
        gpu = ResourcePolicy("gpu", mode=ResourcePolicy.EXCLUSIVE)
        order = []

        @f.compute(resource=gpu)
        def step_a(x):
            order.append("a")
            return x + 1

        @f.compute(resource=gpu)
        def step_b(step_a):
            order.append("b")
            return step_a * 2

        assert f.demand("step_b", x=1) == 4
        assert order == ["a", "b"]

    def test_shared_resource_with_capacity(self):
        f = DemandGraph()
        cpu = ResourcePolicy("cpu", mode=ResourcePolicy.SHARED, capacity=4)

        @f.compute(resource=cpu)
        def process(x):
            return x * 3

        assert f.demand("process", x=7) == 21

    def test_different_resources_independent(self):
        f = DemandGraph()
        gpu = ResourcePolicy("gpu", mode=ResourcePolicy.EXCLUSIVE)
        io = ResourcePolicy("io", mode=ResourcePolicy.SHARED, capacity=2)

        @f.compute(resource=gpu)
        def train(x):
            return x + 1

        @f.compute(resource=io)
        def export(train):
            return train * 10

        assert f.demand("export", x=5) == 60


class TestDemandErrors:
    def test_demand_unregistered_target(self):
        f = DemandGraph()
        with pytest.raises(KeyError, match="unknown node"):
            f.demand("nonexistent")
