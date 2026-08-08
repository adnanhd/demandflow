import logging

import pytest

from ondemand import CachePolicy, CycleError, Graph


class TestBasicDemand:
    def test_single_node(self):
        g = Graph()

        @g.node
        def double(x):
            return x * 2

        assert g.demand("double", x=5) == 10

    def test_chain(self):
        g = Graph()

        @g.node
        def add_one(x):
            return x + 1

        @g.node
        def double(add_one):
            return add_one * 2

        assert g.demand("double", x=3) == 8

    def test_diamond(self):
        g = Graph()

        @g.node
        def source(x):
            return x

        @g.node
        def left(source):
            return source + 10

        @g.node
        def right(source):
            return source * 2

        @g.node
        def merge(left, right):
            return left + right

        assert g.demand("merge", x=5) == 25

    def test_unknown_node_raises(self):
        g = Graph()
        with pytest.raises(KeyError, match="unknown node"):
            g.demand("nonexistent")

    def test_missing_input_raises(self):
        g = Graph()

        @g.node
        def needs_input(x, y):
            return x + y

        with pytest.raises(TypeError, match="not provided"):
            g.demand("needs_input", x=1)


class TestMemoization:
    def test_cache_hit_skips_compute(self):
        g = Graph()
        call_count = 0

        @g.node
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        g.demand("expensive", x=4)
        assert call_count == 1
        g.demand("expensive", x=4)
        assert call_count == 1

    def test_different_input_recomputes(self):
        g = Graph()
        call_count = 0

        @g.node
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        g.demand("expensive", x=4)
        g.demand("expensive", x=5)
        assert call_count == 2

    def test_upstream_change_invalidates_downstream(self):
        g = Graph()
        calls = []

        @g.node
        def step_a(x):
            calls.append("a")
            return x + 1

        @g.node
        def step_b(step_a):
            calls.append("b")
            return step_a * 2

        g.demand("step_b", x=1)
        assert calls == ["a", "b"]

        calls.clear()
        g.demand("step_b", x=1)
        assert calls == []

        calls.clear()
        g.demand("step_b", x=2)
        assert calls == ["a", "b"]


class TestCachePolicy:
    def test_accept_uses_cache(self):
        g = Graph()
        call_count = 0

        @g.node
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x

        g.demand("compute", policy=CachePolicy.ACCEPT, x=1)
        g.demand("compute", policy=CachePolicy.ACCEPT, x=1)
        assert call_count == 1

    def test_check_recomputes_and_warns_on_divergence(self, caplog):
        g = Graph()
        counter = 0

        @g.node
        def impure(x):
            nonlocal counter
            counter += 1
            return x + counter

        g.demand("impure", policy=CachePolicy.ACCEPT, x=0)

        with caplog.at_level(logging.WARNING):
            result = g.demand("impure", policy=CachePolicy.CHECK, x=0)

        assert result == 2
        assert "CHECK divergence" in caplog.text

    def test_check_no_warning_when_reproducible(self, caplog):
        g = Graph()

        @g.node
        def pure(x):
            return x * 2

        g.demand("pure", policy=CachePolicy.ACCEPT, x=3)
        with caplog.at_level(logging.WARNING):
            g.demand("pure", policy=CachePolicy.CHECK, x=3)
        assert "divergence" not in caplog.text

    def test_fresh_recomputes_target_only(self):
        g = Graph()
        calls = []

        @g.node
        def upstream(x):
            calls.append("upstream")
            return x

        @g.node
        def target(upstream):
            calls.append("target")
            return upstream + 1

        g.demand("target", x=1)
        calls.clear()

        g.demand("target", policy=CachePolicy.FRESH, x=1)
        assert calls == ["target"]

    def test_force_recomputes_everything(self):
        g = Graph()
        calls = []

        @g.node
        def upstream(x):
            calls.append("upstream")
            return x

        @g.node
        def target(upstream):
            calls.append("target")
            return upstream + 1

        g.demand("target", x=1)
        calls.clear()

        g.demand("target", policy=CachePolicy.FORCE, x=1)
        assert calls == ["upstream", "target"]


class TestCycleDetection:
    def test_cycle_raises(self):
        g = Graph()

        @g.node
        def a(b):
            return b

        @g.node
        def b(a):
            return a

        with pytest.raises(CycleError, match="cycle"):
            g.demand("a", x=1)


class TestCustomName:
    def test_node_with_custom_name(self):
        g = Graph()

        @g.node(name="my_node")
        def some_function(x):
            return x + 1

        assert g.demand("my_node", x=5) == 6


class TestContentHash:
    def test_dict_input_order_independent(self):
        g = Graph()
        call_count = 0

        @g.node
        def f(config):
            nonlocal call_count
            call_count += 1
            return sum(config.values())

        g.demand("f", config={"a": 1, "b": 2})
        g.demand("f", config={"b": 2, "a": 1})
        assert call_count == 1

    def test_custom_content_hash(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

            def __content_hash__(self):
                return "fixed_hash"

        g = Graph()
        call_count = 0

        @g.node
        def f(obj):
            nonlocal call_count
            call_count += 1
            return obj.val

        g.demand("f", obj=MyObj(1))
        g.demand("f", obj=MyObj(999))
        assert call_count == 1
