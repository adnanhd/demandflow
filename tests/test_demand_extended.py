"""Extended tests for DemandGraph: topic resolution, flow mode, recurrence,
triggers, error handling, and MemoStore."""

import threading

import pytest

from demandflow.demand import DemandGraph
from demandflow.memo import MemoStore
from demandflow.policy import CachePolicy


class TestTopicResolution:
    """demand() resolves inputs by topic name, not just node name."""

    def test_node_name_differs_from_topic(self):
        g = DemandGraph()
        g.node("preprocessor", inputs={"x": "x"},
               fn=lambda x: x * 2, outs=["cleaned"])
        g.node("consumer", inputs={"cleaned": "cleaned"},
               fn=lambda cleaned: cleaned + 1, outs=["result"])
        assert g.demand("consumer", x=5) == 11

    def test_demand_by_topic_name(self):
        g = DemandGraph()
        g.node("producer", inputs={"x": "x"},
               fn=lambda x: x + 10, outs=["output"])
        assert g.demand("output", x=3) == 13

    def test_multi_hop_topic_chain(self):
        g = DemandGraph()
        g.node("a", inputs={"x": "x"}, fn=lambda x: x + 1, outs=["a_out"])
        g.node("b", inputs={"a_out": "a_out"}, fn=lambda a_out: a_out * 2, outs=["b_out"])
        g.node("c", inputs={"b_out": "b_out"}, fn=lambda b_out: b_out - 3, outs=["c_out"])
        assert g.demand("c_out", x=4) == 7  # (4+1)*2 - 3


class TestTickMode:
    """Source-driven streaming via subscribe + flow."""

    def test_basic_tick(self):
        g = DemandGraph()
        data = iter([10, 20, 30])
        g.source("src", outs=["val"], pull=lambda: next(data))
        g.node("double", inputs={"val": "val"},
               fn=lambda val: val * 2, outs=["result"])

        results = []
        g.subscribe("result", lambda v: results.append(v))
        while g.flow():
            pass
        assert results == [20, 40, 60]

    def test_unsubscribed_branch_dormant(self):
        g = DemandGraph()
        data = iter([1, 2])
        g.source("src", outs=["val"], pull=lambda: next(data))
        calls = {"a": 0, "b": 0}
        g.node("a", inputs={"val": "val"},
               fn=lambda val: (calls.__setitem__("a", calls["a"] + 1), val)[-1],
               outs=["a_out"])
        g.node("b", inputs={"val": "val"},
               fn=lambda val: (calls.__setitem__("b", calls["b"] + 1), val)[-1],
               outs=["b_out"])

        g.subscribe("a_out", lambda v: None)
        while g.flow():
            pass
        assert calls["a"] == 2
        assert calls["b"] == 0  # unsubscribed, never fired

    def test_subscribe_unsubscribe(self):
        g = DemandGraph()
        data = iter([1, 2, 3])
        g.source("src", outs=["val"], pull=lambda: next(data))
        g.node("id", inputs={"val": "val"}, fn=lambda val: val, outs=["out"])

        results = []
        sub = g.subscribe("out", lambda v: results.append(v))
        g.flow()
        g.unsubscribe(sub)
        g.flow()  # should not fire subscriber
        assert results == [1]


class TestRecurrence:
    """Iterative computation with feedback."""

    def test_simple_recurrence(self):
        g = DemandGraph()
        data = iter([1])
        g.source("init", outs=["state"], pull=lambda: next(data))
        g.recurrence(
            "counter",
            inputs={"state": "state"},
            fn=lambda state: state + 1,
            iterations=5,
            feedback={"counter_out": "state"},
            out="counter_out",
        )
        results = []
        g.subscribe("counter_out", lambda v: results.append(v))
        g.flow()
        g.run_recurrences()
        assert results[-1] == 6  # 1 + 5 iterations

    def test_recurrence_with_memo(self):
        g = DemandGraph()
        data = iter([0])
        g.source("init", outs=["state"], pull=lambda: next(data))
        compute_count = 0

        def step(state):
            nonlocal compute_count
            compute_count += 1
            return state + 1

        g.recurrence(
            "stepper",
            inputs={"state": "state"},
            fn=step,
            iterations=3,
            feedback={"step_out": "state"},
            out="step_out",
            memo=lambda iteration, kwargs: "key_%d" % iteration,
            memo_load=lambda key: int(key.split("_")[1]) + 1,
            memo_save=lambda key, val: None,
        )
        results = []
        g.subscribe("step_out", lambda v: results.append(v))
        g.flow()
        g.run_recurrences()
        # memo_load returns iteration+1 for each, so fn never runs
        assert compute_count == 0


class TestTriggers:
    """Nodes with event triggers fire only when triggered."""

    def test_trigger_guards_execution(self):
        g = DemandGraph()
        data = iter([10, 20])
        g.source("src", outs=["val"], pull=lambda: next(data))
        fired = []
        g.node("guarded", inputs={"val": "val"}, triggers=["go"],
               fn=lambda val: (fired.append(val), val)[-1], outs=["out"])
        g.subscribe("out", lambda v: None)

        g.flow()  # source pulls val=10, but trigger not fired
        assert fired == []

        g.inject("go", True)
        # node should fire now that trigger is set and input is fresh
        assert len(fired) == 1


class TestMemoStore:
    def test_get_miss(self):
        m = MemoStore()
        hit, val = m.get("missing")
        assert hit is False
        assert val is None

    def test_put_and_get(self):
        m = MemoStore()
        m.put("key1", 42)
        hit, val = m.get("key1")
        assert hit is True
        assert val == 42

    def test_overwrite(self):
        m = MemoStore()
        m.put("k", 1)
        m.put("k", 2)
        _, val = m.get("k")
        assert val == 2

    def test_clear(self):
        m = MemoStore()
        m.put("a", 1)
        m.put("b", 2)
        m.clear()
        assert not m.has("a")
        assert not m.has("b")

    def test_keys(self):
        m = MemoStore()
        m.put("x", 1)
        m.put("y", 2)
        assert sorted(m.keys()) == ["x", "y"]


class TestDemandErrors:
    def test_unknown_target(self):
        g = DemandGraph()
        with pytest.raises(KeyError, match="unknown"):
            g.demand("nope")

    def test_missing_input_kwarg(self):
        g = DemandGraph()

        @g.compute
        def needs_x(x):
            return x

        with pytest.raises(TypeError, match="requires input"):
            g.demand("needs_x")

    def test_cycle_detection(self):
        g = DemandGraph()
        g.node("a", inputs={"b": "b"}, fn=lambda b: b, outs=["a"])
        g.node("b", inputs={"a": "a"}, fn=lambda a: a, outs=["b"])
        with pytest.raises(ValueError, match="cycle"):
            g.demand("a", b=1)

    def test_duplicate_topic_producer(self):
        g = DemandGraph()
        g.node("p1", inputs={"x": "x"}, fn=lambda x: x, outs=["shared"])
        with pytest.raises(ValueError, match="already produced"):
            g.node("p2", inputs={"x": "x"}, fn=lambda x: x, outs=["shared"])


class TestCachePolicy:
    def test_accept_uses_cache(self):
        g = DemandGraph()
        count = 0

        @g.compute
        def fn(x):
            nonlocal count
            count += 1
            return x

        g.demand("fn", x=1)
        assert count == 1
        g.demand("fn", policy=CachePolicy(CachePolicy.ACCEPT), x=1)
        assert count == 1  # cache hit, no recompute

    def test_force_recomputes(self):
        g = DemandGraph()
        count = 0

        @g.compute
        def fn(x):
            nonlocal count
            count += 1
            return x

        g.demand("fn", x=1)
        assert count == 1
        g.demand("fn", policy=CachePolicy(CachePolicy.FORCE), x=1)
        assert count == 2

    def test_fresh_recomputes(self):
        g = DemandGraph()
        count = 0

        @g.compute
        def fn(x):
            nonlocal count
            count += 1
            return x

        g.demand("fn", x=1)
        assert count == 1
        g.demand("fn", policy=CachePolicy(CachePolicy.FRESH), x=1)
        assert count == 2

    def test_different_inputs_recompute(self):
        g = DemandGraph()
        count = 0

        @g.compute
        def fn(x):
            nonlocal count
            count += 1
            return x * 2

        assert g.demand("fn", x=1) == 2
        assert g.demand("fn", x=2) == 4
        assert count == 2
