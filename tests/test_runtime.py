import threading
import time

import pytest

import demandflow as od


@pytest.fixture
def ctx():
    c = od.Context()
    yield c
    c.close()


class TestPubSub:
    def test_publish_subscribe(self, ctx):
        received = threading.Event()
        messages = []

        def on_msg(msg):
            messages.append(msg)
            received.set()

        pub = od.Publisher(ctx, "test/greet")
        od.Subscriber(ctx, "test/greet", callback=on_msg)
        time.sleep(0.05)

        pub.publish({"hello": "world"})
        assert received.wait(timeout=2)
        assert messages == [{"hello": "world"}]

    def test_polling_subscription(self, ctx):
        got_data = threading.Event()
        pub = od.Publisher(ctx, "test/poll")

        class PollSub(od.Subscriber):
            pass

        sub = od.Subscriber(ctx, "test/poll")
        # wrap to detect arrival
        orig_deliver = type(sub)._Subscription__on_message if hasattr(type(sub), '_Subscription__on_message') else None
        pub.publish(42)
        time.sleep(0.15)
        assert sub.has_data()
        assert sub.take() == 42
        assert not sub.has_data()

    def test_multiple_subscribers(self, ctx):
        barrier = threading.Barrier(3, timeout=2)
        a_msgs, b_msgs = [], []

        def on_a(m):
            a_msgs.append(m)
            barrier.wait()

        def on_b(m):
            b_msgs.append(m)
            barrier.wait()

        pub = od.Publisher(ctx, "test/multi")
        od.Subscriber(ctx, "test/multi", callback=on_a)
        od.Subscriber(ctx, "test/multi", callback=on_b)
        time.sleep(0.05)

        pub.publish("ping")
        barrier.wait()
        assert a_msgs == ["ping"]
        assert b_msgs == ["ping"]


class TestService:
    def test_request_response(self, ctx):
        od.ServiceServer(ctx, "add", handler=lambda req: req["a"] + req["b"])
        time.sleep(0.05)

        client = od.ServiceClient(ctx, "add")
        result = client.call({"a": 3, "b": 4})
        assert result == 7


class TestAction:
    def test_action_with_feedback(self, ctx):
        feedbacks = []
        result_box = []
        result_done = threading.Event()

        def execute(goal_handle):
            for i in range(3):
                goal_handle.publish_feedback({"step": i})
                time.sleep(0.05)
            return {"total": 3}

        od.ActionServer(ctx, "count", execute_fn=execute)
        time.sleep(0.05)

        client = od.ActionClient(ctx, "count")
        goal_id = client.send_goal(
            None,
            on_feedback=lambda fb: feedbacks.append(fb),
            on_result=lambda r: (result_box.append(r), result_done.set()),
        )
        assert goal_id
        assert result_done.wait(timeout=5)
        assert result_box[0]["status"] == "succeeded"
        assert result_box[0]["result"]["total"] == 3
        assert len(feedbacks) >= 2

    def test_action_cancel(self, ctx):
        cancel_seen = threading.Event()

        def execute(goal_handle):
            while not goal_handle.cancel_requested:
                time.sleep(0.01)
            cancel_seen.set()
            return {"canceled": True}

        od.ActionServer(ctx, "slow", execute_fn=execute)
        time.sleep(0.05)

        client = od.ActionClient(ctx, "slow")
        goal_id = client.send_goal(None)
        time.sleep(0.1)
        assert client.cancel(goal_id)
        assert cancel_seen.wait(timeout=2)


class TestNode:
    def test_node_creates_pub_sub(self, ctx):
        node = od.Node("test_node", context=ctx)
        pub = node.create_publisher("node/out")
        received = threading.Event()
        messages = []

        def on_msg(m):
            messages.append(m)
            received.set()

        node.create_subscription("node/out", callback=on_msg)
        time.sleep(0.05)

        pub.publish("hello")
        assert received.wait(timeout=2)
        assert messages == ["hello"]
        node.destroy()

    def test_node_timer(self, ctx):
        node = od.Node("timer_node", context=ctx)
        count_event = threading.Event()
        count = 0

        def tick():
            nonlocal count
            count += 1
            if count >= 3:
                count_event.set()

        node.create_timer(0.05, tick)
        executor = od.SingleThreadedExecutor()
        executor.add_node(node)

        deadline = time.monotonic() + 2
        while not count_event.is_set() and time.monotonic() < deadline:
            executor.spin_once(timeout_s=0.01)
        assert count >= 3
        node.destroy()

    def test_node_service(self, ctx):
        node = od.Node("srv_node", context=ctx)
        node.create_service("echo", handler=lambda req: req)
        time.sleep(0.05)

        client = node.create_service_client("echo")
        assert client.call("ping") == "ping"
        node.destroy()


class TestLifecycleNode:
    def test_lifecycle_transitions(self, ctx):
        transitions = []

        class MyNode(od.LifecycleNode):
            def on_configure(self):
                transitions.append("configure")

            def on_activate(self):
                transitions.append("activate")

            def on_deactivate(self):
                transitions.append("deactivate")

            def on_cleanup(self):
                transitions.append("cleanup")

            def on_shutdown(self):
                transitions.append("shutdown")

        node = MyNode("lifecycle", context=ctx)
        assert node.state == od.LifecycleState.UNCONFIGURED

        node.configure()
        assert node.state == od.LifecycleState.INACTIVE
        node.activate()
        assert node.state == od.LifecycleState.ACTIVE
        node.deactivate()
        assert node.state == od.LifecycleState.INACTIVE
        node.cleanup()
        assert node.state == od.LifecycleState.UNCONFIGURED
        node.shutdown()
        assert node.state == od.LifecycleState.FINALIZED

        assert transitions == ["configure", "activate", "deactivate", "cleanup", "shutdown"]

    def test_invalid_transition_raises(self, ctx):
        node = od.LifecycleNode("bad", context=ctx)
        with pytest.raises(od.TransitionError, match="cannot activate"):
            node.activate()


class TestQoS:
    def test_compatible(self):
        reliable = od.QoS(reliability="reliable")
        best_effort = od.QoS(reliability="best_effort")
        assert reliable.compatible_with(reliable)
        assert best_effort.compatible_with(best_effort)
        assert reliable.compatible_with(best_effort)
        assert not best_effort.compatible_with(reliable)


class TestSingleThreadedExecutor:
    def test_spin_once_fires_timers(self, ctx):
        node = od.Node("exec_test", context=ctx)
        fired = []
        node.create_timer(0.0, lambda: fired.append(1))

        executor = od.SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin_once()
        assert fired == [1]
        node.destroy()
