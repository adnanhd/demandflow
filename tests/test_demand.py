import pytest

from demandflow.demand import DemandGraph
from demandflow.qos import QoS


class TestTopicSchema:
    def test_publish_validates_topic(self):
        from pydantic import BaseModel

        class TensorMeta(BaseModel):
            shape: list[int]
            dtype: str

        g = DemandGraph()
        g.topic("tensor_meta", schema=TensorMeta)
        g.inject("tensor_meta", {"shape": [2, 3], "dtype": "float32"})
        result = g.serialize_topic("tensor_meta")
        assert result == {"shape": [2, 3], "dtype": "float32"}

    def test_publish_rejects_invalid(self):
        from pydantic import BaseModel, ValidationError

        class TensorMeta(BaseModel):
            shape: list[int]

        g = DemandGraph()
        g.topic("tensor_meta", schema=TensorMeta)
        with pytest.raises(ValidationError):
            g.inject("tensor_meta", {"shape": "not_a_list"})

    def test_no_schema_skips_validation(self):
        g = DemandGraph()
        g.inject("out", "anything")

    def test_serialize_without_schema_raises(self):
        g = DemandGraph()
        g.inject("out", 42)
        with pytest.raises(TypeError):
            g.serialize_topic("out")

    def test_topic_with_qos(self):
        from pydantic import BaseModel

        class Msg(BaseModel):
            value: int

        g = DemandGraph()
        qos = QoS(reliability="best_effort", deadline=1.0)
        g.topic("sensor", schema=Msg, qos=qos)
        g.inject("sensor", {"value": 42})
        assert g._qos["sensor"] is qos
        assert g._schemas["sensor"] is Msg

    def test_topic_schema_compat(self):
        """topic_schema() still works as alias."""
        from pydantic import BaseModel, ValidationError

        class Epoch(BaseModel):
            value: int

        g = DemandGraph()
        g.topic_schema("epoch", Epoch)
        g.inject("epoch", {"value": 5})
        with pytest.raises(ValidationError):
            g.inject("epoch", {"value": "bad"})
