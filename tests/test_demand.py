import pytest

from demandflow.demand import DemandGraph


class TestTopicSchema:
    def test_publish_validates_topic(self):
        from pydantic import BaseModel

        class TensorMeta(BaseModel):
            shape: list[int]
            dtype: str

        g = DemandGraph()
        g.source("src", pull=lambda: {"shape": [2, 3], "dtype": "float32"}, out="tensor_meta")
        g.topic_schema("tensor_meta", TensorMeta)
        g.inject("tensor_meta", {"shape": [2, 3], "dtype": "float32"})
        result = g.serialize_topic("tensor_meta")
        assert result == {"shape": [2, 3], "dtype": "float32"}

    def test_publish_rejects_invalid(self):
        from pydantic import BaseModel, ValidationError

        class TensorMeta(BaseModel):
            shape: list[int]

        g = DemandGraph()
        g.topic_schema("tensor_meta", TensorMeta)
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
