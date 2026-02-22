"""Tests for the bcmodel Python bindings."""

import struct
import json
import pytest
import numpy as np

import bcmodel
from bcmodel import BcmodelFile


def make_bcmodel(header_json: str, tensor_data: bytes = b"") -> bytes:
    """Build a .bcmodel file from a JSON header string and raw tensor bytes."""
    header_bytes = header_json.encode("utf-8")
    header_size = len(header_bytes)
    return struct.pack("<Q", header_size) + header_bytes + tensor_data


def minimal_header(tensor_bytes: int = 8) -> str:
    return json.dumps({
        "bcmodel_version": "1.0",
        "metadata": {"name": "Test", "type": "brain-extraction"},
        "input": {"shape": [1, 1, 4, 4, 4], "dtype": "float32", "data_layout": "channels_first"},
        "output": {"num_classes": 2},
        "graph": [{"id": "conv0", "op": "conv3d", "params": {}, "inputs": []}],
        "tensors": {
            "conv0.weight": {"dtype": "float32", "shape": [2], "data_offsets": [0, tensor_bytes]}
        },
        "labels": [{"index": 0, "name": "bg", "color": [0, 0, 0]}],
    })


def make_tensor_floats(*values: float) -> bytes:
    """Pack float32 values into little-endian bytes."""
    return struct.pack(f"<{len(values)}f", *values)


# ---------------------------------------------------------------------------
# BcmodelFile.from_bytes — basic parsing
# ---------------------------------------------------------------------------

class TestFromBytes:
    def test_parse_minimal(self):
        data = make_bcmodel(minimal_header(8), b"\x00" * 8)
        model = BcmodelFile.from_bytes(data)
        assert model.name == "Test"
        assert model.model_type == "brain-extraction"
        assert model.num_classes == 2
        assert model.input_shape == [1, 1, 4, 4, 4]
        assert model.graph_length == 1
        assert model.total_params == 2
        assert model.weight_size_bytes == 8

    def test_tensor_values(self):
        floats = (1.0, -2.5, 3.14)
        tensor_data = make_tensor_floats(*floats)
        header = json.dumps({
            "bcmodel_version": "1.0",
            "metadata": {"name": "T", "type": "brain-extraction"},
            "input": {"shape": [1, 1, 1, 1, 1], "dtype": "float32", "data_layout": "channels_first"},
            "output": {"num_classes": 2},
            "graph": [{"id": "w", "op": "relu", "params": {}, "inputs": []}],
            "tensors": {"w.weight": {"dtype": "float32", "shape": [3], "data_offsets": [0, 12]}},
            "labels": [],
        })
        model = BcmodelFile.from_bytes(make_bcmodel(header, tensor_data))
        arr = model.tensor("w.weight")
        assert arr is not None
        assert arr.dtype == np.float32
        assert len(arr) == 3
        np.testing.assert_allclose(arr, floats, rtol=1e-6)

    def test_missing_tensor_returns_none(self):
        data = make_bcmodel(minimal_header(8), b"\x00" * 8)
        model = BcmodelFile.from_bytes(data)
        assert model.tensor("nonexistent") is None
        assert model.tensor_shape("nonexistent") is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_file_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            BcmodelFile.from_bytes(b"\x00\x01\x02")

    def test_header_overflow(self):
        buf = struct.pack("<Q", 9999) + b"\x00" * 8
        with pytest.raises(ValueError):
            BcmodelFile.from_bytes(buf)

    def test_tensor_out_of_bounds(self):
        header = json.dumps({
            "bcmodel_version": "1.0",
            "metadata": {"name": "T", "type": "brain-extraction"},
            "input": {"shape": [1, 1, 1, 1, 1], "dtype": "float32", "data_layout": "channels_first"},
            "output": {"num_classes": 2},
            "graph": [],
            "tensors": {"bad.weight": {"dtype": "float32", "shape": [100], "data_offsets": [0, 400]}},
            "labels": [],
        })
        with pytest.raises(ValueError, match="exceeds"):
            BcmodelFile.from_bytes(make_bcmodel(header, b"\x00" * 8))

    def test_invalid_json(self):
        buf = struct.pack("<Q", 5) + b"hello"
        with pytest.raises(ValueError):
            BcmodelFile.from_bytes(buf)

    def test_load_nonexistent_file(self):
        with pytest.raises(ValueError):
            BcmodelFile.load("/nonexistent/path/model.bcmodel")


# ---------------------------------------------------------------------------
# Properties and methods
# ---------------------------------------------------------------------------

class TestProperties:
    @pytest.fixture()
    def model(self):
        header = json.dumps({
            "bcmodel_version": "1.0",
            "metadata": {
                "name": "MyModel",
                "type": "tissue-segmentation",
                "description": "A test model",
            },
            "input": {"shape": [1, 1, 8, 8, 8], "dtype": "float32", "data_layout": "channels_first"},
            "output": {"num_classes": 3},
            "graph": [
                {"id": "conv0", "op": "conv3d", "params": {"kernel_size": 3}, "inputs": []},
                {"id": "relu0", "op": "relu", "params": {}, "inputs": ["conv0"]},
            ],
            "tensors": {
                "conv0.weight": {"dtype": "float32", "shape": [4, 2], "data_offsets": [0, 32]},
                "conv0.bias": {"dtype": "float32", "shape": [4], "data_offsets": [32, 48]},
            },
            "labels": [
                {"index": 0, "name": "bg", "color": [0, 0, 0]},
                {"index": 1, "name": "gm", "color": [255, 0, 0]},
                {"index": 2, "name": "wm", "color": [0, 255, 0]},
            ],
            "inference": {"enable_seq_conv": True, "crop_padding": 4},
        })
        return BcmodelFile.from_bytes(make_bcmodel(header, b"\x00" * 48))

    def test_name(self, model):
        assert model.name == "MyModel"

    def test_model_type(self, model):
        assert model.model_type == "tissue-segmentation"

    def test_description(self, model):
        assert model.description == "A test model"

    def test_num_classes(self, model):
        assert model.num_classes == 3

    def test_input_shape(self, model):
        assert model.input_shape == [1, 1, 8, 8, 8]

    def test_graph_length(self, model):
        assert model.graph_length == 2

    def test_total_params(self, model):
        # 4*2 + 4 = 12
        assert model.total_params == 12

    def test_weight_size_bytes(self, model):
        assert model.weight_size_bytes == 48

    def test_tensor_names(self, model):
        names = model.tensor_names()
        assert set(names) == {"conv0.weight", "conv0.bias"}

    def test_tensor_shape(self, model):
        assert model.tensor_shape("conv0.weight") == [4, 2]
        assert model.tensor_shape("conv0.bias") == [4]

    def test_tensor_returns_numpy_float32(self, model):
        arr = model.tensor("conv0.weight")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (8,)  # returned flat; user reshapes

    def test_labels(self, model):
        labels = model.labels()
        assert isinstance(labels, list)
        assert len(labels) == 3
        assert labels[0]["name"] == "bg"
        assert labels[1]["color"] == [255, 0, 0]

    def test_graph(self, model):
        graph = model.graph()
        assert isinstance(graph, list)
        assert len(graph) == 2
        assert graph[0]["op"] == "conv3d"
        assert graph[1]["inputs"] == ["conv0"]

    def test_inference_config(self, model):
        cfg = model.inference_config()
        assert isinstance(cfg, dict)
        assert cfg["enable_seq_conv"] is True
        assert cfg["crop_padding"] == 4

    def test_header_is_dict(self, model):
        h = model.header
        assert isinstance(h, dict)
        assert h["bcmodel_version"] == "1.0"
        assert h["metadata"]["name"] == "MyModel"

    def test_repr(self, model):
        r = repr(model)
        assert "MyModel" in r
        assert "tissue-segmentation" in r


# ---------------------------------------------------------------------------
# load_bcmodel() compatibility function
# ---------------------------------------------------------------------------

class TestLoadBcmodel:
    def test_load_bcmodel_returns_tuple(self, tmp_path):
        floats = (1.0, 2.0, 3.0, 4.0)
        tensor_data = make_tensor_floats(*floats)
        header = json.dumps({
            "bcmodel_version": "1.0",
            "metadata": {"name": "Compat", "type": "brain-extraction"},
            "input": {"shape": [1, 1, 2, 2, 2], "dtype": "float32", "data_layout": "channels_first"},
            "output": {"num_classes": 2},
            "graph": [{"id": "c", "op": "conv3d", "params": {}, "inputs": []}],
            "tensors": {"c.weight": {"dtype": "float32", "shape": [2, 2], "data_offsets": [0, 16]}},
            "labels": [],
        })
        path = tmp_path / "model.bcmodel"
        path.write_bytes(make_bcmodel(header, tensor_data))

        header_dict, tensors = bcmodel.load_bcmodel(str(path))

        assert isinstance(header_dict, dict)
        assert header_dict["metadata"]["name"] == "Compat"

        assert isinstance(tensors, dict)
        assert "c.weight" in tensors
        arr = tensors["c.weight"]
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr.flatten(), floats)


# ---------------------------------------------------------------------------
# BcmodelFile.load() from file
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_load_from_path(self, tmp_path):
        data = make_bcmodel(minimal_header(8), b"\x00" * 8)
        path = tmp_path / "test.bcmodel"
        path.write_bytes(data)

        model = BcmodelFile.load(str(path))
        assert model.name == "Test"
        assert model.total_params == 2


# ---------------------------------------------------------------------------
# Optional / default fields
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_optional_fields_have_defaults(self):
        header = json.dumps({
            "bcmodel_version": "1.0",
            "metadata": {"name": "T", "type": "brain-extraction"},
            "input": {"shape": [1, 1, 1, 1, 1], "dtype": "float32", "data_layout": "channels_first"},
            "output": {"num_classes": 2},
            "graph": [],
            "tensors": {},
            "labels": [],
        })
        model = BcmodelFile.from_bytes(make_bcmodel(header))
        cfg = model.inference_config()
        assert cfg["enable_seq_conv"] is False
        assert cfg["crop_padding"] == 0
        assert cfg["enable_transpose"] is True


# ---------------------------------------------------------------------------
# Real .bcmodel file integration test
# ---------------------------------------------------------------------------

class TestRealFile:
    REAL_MODEL = "meshnet/model5_gw_ae/model.bcmodel"

    @pytest.fixture()
    def real_model_path(self):
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base, self.REAL_MODEL)
        if not os.path.exists(path):
            pytest.skip(f"Real model not found: {path}")
        return path

    def test_load_real_model(self, real_model_path):
        model = BcmodelFile.load(real_model_path)
        assert model.name
        assert model.model_type == "tissue-segmentation"
        assert model.num_classes == 3
        assert model.graph_length > 0
        assert model.total_params > 0

    def test_real_model_tensors(self, real_model_path):
        model = BcmodelFile.load(real_model_path)
        names = model.tensor_names()
        assert len(names) > 0
        for name in names:
            arr = model.tensor(name)
            shape = model.tensor_shape(name)
            assert arr is not None
            assert shape is not None
            expected_len = 1
            for s in shape:
                expected_len *= s
            assert len(arr) == expected_len

    def test_real_model_labels(self, real_model_path):
        model = BcmodelFile.load(real_model_path)
        labels = model.labels()
        assert len(labels) > 0
        for label in labels:
            assert "index" in label
            assert "name" in label
            assert "color" in label
            assert len(label["color"]) == 3

    def test_load_bcmodel_compat(self, real_model_path):
        header, tensors = bcmodel.load_bcmodel(real_model_path)
        assert header["bcmodel_version"] == "1.0"
        assert len(tensors) > 0
        for name, arr in tensors.items():
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float32
