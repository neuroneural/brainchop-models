from typing import Optional
import numpy as np
import numpy.typing as npt

class BcmodelFile:
    """A parsed .bcmodel file."""

    @staticmethod
    def load(path: str) -> "BcmodelFile":
        """Load from a file path."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> "BcmodelFile":
        """Parse from bytes."""
        ...

    @property
    def header(self) -> dict:
        """Full header as a Python dict."""
        ...

    @property
    def name(self) -> str:
        """Model name."""
        ...

    @property
    def model_type(self) -> str:
        """Model type (e.g. 'brain-extraction', 'tissue-segmentation')."""
        ...

    @property
    def description(self) -> str:
        """Model description."""
        ...

    @property
    def num_classes(self) -> int:
        """Number of output segmentation classes."""
        ...

    @property
    def input_shape(self) -> list[int]:
        """Input tensor shape."""
        ...

    @property
    def graph_length(self) -> int:
        """Number of graph nodes."""
        ...

    @property
    def total_params(self) -> int:
        """Total parameter count across all tensors."""
        ...

    @property
    def weight_size_bytes(self) -> int:
        """Total weight data size in bytes."""
        ...

    def tensor_names(self) -> list[str]:
        """Get all tensor names."""
        ...

    def tensor(self, name: str) -> Optional[npt.NDArray[np.float32]]:
        """Get tensor data as a 1D numpy float32 array."""
        ...

    def tensor_shape(self, name: str) -> Optional[list[int]]:
        """Get tensor shape by name."""
        ...

    def labels(self) -> list[dict]:
        """Get labels as a list of dicts."""
        ...

    def graph(self) -> list[dict]:
        """Get graph as a list of dicts."""
        ...

    def inference_config(self) -> dict:
        """Get inference config as a dict."""
        ...

def load_bcmodel(
    path: str,
) -> tuple[dict, dict[str, npt.NDArray[np.float32]]]:
    """Load a .bcmodel file, returning (header_dict, tensors_dict).

    Compatible with the existing load_bcmodel.py API.
    """
    ...
