use bcmodel_core::parser::BcmodelFile;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A parsed .bcmodel file.
#[pyclass(name = "BcmodelFile")]
struct PyBcmodelFile {
    inner: BcmodelFile,
}

#[pymethods]
impl PyBcmodelFile {
    /// Load from a file path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = BcmodelFile::from_path(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Parse from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = BcmodelFile::from_bytes(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Full header as a Python dict.
    #[getter]
    fn header(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = serde_json::to_string(self.inner.header())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    /// Model name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.metadata().name
    }

    /// Model type (e.g. "brain-extraction", "tissue-segmentation").
    #[getter]
    fn model_type(&self) -> &str {
        &self.inner.metadata().model_type
    }

    /// Model description.
    #[getter]
    fn description(&self) -> &str {
        &self.inner.metadata().description
    }

    /// Number of output segmentation classes.
    #[getter]
    fn num_classes(&self) -> usize {
        self.inner.output_spec().num_classes
    }

    /// Input tensor shape.
    #[getter]
    fn input_shape(&self) -> Vec<usize> {
        self.inner.input_spec().shape.clone()
    }

    /// Number of graph nodes.
    #[getter]
    fn graph_length(&self) -> usize {
        self.inner.graph().len()
    }

    /// Total parameter count across all tensors.
    #[getter]
    fn total_params(&self) -> usize {
        self.inner.total_params()
    }

    /// Total weight data size in bytes.
    #[getter]
    fn weight_size_bytes(&self) -> usize {
        self.inner.weight_size_bytes()
    }

    /// Get all tensor names.
    fn tensor_names(&self) -> Vec<String> {
        self.inner
            .tensor_names()
            .into_iter()
            .map(|s| s.to_owned())
            .collect()
    }

    /// Get tensor data as a 1D numpy float32 array. Reshape using tensor_shape().
    fn tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        match self.inner.get_tensor_data(name) {
            Some(data) => Ok(Some(PyArray1::from_slice(py, data))),
            None => Ok(None),
        }
    }

    /// Get tensor shape by name.
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.inner.get_tensor_shape(name).map(|s| s.to_vec())
    }

    /// Get labels as a list of dicts.
    fn labels(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = serde_json::to_string(self.inner.labels())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    /// Get graph as a list of dicts.
    fn graph(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = serde_json::to_string(self.inner.graph())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    /// Get inference config as a dict.
    fn inference_config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = serde_json::to_string(self.inner.inference_config())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "BcmodelFile(name='{}', type='{}', classes={}, params={}, graph_nodes={})",
            self.inner.metadata().name,
            self.inner.metadata().model_type,
            self.inner.output_spec().num_classes,
            self.inner.total_params(),
            self.inner.graph().len(),
        )
    }
}

/// Load a .bcmodel file, returning (header_dict, tensors_dict).
///
/// Compatible with the existing load_bcmodel.py API.
#[pyfunction]
fn load_bcmodel(py: Python<'_>, path: &str) -> PyResult<(PyObject, PyObject)> {
    let file = PyBcmodelFile::load(path)?;
    let header = file.header(py)?;

    let tensors_dict = PyDict::new(py);
    for name in file.inner.tensor_names() {
        if let Some(data) = file.inner.get_tensor_data(name) {
            let shape = file.inner.get_tensor_shape(name).unwrap();
            let arr = PyArray1::from_slice(py, data);
            let np = py.import("numpy")?;
            let reshaped = np.call_method1("reshape", (arr, shape.to_vec()))?;
            tensors_dict.set_item(name, reshaped)?;
        }
    }

    Ok((header, tensors_dict.into()))
}

#[pymodule]
fn bcmodel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBcmodelFile>()?;
    m.add_function(wrap_pyfunction!(load_bcmodel, m)?)?;
    Ok(())
}
