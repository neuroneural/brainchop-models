use bcmodel_core::parser::BcmodelFile;
use wasm_bindgen::prelude::*;

/// A parsed .bcmodel file handle for JavaScript.
#[wasm_bindgen]
pub struct WasmBcmodel {
    inner: BcmodelFile,
}

#[wasm_bindgen]
impl WasmBcmodel {
    /// Parse a .bcmodel file from a Uint8Array or ArrayBuffer.
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Result<WasmBcmodel, JsError> {
        let inner =
            BcmodelFile::from_bytes(data).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Full JSON header as a JS object.
    #[wasm_bindgen(getter)]
    pub fn header(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.header())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Model metadata as a JS object.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.metadata())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Model name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.metadata().name.clone()
    }

    /// Model type (e.g. "brain-extraction", "tissue-segmentation", "parcellation").
    #[wasm_bindgen(getter, js_name = "modelType")]
    pub fn model_type(&self) -> String {
        self.inner.metadata().model_type.clone()
    }

    /// Number of output segmentation classes.
    #[wasm_bindgen(getter, js_name = "numClasses")]
    pub fn num_classes(&self) -> usize {
        self.inner.output_spec().num_classes
    }

    /// Input tensor shape as an array.
    #[wasm_bindgen(getter, js_name = "inputShape")]
    pub fn input_shape(&self) -> Vec<usize> {
        self.inner.input_spec().shape.clone()
    }

    /// Number of graph nodes.
    #[wasm_bindgen(getter, js_name = "graphLength")]
    pub fn graph_length(&self) -> usize {
        self.inner.graph().len()
    }

    /// Architecture graph as a JS array of node objects.
    #[wasm_bindgen(getter)]
    pub fn graph(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.graph())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Labels as a JS array of {index, name, color} objects.
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.labels())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Inference configuration as a JS object.
    #[wasm_bindgen(getter, js_name = "inferenceConfig")]
    pub fn inference_config(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.inference_config())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Pipeline configuration as a JS object.
    #[wasm_bindgen(getter, js_name = "pipelineConfig")]
    pub fn pipeline_config(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.pipeline_config())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Performance hints as a JS object.
    #[wasm_bindgen(getter, js_name = "performanceHints")]
    pub fn performance_hints(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(self.inner.performance_hints())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// All tensor names as an array of strings.
    #[wasm_bindgen(js_name = "tensorNames")]
    pub fn tensor_names(&self) -> Vec<String> {
        self.inner
            .tensor_names()
            .into_iter()
            .map(|s| s.to_owned())
            .collect()
    }

    /// Get tensor shape by name. Returns undefined if not found.
    #[wasm_bindgen(js_name = "tensorShape")]
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.inner.get_tensor_shape(name).map(|s| s.to_vec())
    }

    /// Get tensor data as a Float32Array view into WASM memory.
    ///
    /// Note: The returned view is invalidated if WASM memory grows.
    /// Copy with `new Float32Array(view)` if you need to retain the data.
    #[wasm_bindgen(js_name = "tensorData")]
    pub fn tensor_data(&self, name: &str) -> Option<js_sys::Float32Array> {
        self.inner.get_tensor_data(name).map(|data| {
            unsafe { js_sys::Float32Array::view(data) }
        })
    }

    /// Total parameter count across all tensors.
    #[wasm_bindgen(getter, js_name = "totalParams")]
    pub fn total_params(&self) -> usize {
        self.inner.total_params()
    }

    /// Total weight data size in bytes.
    #[wasm_bindgen(getter, js_name = "weightSizeBytes")]
    pub fn weight_size_bytes(&self) -> usize {
        self.inner.weight_size_bytes()
    }
}
