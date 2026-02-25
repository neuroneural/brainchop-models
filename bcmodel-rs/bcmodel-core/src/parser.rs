use crate::error::BcmodelError;
use crate::types::*;

/// A parsed .bcmodel file. Owns the raw bytes for zero-copy tensor access.
pub struct BcmodelFile {
    header: Header,
    data: Vec<u8>,
}

impl BcmodelFile {
    /// Parse from an in-memory byte buffer (the entire .bcmodel file contents).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BcmodelError> {
        if bytes.len() < 8 {
            return Err(BcmodelError::FileTooSmall(bytes.len()));
        }

        let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let header_end = 8u64
            .checked_add(header_size)
            .ok_or(BcmodelError::HeaderOverflow {
                header_size,
                data_len: bytes.len(),
            })? as usize;

        if header_end > bytes.len() {
            return Err(BcmodelError::HeaderOverflow {
                header_size,
                data_len: bytes.len(),
            });
        }

        let header_str = std::str::from_utf8(&bytes[8..header_end])?;
        let header: Header = serde_json::from_str(header_str)?;

        let data = bytes[header_end..].to_vec();

        // Validate tensor offsets
        for (name, info) in &header.tensors {
            let [begin, end] = info.data_offsets;
            if end > data.len() {
                return Err(BcmodelError::TensorOutOfBounds {
                    name: name.clone(),
                    begin,
                    end,
                    data_len: data.len(),
                });
            }
            let size = end - begin;
            if size % 4 != 0 {
                return Err(BcmodelError::TensorAlignment {
                    name: name.clone(),
                    size,
                });
            }
        }

        Ok(Self { header, data })
    }

    /// Parse from a file path (not available in WASM).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_path(path: &std::path::Path) -> Result<Self, BcmodelError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn metadata(&self) -> &Metadata {
        &self.header.metadata
    }

    pub fn input_spec(&self) -> &InputSpec {
        &self.header.input
    }

    pub fn output_spec(&self) -> &OutputSpec {
        &self.header.output
    }

    pub fn graph(&self) -> &[GraphNode] {
        &self.header.graph
    }

    pub fn labels(&self) -> &[Label] {
        &self.header.labels
    }

    pub fn inference_config(&self) -> &InferenceConfig {
        &self.header.inference
    }

    pub fn pipeline_config(&self) -> &PipelineConfig {
        &self.header.pipeline
    }

    pub fn performance_hints(&self) -> &PerformanceHints {
        &self.header.performance
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }

    /// Zero-copy access to raw float32 tensor data.
    /// Returns None if the tensor name doesn't exist.
    ///
    /// # Safety note
    /// The pointer cast from &[u8] to &[f32] is sound because:
    /// - The format specifies little-endian float32
    /// - All target platforms (x86, ARM, wasm32) are little-endian
    /// - Alignment and size are validated at parse time
    pub fn get_tensor_data(&self, name: &str) -> Option<&[f32]> {
        let info = self.header.tensors.get(name)?;
        let [begin, end] = info.data_offsets;
        let byte_slice = &self.data[begin..end];
        let float_slice = unsafe {
            std::slice::from_raw_parts(byte_slice.as_ptr() as *const f32, byte_slice.len() / 4)
        };
        Some(float_slice)
    }

    pub fn get_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.header.tensors.get(name).map(|i| i.shape.as_slice())
    }

    /// Total number of parameters across all tensors.
    pub fn total_params(&self) -> usize {
        self.header
            .tensors
            .values()
            .map(|info| info.shape.iter().product::<usize>())
            .sum()
    }

    /// Total weight data size in bytes.
    pub fn weight_size_bytes(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bcmodel(header_json: &str, tensor_data: &[u8]) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(tensor_data);
        buf
    }

    fn minimal_header(tensor_bytes: usize) -> String {
        format!(
            r#"{{
            "bcmodel_version": "1.0",
            "metadata": {{"name": "Test", "type": "brain-extraction"}},
            "input": {{"shape": [1,1,4,4,4], "dtype": "float32", "data_layout": "channels_first"}},
            "output": {{"num_classes": 2}},
            "graph": [{{"id": "conv0", "op": "conv3d", "params": {{}}, "inputs": []}}],
            "tensors": {{
                "conv0.weight": {{"dtype": "float32", "shape": [2], "data_offsets": [0, {}]}}
            }},
            "labels": [{{"index": 0, "name": "bg", "color": [0,0,0]}}]
        }}"#,
            tensor_bytes
        )
    }

    #[test]
    fn parse_minimal() {
        let data: Vec<u8> = vec![0u8; 8]; // 2 floats worth of zeros
        let header = minimal_header(8);
        let buf = make_bcmodel(&header, &data);

        let model = BcmodelFile::from_bytes(&buf).unwrap();
        assert_eq!(model.metadata().name, "Test");
        assert_eq!(model.metadata().model_type, "brain-extraction");
        assert_eq!(model.output_spec().num_classes, 2);
        assert_eq!(model.graph().len(), 1);
        assert_eq!(model.graph()[0].op, "conv3d");
        assert_eq!(model.labels().len(), 1);
        assert_eq!(model.total_params(), 2);
        assert_eq!(model.weight_size_bytes(), 8);

        let tensor = model.get_tensor_data("conv0.weight").unwrap();
        assert_eq!(tensor.len(), 2);
        assert_eq!(tensor[0], 0.0);
    }

    #[test]
    fn parse_tensor_values() {
        let floats: [f32; 3] = [1.0, -2.5, 3.14];
        let mut data = Vec::new();
        for f in &floats {
            data.extend_from_slice(&f.to_le_bytes());
        }

        let header = format!(
            r#"{{
            "bcmodel_version": "1.0",
            "metadata": {{"name": "T", "type": "brain-extraction"}},
            "input": {{"shape": [1,1,1,1,1], "dtype": "float32", "data_layout": "channels_first"}},
            "output": {{"num_classes": 2}},
            "graph": [{{"id": "w", "op": "relu", "params": {{}}, "inputs": []}}],
            "tensors": {{
                "w.weight": {{"dtype": "float32", "shape": [3], "data_offsets": [0, 12]}}
            }},
            "labels": []
        }}"#
        );
        let buf = make_bcmodel(&header, &data);
        let model = BcmodelFile::from_bytes(&buf).unwrap();
        let tensor = model.get_tensor_data("w.weight").unwrap();
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor[1], -2.5);
        assert!((tensor[2] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn error_file_too_small() {
        let result = BcmodelFile::from_bytes(&[0, 1, 2]);
        assert!(matches!(result, Err(BcmodelError::FileTooSmall(3))));
    }

    #[test]
    fn error_header_overflow() {
        // header_size = 9999 but only 10 bytes total
        let mut buf = vec![0u8; 16];
        buf[0..8].copy_from_slice(&9999u64.to_le_bytes());
        let result = BcmodelFile::from_bytes(&buf);
        assert!(matches!(result, Err(BcmodelError::HeaderOverflow { .. })));
    }

    #[test]
    fn error_tensor_out_of_bounds() {
        let header = format!(
            r#"{{
            "bcmodel_version": "1.0",
            "metadata": {{"name": "T", "type": "brain-extraction"}},
            "input": {{"shape": [1,1,1,1,1], "dtype": "float32", "data_layout": "channels_first"}},
            "output": {{"num_classes": 2}},
            "graph": [],
            "tensors": {{
                "bad.weight": {{"dtype": "float32", "shape": [100], "data_offsets": [0, 400]}}
            }},
            "labels": []
        }}"#
        );
        let buf = make_bcmodel(&header, &[0u8; 8]); // only 8 bytes of data
        let result = BcmodelFile::from_bytes(&buf);
        assert!(matches!(result, Err(BcmodelError::TensorOutOfBounds { .. })));
    }

    #[test]
    fn missing_tensor_returns_none() {
        let data = vec![0u8; 8];
        let header = minimal_header(8);
        let buf = make_bcmodel(&header, &data);
        let model = BcmodelFile::from_bytes(&buf).unwrap();
        assert!(model.get_tensor_data("nonexistent").is_none());
        assert!(model.get_tensor_shape("nonexistent").is_none());
    }

    #[test]
    fn optional_fields_default() {
        let header = r#"{
            "bcmodel_version": "1.0",
            "metadata": {"name": "T", "type": "brain-extraction"},
            "input": {"shape": [1,1,1,1,1], "dtype": "float32", "data_layout": "channels_first"},
            "output": {"num_classes": 2},
            "graph": [],
            "tensors": {},
            "labels": []
        }"#;
        let buf = make_bcmodel(header, &[]);
        let model = BcmodelFile::from_bytes(&buf).unwrap();
        // Inference defaults
        assert!(!model.inference_config().enable_seq_conv);
        assert_eq!(model.inference_config().crop_padding, 0);
        assert!(model.inference_config().enable_transpose); // defaults to true
        // Pipeline defaults
        assert!(model.pipeline_config().requires_pre_model.is_none());
        assert!(!model.pipeline_config().filter_with_pre_mask);
        // Performance defaults
        assert_eq!(model.performance_hints().estimated_time_seconds, 0);
    }

    #[test]
    fn load_real_bcmodel_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("meshnet/model5_gw_ae/model.bcmodel");

        if !path.exists() {
            eprintln!("Skipping real file test: {:?} not found", path);
            return;
        }

        let model = BcmodelFile::from_path(&path).unwrap();
        assert_eq!(model.header().bcmodel_version, "1.0");
        assert_eq!(model.metadata().model_type, "tissue-segmentation");
        assert_eq!(model.output_spec().num_classes, 3);
        assert!(model.graph().len() > 0);
        assert!(model.total_params() > 0);
        assert!(model.labels().len() > 0);

        // Verify we can access tensor data
        let names = model.tensor_names();
        assert!(!names.is_empty());
        for name in &names {
            let data = model.get_tensor_data(name).unwrap();
            let shape = model.get_tensor_shape(name).unwrap();
            let expected_len: usize = shape.iter().product();
            assert_eq!(data.len(), expected_len, "tensor {name} length mismatch");
        }
    }
}
