use wasm_bindgen_test::*;

/// Build a .bcmodel byte buffer from a JSON header and optional tensor data.
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

fn make_tensor_floats(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    for f in values {
        data.extend_from_slice(&f.to_le_bytes());
    }
    data
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn parse_minimal() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    assert_eq!(model.name(), "Test");
    assert_eq!(model.model_type(), "brain-extraction");
    assert_eq!(model.num_classes(), 2);
    assert_eq!(model.graph_length(), 1);
    assert_eq!(model.total_params(), 2);
    assert_eq!(model.weight_size_bytes(), 8);
}

#[wasm_bindgen_test]
fn input_shape() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    assert_eq!(model.input_shape(), vec![1, 1, 4, 4, 4]);
}

// ---------------------------------------------------------------------------
// Tensor access
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn tensor_values() {
    let floats = [1.0f32, -2.5, 3.14];
    let tensor_data = make_tensor_floats(&floats);
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
    let buf = make_bcmodel(&header, &tensor_data);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();

    let names = model.tensor_names();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "w.weight");

    let shape = model.tensor_shape("w.weight").unwrap();
    assert_eq!(shape, vec![3]);

    let data = model.tensor_data("w.weight").unwrap();
    assert_eq!(data.length(), 3);
    assert_eq!(data.get_index(0), 1.0);
    assert_eq!(data.get_index(1), -2.5);
    assert!((data.get_index(2) - 3.14).abs() < 1e-5);
}

#[wasm_bindgen_test]
fn missing_tensor_returns_none() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    assert!(model.tensor_data("nonexistent").is_none());
    assert!(model.tensor_shape("nonexistent").is_none());
}

#[wasm_bindgen_test]
fn tensor_names_multiple() {
    let header = r#"{
        "bcmodel_version": "1.0",
        "metadata": {"name": "T", "type": "brain-extraction"},
        "input": {"shape": [1,1,1,1,1], "dtype": "float32", "data_layout": "channels_first"},
        "output": {"num_classes": 2},
        "graph": [],
        "tensors": {
            "layer0.weight": {"dtype": "float32", "shape": [2], "data_offsets": [0, 8]},
            "layer0.bias":   {"dtype": "float32", "shape": [1], "data_offsets": [8, 12]}
        },
        "labels": []
    }"#;
    let buf = make_bcmodel(header, &[0u8; 12]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let mut names = model.tensor_names();
    names.sort();
    assert_eq!(names, vec!["layer0.bias", "layer0.weight"]);
    assert_eq!(model.total_params(), 3); // 2 + 1
    assert_eq!(model.weight_size_bytes(), 12);
}

// ---------------------------------------------------------------------------
// Header / metadata getters
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn header_is_js_object() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let header = model.header().unwrap();
    assert!(header.is_object());
}

#[wasm_bindgen_test]
fn metadata_getter() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let meta = model.metadata().unwrap();
    assert!(meta.is_object());
}

#[wasm_bindgen_test]
fn graph_getter() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let graph = model.graph().unwrap();
    // graph is a JS array
    assert!(js_sys::Array::is_array(&graph));
    let arr = js_sys::Array::from(&graph);
    assert_eq!(arr.length(), 1);
}

#[wasm_bindgen_test]
fn labels_getter() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let labels = model.labels().unwrap();
    assert!(js_sys::Array::is_array(&labels));
    let arr = js_sys::Array::from(&labels);
    assert_eq!(arr.length(), 1);
}

#[wasm_bindgen_test]
fn inference_config_getter() {
    let header = r#"{
        "bcmodel_version": "1.0",
        "metadata": {"name": "T", "type": "brain-extraction"},
        "input": {"shape": [1,1,1,1,1], "dtype": "float32", "data_layout": "channels_first"},
        "output": {"num_classes": 2},
        "graph": [],
        "tensors": {},
        "labels": [],
        "inference": {"enable_seq_conv": true, "crop_padding": 4}
    }"#;
    let buf = make_bcmodel(header, &[]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let cfg = model.inference_config().unwrap();
    assert!(cfg.is_object());
}

#[wasm_bindgen_test]
fn pipeline_config_getter() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let cfg = model.pipeline_config().unwrap();
    assert!(cfg.is_object());
}

#[wasm_bindgen_test]
fn performance_hints_getter() {
    let buf = make_bcmodel(&minimal_header(8), &[0u8; 8]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    let hints = model.performance_hints().unwrap();
    assert!(hints.is_object());
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn error_file_too_small() {
    let result = bcmodel_wasm::WasmBcmodel::new(&[0, 1, 2]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn error_header_overflow() {
    let mut buf = vec![0u8; 16];
    buf[0..8].copy_from_slice(&9999u64.to_le_bytes());
    let result = bcmodel_wasm::WasmBcmodel::new(&buf);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn error_tensor_out_of_bounds() {
    let header = r#"{
        "bcmodel_version": "1.0",
        "metadata": {"name": "T", "type": "brain-extraction"},
        "input": {"shape": [1,1,1,1,1], "dtype": "float32", "data_layout": "channels_first"},
        "output": {"num_classes": 2},
        "graph": [],
        "tensors": {
            "bad.weight": {"dtype": "float32", "shape": [100], "data_offsets": [0, 400]}
        },
        "labels": []
    }"#;
    let buf = make_bcmodel(header, &[0u8; 8]);
    let result = bcmodel_wasm::WasmBcmodel::new(&buf);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn error_invalid_json() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&5u64.to_le_bytes());
    buf.extend_from_slice(b"hello");
    let result = bcmodel_wasm::WasmBcmodel::new(&buf);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Optional / default fields
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
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
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();
    // Just verify the model parses successfully with no optional fields
    assert_eq!(model.name(), "T");
    assert_eq!(model.num_classes(), 2);
    assert_eq!(model.total_params(), 0);
    assert_eq!(model.weight_size_bytes(), 0);
}

// ---------------------------------------------------------------------------
// Complex model with multiple tensors and full metadata
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn complex_model() {
    let header = r#"{
        "bcmodel_version": "1.0",
        "metadata": {
            "name": "FullModel",
            "type": "tissue-segmentation",
            "description": "A complex test model"
        },
        "input": {"shape": [1,1,8,8,8], "dtype": "float32", "data_layout": "channels_first"},
        "output": {"num_classes": 3},
        "graph": [
            {"id": "conv0", "op": "conv3d", "params": {"kernel_size": 3}, "inputs": []},
            {"id": "relu0", "op": "relu", "params": {}, "inputs": ["conv0"]}
        ],
        "tensors": {
            "conv0.weight": {"dtype": "float32", "shape": [4, 2], "data_offsets": [0, 32]},
            "conv0.bias":   {"dtype": "float32", "shape": [4],    "data_offsets": [32, 48]}
        },
        "labels": [
            {"index": 0, "name": "bg", "color": [0,0,0]},
            {"index": 1, "name": "gm", "color": [255,0,0]},
            {"index": 2, "name": "wm", "color": [0,255,0]}
        ],
        "inference": {"enable_seq_conv": true, "crop_padding": 4}
    }"#;
    let buf = make_bcmodel(header, &[0u8; 48]);
    let model = bcmodel_wasm::WasmBcmodel::new(&buf).unwrap();

    assert_eq!(model.name(), "FullModel");
    assert_eq!(model.model_type(), "tissue-segmentation");
    assert_eq!(model.num_classes(), 3);
    assert_eq!(model.input_shape(), vec![1, 1, 8, 8, 8]);
    assert_eq!(model.graph_length(), 2);
    assert_eq!(model.total_params(), 12); // 4*2 + 4
    assert_eq!(model.weight_size_bytes(), 48);

    let labels = model.labels().unwrap();
    let labels_arr = js_sys::Array::from(&labels);
    assert_eq!(labels_arr.length(), 3);

    let mut names = model.tensor_names();
    names.sort();
    assert_eq!(names, vec!["conv0.bias", "conv0.weight"]);

    assert_eq!(model.tensor_shape("conv0.weight").unwrap(), vec![4, 2]);
    assert_eq!(model.tensor_shape("conv0.bias").unwrap(), vec![4]);
}
