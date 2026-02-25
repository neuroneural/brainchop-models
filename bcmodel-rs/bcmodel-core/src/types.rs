use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub bcmodel_version: String,
    pub metadata: Metadata,
    pub input: InputSpec,
    pub output: OutputSpec,
    pub graph: Vec<GraphNode>,
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub labels: Vec<Label>,
    #[serde(default)]
    pub pipeline: PipelineConfig,
    #[serde(default)]
    pub performance: PerformanceHints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "type")]
    pub model_type: String,
    #[serde(default)]
    pub source_framework: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data_layout: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    pub num_classes: usize,
    #[serde(default = "default_channels_first")]
    pub data_layout: String,
}

fn default_channels_first() -> String {
    "channels_first".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub op: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default)]
    pub enable_seq_conv: bool,
    #[serde(default)]
    pub crop_padding: u32,
    #[serde(default)]
    pub auto_threshold: f64,
    #[serde(default)]
    pub enable_quantile_norm: bool,
    #[serde(default = "default_true")]
    pub enable_transpose: bool,
}

fn default_true() -> bool {
    true
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enable_seq_conv: false,
            crop_padding: 0,
            auto_threshold: 0.0,
            enable_quantile_norm: false,
            enable_transpose: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub index: usize,
    pub name: String,
    pub color: [u8; 3],
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub requires_pre_model: Option<String>,
    #[serde(default)]
    pub filter_with_pre_mask: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceHints {
    #[serde(default)]
    pub estimated_time_seconds: u32,
    #[serde(default)]
    pub memory_requirement_mb: u32,
}
