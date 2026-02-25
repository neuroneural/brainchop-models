import initWasm, { initSync, WasmBcmodel } from '../../bcmodel-wasm/pkg/bcmodel_wasm.js';
import type { SyncInitInput } from '../../bcmodel-wasm/pkg/bcmodel_wasm.js';

// ── Types ──────────────────────────────────────────────────────────

export interface GraphNode {
  id: string;
  op: string;
  params: Record<string, unknown>;
  inputs: string[];
}

export interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface LabelEntry {
  index: number;
  name: string;
  color: [number, number, number];
}

export interface Metadata {
  name: string;
  type: string;
  description?: string;
  source_framework?: string;
  authors?: string[];
  license?: string;
}

export interface InputSpec {
  shape: number[];
  dtype: string;
  data_layout: string;
}

export interface OutputSpec {
  num_classes: number;
  data_layout: string;
}

export interface InferenceConfig {
  enable_seq_conv: boolean;
  crop_padding: number;
  auto_threshold: number;
  enable_quantile_norm: boolean;
  enable_transpose: boolean;
}

export interface PipelineConfig {
  requires_pre_model?: string;
  filter_with_pre_mask: boolean;
}

export interface PerformanceHints {
  estimated_time_seconds: number;
  memory_requirement_mb: number;
}

export interface BcmodelHeader {
  bcmodel_version: string;
  metadata: Metadata;
  input: InputSpec;
  output: OutputSpec;
  graph: GraphNode[];
  tensors: Record<string, TensorInfo>;
  labels: LabelEntry[];
  inference: InferenceConfig;
  pipeline: PipelineConfig;
  performance: PerformanceHints;
}

export interface TensorStats {
  name: string;
  shape: number[];
  dtype: string;
  numElements: number;
  sizeBytes: number;
  min: number;
  max: number;
  mean: number;
  std: number;
}

// ── BcmodelFile wrapper ────────────────────────────────────────────

/**
 * Typed wrapper around WasmBcmodel that replaces `any` return types
 * with concrete interfaces for full IntelliSense support.
 */
export class BcmodelFile {
  private wasm: WasmBcmodel;

  constructor(wasm: WasmBcmodel) {
    this.wasm = wasm;
  }

  get header(): BcmodelHeader {
    return this.wasm.header as BcmodelHeader;
  }

  get metadata(): Metadata {
    return this.wasm.metadata as Metadata;
  }

  get graph(): GraphNode[] {
    return this.wasm.graph as GraphNode[];
  }

  get labels(): LabelEntry[] {
    return this.wasm.labels as LabelEntry[];
  }

  get inferenceConfig(): InferenceConfig {
    return this.wasm.inferenceConfig as InferenceConfig;
  }

  get pipelineConfig(): PipelineConfig {
    return this.wasm.pipelineConfig as PipelineConfig;
  }

  get performanceHints(): PerformanceHints {
    return this.wasm.performanceHints as PerformanceHints;
  }

  get name(): string {
    return this.wasm.name;
  }

  get modelType(): string {
    return this.wasm.modelType;
  }

  get numClasses(): number {
    return this.wasm.numClasses;
  }

  get inputShape(): number[] {
    return Array.from(this.wasm.inputShape);
  }

  get graphLength(): number {
    return this.wasm.graphLength;
  }

  get totalParams(): number {
    return this.wasm.totalParams;
  }

  get weightSizeBytes(): number {
    return this.wasm.weightSizeBytes;
  }

  tensorNames(): string[] {
    return this.wasm.tensorNames();
  }

  tensorShape(name: string): number[] | undefined {
    const shape = this.wasm.tensorShape(name);
    return shape ? Array.from(shape) : undefined;
  }

  tensorData(name: string): Float32Array | undefined {
    return this.wasm.tensorData(name);
  }

  free(): void {
    this.wasm.free();
  }

  [Symbol.dispose](): void {
    this.wasm.free();
  }
}

// ── Loaders ────────────────────────────────────────────────────────

/**
 * Initialize the WASM module asynchronously. Call once before using any loading functions.
 */
export async function initBcmodel(wasmPath?: string | URL): Promise<void> {
  if (wasmPath) {
    await initWasm({ module_or_path: wasmPath });
  } else {
    await initWasm();
  }
}

/**
 * Initialize the WASM module synchronously from a buffer (Node.js).
 */
export function initBcmodelSync(module: SyncInitInput): void {
  initSync({ module });
}

/**
 * Load a .bcmodel file from a URL.
 */
export async function loadFromUrl(url: string | URL): Promise<BcmodelFile> {
  const response = await fetch(url instanceof URL ? url.href : url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  return loadFromBuffer(buffer);
}

/**
 * Load a .bcmodel file from an ArrayBuffer or Uint8Array.
 */
export function loadFromBuffer(data: ArrayBuffer | Uint8Array): BcmodelFile {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  return new BcmodelFile(new WasmBcmodel(bytes));
}

/**
 * Load a .bcmodel file from a browser File object.
 */
export async function loadFromFile(file: File): Promise<BcmodelFile> {
  const buffer = await file.arrayBuffer();
  return loadFromBuffer(buffer);
}

// ── Tensor utilities ───────────────────────────────────────────────

/**
 * Compute statistics for a single tensor.
 */
export function computeTensorStats(
  name: string,
  data: Float32Array,
  shape: number[],
  dtype = 'float32',
): TensorStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }

  const mean = data.length > 0 ? sum / data.length : 0;

  let variance = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    variance += d * d;
  }
  const std = data.length > 0 ? Math.sqrt(variance / data.length) : 0;

  return {
    name,
    shape,
    dtype,
    numElements: data.length,
    sizeBytes: data.byteLength,
    min: data.length > 0 ? min : 0,
    max: data.length > 0 ? max : 0,
    mean,
    std,
  };
}

/**
 * Compute statistics for all tensors in a model.
 */
export function computeAllTensorStats(model: BcmodelFile): TensorStats[] {
  const stats: TensorStats[] = [];
  for (const name of model.tensorNames()) {
    const data = model.tensorData(name);
    const shape = model.tensorShape(name);
    if (data && shape) {
      stats.push(computeTensorStats(name, data, shape));
    }
  }
  return stats;
}

/**
 * Transpose a 5D conv kernel from channels-first to channels-last for TFJS.
 * [O, I, D, H, W] → [D, H, W, I, O]
 */
export function transposeConvKernel(
  data: Float32Array,
  shape: number[],
): { data: Float32Array; shape: number[] } {
  if (shape.length !== 5) {
    throw new Error(`Expected 5D tensor, got ${shape.length}D`);
  }

  const [O, I, D, H, W] = shape;
  const newShape = [D, H, W, I, O];
  const result = new Float32Array(data.length);

  for (let o = 0; o < O; o++) {
    for (let i = 0; i < I; i++) {
      for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            const srcIdx = ((((o * I + i) * D + d) * H + h) * W + w);
            const dstIdx = ((((d * H + h) * W + w) * I + i) * O + o);
            result[dstIdx] = data[srcIdx];
          }
        }
      }
    }
  }

  return { data: result, shape: newShape };
}

type NestedArray = number | NestedArray[];

/**
 * Reshape a flat Float32Array into a nested array matching the given shape.
 */
export function reshape(data: Float32Array, shape: number[]): NestedArray {
  if (shape.length === 0) {
    return data[0];
  }
  if (shape.length === 1) {
    return Array.from(data.subarray(0, shape[0]));
  }

  const [first, ...rest] = shape;
  const stride = rest.reduce((a, b) => a * b, 1);
  const result: NestedArray[] = [];

  for (let i = 0; i < first; i++) {
    const slice = data.subarray(i * stride, (i + 1) * stride);
    result.push(reshape(new Float32Array(slice), rest));
  }

  return result;
}

/**
 * Iterate over graph nodes, yielding each node with its associated tensor data.
 */
export function* iterateLayers(
  model: BcmodelFile,
): Generator<{ node: GraphNode; tensors: Map<string, Float32Array> }> {
  const allTensors = new Set(model.tensorNames());

  for (const node of model.graph) {
    const tensors = new Map<string, Float32Array>();

    for (const tensorName of allTensors) {
      if (tensorName.startsWith(node.id + '.')) {
        const data = model.tensorData(tensorName);
        if (data) {
          tensors.set(tensorName, data);
        }
      }
    }

    yield { node, tensors };
  }
}
