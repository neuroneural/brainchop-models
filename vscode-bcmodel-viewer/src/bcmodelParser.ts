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

export interface BcmodelHeader {
  bcmodel_version: string;
  metadata: {
    name: string;
    type: string;
    description?: string;
    source_framework?: string;
    authors?: string[];
    license?: string;
  };
  input: { shape: number[]; dtype: string; data_layout: string };
  output: { num_classes: number; data_layout: string };
  graph: GraphNode[];
  tensors: Record<string, TensorInfo>;
  labels?: LabelEntry[];
  inference?: Record<string, unknown>;
  performance?: Record<string, unknown>;
  pipeline?: Record<string, unknown>;
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

export interface ParsedBcmodel {
  header: BcmodelHeader;
  tensorStats: TensorStats[];
  totalParams: number;
  totalWeightBytes: number;
  fileSizeBytes: number;
}

export function parseBcmodel(buffer: ArrayBuffer): ParsedBcmodel {
  const view = new DataView(buffer);

  // Read header size (uint64 LE — use lower 32 bits)
  const headerSize = view.getUint32(0, true);
  if (view.getUint32(4, true) !== 0) {
    throw new Error('Header size exceeds 4GB — unsupported');
  }

  // Parse JSON header
  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const header: BcmodelHeader = JSON.parse(new TextDecoder().decode(headerBytes));

  if (!header.bcmodel_version) {
    throw new Error('Not a valid .bcmodel file: missing bcmodel_version');
  }

  // Compute tensor statistics from binary data
  const dataOffset = 8 + headerSize;
  const tensorStats: TensorStats[] = [];
  let totalParams = 0;

  for (const [name, info] of Object.entries(header.tensors)) {
    const [begin, end] = info.data_offsets;
    const count = (end - begin) / 4;

    // Create a copy to avoid alignment issues
    const slice = buffer.slice(dataOffset + begin, dataOffset + end);
    const data = new Float32Array(slice);

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < min) { min = v; }
      if (v > max) { max = v; }
      sum += v;
    }
    const mean = data.length > 0 ? sum / data.length : 0;

    let variance = 0;
    for (let i = 0; i < data.length; i++) {
      const d = data[i] - mean;
      variance += d * d;
    }
    const std = data.length > 0 ? Math.sqrt(variance / data.length) : 0;

    tensorStats.push({
      name,
      shape: info.shape,
      dtype: info.dtype,
      numElements: count,
      sizeBytes: end - begin,
      min: data.length > 0 ? min : 0,
      max: data.length > 0 ? max : 0,
      mean,
      std,
    });
    totalParams += count;
  }

  return {
    header,
    tensorStats,
    totalParams,
    totalWeightBytes: tensorStats.reduce((s, t) => s + t.sizeBytes, 0),
    fileSizeBytes: buffer.byteLength,
  };
}
