/**
 * BrainChop Model (.bcmodel) loader for JavaScript runtimes.
 *
 * Loads .bcmodel files containing architecture, weights, inference config,
 * and labels in a single file. Compatible with TensorFlow.js and WebGPU.
 *
 * Usage (Node.js):
 *   const { loadBcmodel } = require('./load_bcmodel');
 *   const model = await loadBcmodel('meshnet/model5_gw_ae/model.bcmodel');
 *   console.log(model.header.metadata.name);
 *
 * Usage (Browser):
 *   const model = await loadBcmodel(url);
 *   const weight = model.tensors['conv0.weight'];
 */

/**
 * @typedef {Object} BcmodelTensor
 * @property {Float32Array} data - Raw tensor data
 * @property {number[]} shape - Tensor dimensions
 */

/**
 * @typedef {Object} BcmodelFile
 * @property {Object} header - Parsed JSON header
 * @property {Object<string, BcmodelTensor>} tensors - Weight tensors by name
 */

/**
 * Load a .bcmodel file from a URL or ArrayBuffer.
 *
 * @param {string|ArrayBuffer} source - URL string or ArrayBuffer
 * @returns {Promise<BcmodelFile>}
 */
async function loadBcmodel(source) {
  let buffer;
  if (typeof source === 'string') {
    // Fetch from URL
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${source}: ${response.statusText}`);
    }
    buffer = await response.arrayBuffer();
  } else if (source instanceof ArrayBuffer) {
    buffer = source;
  } else {
    throw new TypeError('source must be a URL string or ArrayBuffer');
  }

  return parseBcmodel(buffer);
}

/**
 * Parse a .bcmodel ArrayBuffer into header and tensors.
 *
 * @param {ArrayBuffer} buffer
 * @returns {BcmodelFile}
 */
function parseBcmodel(buffer) {
  const view = new DataView(buffer);

  // Read header size (uint64 LE — use lower 32 bits, sufficient for <4GB headers)
  const headerSize = view.getUint32(0, true);
  // Verify high 32 bits are zero
  if (view.getUint32(4, true) !== 0) {
    throw new Error('Header size exceeds 4GB — unsupported');
  }

  // Parse JSON header
  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerString = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerString);

  if (!header.bcmodel_version) {
    throw new Error('Not a valid .bcmodel file: missing bcmodel_version');
  }

  // Extract tensors from binary data section
  const dataOffset = 8 + headerSize;
  const tensors = {};

  for (const [name, info] of Object.entries(header.tensors)) {
    const [begin, end] = info.data_offsets;
    const byteLength = end - begin;
    const count = byteLength / 4; // float32 = 4 bytes

    // Create a view into the buffer (zero-copy where possible)
    const data = new Float32Array(buffer, dataOffset + begin, count);
    tensors[name] = { data, shape: info.shape };
  }

  return { header, tensors };
}

/**
 * Transpose a tensor from channels-first to channels-last for TFJS.
 * Conv3D kernel: [out, in, D, H, W] -> [D, H, W, in, out]
 *
 * @param {Float32Array} data - Source tensor data
 * @param {number[]} shape - Source shape [O, I, D, H, W]
 * @returns {{ data: Float32Array, shape: number[] }}
 */
function transposeConvKernel(data, shape) {
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

/**
 * Get model metadata summary.
 *
 * @param {Object} header
 * @returns {Object}
 */
function getModelInfo(header) {
  const meta = header.metadata;
  const totalParams = Object.values(header.tensors).reduce((sum, t) => {
    return sum + t.shape.reduce((a, b) => a * b, 1);
  }, 0);

  return {
    name: meta.name,
    type: meta.type,
    description: meta.description || '',
    inputShape: header.input.shape,
    numClasses: header.output.num_classes,
    graphNodes: header.graph.length,
    totalParams,
    labels: (header.labels || []).map(l => l.name),
    inference: header.inference || {},
  };
}

// Node.js exports
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { loadBcmodel, parseBcmodel, transposeConvKernel, getModelInfo };
}
