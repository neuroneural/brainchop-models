"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseBcmodel = parseBcmodel;
function parseBcmodel(buffer) {
    const view = new DataView(buffer);
    // Read header size (uint64 LE — use lower 32 bits)
    const headerSize = view.getUint32(0, true);
    if (view.getUint32(4, true) !== 0) {
        throw new Error('Header size exceeds 4GB — unsupported');
    }
    // Parse JSON header
    const headerBytes = new Uint8Array(buffer, 8, headerSize);
    const header = JSON.parse(new TextDecoder().decode(headerBytes));
    if (!header.bcmodel_version) {
        throw new Error('Not a valid .bcmodel file: missing bcmodel_version');
    }
    // Compute tensor statistics from binary data
    const dataOffset = 8 + headerSize;
    const tensorStats = [];
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
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
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
//# sourceMappingURL=bcmodelParser.js.map