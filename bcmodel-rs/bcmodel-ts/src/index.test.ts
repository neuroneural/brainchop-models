import { describe, it, expect } from 'vitest';
import { computeTensorStats, computeAllTensorStats, transposeConvKernel, reshape, iterateLayers } from './index.js';
import type { BcmodelFile } from './index.js';

// ---------------------------------------------------------------------------
// computeTensorStats
// ---------------------------------------------------------------------------

describe('computeTensorStats', () => {
  it('computes stats for a simple tensor', () => {
    const data = new Float32Array([1, 2, 3, 4, 5]);
    const stats = computeTensorStats('test', data, [5]);

    expect(stats.name).toBe('test');
    expect(stats.shape).toEqual([5]);
    expect(stats.dtype).toBe('float32');
    expect(stats.numElements).toBe(5);
    expect(stats.sizeBytes).toBe(20);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
    expect(stats.mean).toBe(3);
    expect(stats.std).toBeCloseTo(Math.sqrt(2), 5);
  });

  it('handles single element', () => {
    const data = new Float32Array([42]);
    const stats = computeTensorStats('single', data, [1]);

    expect(stats.min).toBe(42);
    expect(stats.max).toBe(42);
    expect(stats.mean).toBe(42);
    expect(stats.std).toBe(0);
  });

  it('handles negative values', () => {
    const data = new Float32Array([-3, -1, 0, 1, 3]);
    const stats = computeTensorStats('neg', data, [5]);

    expect(stats.min).toBe(-3);
    expect(stats.max).toBe(3);
    expect(stats.mean).toBe(0);
  });

  it('handles empty tensor', () => {
    const data = new Float32Array([]);
    const stats = computeTensorStats('empty', data, [0]);

    expect(stats.numElements).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.max).toBe(0);
    expect(stats.mean).toBe(0);
    expect(stats.std).toBe(0);
  });

  it('accepts custom dtype', () => {
    const data = new Float32Array([1]);
    const stats = computeTensorStats('t', data, [1], 'float16');
    expect(stats.dtype).toBe('float16');
  });

  it('reports correct sizeBytes for multidimensional shape', () => {
    const data = new Float32Array(6);
    const stats = computeTensorStats('t', data, [2, 3]);
    expect(stats.numElements).toBe(6);
    expect(stats.sizeBytes).toBe(24);
  });
});

// ---------------------------------------------------------------------------
// transposeConvKernel
// ---------------------------------------------------------------------------

describe('transposeConvKernel', () => {
  it('transposes [O,I,D,H,W] to [D,H,W,I,O]', () => {
    // 2x1x1x1x1 kernel — simplest 5D case
    const data = new Float32Array([10, 20]);
    const result = transposeConvKernel(data, [2, 1, 1, 1, 1]);

    expect(result.shape).toEqual([1, 1, 1, 1, 2]);
    expect(Array.from(result.data)).toEqual([10, 20]);
  });

  it('transposes a 2x2x1x1x1 kernel correctly', () => {
    // [O=2, I=2, D=1, H=1, W=1]
    // src layout: [o0i0, o0i1, o1i0, o1i1]
    const data = new Float32Array([1, 2, 3, 4]);
    const result = transposeConvKernel(data, [2, 2, 1, 1, 1]);

    // dst: [D=1, H=1, W=1, I=2, O=2] → [i0o0, i0o1, i1o0, i1o1] = [1, 3, 2, 4]
    expect(result.shape).toEqual([1, 1, 1, 2, 2]);
    expect(Array.from(result.data)).toEqual([1, 3, 2, 4]);
  });

  it('transposes a 1x1x2x2x2 kernel correctly', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = transposeConvKernel(data, [1, 1, 2, 2, 2]);

    // [D=2,H=2,W=2,I=1,O=1] — same data, just different shape
    expect(result.shape).toEqual([2, 2, 2, 1, 1]);
    expect(Array.from(result.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('throws for non-5D tensor', () => {
    expect(() => transposeConvKernel(new Float32Array(4), [2, 2])).toThrow('5D');
    expect(() => transposeConvKernel(new Float32Array(1), [1])).toThrow('5D');
  });

  it('preserves data length', () => {
    const data = new Float32Array(2 * 3 * 4 * 5 * 6);
    for (let i = 0; i < data.length; i++) data[i] = i;
    const result = transposeConvKernel(data, [2, 3, 4, 5, 6]);

    expect(result.data.length).toBe(data.length);
    expect(result.shape).toEqual([4, 5, 6, 3, 2]);
  });
});

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

describe('reshape', () => {
  it('returns scalar for empty shape', () => {
    expect(reshape(new Float32Array([42]), [])).toBe(42);
  });

  it('returns flat array for 1D shape', () => {
    const data = new Float32Array([1, 2, 3]);
    expect(reshape(data, [3])).toEqual([1, 2, 3]);
  });

  it('reshapes to 2D', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    expect(reshape(data, [2, 3])).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it('reshapes to 3D', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(reshape(data, [2, 2, 2])).toEqual([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ]);
  });

  it('reshapes to 1x1x4', () => {
    const data = new Float32Array([10, 20, 30, 40]);
    expect(reshape(data, [1, 1, 4])).toEqual([[[10, 20, 30, 40]]]);
  });
});

// ---------------------------------------------------------------------------
// computeAllTensorStats
// ---------------------------------------------------------------------------

describe('computeAllTensorStats', () => {
  it('computes stats for all tensors in a mock model', () => {
    const mockModel = {
      tensorNames: () => ['a', 'b'],
      tensorData: (name: string) => {
        if (name === 'a') return new Float32Array([1, 2, 3]);
        if (name === 'b') return new Float32Array([10, 20]);
        return undefined;
      },
      tensorShape: (name: string) => {
        if (name === 'a') return [3];
        if (name === 'b') return [2];
        return undefined;
      },
    } as unknown as BcmodelFile;

    const stats = computeAllTensorStats(mockModel);
    expect(stats).toHaveLength(2);
    expect(stats[0].name).toBe('a');
    expect(stats[0].mean).toBe(2);
    expect(stats[1].name).toBe('b');
    expect(stats[1].mean).toBe(15);
  });

  it('skips tensors with missing data', () => {
    const mockModel = {
      tensorNames: () => ['exists', 'missing'],
      tensorData: (name: string) => name === 'exists' ? new Float32Array([1]) : undefined,
      tensorShape: (name: string) => name === 'exists' ? [1] : undefined,
    } as unknown as BcmodelFile;

    const stats = computeAllTensorStats(mockModel);
    expect(stats).toHaveLength(1);
    expect(stats[0].name).toBe('exists');
  });
});

// ---------------------------------------------------------------------------
// iterateLayers
// ---------------------------------------------------------------------------

describe('iterateLayers', () => {
  it('yields nodes with matched tensors', () => {
    const mockModel = {
      graph: [
        { id: 'conv0', op: 'conv3d', params: {}, inputs: [] },
        { id: 'relu0', op: 'relu', params: {}, inputs: ['conv0'] },
      ],
      tensorNames: () => ['conv0.weight', 'conv0.bias', 'relu0.dummy'],
      tensorData: (name: string) => new Float32Array([1]),
    } as unknown as BcmodelFile;

    const layers = [...iterateLayers(mockModel)];
    expect(layers).toHaveLength(2);

    expect(layers[0].node.id).toBe('conv0');
    expect([...layers[0].tensors.keys()].sort()).toEqual(['conv0.bias', 'conv0.weight']);

    expect(layers[1].node.id).toBe('relu0');
    expect([...layers[1].tensors.keys()]).toEqual(['relu0.dummy']);
  });

  it('yields empty tensors map for nodes without matching tensors', () => {
    const mockModel = {
      graph: [{ id: 'relu0', op: 'relu', params: {}, inputs: [] }],
      tensorNames: () => ['conv0.weight'],
      tensorData: () => new Float32Array([1]),
    } as unknown as BcmodelFile;

    const layers = [...iterateLayers(mockModel)];
    expect(layers).toHaveLength(1);
    expect(layers[0].tensors.size).toBe(0);
  });
});
