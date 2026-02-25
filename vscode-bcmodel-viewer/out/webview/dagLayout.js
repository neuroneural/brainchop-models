"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.computeDagLayout = computeDagLayout;
const NODE_WIDTH = 180;
const NODE_HEIGHT = 36;
const LAYER_GAP = 50;
const NODE_GAP = 24;
const PADDING = 40;
function computeDagLayout(graph) {
    if (graph.length === 0) {
        return { nodes: [], edges: [], width: 100, height: 100 };
    }
    // Step 1: Assign layers via longest path from sources
    const layerOf = {};
    for (const node of graph) {
        if (node.inputs.length === 0) {
            layerOf[node.id] = 0;
        }
        else {
            let maxParent = 0;
            for (const inp of node.inputs) {
                if (layerOf[inp] !== undefined && layerOf[inp] >= maxParent) {
                    maxParent = layerOf[inp] + 1;
                }
            }
            layerOf[node.id] = maxParent;
        }
    }
    // Step 2: Group nodes by layer
    const layerGroups = new Map();
    for (const node of graph) {
        const l = layerOf[node.id];
        if (!layerGroups.has(l)) {
            layerGroups.set(l, []);
        }
        layerGroups.get(l).push(node.id);
    }
    const numLayers = Math.max(...Array.from(layerGroups.keys())) + 1;
    // Step 3: Order nodes within layers to minimize edge crossings
    // Simple heuristic: order by average position of input nodes
    const columnOf = {};
    for (let l = 0; l < numLayers; l++) {
        const nodesInLayer = layerGroups.get(l) || [];
        if (l === 0) {
            nodesInLayer.forEach((id, i) => { columnOf[id] = i; });
        }
        else {
            // Sort by average column of parents
            const nodeMap = new Map(graph.map(n => [n.id, n]));
            const scored = nodesInLayer.map(id => {
                const node = nodeMap.get(id);
                const parentCols = node.inputs
                    .filter(inp => columnOf[inp] !== undefined)
                    .map(inp => columnOf[inp]);
                const avg = parentCols.length > 0
                    ? parentCols.reduce((a, b) => a + b, 0) / parentCols.length
                    : 0;
                return { id, avg };
            });
            scored.sort((a, b) => a.avg - b.avg);
            scored.forEach((s, i) => { columnOf[s.id] = i; });
        }
    }
    // Step 4: Compute pixel coordinates
    // Center each layer horizontally
    const maxCols = Math.max(...Array.from(layerGroups.values()).map(g => g.length));
    const totalContentWidth = maxCols * NODE_WIDTH + (maxCols - 1) * NODE_GAP;
    const nodeMap = new Map(graph.map(n => [n.id, n]));
    const layoutNodes = [];
    const positions = {};
    for (let l = 0; l < numLayers; l++) {
        const nodesInLayer = layerGroups.get(l) || [];
        const layerWidth = nodesInLayer.length * NODE_WIDTH + (nodesInLayer.length - 1) * NODE_GAP;
        const offsetX = PADDING + (totalContentWidth - layerWidth) / 2;
        // Sort by assigned column
        const sorted = [...nodesInLayer].sort((a, b) => columnOf[a] - columnOf[b]);
        for (let c = 0; c < sorted.length; c++) {
            const id = sorted[c];
            const node = nodeMap.get(id);
            const x = offsetX + c * (NODE_WIDTH + NODE_GAP);
            const y = PADDING + l * (NODE_HEIGHT + LAYER_GAP);
            const cx = x + NODE_WIDTH / 2;
            const cy = y + NODE_HEIGHT / 2;
            positions[id] = { cx, cy };
            layoutNodes.push({
                id,
                op: node.op,
                params: node.params,
                inputs: node.inputs,
                layer: l,
                column: c,
                x,
                y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
            });
        }
    }
    // Step 5: Compute edges
    const layoutEdges = [];
    for (const node of graph) {
        for (const inputId of node.inputs) {
            const from = positions[inputId];
            const to = positions[node.id];
            if (!from || !to) {
                continue;
            }
            layoutEdges.push({
                from: inputId,
                to: node.id,
                fromX: from.cx,
                fromY: from.cy + NODE_HEIGHT / 2,
                toX: to.cx,
                toY: to.cy - NODE_HEIGHT / 2,
                isSkip: layerOf[node.id] - layerOf[inputId] > 1,
            });
        }
    }
    const width = PADDING * 2 + totalContentWidth;
    const height = PADDING * 2 + numLayers * NODE_HEIGHT + (numLayers - 1) * LAYER_GAP;
    return { nodes: layoutNodes, edges: layoutEdges, width, height };
}
//# sourceMappingURL=dagLayout.js.map