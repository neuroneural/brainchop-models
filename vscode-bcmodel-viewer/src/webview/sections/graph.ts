import { ParsedBcmodel } from '../../bcmodelParser';
import { computeDagLayout, LayoutNode, LayoutEdge } from '../dagLayout';
import { escapeHtml } from '../utils';

const OP_COLORS: Record<string, string> = {
  conv3d: '#4a9eff',
  conv_transpose3d: '#4a9eff',
  batch_norm3d: '#e8a838',
  group_norm: '#e8a838',
  instance_norm3d: '#e8a838',
  relu: '#4ec9b0',
  elu: '#4ec9b0',
  gelu: '#4ec9b0',
  sigmoid: '#4ec9b0',
  softmax: '#4ec9b0',
  add: '#c586c0',
  cat: '#c586c0',
  max_pool3d: '#d4d4d4',
  avg_pool3d: '#d4d4d4',
  upsample: '#dcdcaa',
  dropout: '#808080',
  linear: '#4a9eff',
};

function getOpColor(op: string): string {
  return OP_COLORS[op] || '#888888';
}

function renderEdgeSvg(edge: LayoutEdge): string {
  if (edge.isSkip) {
    // Curved path for skip connections
    const dx = edge.toX - edge.fromX;
    const dy = edge.toY - edge.fromY;
    const offsetX = dx === 0 ? 40 : 0;
    const midY = edge.fromY + dy / 2;
    return `<path d="M${edge.fromX},${edge.fromY} C${edge.fromX + offsetX},${midY} ${edge.toX + offsetX},${midY} ${edge.toX},${edge.toY}"
      fill="none" stroke="var(--vscode-foreground)" stroke-opacity="0.25" stroke-width="1.5"
      stroke-dasharray="4,3" marker-end="url(#arrowhead)"/>`;
  }
  // Straight line
  return `<line x1="${edge.fromX}" y1="${edge.fromY}" x2="${edge.toX}" y2="${edge.toY}"
    stroke="var(--vscode-foreground)" stroke-opacity="0.3" stroke-width="1.5"
    marker-end="url(#arrowhead)"/>`;
}

function renderNodeSvg(node: LayoutNode): string {
  const color = getOpColor(node.op);
  const shortLabel = node.op.replace(/_/g, ' ');
  const idLabel = node.id.length > 16 ? node.id.slice(0, 14) + '..' : node.id;

  // Encode params as data attribute for tooltip
  const paramsJson = escapeHtml(JSON.stringify(node.params));

  return `<g class="dag-node" data-id="${escapeHtml(node.id)}" data-op="${escapeHtml(node.op)}" data-params="${paramsJson}" data-inputs="${escapeHtml(node.inputs.join(','))}">
    <rect x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}"
      rx="4" ry="4" fill="${color}" fill-opacity="0.15" stroke="${color}" stroke-width="1.5"/>
    <text x="${node.x + 8}" y="${node.y + node.height / 2 + 1}" dominant-baseline="middle"
      fill="var(--vscode-foreground)" font-size="11" font-family="var(--vscode-editor-font-family, monospace)">
      <tspan font-weight="600">${escapeHtml(shortLabel)}</tspan>
      <tspan fill="var(--vscode-descriptionForeground)" font-size="10"> ${escapeHtml(idLabel)}</tspan>
    </text>
  </g>`;
}

export function renderGraph(parsed: ParsedBcmodel): string {
  const graph = parsed.header.graph;
  if (graph.length === 0) {
    return '';
  }

  const layout = computeDagLayout(graph);

  // Count unique ops
  const uniqueOps = new Set(graph.map(n => n.op));

  // Build legend
  const legendItems = Array.from(uniqueOps)
    .sort()
    .map(op => {
      const color = getOpColor(op);
      return `<span style="display:inline-flex;align-items:center;margin-right:12px;font-size:0.85em">
        <span style="width:10px;height:10px;border-radius:2px;background:${color};display:inline-block;margin-right:4px"></span>
        ${escapeHtml(op)}
      </span>`;
    })
    .join('');

  const edges = layout.edges.map(renderEdgeSvg).join('\n');
  const nodes = layout.nodes.map(renderNodeSvg).join('\n');

  // Determine max viewport height: cap at 700px for large graphs
  const viewportHeight = Math.min(layout.height + 20, 700);

  return `
  <section class="card">
    <h2>Architecture Graph <span class="count">${graph.length} nodes &middot; ${uniqueOps.size} op types</span></h2>
    <div class="graph-controls">
      <button class="btn" id="zoom-in" title="Zoom in">+</button>
      <button class="btn" id="zoom-out" title="Zoom out">&minus;</button>
      <button class="btn" id="zoom-fit" title="Fit to view">Fit</button>
      <span style="flex:1"></span>
      ${legendItems}
    </div>
    <div class="graph-viewport" id="graph-viewport" style="max-height:${viewportHeight}px">
      <div id="graph-transform" style="transform-origin:0 0">
        <svg id="dag-svg" width="${layout.width}" height="${layout.height}"
             viewBox="0 0 ${layout.width} ${layout.height}" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="var(--vscode-foreground)" opacity="0.4"/>
            </marker>
          </defs>
          <g id="edges-group">${edges}</g>
          <g id="nodes-group">${nodes}</g>
        </svg>
      </div>
    </div>
    <div id="node-tooltip" class="node-tooltip" style="display:none"></div>
  </section>`;
}
