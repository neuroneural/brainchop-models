"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.renderInference = renderInference;
const utils_1 = require("../utils");
function renderInference(parsed) {
    const { inference, performance, pipeline } = parsed.header;
    const hasInference = inference && Object.keys(inference).length > 0;
    const hasPerformance = performance && Object.keys(performance).length > 0;
    const hasPipeline = pipeline && Object.keys(pipeline).length > 0;
    if (!hasInference && !hasPerformance && !hasPipeline) {
        return '';
    }
    let html = '<section class="card"><h2>Configuration</h2><div class="two-col">';
    if (hasInference) {
        html += '<div>';
        html += '<h3 style="font-size:0.95em;margin-bottom:8px;color:var(--vscode-foreground)">Inference</h3>';
        html += '<div class="kv-grid">';
        for (const [k, v] of Object.entries(inference)) {
            html += `<div class="key">${(0, utils_1.escapeHtml)(formatKey(k))}</div><div class="val">${(0, utils_1.escapeHtml)(formatValue(v))}</div>`;
        }
        html += '</div></div>';
    }
    if (hasPerformance) {
        html += '<div>';
        html += '<h3 style="font-size:0.95em;margin-bottom:8px;color:var(--vscode-foreground)">Performance</h3>';
        html += '<div class="kv-grid">';
        for (const [k, v] of Object.entries(performance)) {
            html += `<div class="key">${(0, utils_1.escapeHtml)(formatKey(k))}</div><div class="val">${(0, utils_1.escapeHtml)(formatValue(v))}</div>`;
        }
        html += '</div></div>';
    }
    if (hasPipeline) {
        html += '<div>';
        html += '<h3 style="font-size:0.95em;margin-bottom:8px;color:var(--vscode-foreground)">Pipeline</h3>';
        html += '<div class="kv-grid">';
        for (const [k, v] of Object.entries(pipeline)) {
            html += `<div class="key">${(0, utils_1.escapeHtml)(formatKey(k))}</div><div class="val">${(0, utils_1.escapeHtml)(formatValue(v))}</div>`;
        }
        html += '</div></div>';
    }
    html += '</div></section>';
    return html;
}
function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
function formatValue(v) {
    if (v === null) {
        return 'none';
    }
    if (typeof v === 'boolean') {
        return v ? 'yes' : 'no';
    }
    return String(v);
}
//# sourceMappingURL=inference.js.map