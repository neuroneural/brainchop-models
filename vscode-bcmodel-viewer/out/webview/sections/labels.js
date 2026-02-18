"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.renderLabels = renderLabels;
const utils_1 = require("../utils");
function renderLabels(parsed) {
    const labels = parsed.header.labels;
    if (!labels || labels.length === 0) {
        return '';
    }
    const items = labels
        .map(l => {
        const [r, g, b] = l.color;
        return `<div class="label-item">
        <span class="color-swatch" style="background:rgb(${r},${g},${b})"></span>
        <span class="label-index">${l.index}</span>
        <span>${(0, utils_1.escapeHtml)(l.name)}</span>
      </div>`;
    })
        .join('\n');
    return `
  <section class="card">
    <h2>Labels <span class="count">${labels.length} classes</span></h2>
    <div class="labels-grid">
      ${items}
    </div>
  </section>`;
}
//# sourceMappingURL=labels.js.map