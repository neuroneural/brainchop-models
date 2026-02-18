"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.renderTensors = renderTensors;
const utils_1 = require("../utils");
function renderTensors(parsed) {
    if (parsed.tensorStats.length === 0) {
        return '';
    }
    const rows = parsed.tensorStats
        .map((t, i) => `<tr data-idx="${i}" data-name="${t.name.toLowerCase()}">
      <td class="mono">${t.name}</td>
      <td class="mono">[${t.shape.join(', ')}]</td>
      <td class="num">${t.numElements.toLocaleString()}</td>
      <td class="num">${(0, utils_1.formatBytes)(t.sizeBytes)}</td>
      <td class="num">${t.min.toFixed(4)}</td>
      <td class="num">${t.max.toFixed(4)}</td>
      <td class="num">${t.mean.toFixed(4)}</td>
      <td class="num">${t.std.toFixed(4)}</td>
    </tr>`)
        .join('\n');
    return `
  <section class="card">
    <h2>Weight Tensors <span class="count">${parsed.tensorStats.length} tensors &middot; ${parsed.totalParams.toLocaleString()} params &middot; ${(0, utils_1.formatBytes)(parsed.totalWeightBytes)}</span></h2>
    <input type="text" id="tensor-filter" placeholder="Filter by name..." class="filter-input">
    <div class="table-scroll">
      <table class="data-table" id="tensor-table">
        <thead>
          <tr>
            <th data-col="0">Name <span class="sort-arrow">&#9650;</span></th>
            <th data-col="1">Shape <span class="sort-arrow">&#9650;</span></th>
            <th data-col="2">Elements <span class="sort-arrow">&#9650;</span></th>
            <th data-col="3">Size <span class="sort-arrow">&#9650;</span></th>
            <th data-col="4">Min <span class="sort-arrow">&#9650;</span></th>
            <th data-col="5">Max <span class="sort-arrow">&#9650;</span></th>
            <th data-col="6">Mean <span class="sort-arrow">&#9650;</span></th>
            <th data-col="7">Std <span class="sort-arrow">&#9650;</span></th>
          </tr>
        </thead>
        <tbody id="tensor-tbody">
          ${rows}
        </tbody>
      </table>
    </div>
  </section>`;
}
//# sourceMappingURL=tensors.js.map