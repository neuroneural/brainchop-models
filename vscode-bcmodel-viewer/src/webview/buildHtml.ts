import { ParsedBcmodel } from '../bcmodelParser';
import { getBaseStyles } from './theme';
import { getNonce, escapeHtml, formatBytes } from './utils';
import { renderMetadata } from './sections/metadata';
import { renderInference } from './sections/inference';
import { renderLabels } from './sections/labels';
import { renderTensors } from './sections/tensors';
import { renderGraph } from './sections/graph';

export function buildWebviewHtml(parsed: ParsedBcmodel): string {
  const nonce = getNonce();
  const name = escapeHtml(parsed.header.metadata.name);
  const version = escapeHtml(parsed.header.bcmodel_version);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src 'nonce-${nonce}' 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style nonce="${nonce}">${getBaseStyles()}</style>
</head>
<body>
  <div class="container">
    <h1 class="title">${name}<span class="badge">bcmodel v${version}</span></h1>
    <div class="subtitle">
      ${formatBytes(parsed.fileSizeBytes)} &middot;
      ${parsed.totalParams.toLocaleString()} parameters &middot;
      ${parsed.header.graph.length} graph nodes
    </div>

    ${renderMetadata(parsed)}
    ${renderInference(parsed)}
    ${renderGraph(parsed)}
    ${renderTensors(parsed)}
    ${renderLabels(parsed)}
  </div>
  <script nonce="${nonce}">${getInteractiveScript()}</script>
</body>
</html>`;
}

function getInteractiveScript(): string {
  return `(function() {
  // ---- Graph zoom/pan ----
  const viewport = document.getElementById('graph-viewport');
  const transform = document.getElementById('graph-transform');
  const svg = document.getElementById('dag-svg');

  if (viewport && transform && svg) {
    let scale = 1;
    const svgW = parseFloat(svg.getAttribute('width') || '100');
    const svgH = parseFloat(svg.getAttribute('height') || '100');

    function applyScale() {
      transform.style.transform = 'scale(' + scale + ')';
      transform.style.width = (svgW * scale) + 'px';
      transform.style.height = (svgH * scale) + 'px';
    }

    function fitToView() {
      const vw = viewport.clientWidth - 4;
      const vh = viewport.clientHeight - 4;
      scale = Math.min(vw / svgW, vh / svgH, 1);
      applyScale();
    }

    var zoomIn = document.getElementById('zoom-in');
    var zoomOut = document.getElementById('zoom-out');
    var zoomFit = document.getElementById('zoom-fit');

    if (zoomIn) zoomIn.addEventListener('click', function() { scale = Math.min(scale * 1.25, 3); applyScale(); });
    if (zoomOut) zoomOut.addEventListener('click', function() { scale = Math.max(scale / 1.25, 0.2); applyScale(); });
    if (zoomFit) zoomFit.addEventListener('click', fitToView);

    // Mouse wheel zoom
    viewport.addEventListener('wheel', function(e) {
      e.preventDefault();
      var delta = e.deltaY > 0 ? 0.9 : 1.1;
      scale = Math.min(Math.max(scale * delta, 0.2), 3);
      applyScale();
    }, { passive: false });

    // Auto-fit on load
    fitToView();
  }

  // ---- Node click tooltip ----
  var tooltip = document.getElementById('node-tooltip');
  var dagNodes = document.querySelectorAll('.dag-node');

  dagNodes.forEach(function(node) {
    node.style.cursor = 'pointer';
    node.addEventListener('click', function(e) {
      e.stopPropagation();
      var id = node.getAttribute('data-id') || '';
      var op = node.getAttribute('data-op') || '';
      var paramsStr = node.getAttribute('data-params') || '{}';
      var inputsStr = node.getAttribute('data-inputs') || '';
      var params = {};
      try { params = JSON.parse(paramsStr); } catch(_) {}

      var html = '<h3>' + op + ' <span style="color:var(--vscode-descriptionForeground);font-weight:400">' + id + '</span></h3>';

      if (inputsStr) {
        html += '<div class="param-row"><span class="param-key">inputs</span><span class="param-val">' + inputsStr + '</span></div>';
      }

      var keys = Object.keys(params);
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        html += '<div class="param-row"><span class="param-key">' + k + '</span><span class="param-val">' + String(params[k]) + '</span></div>';
      }

      tooltip.innerHTML = html;
      tooltip.style.display = 'block';

      // Position near click
      var rect = viewport.getBoundingClientRect();
      var x = e.clientX + 12;
      var y = e.clientY - 20;

      // Keep within viewport
      if (x + 380 > window.innerWidth) x = e.clientX - 390;
      if (y + 200 > window.innerHeight) y = window.innerHeight - 210;
      if (y < 0) y = 10;

      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    });
  });

  // Click outside to dismiss tooltip
  document.addEventListener('click', function() {
    if (tooltip) tooltip.style.display = 'none';
  });

  // ---- Tensor table filter ----
  var filterInput = document.getElementById('tensor-filter');
  var tbody = document.getElementById('tensor-tbody');

  if (filterInput && tbody) {
    filterInput.addEventListener('input', function() {
      var query = filterInput.value.toLowerCase();
      var rows = tbody.querySelectorAll('tr');
      rows.forEach(function(row) {
        var name = row.getAttribute('data-name') || '';
        row.style.display = name.indexOf(query) >= 0 ? '' : 'none';
      });
    });
  }

  // ---- Tensor table sort ----
  var tensorTable = document.getElementById('tensor-table');
  if (tensorTable && tbody) {
    var headers = tensorTable.querySelectorAll('th[data-col]');
    var sortCol = -1;
    var sortAsc = true;

    headers.forEach(function(th) {
      th.addEventListener('click', function() {
        var col = parseInt(th.getAttribute('data-col') || '0');
        if (sortCol === col) {
          sortAsc = !sortAsc;
        } else {
          sortCol = col;
          sortAsc = true;
        }

        // Update sort indicators
        headers.forEach(function(h) { h.classList.remove('sorted'); });
        th.classList.add('sorted');
        th.querySelector('.sort-arrow').textContent = sortAsc ? '\\u25B2' : '\\u25BC';

        // Sort rows
        var rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort(function(a, b) {
          var aCell = a.children[col];
          var bCell = b.children[col];
          var aText = aCell ? aCell.textContent.trim() : '';
          var bText = bCell ? bCell.textContent.trim() : '';

          // Try numeric comparison
          var aNum = parseFloat(aText.replace(/[^\\d.e\\-]/g, ''));
          var bNum = parseFloat(bText.replace(/[^\\d.e\\-]/g, ''));

          if (!isNaN(aNum) && !isNaN(bNum)) {
            return sortAsc ? aNum - bNum : bNum - aNum;
          }
          return sortAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
        });

        rows.forEach(function(row) { tbody.appendChild(row); });
      });
    });
  }
})();`;
}
