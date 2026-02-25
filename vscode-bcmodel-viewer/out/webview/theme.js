"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getBaseStyles = getBaseStyles;
function getBaseStyles() {
    return `
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--vscode-font-family, system-ui, sans-serif);
  font-size: var(--vscode-font-size, 13px);
  color: var(--vscode-foreground);
  background: var(--vscode-editor-background);
  line-height: 1.5;
  padding: 0;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.title {
  font-size: 1.6em;
  font-weight: 600;
  margin: 0 0 4px 0;
}

.subtitle {
  color: var(--vscode-descriptionForeground);
  margin-bottom: 20px;
  font-size: 0.95em;
}

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.8em;
  font-weight: 500;
  background: var(--vscode-badge-background);
  color: var(--vscode-badge-foreground);
  margin-left: 8px;
  vertical-align: middle;
}

.card {
  background: var(--vscode-editorWidget-background, var(--vscode-editor-background));
  border: 1px solid var(--vscode-widget-border, rgba(128,128,128,0.2));
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 16px;
}

.card h2 {
  font-size: 1.1em;
  font-weight: 600;
  margin: 0 0 12px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--vscode-widget-border, rgba(128,128,128,0.2));
  display: flex;
  align-items: center;
  gap: 8px;
}

.card h2 .count {
  font-weight: 400;
  color: var(--vscode-descriptionForeground);
  font-size: 0.85em;
}

.kv-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 16px;
  font-size: 0.95em;
}

.kv-grid .key {
  color: var(--vscode-descriptionForeground);
  white-space: nowrap;
}

.kv-grid .val {
  font-family: var(--vscode-editor-font-family, monospace);
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
}

.data-table th {
  background: var(--vscode-editorGroupHeader-tabsBackground, rgba(128,128,128,0.1));
  color: var(--vscode-foreground);
  text-align: left;
  padding: 6px 10px;
  border-bottom: 2px solid var(--vscode-widget-border, rgba(128,128,128,0.2));
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 1;
}

.data-table th:hover {
  background: var(--vscode-list-hoverBackground, rgba(128,128,128,0.15));
}

.data-table th .sort-arrow {
  opacity: 0.4;
  margin-left: 4px;
}

.data-table th.sorted .sort-arrow {
  opacity: 1;
}

.data-table td {
  padding: 4px 10px;
  border-bottom: 1px solid var(--vscode-widget-border, rgba(128,128,128,0.1));
}

.data-table tr:hover td {
  background: var(--vscode-list-hoverBackground, rgba(128,128,128,0.05));
}

.data-table .mono {
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 0.92em;
}

.data-table .num {
  text-align: right;
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 0.92em;
}

.table-scroll {
  overflow: auto;
  max-height: 500px;
  border: 1px solid var(--vscode-widget-border, rgba(128,128,128,0.2));
  border-radius: 4px;
}

.filter-input {
  background: var(--vscode-input-background);
  color: var(--vscode-input-foreground);
  border: 1px solid var(--vscode-input-border, rgba(128,128,128,0.3));
  padding: 5px 10px;
  border-radius: 4px;
  width: 100%;
  max-width: 300px;
  margin-bottom: 10px;
  font-size: 0.95em;
  outline: none;
}

.filter-input:focus {
  border-color: var(--vscode-focusBorder);
}

.color-swatch {
  display: inline-block;
  width: 16px;
  height: 16px;
  border-radius: 3px;
  border: 1px solid var(--vscode-widget-border, rgba(128,128,128,0.3));
  vertical-align: middle;
  margin-right: 8px;
  flex-shrink: 0;
}

.labels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 4px 12px;
}

.label-item {
  display: flex;
  align-items: center;
  padding: 3px 0;
  font-size: 0.92em;
}

.label-item .label-index {
  color: var(--vscode-descriptionForeground);
  min-width: 30px;
  margin-right: 6px;
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 0.9em;
}

.graph-controls {
  display: flex;
  gap: 4px;
  margin-bottom: 8px;
  flex-wrap: wrap;
  align-items: center;
}

.btn {
  background: var(--vscode-button-secondaryBackground);
  color: var(--vscode-button-secondaryForeground);
  border: none;
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.85em;
  font-family: inherit;
}

.btn:hover {
  background: var(--vscode-button-secondaryHoverBackground);
}

.btn.active {
  background: var(--vscode-button-background);
  color: var(--vscode-button-foreground);
}

.graph-viewport {
  overflow: auto;
  border: 1px solid var(--vscode-widget-border, rgba(128,128,128,0.2));
  border-radius: 4px;
  background: var(--vscode-editor-background);
  position: relative;
}

.graph-viewport svg {
  display: block;
}

.graph-info {
  color: var(--vscode-descriptionForeground);
  font-size: 0.9em;
  margin-bottom: 8px;
}

.node-tooltip {
  position: fixed;
  background: var(--vscode-editorHoverWidget-background, var(--vscode-editorWidget-background));
  border: 1px solid var(--vscode-editorHoverWidget-border, var(--vscode-widget-border));
  border-radius: 4px;
  padding: 10px 14px;
  font-size: 0.9em;
  max-width: 380px;
  z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  pointer-events: none;
}

.node-tooltip h3 {
  font-size: 1em;
  margin-bottom: 6px;
}

.node-tooltip .param-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.node-tooltip .param-key {
  color: var(--vscode-descriptionForeground);
}

.node-tooltip .param-val {
  font-family: var(--vscode-editor-font-family, monospace);
}

.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

@media (max-width: 700px) {
  .two-col { grid-template-columns: 1fr; }
}
`;
}
//# sourceMappingURL=theme.js.map