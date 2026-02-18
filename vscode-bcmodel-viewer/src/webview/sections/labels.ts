import { ParsedBcmodel } from '../../bcmodelParser';
import { escapeHtml } from '../utils';

export function renderLabels(parsed: ParsedBcmodel): string {
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
        <span>${escapeHtml(l.name)}</span>
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
