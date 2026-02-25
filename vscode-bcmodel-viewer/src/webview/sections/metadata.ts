import { ParsedBcmodel } from '../../bcmodelParser';
import { escapeHtml, formatBytes, formatShape } from '../utils';

export function renderMetadata(parsed: ParsedBcmodel): string {
  const { metadata, input, output } = parsed.header;

  const rows: [string, string][] = [
    ['Type', metadata.type],
    ['Input Shape', formatShape(input.shape)],
    ['Input dtype', input.dtype],
    ['Data Layout', input.data_layout],
    ['Output Classes', String(output.num_classes)],
  ];

  if (metadata.description) {
    rows.unshift(['Description', metadata.description]);
  }
  if (metadata.source_framework) {
    rows.push(['Source Framework', metadata.source_framework]);
  }
  if (metadata.authors && metadata.authors.length > 0) {
    rows.push(['Authors', metadata.authors.join(', ')]);
  }
  if (metadata.license) {
    rows.push(['License', metadata.license]);
  }

  const kvHtml = rows
    .map(([k, v]) => `<div class="key">${escapeHtml(k)}</div><div class="val">${escapeHtml(v)}</div>`)
    .join('\n');

  return `
  <section class="card">
    <h2>Model Info</h2>
    <div class="kv-grid">
      ${kvHtml}
    </div>
  </section>`;
}
