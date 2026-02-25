import * as vscode from 'vscode';
import { parseBcmodel } from './bcmodelParser';
import { buildWebviewHtml } from './webview/buildHtml';

export class BcmodelEditorProvider implements vscode.CustomReadonlyEditorProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  async openCustomDocument(
    uri: vscode.Uri,
    _openContext: vscode.CustomDocumentOpenContext,
    _token: vscode.CancellationToken
  ): Promise<vscode.CustomDocument> {
    return { uri, dispose: () => {} };
  }

  async resolveCustomEditor(
    document: vscode.CustomDocument,
    webviewPanel: vscode.WebviewPanel,
    _token: vscode.CancellationToken
  ): Promise<void> {
    webviewPanel.webview.options = { enableScripts: true };

    try {
      const fileData = await vscode.workspace.fs.readFile(document.uri);
      const buffer: ArrayBuffer = fileData.buffer.slice(
        fileData.byteOffset,
        fileData.byteOffset + fileData.byteLength
      ) as ArrayBuffer;
      const parsed = parseBcmodel(buffer);
      webviewPanel.webview.html = buildWebviewHtml(parsed);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      webviewPanel.webview.html = `<!DOCTYPE html>
<html><body style="color:var(--vscode-errorForeground);padding:20px;font-family:sans-serif;">
<h2>Failed to load .bcmodel file</h2>
<pre>${escapeHtml(message)}</pre>
</body></html>`;
    }
  }
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
