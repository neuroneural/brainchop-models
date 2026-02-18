import * as vscode from 'vscode';
import { BcmodelEditorProvider } from './bcmodelEditorProvider';

export function activate(context: vscode.ExtensionContext) {
  const provider = new BcmodelEditorProvider(context);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      'bcmodel.preview',
      provider,
      {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false,
      }
    )
  );
}

export function deactivate() {}
