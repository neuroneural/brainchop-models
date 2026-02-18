"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.BcmodelEditorProvider = void 0;
const vscode = __importStar(require("vscode"));
const bcmodelParser_1 = require("./bcmodelParser");
const buildHtml_1 = require("./webview/buildHtml");
class BcmodelEditorProvider {
    constructor(context) {
        this.context = context;
    }
    async openCustomDocument(uri, _openContext, _token) {
        return { uri, dispose: () => { } };
    }
    async resolveCustomEditor(document, webviewPanel, _token) {
        webviewPanel.webview.options = { enableScripts: true };
        try {
            const fileData = await vscode.workspace.fs.readFile(document.uri);
            const buffer = fileData.buffer.slice(fileData.byteOffset, fileData.byteOffset + fileData.byteLength);
            const parsed = (0, bcmodelParser_1.parseBcmodel)(buffer);
            webviewPanel.webview.html = (0, buildHtml_1.buildWebviewHtml)(parsed);
        }
        catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            webviewPanel.webview.html = `<!DOCTYPE html>
<html><body style="color:var(--vscode-errorForeground);padding:20px;font-family:sans-serif;">
<h2>Failed to load .bcmodel file</h2>
<pre>${escapeHtml(message)}</pre>
</body></html>`;
        }
    }
}
exports.BcmodelEditorProvider = BcmodelEditorProvider;
function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
//# sourceMappingURL=bcmodelEditorProvider.js.map