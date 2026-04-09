# Instruction: Avoid Whole Thirdparty Context

- Do not load the entire `thirdparty` folder into model context.
- Treat `thirdparty` as a reference-only area: use targeted searches (`rg`) and specific file reads only when needed.
- If third-party code is relevant, cite exact files or snippets instead of broad folder-wide ingestion.
- If uncertain whether a read is too broad, prefer narrowing scope first and ask the user before expanding.
