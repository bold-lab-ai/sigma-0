# sigma-0 blog

**Website: https://bold-lab-ai.github.io/sigma-0/**

This orphan branch holds only the blog. It shares no history with `main`.

## Write a post

1. Add `src/posts/<slug>.md`, starting with:

   ```
   ---
   title: Post title
   date: 2026-10-18
   description: One sentence for the post list.
   tags: [tag1, tag2]
   ---
   ```

   Math: `$...$` inline, `$$...$$` display (rendered with KaTeX).
2. Run `uv run build.py`. It writes `posts/<slug>/index.html` and `index.html`.
3. Commit the source and the generated HTML, then push. GitHub Pages serves the branch as is (no build step).

Preview locally: `python3 -m http.server` in this folder, then open http://localhost:8000/.

## Layout

| Path | What |
|---|---|
| `src/posts/` | Markdown sources |
| `posts/` | generated pages |
| `build.py` | Markdown → HTML (markdown-it-py + dollarmath) |
| `site.css`, `fonts/` | style shared with the nanoSigma site (CMU Serif, yellow→pink palette) |
| `math.js`, `vendor/katex/` | KaTeX 0.16 (MIT licence), renders the math in the browser |
