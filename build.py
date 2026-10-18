# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py>=3", "mdit-py-plugins>=0.4"]
# ///
"""Build the sigma-0 blog: src/posts/*.md -> posts/<slug>/index.html + index.html.

Usage:  uv run build.py

A post is one Markdown file with a front-matter block (title, date, tags,
optional description); `name.md` becomes posts/name/. Math in $...$ and
$$...$$ is rendered in the browser by the vendored KaTeX.
"""
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "posts"
SITE_TITLE = "sigma-0 blog"
REPO_URL = "https://github.com/bold-lab-ai/sigma-0"

md = (
    MarkdownIt("commonmark", {"html": True})
    .enable("table")
    .use(front_matter_plugin)
    .use(dollarmath_plugin, double_inline=True)
)


def render(body):
    """Markdown -> HTML; each table gets a scrollable, bordered wrapper."""
    out = md.render(body)
    return out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def parse_front_matter(text):
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return meta, text
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip() for v in val[1:-1].split(",") if v.strip()]
        meta[key.strip()] = val
    return meta, text[m.end():]


def page(title, body, depth, description=""):
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="author" content="Bold.ai">
  <meta name="description" content="{html.escape(description)}">
  <meta name="theme-color" content="#ffffff">
  <title>{html.escape(title)}</title>
  <link rel="icon" href="{up}favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="{up}vendor/katex/katex.min.css">
  <link rel="stylesheet" href="{up}site.css">
  <script defer src="{up}vendor/katex/katex.min.js"></script>
  <script defer src="{up}math.js"></script>
</head>
<body>
<div class="wrap">
{body}
  <footer><a href="{up}">{SITE_TITLE}</a> · <a href="{REPO_URL}">{REPO_URL.split('//')[1]}</a></footer>
</div>
</body>
</html>
"""


def build_post(path):
    slug = path.name[: -len(".md")]
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    body = re.sub(r"^\s*# .*\n", "", body, count=1)  # title comes from front matter
    title = meta.get("title", slug)
    date = meta.get("date", "")
    tags = meta.get("tags", [])
    tag_html = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in tags)
    head = f"""  <header class="post-head">
    <p class="crumb"><a href="../../">{SITE_TITLE}</a></p>
    <h1>{html.escape(title)}</h1>
    <p class="meta"><time>{html.escape(date)}</time>{tag_html}</p>
  </header>
  <article class="prose">
{render(body)}  </article>"""
    out = ROOT / "posts" / slug / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(title, head, 2, meta.get("description", title)), encoding="utf-8")
    return dict(slug=slug, title=title, date=date, tags=tags,
                description=meta.get("description", ""))


def build_index(posts):
    items = []
    for p in sorted(posts, key=lambda p: p["date"], reverse=True):
        desc = f'<p>{html.escape(p["description"])}</p>' if p["description"] else ""
        items.append(f'    <li><time>{html.escape(p["date"])}</time>'
                     f'<a class="title" href="posts/{p["slug"]}/">{html.escape(p["title"])}</a>{desc}</li>')
    body = f"""  <header>
    <h1>sigma-0</h1>
    <p class="lede">Notes on modelling limit order books with language models.</p>
  </header>
  <ul class="posts">
{chr(10).join(items)}
  </ul>"""
    (ROOT / "index.html").write_text(
        page(SITE_TITLE, body, 0, "Notes from the sigma-0 project."), encoding="utf-8")


if __name__ == "__main__":
    posts = [build_post(p) for p in sorted(SRC.glob("*.md"))]
    build_index(posts)
    print(f"built {len(posts)} post page(s) + index.html")
