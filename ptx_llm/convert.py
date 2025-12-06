#!/usr/bin/env python3
"""
Convert PTX ISA HTML (ptx_isa.html) into:
- One big markdown file: ptx_export/ptx_full.md
- A tree structure (JSON but saved as .txt): ptx_export/ptx_tree.txt
- Optional: per-section markdown files in ptx_export/sections/
"""

import re
import json
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ---------- Config ----------
HTML_PATH = Path("ptx_isa.html")          # your downloaded file
OUT_DIR = Path("ptx_export")              # output directory
OUT_DIR.mkdir(exist_ok=True)

SECTIONS_DIR = OUT_DIR / "sections"       # per-section markdown
SECTIONS_DIR.mkdir(exist_ok=True)

HEADING_TAGS = [f"h{i}" for i in range(1, 7)]  # h1..h6

# ---------- 1. Load HTML ----------
if not HTML_PATH.exists():
    raise FileNotFoundError(f"{HTML_PATH} not found. Put ptx_isa.html in this directory.")

# Some downloads of the PTX docs come from a browser “view source” page where the
# real HTML is HTML-escaped inside <span id=\"lineX\"> wrappers.  If that form is
# detected, unwrap it back to the real HTML before parsing.
raw_html = HTML_PATH.read_text(encoding="utf-8")
soup = BeautifulSoup(raw_html, "lxml")

if soup.body and soup.body.get("id") == "viewsource":
    line_spans = soup.select("span[id^=line]")
    lines = [span.get_text() for span in line_spans]
    if lines:
        unwrapped_html = "\n".join(lines)
        soup = BeautifulSoup(unwrapped_html, "lxml")

# Optional: strip nav/sidebar/noise, if present
for tag in soup.select("nav, header, footer, .sphinxsidebar, .wy-nav-side"):
    tag.decompose()

# ---------- 2. Collect headings ----------
headings = []
for h in soup.find_all(HEADING_TAGS):
    level = int(h.name[1])        # 1..6
    text = " ".join(h.stripped_strings)
    anchor = h.get("id") or h.get("name")
    headings.append({
        "tag": h,
        "level": level,
        "text": text,
        "anchor": anchor,
    })

# ---------- 3. Extract section content between headings ----------
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "section"

nodes = []

for idx, h in enumerate(headings):
    start_tag = h["tag"]
    contents = []

    # Grab everything until the next heading
    for sib in start_tag.next_siblings:
        if isinstance(sib, str):
            contents.append(sib)
            continue
        if sib.name in HEADING_TAGS:
            break
        contents.append(sib)

    section_html = "".join(str(c) for c in contents).strip()
    section_md = md(section_html) if section_html else ""

    node = {
        "title": h["text"],
        "level": h["level"],
        "anchor": h["anchor"],
        "slug": h["anchor"] or slugify(h["text"]),
        "html": section_html,
        "markdown_body": section_md,
        "children": [],
    }
    nodes.append(node)

# ---------- 4. Build a tree from the flat list ----------
root = {"title": "PTX ISA", "level": 0, "children": []}
stack = [root]  # stack of last node at each depth

for node in nodes:
    # Pop until parent has smaller level
    while stack and stack[-1]["level"] >= node["level"]:
        stack.pop()

    parent = stack[-1]
    parent["children"].append(node)

    # Path from root to this node
    node["path"] = parent.get("path", []) + [node["title"]]

    stack.append(node)

# ---------- 5. Add full markdown per node ----------
def add_full_markdown(node):
    if node["level"] > 0:
        header = "#" * node["level"] + " " + node["title"]
        node["markdown_full"] = header + "\n\n" + (node["markdown_body"] or "")
    else:
        node["markdown_full"] = "# PTX ISA\n\n"

    for child in node["children"]:
        add_full_markdown(child)

add_full_markdown(root)

# ---------- 6. Build a concise tree index ----------
def build_tree_index(node):
    entry = {
        "title": node["title"],
        "level": node["level"],
        "slug": node.get("slug"),
        "anchor": node.get("anchor"),
        "path": node.get("path", []),
    }
    if node["level"] > 0:
        entry["file"] = f"sections/{node['slug']}.md"

    entry["children"] = [build_tree_index(child) for child in node["children"]]
    return entry

tree_path = OUT_DIR / "ptx_tree.txt"
tree_path.write_text(
    json.dumps(build_tree_index(root), indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"Wrote tree to {tree_path}")

# ---------- 7. Write one big markdown ----------
all_md_parts = []

def collect_markdown(node):
    # Skip root's own header if you don’t want it duplicated
    if node["level"] > 0:
        all_md_parts.append(node["markdown_full"])
    for child in node["children"]:
        collect_markdown(child)

collect_markdown(root)

full_md_path = OUT_DIR / "ptx_full.md"
full_md_path.write_text("\n\n".join(all_md_parts), encoding="utf-8")
print(f"Wrote full markdown to {full_md_path}")

# ---------- 8. Optional: per-section markdown files ----------
def dump_sections(node):
    if node.get("level", 0) > 0:  # skip synthetic root
        filename = f"{node['slug'] or 'section'}.md"
        (SECTIONS_DIR / filename).write_text(node["markdown_full"], encoding="utf-8")
    for child in node["children"]:
        dump_sections(child)

dump_sections(root)
print(f"Wrote per-section markdown files to {SECTIONS_DIR}")
