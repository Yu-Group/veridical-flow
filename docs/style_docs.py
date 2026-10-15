"""Post-process the pdoc-generated index page.

Run after pdoc has written the API pages into docs/; build_pages.py then
builds the article pages from the result.
"""

import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")

with open(INDEX, encoding="utf-8") as f:
    data = f.read()


def drop(pattern, text, what):
    """Remove `pattern` from `text`, and say so when it matches nothing."""
    new_text, n = re.subn(pattern, "", text, flags=re.S)
    if n == 0:
        print(f"  style_docs.py: nothing removed for {what}; has the readme changed?")
    return new_text


# the readme opens with a large centered logo, which the header bar now carries
data = re.sub(
    r'<p align="center">\s*<img src="https://yu-group.github.io/veridical-flow/logo_vflow_straight.png"[^>]*>\s*</p>',
    "",
    data,
    count=1,
)

# a self-referential "Docs" link from the readme is noise on the docs site itself
data = data.replace('<a href="https://yu-group.github.io/veridical-flow/">Docs</a>', "")

# The index is the readme, which ends with References. Everything pdoc appends
# after it (the sub-module list and the re-exported functions and classes) is
# documented on the module pages, so cut it from the article.
data = drop(
    r'<section>\s*<h2 class="section-title" id="header-submodules">.*?(?=</article>)',
    data,
    "the API sections after References",
)

# pdoc's "Expand source code" toggle for the package __init__ would be the last
# thing on the page, after References
data = drop(
    r'<details class="source">.*?</details>\s*(?=</section>)',
    data,
    "the package source toggle",
)

# The sidebar section headings pointed at those sections; keep them as plain
# labels.
data = re.sub(r'<h3><a href="#header-[a-z]+">(.*?)</a></h3>', r"<h3>\1</h3>", data)

# The sidebar's function and class links pointed at anchors on this page that
# no longer exist. Point each at the module page that documents it:
# "#vflow.build_vset" -> "helpers.html#vflow.helpers.build_vset".
anchors = {}
for page in glob.glob(os.path.join(HERE, "*.html")):
    name = os.path.basename(page)
    if name == "index.html":
        continue
    with open(page, encoding="utf-8") as f:
        for ref in re.findall(r'<dt id="(vflow\.[\w.]+)">', f.read()):
            # vflow.<module>.<qualname> -> <qualname>
            anchors.setdefault(ref.split(".", 2)[-1], f"{name}#{ref}")


def _repoint(match):
    target = anchors.get(match.group(1))
    return f'href="{target}"' if target else match.group(0)


data, n_repointed = re.subn(r'href="#vflow\.([\w.]+)"', _repoint, data)
left = sorted(set(re.findall(r'href="#vflow\.([\w.]+)"', data)))
print(f"  style_docs.py: repointed {n_repointed} sidebar links to module pages")
if left:
    print(f"  style_docs.py: no module page documents {left}")

with open(INDEX, "w", encoding="utf-8") as f:
    f.write(data)
print("  style_docs.py: cleaned index.html")
