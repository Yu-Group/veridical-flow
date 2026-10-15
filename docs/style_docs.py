"""Post-process the pdoc-generated index page.

Run from docs/ after pdoc has written index.html; build_pages.py then builds
the article pages from the result.
"""

import os
import re

INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

with open(INDEX, encoding="utf-8") as f:
    data = f.read()

# the readme opens with a large centered logo, which the header bar now carries
data = re.sub(
    r'<p align="center">\s*<img src="https://yu-group.github.io/veridical-flow/logo_vflow_straight.png"[^>]*>\s*</p>',
    "",
    data,
    count=1,
)

# a self-referential "Docs" link from the readme is noise on the docs site itself
data = data.replace('<a href="https://yu-group.github.io/veridical-flow/">Docs</a>', "")

with open(INDEX, "w", encoding="utf-8") as f:
    f.write(data)
print("  style_docs.py: cleaned index.html")
