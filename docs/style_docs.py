# Read in the file
with open('index.html', 'r') as f:
    data = f.read()

with open('how-vflow-works.html', 'r') as f:
    how_vflow_works = f.read()

# Fix a url
# data = data.replace('&lt;https://github.com/csinva/imodels</code>&gt;',
#                     'https://github.com/csinva/imodels</code>')
# data = data.replace('&lt;https://doi.org/10.5281/zenodo.4026887}&gt;',
#                     'https://doi.org/10.5281/zenodo.4026887}')

data = data.replace('<h1>Index</h1>',
                    '<h1>Index 🔍</h1>')

data = data.replace('<a href="https://yu-group.github.io/veridical-flow/">Docs</a>',
                    '')
# '<a href="https://github.com/Yu-Group/pcs-pipeline">Github</a>')

# data = data.replace('.html">imodels.', '.html">')
# data = data.replace('<h1 class="title">Package <code>vflow</code></h1>', '') # remove header
# data = data.replace('<th>Reference</th>', '<th white-space: nowrap>Reference</th>')

header_start = data.find('<header>')
header_end = data.find('</header>')
header = data[header_start:header_end]

# replace possibly out-of-date how-vflow-works header with index's
header_start = how_vflow_works.find('<header>')
header_end = how_vflow_works.find('</header>')
old_header = how_vflow_works[header_start:header_end]
how_vflow_works = how_vflow_works.replace(old_header, header)

sidebar_start = data.find('<nav id="sidebar">')
sidebar_end = data.find('</nav>')
sidebar = data[sidebar_start:sidebar_end]

# replace possibly out-of-date how-vflow-works sidebar with index's
sidebar_start = how_vflow_works.find('<nav id="sidebar">')
sidebar_end = how_vflow_works.find('</nav>')
old_sidebar = how_vflow_works[sidebar_start:sidebar_end]
how_vflow_works = how_vflow_works.replace(old_sidebar, sidebar)

# data = data.replace('<head>', "<head>\n<link rel='icon' href='https://csinva.io/imodels/docs/favicon.ico' type='image/x-icon'/ >\n")

# Write the files out again
with open('index.html', 'w') as f:
    f.write(data)

with open('how-vflow-works.html', 'w') as f:
    f.write(how_vflow_works)
