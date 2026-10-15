<%
  import os
  import re

  import pdoc
  from pdoc.html_helpers import extract_toc, glimpse, to_html as _to_html, format_git_link


  def link(dobj: pdoc.Doc, name=None):
    name = name or dobj.qualname + ('()' if isinstance(dobj, pdoc.Function) else '')
    if isinstance(dobj, pdoc.External) and not external_links:
        return name
    url = dobj.url(relative_to=module, link_prefix=link_prefix,
                   top_ancestor=not show_inherited_members)
    return f'<a title="{dobj.refname}" href="{url}">{name}</a>'


  def to_html(text):
    return _to_html(text, docformat=docformat, module=module, link=link, latex_math=latex_math)


  def strip_markup(text):
      """Plain prose for the description meta tag: the vflow docstring is the
      readme, which opens with the logo <img>, and escaped tags make a useless
      description in search results and link previews."""
      return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text or '')).strip()


  def get_annotation(bound_method, sep=':'):
    annot = show_type_annotations and bound_method(link=link) or ''
    if annot:
        annot = ' ' + sep + '\N{NBSP}' + annot
    return annot
%>

<%def name="ident(name)"><span class="ident">${name}</span></%def>

<%def name="show_source(d)">
  % if (show_source_code or git_link_template) and d.source and d.obj is not getattr(d.inherits, 'obj', None):
    <% git_link = format_git_link(git_link_template, d) %>
    % if show_source_code:
      <details class="source">
        <summary>
            <span>Expand source code</span>
            % if git_link:
              <a href="${git_link}" class="git-link">Browse git</a>
            %endif
        </summary>
        <pre><code class="python">${d.source | h}</code></pre>
      </details>
    % elif git_link:
      <div class="git-link-div"><a href="${git_link}" class="git-link">Browse git</a></div>
    %endif
  %endif
</%def>

<%def name="show_desc(d, short=False)">
  <%
  inherits = ' inherited' if d.inherits else ''
  docstring = glimpse(d.docstring) if short or inherits else d.docstring
  %>
  % if d.inherits:
      <p class="inheritance">
          <em>Inherited from:</em>
          % if hasattr(d.inherits, 'cls'):
              <code>${link(d.inherits.cls)}</code>.<code>${link(d.inherits, d.name)}</code>
          % else:
              <code>${link(d.inherits)}</code>
          % endif
      </p>
  % endif
  <div class="desc${inherits}">${docstring | to_html}</div>
  % if not isinstance(d, pdoc.Module):
  ${show_source(d)}
  % endif
</%def>

<%def name="show_module_list(modules)">
<h1>Python module list</h1>

% if not modules:
  <p>No modules found.</p>
% else:
  <dl id="http-server-module-list">
  % for name, desc in modules:
      <div class="flex">
      <dt><a href="${link_prefix}${name}">${name}</a></dt>
      <dd>${desc | glimpse, to_html}</dd>
      </div>
  % endfor
  </dl>
% endif
</%def>

<%def name="show_column_list(items)">
  <%
      two_column = len(items) >= 6 and all(len(i.name) < 20 for i in items)
  %>
  <ul class="${'two-column' if two_column else ''}">
  % for item in items:
    <li><code>${link(item, item.name)}</code></li>
  % endfor
  </ul>
</%def>

<%def name="show_module(module)">
  <%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  %>

  <%def name="show_func(f)">
    <dt id="${f.refname}"><code class="name flex">
        <%
            params = ', '.join(f.params(annotate=show_type_annotations, link=link))
            return_type = get_annotation(f.return_annotation, '\N{non-breaking hyphen}>')
        %>
        <span>${f.funcdef()} ${ident(f.name)}</span>(<span>${params})${return_type}</span>
    </code></dt>
    <dd>${show_desc(f)}</dd>
  </%def>

  <section id="section-intro">
  ${module.docstring | to_html}
  ${show_source(module)}
  </section>

  <section>
    % if submodules:
    <h2 class="section-title" id="header-submodules">Sub-modules</h2>
    <dl>
    % for m in submodules:
      <dt><code class="name">${link(m)}</code></dt>
      <dd>${show_desc(m, short=True)}</dd>
    % endfor
    </dl>
    % endif
  </section>

  <section>
    % if variables:
    <h2 class="section-title" id="header-variables">Global variables</h2>
    <dl>
    % for v in variables:
      <% return_type = get_annotation(v.type_annotation) %>
      <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
      <dd>${show_desc(v)}</dd>
    % endfor
    </dl>
    % endif
  </section>

  <section>
    % if functions:
    <h2 class="section-title" id="header-functions">Functions</h2>
    <dl>
    % for f in functions:
      ${show_func(f)}
    % endfor
    </dl>
    % endif
  </section>

  <section>
    % if classes:
    <h2 class="section-title" id="header-classes">Classes</h2>
    <dl>
    % for c in classes:
      <%
      class_vars = c.class_variables(show_inherited_members, sort=sort_identifiers)
      smethods = c.functions(show_inherited_members, sort=sort_identifiers)
      inst_vars = c.instance_variables(show_inherited_members, sort=sort_identifiers)
      methods = c.methods(show_inherited_members, sort=sort_identifiers)
      mro = c.mro()
      subclasses = c.subclasses()
      params = ', '.join(c.params(annotate=show_type_annotations, link=link))
      %>
      <dt id="${c.refname}"><code class="flex name class">
          <span>class ${ident(c.name)}</span>
          % if params:
              <span>(</span><span>${params})</span>
          % endif
      </code></dt>

      <dd>${show_desc(c)}

      % if mro:
          <h3>Ancestors</h3>
          <ul class="hlist">
          % for cls in mro:
              <li>${link(cls)}</li>
          % endfor
          </ul>
      %endif

      % if subclasses:
          <h3>Subclasses</h3>
          <ul class="hlist">
          % for sub in subclasses:
              <li>${link(sub)}</li>
          % endfor
          </ul>
      % endif
      % if class_vars:
          <h3>Class variables</h3>
          <dl>
          % for v in class_vars:
              <% return_type = get_annotation(v.type_annotation) %>
              <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
              <dd>${show_desc(v)}</dd>
          % endfor
          </dl>
      % endif
      % if smethods:
          <h3>Static methods</h3>
          <dl>
          % for f in smethods:
              ${show_func(f)}
          % endfor
          </dl>
      % endif
      % if inst_vars:
          <h3>Instance variables</h3>
          <dl>
          % for v in inst_vars:
              <% return_type = get_annotation(v.type_annotation) %>
              <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
              <dd>${show_desc(v)}</dd>
          % endfor
          </dl>
      % endif
      % if methods:
          <h3>Methods</h3>
          <dl>
          % for f in methods:
              ${show_func(f)}
          % endfor
          </dl>
      % endif

      % if not show_inherited_members:
          <%
              members = c.inherited_members()
          %>
          % if members:
              <h3>Inherited members</h3>
              <ul class="hlist">
              % for cls, mems in members:
                  <li><code><b>${link(cls)}</b></code>:
                      <ul class="hlist">
                          % for m in mems:
                              <li><code>${link(m, name=m.name)}</code></li>
                          % endfor
                      </ul>

                  </li>
              % endfor
              </ul>
          % endif
      % endif

      </dd>
    % endfor
    </dl>
    % endif
  </section>
</%def>

<%def name="module_index(module)">
  <%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  supermodule = module.supermodule
  %>
  <nav id="sidebar">

    % if google_search_query:
        <div class="gcse-search" style="height: 70px"
             data-as_oq="${' '.join(google_search_query.strip().split()) | h }"
             data-gaCategoryParameter="${module.refname | h}">
        </div>
    % endif

    % if lunr_search is not None:
      <%include file="_lunr_search.inc.mako"/>
    % endif

    <ul id="index">
    <li><h3>Guides</h3>
      <%doc>Newest first. Each entry carries its date; keep in step with docs/pages/
      and the PAGES table in docs/build_pages.py.</%doc>
      <ul class="post-list">
        <li><a href="agents.html" title="PCS for analyses done by AI agents">vflow for agents</a><span class="post-tag" title="Claude Code plugin">new</span><span class="post-date">Oct '26</span></li>
        <li><a href="vflow-internals.html" title="Vset, Vfunc and Subkey">How vflow works internally</a><span class="post-date">'22</span></li>
      </ul>
    </li>
    <li><h3>Notebooks</h3>
      <ul>
        <li><a href="notebooks/00_synthetic_classification.html">Synthetic classification</a></li>
        <li><a href="notebooks/01_enhancer.html">Enhancer genomics</a></li>
        <li><a href="notebooks/02_fmri.html">fMRI voxel prediction</a></li>
        <li><a href="notebooks/03_computer_vision_dnn.html">Fashion-MNIST with DNNs</a></li>
        <li><a href="notebooks/04_feat_importance_stability.html">Feature-importance stability</a></li>
      </ul>
    </li>
    % if module.name == 'vflow':
    <li><h3>On this page</h3>
      <div class="toc">${extract_toc(module.docstring)}</div>
    </li>
    % endif
    % if supermodule:
    <li><h3>Super-module</h3>
      <ul>
        <li><code>${link(supermodule)}</code></li>
      </ul>
    </li>
    % endif

    % if submodules:
    <li><h3><a href="#header-submodules">API</a></h3>
      <ul>
      % for m in submodules:
        <li><code>${link(m)}</code></li>
      % endfor
      </ul>
    </li>
    % endif

    % if variables:
    <li><h3><a href="#header-variables">Global variables</a></h3>
      ${show_column_list(variables)}
    </li>
    % endif

    % if functions:
    <li><h3><a href="#header-functions">Functions</a></h3>
      ${show_column_list(functions)}
    </li>
    % endif

    % if classes:
    <li><h3><a href="#header-classes">Classes</a></h3>
      <ul>
      % for c in classes:
        <li>
        <h4><code>${link(c)}</code></h4>
        <%
            members = c.functions(sort=sort_identifiers) + c.methods(sort=sort_identifiers)
            if list_class_variables_in_index:
                members += (c.instance_variables(sort=sort_identifiers) +
                            c.class_variables(sort=sort_identifiers))
            if not show_inherited_members:
                members = [i for i in members if not i.inherits]
            if sort_identifiers:
              members = sorted(members)
        %>
        % if members:
          ${show_column_list(members)}
        % endif
        </li>
      % endfor
      </ul>
    </li>
    % endif

    </ul>
  </nav>
</%def>

<!doctype html>
<html lang="${html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
  <meta name="generator" content="pdoc ${pdoc.__version__}" />

<%
    module_list = 'modules' in context.keys()  # Whether we're showing module list in server mode
%>

  % if module_list:
    <title>Python module list</title>
    <meta name="description" content="A list of documented Python modules." />
  % else:
    <title>${'vflow' if module.name == 'vflow' else module.name}</title>
    <meta name="description" content="${module.docstring | strip_markup, glimpse, trim, h}" />
  % endif

  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/sanitize.min.css" integrity="sha256-PK9q560IAAa6WVRRh76LtCaI8pjTJ2z11v0miyNNjrs=" crossorigin>
  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
  % if syntax_highlighting:
    <link rel="stylesheet preload" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/styles/${hljs_style}.min.css" crossorigin>
  %endif

  <%namespace name="css" file="css.mako" />
  <style>${css.mobile()}</style>
  <style media="screen and (min-width: 700px)">${css.desktop()}</style>
  <style media="print">${css.print()}</style>
  <link rel="stylesheet" href="github.css">
  <link rel="stylesheet" href="style.css">
  <link rel="icon" type="image/png" sizes="64x64" href="favicon.png">
  <link rel="icon" type="image/svg+xml" sizes="any" href="favicon.svg">

  % if google_analytics:
    <script>
    window.ga=window.ga||function(){(ga.q=ga.q||[]).push(arguments)};ga.l=+new Date;
    ga('create', '${google_analytics}', 'auto'); ga('send', 'pageview');
    </script><script async src='https://www.google-analytics.com/analytics.js'></script>
  % endif

  % if google_search_query:
    <link rel="preconnect" href="https://www.google.com">
    <script async src="https://cse.google.com/cse.js?cx=017837193012385208679:pey8ky8gdqw"></script>
    <style>
        .gsc-control-cse {padding:0 !important;margin-top:1em}
        body.gsc-overflow-hidden #sidebar {overflow: visible;}
    </style>
  % endif

  % if latex_math:
    <script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML" integrity="sha256-kZafAc6mZvK3W3v1pHOcUix30OHQN6pU/NO2oFkqZVw=" crossorigin></script>
  % endif

  % if syntax_highlighting:
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/highlight.min.js" integrity="sha256-Uv3H6lx7dJmRfRvH8TH6kJD1TSK1aFcwgx+mdg3epi8=" crossorigin></script>
    <script>window.addEventListener('DOMContentLoaded', () => hljs.initHighlighting())</script>
  % endif

  <%include file="head.mako"/>
</head>
<body>
<header id="site-header">
  <a href="index.html" class="nav-icon" aria-label="Home" title="Home">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M3 10.5 12 3l9 7.5"></path>
      <path d="M5 9.5V21h14V9.5"></path>
      <path d="M9.5 21v-6h5v6"></path>
    </svg>
  </a>
  <p id="site-tagline"><b>vflow</b> &middot; Stability analysis for data-science pipelines, with PCS.</p>
  <a href="index.html" id="site-logo">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 124" role="img" aria-label="vflow"><title>vflow</title><g transform="translate(14,10)"><path d="M6,40 L31,100 L56,40 M84,100 V30 Q84,4 110,4 M68,46 H104 M132,2 V82 Q132,100 150,100 M236,40 L254,100 L271,52 L288,100 L306,40" fill="none" stroke="#1F2A33" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"/><g class="vo"><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2f85e7" transform="translate(190.0,40.0) rotate(-90.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2d9cc6" transform="translate(211.2,48.8) rotate(-45.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2bb3a6" transform="translate(220.0,70.0) rotate(0.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2bb8a0" transform="translate(211.2,91.2) rotate(45.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2bb3a6" transform="translate(190.0,100.0) rotate(90.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2d9cc6" transform="translate(168.8,91.2) rotate(135.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2f85e7" transform="translate(160.0,70.0) rotate(180.0)"/><path d="M0,-12.540000000000001 C3.1350000000000002,-7.695000000000001 5.7,-3.9899999999999998 5.7,0 A5.7,5.7 0 1 1 -5.7,0 C-5.7,-3.9899999999999998 -3.1350000000000002,-7.695000000000001 0,-12.540000000000001 Z" fill="#2f80ed" transform="translate(168.8,48.8) rotate(225.0)"/></g></g></svg>
  </a>
  <a href="https://github.com/Yu-Group/veridical-flow" class="nav-icon"
     aria-label="View source on GitHub" title="GitHub">
    <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
        1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
        0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04
        2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
        0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    </svg>
  </a>
</header>
<main>
  % if module_list:
    <article id="content">
      ${show_module_list(modules)}
    </article>
  % else:
    <article id="content">
      ${show_module(module)}
    </article>
    ${module_index(module)}
  % endif
</main>

<footer id="footer">
    <p>vflow &middot; <a href="https://github.com/Yu-Group/veridical-flow">GitHub</a> &middot; <a href="https://pypi.org/project/vflow/">PyPI</a> &middot; Generated by <a href="https://pdoc3.github.io/pdoc"><cite>pdoc</cite> ${pdoc.__version__}</a></p>
</footer>

% if http_server and module:  ## Auto-reload on file change in dev mode
    <script>
    setInterval(() =>
        fetch(window.location.href, {
            method: "HEAD",
            cache: "no-store",
            headers: {"If-None-Match": "${os.stat(module.obj.__file__).st_mtime}"},
        }).then(response => response.ok && window.location.reload()), 700);
    </script>
% endif
</body>
</html>
