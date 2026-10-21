# Hongshuo Zhao's Homepage

Personal academic website with Markdown content, a rotating background, and a blog supporting images and LaTeX math. Hosted on [GitHub Pages](https://anthony827624.github.io/); no build step is required.

This repository is forked from [Sen Li's academic-homepage-template](https://github.com/senli1073/academic-homepage-template), which is based on [Start Bootstrap New Age](https://github.com/StartBootstrap/startbootstrap-new-age). This fork adds configurable sections, background rotation, and blog pages.

## Repository structure

```text
.
├── index.html                 # Homepage
├── blog.html                  # Individual blog page
├── contents/
│   ├── config.yml             # Site settings, navigation, sections, backgrounds
│   ├── home.md                # Bio, contact details, education, interests
│   ├── publications.md
│   ├── projects.md
│   ├── awards.md
│   └── blogs/
│       ├── index.json         # Filenames of posts to display
│       └── example.md         # Copyable template; not listed by default
├── static/
│   ├── assets/
│   │   ├── background/        # Background images
│   │   └── img/photo.jpg      # Profile photo
│   ├── css/                   # Theme and custom styles
│   └── js/                    # Content loading, Markdown, YAML, MathJax
└── tests/                     # Node.js checks
```

## Edit the website

Edit `contents/config.yml` to change site text, footer links, backgrounds, and section order. Each regular section loads `contents/<id>.md`; navigation is derived from the same configuration. For example:

```yaml
sections:
  - id: publications
    nav: PUBLICATIONS
    title: PUBLICATIONS
    icon: bi-file-text-fill
```

`id` identifies the section, `nav` labels its navigation link, `title` sets its heading, and `icon` is optional. Navigation links default to `#<id>`; use `href` to override them. The `blogs` section loads the blog index instead of a matching Markdown file.

Replace the profile photo at `static/assets/img/photo.jpg`. Add background images under `static/assets/background/` and list their paths in `backgrounds` in the configuration. Adjust `background-interval-ms` and `background-overlay` there as needed.

## Write a blog post

Create a `.md` file directly inside `contents/blogs/`, or copy `example.md`. Start with YAML front matter:

```markdown
---
title: "My article title"
tldr: "A short summary for the homepage."
date: "2026-10-20"
---

## First section

Write the article in Markdown here.
```

All three fields are required. Use a valid `YYYY-MM-DD` date; keep values quoted and escape embedded double quotes as `\"`. The title and TLDR display as plain text. The article page adds the title, date, and TLDR automatically.

Add the filename to `contents/blogs/index.json`:

```json
["my notes.md", "another-post.md"]
```

Filenames may contain spaces or non-English characters and do not determine display text or sorting. Posts appear below AWARDS, newest first; ties are sorted by title. The index is maintained manually because this static site does not scan directories. Renaming a file requires updating the index and changes its article URL.

### Images, math, and code

Image paths are relative to the Markdown file. For example, save an image at `contents/blogs/images/result.png` and write:

```markdown
![Description of the result](images/result.png)
```

External image URLs also work. Images retain their proportions and shrink to fit the page.

Use `$x_i^2$` or `\(x_i^2\)` for inline math. For display math, use `$$` or `\[...\]`, with blank lines around the block:

```latex
$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}(f_\theta(x_i)-y_i)^2
$$
```

MathJax is bundled locally. Use fenced code blocks for code; syntax highlighting is not included. Put literal dollar amounts in inline code, such as `` `$20` ``, to avoid math delimiters. Embedded scripts and iframes are not supported.

## Preview and check

Serve the repository over HTTP rather than opening the HTML files directly:

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000/`. To preview the sample article, add `"example.md"` to the blog index and visit `http://localhost:8000/blog.html?file=example.md`.

Run the checks with Node.js:

```sh
node tests/blogs.test.js
node tests/background-selection.test.js
node tests/template-quality.test.js
```

## Publish

Commit and push your changes to the branch used by GitHub Pages. The site is served directly from the repository root. If reusing this fork, update the personal content, configuration, and repository name for your own account.

## License

[MIT](LICENSE). Original template copyright Sen Li, 2023–2025; see the license for attribution and reuse terms.
