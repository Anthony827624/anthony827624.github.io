const blogDirectory = 'contents/blogs/';

function escapeBlogHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Keep TeX intact: Markdown otherwise consumes backslashes and interprets underscores.
const blogMarkdown = new marked.Marked({
    mangle: false,
    headerIds: false,
    extensions: [
        {
            name: 'displayMath',
            level: 'block',
            start: source => source.search(/\$\$|\\\[/),
            tokenizer(source) {
                const match = /^(?:\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])(?:\n|$)/.exec(source);
                if (match) return { type: 'displayMath', raw: match[0] };
            },
            renderer: token => '<div class="blog-math">' + escapeBlogHtml(token.raw) + '</div>\n',
        },
        {
            name: 'inlineMath',
            level: 'inline',
            start: source => source.search(/\$|\\\(/),
            tokenizer(source) {
                const match = /^(?:\$\$[^\n]+?\$\$|\$(?!\$)(?:\\.|[^$\n\\])+?\$(?!\$)|\\\([^\n]+?\\\))/.exec(source);
                if (match) return { type: 'inlineMath', raw: match[0] };
            },
            renderer: token => '<span>' + escapeBlogHtml(token.raw) + '</span>',
        },
    ],
});

function parseBlog(markdown, file) {
    const match = /^\uFEFF?---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(markdown);
    if (!match) throw new Error(file + ': missing YAML front matter');
    const metadata = jsyaml.load(match[1], { schema: jsyaml.JSON_SCHEMA });
    for (const field of ['title', 'tldr', 'date']) {
        if (typeof metadata?.[field] !== 'string' || !metadata[field].trim()) {
            throw new Error(file + ': ' + field + ' must be a non-empty string');
        }
    }
    const date = metadata.date;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0, 10) !== date) {
        throw new Error(file + ': date must be a valid YYYY-MM-DD date');
    }
    return { file, title: metadata.title.trim(), tldr: metadata.tldr.trim(), date, markdown: markdown.slice(match[0].length) };
}

async function fetchBlogFiles() {
    const response = await fetch(blogDirectory + 'index.json');
    if (!response.ok) throw new Error('Cannot load blog index');
    const files = await response.json();
    if (!Array.isArray(files) || files.some(file => typeof file !== 'string' || !/^[^/\\]+\.md$/i.test(file))) {
        throw new Error('Blog index must contain Markdown filenames without directory paths');
    }
    return [...new Set(files)];
}

async function fetchBlog(file) {
    const response = await fetch(blogDirectory + encodeURIComponent(file));
    if (!response.ok) throw new Error('Cannot load blog: ' + file);
    return parseBlog(await response.text(), file);
}

function blogSummary(blog, headingTag) {
    const article = document.createElement('article');
    const heading = document.createElement(headingTag);
    const date = document.createElement('time');
    const summary = document.createElement('p');
    article.className = 'blog-summary tex2jax_ignore';
    if (headingTag === 'h1') {
        heading.textContent = blog.title;
    } else {
        const link = document.createElement('a');
        link.href = 'blog.html?file=' + encodeURIComponent(blog.file);
        link.textContent = blog.title;
        heading.appendChild(link);
    }
    date.dateTime = blog.date;
    date.textContent = blog.date;
    summary.textContent = 'TLDR: ' + blog.tldr;
    article.append(heading, date, summary);
    return article;
}

async function loadBlogs() {
    const target = document.getElementById('blogs-md');
    try {
        const files = await fetchBlogFiles();
        const results = await Promise.allSettled(files.map(fetchBlog));
        const blogs = results.filter(result => result.status === 'fulfilled').map(result => result.value);
        blogs.sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title));
        target.replaceChildren(...blogs.map(blog => blogSummary(blog, 'h3')));
        const failures = results.filter(result => result.status === 'rejected');
        failures.forEach(result => console.error(result.reason));
        if (!files.length || failures.length) {
            const message = document.createElement('p');
            message.textContent = failures.length ? 'Some posts could not be loaded. Please try again later.' : 'No posts yet.';
            target.appendChild(message);
        }
    } catch (error) {
        console.error(error);
        target.textContent = 'Blog posts could not be loaded. Please try again later.';
    }
}

async function loadBlogPage() {
    const target = document.getElementById('blog-content');
    try {
        const file = new URLSearchParams(window.location.search).get('file');
        const files = await fetchBlogFiles();
        if (!files.includes(file)) throw new Error('Post not found');
        const blog = await fetchBlog(file);
        document.title = blog.title + ' | ' + document.title;
        document.querySelector('meta[name="description"]').content = blog.tldr;
        const body = document.createElement('div');
        body.className = 'blog-body';
        body.innerHTML = sanitizeMarkdownHtml(blogMarkdown.parse(blog.markdown));
        const sourceUrl = new URL(blogDirectory + encodeURIComponent(file), window.location.href);
        body.querySelectorAll('[src], a[href]').forEach(node => {
            const attribute = node.hasAttribute('src') ? 'src' : 'href';
            const value = node.getAttribute(attribute);
            if (value && !value.startsWith('#')) node.setAttribute(attribute, new URL(value, sourceUrl).href);
        });
        target.replaceChildren(blogSummary(blog, 'h1'), body);
    } catch (error) {
        console.error(error);
        target.textContent = 'This post is unavailable. Please return to Blogs and try again.';
    }
}
