const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const context = vm.createContext({
    marked: require('../static/js/marked.min.js'),
    jsyaml: require('../static/js/js-yaml.min.js'),
    URL, URLSearchParams, console,
});
vm.runInContext(read('static/js/blogs.js'), context);
const parse = context.parseBlog;
const source = read('contents/blogs/example.md');
const blog = parse(source, '随手记 #1.md');
assert.equal(blog.title, '博客写作示例：图片与数学公式');
assert.equal(blog.file, '随手记 #1.md');
assert.equal(blog.date, '2026-10-20');
assert.ok(blog.markdown.startsWith('\n这是一篇'));
assert.equal(parse(source.replaceAll('\n', '\r\n'), 'x.md').date, blog.date);
assert.equal(parse('\uFEFF' + source, 'x.md').date, blog.date);
assert.throws(() => parse('No front matter', 'x.md'), /front matter/);
assert.throws(() => parse(source.replace('title:', 'name:'), 'x.md'), /title/);
assert.throws(() => parse(source.replace('2026-10-20', '2026-02-30'), 'x.md'), /date/);
assert.throws(() => parse(source.replace('2026-10-20', '20-10-2026'), 'x.md'), /date/);
assert.throws(() => parse(source.replace(/tldr: .*/, 'tldr: []'), 'x.md'), /tldr/);
const renderer = vm.runInContext('blogMarkdown', context);
const render = markdown => renderer.parse(markdown);
const html = render(blog.markdown);
assert.ok(html.includes('src="../../static/assets/background/background_0.jpg"'));
assert.ok(html.includes('f_\\theta(x_i)'));
assert.ok(html.includes('a_1 &amp; b_1 \\\\'));
assert.ok(!html.includes('<em>'));
assert.match(render('Value $x_i + y_j$.'), /<span>\$x_i \+ y_j\$<\/span>/);
assert.match(render(String.raw`Value \(x_i\).`), /<span>\\\(x_i\\\)<\/span>/);
assert.match(render(String.raw`\[x_i\]`), /class="blog-math"/);
assert.match(render('$x < y$'), /x &lt; y/);
assert.match(render('`$x_i$`'), /<code>\$x_i\$<\/code>/);
assert.match(render('```latex\n$x_i$\n```'), /<code class="language-latex">\$x_i\$/);
const config = context.jsyaml.load(read('contents/config.yml'));
assert.equal(config.sections[config.sections.findIndex(section => section.id === 'awards') + 1].id, 'blogs');

(async () => {
    context.fetch = async () => ({ ok: true, json: async () => ['随手记 #1.md', '随手记 #1.md'] });
    assert.deepEqual(Array.from(await context.fetchBlogFiles()), ['随手记 #1.md']);
    for (const value of [['../secret.md'], ['folder/post.md'], ['post.txt'], null]) {
        context.fetch = async () => ({ ok: true, json: async () => value });
        await assert.rejects(() => context.fetchBlogFiles(), /filenames/);
    }
    context.fetch = async () => ({ ok: false });
    await assert.rejects(() => context.fetchBlogFiles(), /index/);
    await assert.rejects(() => context.fetchBlog('missing.md'), /Cannot load blog/);
    let requested;
    context.fetch = async url => { requested = url; return { ok: true, text: async () => source }; };
    await context.fetchBlog('随手记 #1.md');
    assert.equal(requested, 'contents/blogs/' + encodeURIComponent('随手记 #1.md'));
    class Element {
        constructor(tag) { this.tag = tag; this.children = []; }
        append(...nodes) { this.children.push(...nodes); }
        appendChild(node) { this.children.push(node); }
        replaceChildren(...nodes) { this.children = nodes; }
    }
    const target = new Element('div');
    context.document = { createElement: tag => new Element(tag), getElementById: () => target };
    const posts = {
        'Z filename.md': source.replace('2026-10-20', '2026-10-19').replace('博客写作示例：图片与数学公式', 'Older post'),
        '随手记 #1.md': source.replace('博客写作示例：图片与数学公式', 'Newest title'),
    };
    context.fetch = async url => ({
        ok: true,
        json: async () => Object.keys(posts),
        text: async () => posts[decodeURIComponent(url.slice('contents/blogs/'.length))],
    });
    await context.loadBlogs();
    assert.equal(target.children.length, 2);
    const latest = target.children[0];
    assert.equal(latest.children[0].children[0].textContent, 'Newest title');
    assert.equal(latest.children[0].children[0].href, 'blog.html?file=' + encodeURIComponent('随手记 #1.md'));
    assert.equal(latest.children[1].dateTime, '2026-10-20');
    assert.ok(latest.children[2].textContent.startsWith('TLDR: '));
    assert.ok(latest.className.includes('tex2jax_ignore'));
    context.fetch = async () => ({ ok: true, json: async () => [] });
    await context.loadBlogs();
    assert.equal(target.children[0].textContent, 'No posts yet.');
    for (const file of JSON.parse(read('contents/blogs/index.json'))) parse(read('contents/blogs/' + file), file);
    console.log('blog checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
