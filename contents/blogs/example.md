---
title: "博客写作示例：图片与数学公式"
tldr: "演示博客的元数据、图片、数学公式和代码写法。"
date: "2026-10-20"
---

这是一篇写作模板，尚未加入 `index.json`，因此不会显示在主页。
复制这个文件、修改开头的元数据和正文，再把文件名加入清单即可发布。

## 图片

下面的路径相对于这篇 Markdown 所在的 `contents/blogs/` 目录：

![网站背景示例](../../static/assets/background/background_0.jpg)

也可以创建 `contents/blogs/images/` 存放文章插图，使用 `![图片说明](images/your-image.png)`。

## 数学公式

行内公式：$E = mc^2$，或者 $x_i + x_j$。

独立公式：

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left\| f_\theta(x_i) - y_i \right\|^2
$$

矩阵、下标、反斜杠和换行也会保留：

$$
A = \begin{bmatrix}
a_1 & b_1 \\
a_2 & b_2
\end{bmatrix}
$$

## 代码

```python
def square(x):
    return x ** 2
```

代码块里的 `$x_i$` 会保持代码形式，不会被渲染成公式。
