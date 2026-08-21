---
description: Learn how to write, preview, and publish a post on the Earth2Studio Blog
draft: false
date: 2026-08-20
categories:
  - Guide
authors:
  - negin513
---

<!-- markdownlint-disable MD046 -->

# How to Contribute to the Earth2Studio Blog

The Earth2Studio Blog is a place for the community to share what they are
building, learning, and discovering with Earth2Studio and across the broader
Earth-2 ecosystem.

Posts can cover new features, real-world workflows, performance results,
technical deep dives, community use cases, tutorials, data sources, models,
and practical tips that help others get more from Earth2Studio.

Contributions are welcome from users, developers, researchers, and partners.
If you have a useful workflow, lesson learned, benchmark, integration, or idea
to share, we would love to feature it.

!!! tip "The Earth2Studio Blog welcomes contributions from the community!"

<!-- more -->

## Write and publish a post

### 1. Fork the repository

If you have not already, fork
[NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio).

Then clone your fork and create a branch from `main`:

```bash
git clone https://github.com/<your-username>/earth2studio.git
cd earth2studio
git switch main
git switch -c blog/<your-post-title>
```

### 2. Create your post

Create a Markdown file under `docs/blog/posts/` using a short, lowercase,
hyphenated filename:

```text
docs/blog/posts/<your-post-title>.md
```

For example:

```text
docs/blog/posts/how-to-contribute-to-earth2studio-blog.md
```

The filename determines the published URL:

```text
/blog/posts/<your-post-title>/
```

The publication date comes from the `date` field in the front matter, so you
do not need to include a date in the filename.

The blog directory looks like:

```text
.
├─ docs/
│  └─ blog/
│     ├─ posts/
│     │  ├─ mkdocs-upgrade.md
│     │  └─ your-post-title.md
│     ├─ .authors.yml
│     └─ .categories.yml
└─ mkdocs.yml
```

### 3. Add the front matter

Start each post with YAML front matter followed by a single `#` heading for
the post title:

```yaml
---
description: <your-post-description>
draft: false
date: YYYY-MM-DD
categories:
  - <category-1>
  - <category-2>
authors:
  - <author-identifier>
---

# <Your post title>
```

The fields are:

* `description`: A short summary used in blog listings and the RSS feed.
* `draft`: Set to `true` while a post should remain unpublished.
* `date`: Publication date in `YYYY-MM-DD` format.
* `categories`: Topics associated with the post. Use an existing category,
  such as `Documentation` or `Guide`, or add a new category to
  `docs/blog/.categories.yml`.
* `authors`: Author identifiers defined in `docs/blog/.authors.yml`.

See the
[Material for MkDocs blog metadata documentation](https://squidfunk.github.io/mkdocs-material/plugins/blog/#metadata)
for additional supported metadata.

Do not add a `title` field to the front matter. The first Markdown `#` heading
is used as the post title. Defining both can also trigger the `MD025`
markdownlint rule.

The author, publication date, and category metadata displayed below the title
are generated automatically from the front matter and
`docs/blog/.authors.yml`.

### 4. Write the post

Write the rest of the post in Markdown.

Aim for content that is useful and reusable by the broader Earth2Studio
community. Good topics include:

* New Earth2Studio features and integrations
* End-to-end workflows and applications
* Performance results and optimization techniques
* Models and data sources
* Tutorials and practical examples
* Lessons learned from real-world deployments
* Community projects built with or around Earth2Studio

For Markdown syntax, see the
[Markdown Guide cheat sheet](https://www.markdownguide.org/cheat-sheet/).

### 5. Add an excerpt

Use `<!-- more -->` to control where the preview shown in blog listings ends:

```markdown
This introduction appears in the blog listing.

<!-- more -->

The full post continues here.
```

A useful excerpt should quickly tell readers what the post covers and why it
matters.

### 6. Add yourself as an author

List each author by identifier in the post front matter:

```yaml
authors:
  - <author-1-identifier>
  - <author-2-identifier>
```

Each identifier must have a corresponding entry in
`docs/blog/.authors.yml`.

If you are a first-time contributor, add yourself using a unique identifier.
Your GitHub username is a good default:

```yaml
authors:
  negin513:
    name: Negin Sobhani
    description: Earth2Studio
    avatar: https://github.com/negin513.png
    slug: negin513
    url: https://github.com/negin513
```

The `name`, `avatar`, and `url` fields are used to generate the author
information displayed with the post.

### 7. Add or reuse a category

Whenever possible, use an existing blog category.

If your post introduces a new topic, add the category and an optional
description to:

```text
docs/blog/.categories.yml
```

You do not need to create or update category pages manually.

### 8. Preview the post locally

From the root of the Earth2Studio repository, install the documentation
dependencies:

```bash
uv sync --group docs
```

Start the documentation server:

```bash
make docs-serve
```

Your post is available at:

```text
http://127.0.0.1:8001/earth2studio/blog/posts/<your-post-title>/
```

Before submitting your post, confirm that:

* The post renders correctly.
* The post appears on the blog overview.
* The title, author, date, and categories are correct.
* The excerpt ends in the right place.
* The post appears on the expected category page.
* Links, images, code blocks, and other Markdown elements render correctly.

`make docs-dev` can also be used for local development, but it installs
additional package extras and builds the example gallery. For a blog-only
preview, `make docs-serve` is the lighter option.

### 9. Do not edit generated blog pages

The blog overview, archive, and category pages are generated from the metadata
in individual posts.

You should not manually update:

```text
docs/blog/index.md
docs/blog/archive.md
docs/blog/categories.md
docs/blog/categories/*.md
```

These generated files are git-ignored and should not be committed.

When adding a post, the Markdown file and its metadata are the source of truth.

### 10. Commit and open a pull request

When the post is ready, commit your changes and push your branch:

```bash
git add docs/blog/posts/<your-post-title>.md
git add docs/blog/.authors.yml       # If you added an author
git add docs/blog/.categories.yml    # If you added a category

git commit -s -m "docs: add blog post <your-post-title>"
git push -u origin blog/<your-post-title>
```

The `-s` flag adds the Developer Certificate of Origin sign-off required for
Earth2Studio contributions.

Open a pull request against `NVIDIA/earth2studio:main` and complete the pull
request template.

For first-time contributors, a maintainer may need to approve or trigger CI.
Address any CI failures or review feedback as you would for other
Earth2Studio contributions.

### 11. Publish

Once the pull request is merged, the post will be included in the
Earth2Studio Blog with the next documentation deployment.

Thanks for contributing!

---

Have something useful to share? Open a pull request and contribute it to the
Earth2Studio community. We look forward to seeing what you build.
