# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate an RSS 2.0 feed from Earth2Studio blog posts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
POSTS_DIR = DOCS_ROOT / "blog" / "posts"
FEED_PATH = DOCS_ROOT / "blog" / "feed.xml"
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"
ATOM_NS = "http://www.w3.org/2005/Atom"
EXCERPT_SPLIT = "<!-- more -->"

ET.register_namespace("atom", ATOM_NS)


@dataclass(frozen=True)
class BlogPost:
    """A published blog post used as an RSS item."""

    title: str
    description: str
    date: date
    slug: str
    categories: tuple[str, ...]


def main() -> None:
    """Write ``docs/blog/feed.xml`` from published posts."""
    site_url, site_name, site_description = _site_metadata()
    public_base = _public_base_url(site_url)
    posts = _load_posts()
    xml = _render_feed(
        posts,
        site_name=site_name,
        site_description=site_description,
        public_base=public_base,
    )
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEED_PATH.write_text(xml, encoding="utf-8")


def _site_metadata() -> tuple[str, str, str]:
    """Return ``(site_url, site_name, site_description)`` from ``mkdocs.yml``."""
    raw = MKDOCS_CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(re.sub(r"!!python/name:\S+", "null", raw)) or {}
    site_url = str(config.get("site_url") or "").rstrip("/") + "/"
    site_name = str(config.get("site_name") or "Earth2Studio")
    site_description = str(
        config.get("site_description")
        or "Earth-2 product and engineering blog posts and updates."
    )
    return site_url, site_name, site_description


def _public_base_url(site_url: str) -> str:
    """Return the versioned public docs URL prefix used in feed links."""
    version = os.getenv("DOC_VERSION", "main").strip() or "main"
    return f"{site_url.rstrip('/')}/{version}/"


def _load_posts() -> list[BlogPost]:
    """Return published posts newest-first."""
    posts: list[BlogPost] = []
    if not POSTS_DIR.is_dir():
        return posts

    for path in sorted(POSTS_DIR.glob("*.md")):
        post = _parse_post(path)
        if post is not None:
            posts.append(post)

    posts.sort(key=lambda item: (item.date, item.slug), reverse=True)
    return posts


def _parse_post(path: Path) -> BlogPost | None:
    """Return a published post, or ``None`` for drafts and undated files."""
    text = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(text)
    if meta.get("draft") is True:
        return None

    published = _parse_date(meta.get("date"))
    if published is None:
        return None

    title = str(meta.get("title") or "").strip() or _heading_title(body) or path.stem
    description = str(meta.get("description") or "").strip() or _excerpt(body)
    categories = tuple(
        str(item).strip() for item in meta.get("categories", []) if str(item).strip()
    )
    return BlogPost(
        title=title,
        description=description,
        date=published,
        slug=path.stem,
        categories=categories,
    )


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Split YAML front matter from Markdown body."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2]


def _parse_date(value: object) -> date | None:
    """Parse a blog ``date`` field from front matter."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, dict):
        return _parse_date(value.get("created") or value.get("date"))
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _heading_title(body: str) -> str:
    """Return the first Markdown heading in ``body``."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _excerpt(body: str) -> str:
    """Return a plain-text excerpt for the RSS description."""
    if EXCERPT_SPLIT in body:
        body = body.split(EXCERPT_SPLIT, 1)[0]

    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "```", ":::")):
            continue
        lines.append(stripped)

    text = " ".join(lines)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*`_]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _render_feed(
    posts: list[BlogPost],
    *,
    site_name: str,
    site_description: str,
    public_base: str,
) -> str:
    """Return RSS 2.0 XML for ``posts``."""
    channel_link = f"{public_base}blog/"
    feed_link = f"{public_base}blog/feed.xml"

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    _set_text(channel, "title", f"{site_name} Blog")
    _set_text(channel, "link", channel_link)
    _set_text(channel, "description", site_description)
    _set_text(channel, "language", "en")
    _set_text(channel, "ttl", "1440")

    atom_link = ET.SubElement(channel, f"{{{ATOM_NS}}}link")
    atom_link.set("href", feed_link)
    atom_link.set("rel", "self")
    atom_link.set("type", "application/rss+xml")

    for post in posts:
        item_link = f"{public_base}blog/posts/{post.slug}/"
        item = ET.SubElement(channel, "item")
        _set_text(item, "title", post.title)
        _set_text(item, "link", item_link)
        _set_text(item, "guid", item_link)
        _set_text(item, "pubDate", _rfc822(post.date))
        _set_text(
            item,
            "description",
            post.description or post.title,
        )
        for category in post.categories:
            _set_text(item, "category", category)

    xml = ET.tostring(rss, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml + "\n"


def _set_text(parent: ET.Element, tag: str, text: str) -> None:
    """Append a child element with ``text``."""
    child = ET.SubElement(parent, tag)
    child.text = text


def _rfc822(value: date) -> str:
    """Return an RFC 822 timestamp at midnight UTC."""
    published = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return format_datetime(published)


if __name__ == "__main__":
    main()
