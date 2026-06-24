"""engine/research/discovery/tier1_rss_fetcher.py — Top-5 finance journal RSS feeds.

Configurable list of journal RSS feeds (NOT hardcoded — read from
data/research/tier1_journal_feeds.yaml so adding a new journal is a
config edit not a code change).

RSS feeds give RECENT papers only (no historical backfill via this path;
use journal API or paid services for historical).

Per source_health pattern — each feed marked unhealthy independently;
one bad feed doesn't break others.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml

from engine.data import source_health

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
FEED_CONFIG = REPO_ROOT / "data" / "research" / "tier1_journal_feeds.yaml"

POLITE_DELAY_SEC = 4.0


def _default_feeds() -> list[dict]:
    """Fallback if config file missing. v1 minimal set."""
    return [
        {
            "id":         "jfe_recent",
            "name":       "Journal of Financial Economics (recent)",
            "url":        "https://rss.sciencedirect.com/publication/science/0304405X",
            "publisher":  "Elsevier",
        },
        {
            "id":         "jpm_recent",
            "name":       "Journal of Portfolio Management (recent)",
            "url":        "https://jpm.pm-research.com/rss/current.xml",
            "publisher":  "PMR",
        },
    ]


def load_feed_config() -> list[dict]:
    """Read feed list from YAML config; fall back to defaults if missing."""
    if not FEED_CONFIG.exists():
        return _default_feeds()
    try:
        data = yaml.safe_load(FEED_CONFIG.read_text(encoding="utf-8")) or {}
        return data.get("feeds") or _default_feeds()
    except Exception as exc:
        logger.warning("feed config parse failed: %s; using defaults", exc)
        return _default_feeds()


def fetch_tier1_rss(
    *,
    max_per_feed: int = 50,
    feeds: list[dict] | None = None,
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch RECENT papers from each configured journal RSS.

    Per-feed source_health tracking — one feed's 429/error doesn't block
    others.

    Returns: long-format DataFrame with paper_id (journal:guid), title,
             authors, abstract, journal, submitted_date, abs_url, pdf_url.
    """
    feeds = feeds or load_feed_config()
    rows = []
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent":      "macro-alpha-research/1.0 (research@local; tier1-rss)",
        "Accept-Encoding": "gzip, deflate",
    })

    for feed in feeds:
        feed_id = feed.get("id", "")
        if not skip_health_check:
            healthy, reason = source_health.is_healthy(f"tier1_rss_{feed_id}")
            if not healthy:
                logger.info("feed %s unhealthy: %s", feed_id, reason)
                continue

        time.sleep(POLITE_DELAY_SEC)
        url = feed.get("url")
        if not url:
            continue
        try:
            resp = session.get(url, timeout=30)
        except Exception as exc:
            source_health.mark_failure(
                f"tier1_rss_{feed_id}", "network", str(exc)
            )
            continue
        if resp.status_code == 429:
            source_health.mark_failure(
                f"tier1_rss_{feed_id}", "rate_limited", "HTTP 429"
            )
            continue
        if resp.status_code != 200:
            source_health.mark_failure(
                f"tier1_rss_{feed_id}", "network",
                f"HTTP {resp.status_code}",
            )
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            source_health.mark_failure(
                f"tier1_rss_{feed_id}", "schema_unknown", f"XML: {exc}"
            )
            continue

        feed_rows = _parse_feed(root, feed, max_per_feed)
        if feed_rows:
            rows.extend(feed_rows)
            source_health.mark_success(f"tier1_rss_{feed_id}")

    if not rows:
        return pd.DataFrame(columns=[
            "paper_id", "title", "authors", "abstract", "journal",
            "categories", "submitted_date", "updated_date",
            "pdf_url", "abs_url",
        ])
    return pd.DataFrame(rows)


def _parse_feed(root, feed: dict, max_items: int) -> list[dict]:
    """Parse any of: RSS 1.0, RSS 2.0, Atom. Resilient to schema variants."""
    items = []
    # RSS 2.0 (most common): <channel><item>...
    items.extend(root.findall(".//item"))
    # RSS 1.0 (RDF): different namespace
    items.extend(root.findall(".//{http://purl.org/rss/1.0/}item"))
    # Atom: <feed><entry>
    items.extend(root.findall(".//{http://www.w3.org/2005/Atom}entry"))

    out = []
    for item in items[:max_items]:
        row = _parse_item_generic(item, feed)
        if row:
            out.append(row)
    return out


def _parse_item_generic(item, feed: dict) -> dict | None:
    """Best-effort parse across RSS 2.0 / 1.0 / Atom formats."""
    def _find_text(*tags):
        for tag in tags:
            val = item.findtext(tag, default="")
            if val:
                return val.strip()
        return ""

    try:
        title = _find_text(
            "title",
            "{http://purl.org/rss/1.0/}title",
            "{http://www.w3.org/2005/Atom}title",
        )
        if not title:
            return None
        # Description / abstract
        abstract = _find_text(
            "description",
            "{http://purl.org/rss/1.0/}description",
            "{http://www.w3.org/2005/Atom}summary",
        )
        link = _find_text(
            "link",
            "{http://purl.org/rss/1.0/}link",
        )
        if not link:
            # Atom link is an attribute
            link_el = item.find("{http://www.w3.org/2005/Atom}link")
            if link_el is not None:
                link = link_el.attrib.get("href", "")
        creator = _find_text(
            "{http://purl.org/dc/elements/1.1/}creator",
            "{http://www.w3.org/2005/Atom}author",
        )
        pub_date_str = _find_text(
            "pubDate",
            "{http://purl.org/dc/elements/1.1/}date",
            "{http://www.w3.org/2005/Atom}published",
            "{http://www.w3.org/2005/Atom}updated",
        )
        # Derive paper_id from link (or fall back to guid)
        guid = _find_text("guid", "{http://www.w3.org/2005/Atom}id")
        paper_id = f"{feed.get('id', 'tier1')}:{guid or link or title}"[:200]

        return {
            "paper_id":       paper_id,
            "title":          title.replace("\n", " "),
            "authors":        creator,
            "abstract":       abstract.replace("\n", " "),
            "journal":        feed.get("name", ""),
            "categories":     "",
            "submitted_date": _normalize_date(pub_date_str),
            "updated_date":   None,
            "pdf_url":        None,
            "abs_url":        link,
        }
    except Exception as exc:
        logger.warning("tier1 RSS item parse failed: %s", exc)
        return None


def _normalize_date(s: str) -> str | None:
    if not s:
        return None
    # RFC 2822 (RSS 2.0): "Mon, 15 Jan 2024 00:00:00 GMT"
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).date().isoformat()
    except Exception:
        pass
    # ISO 8601 (Atom): "2024-01-15T00:00:00Z"
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return None
