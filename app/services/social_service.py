"""Service for managing social media channel follows, fetching posts, and generating summaries."""

import logging
from calendar import timegm
from datetime import datetime, timedelta, timezone

import feedparser
import requests as _requests
from sqlalchemy import func

from app.extensions import db
from app.integrations.llm_gateway import BudgetExhaustedError, LLMGateway
from app.models.social import SocialChannel, SocialPost

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
You are a content analyst creating rich, visual markdown summaries. Given a post's title and \
description from a social media channel, produce a structured summary.

## Output Format
- **TL;DR**: One bold sentence summary
- **Key Points**: 3-7 numbered bullets capturing the main ideas
- For technical content, include ONE of these visual elements:
  - A markdown comparison table
  - A mermaid flowchart (use ```mermaid code blocks)
  - An ASCII diagram showing architecture or process
- **Notable Quotes** (1-2 max, only if meaningful)
- **Why It Matters**: 1-2 sentences on significance

Keep under 500 words. Use rich markdown formatting throughout."""

# URL substring -> platform mapping (order matters: first match wins)
_PLATFORM_PATTERNS = [
    ('youtube.com', 'youtube'),
    ('youtu.be', 'youtube'),
    ('substack.com', 'substack'),
    ('rsshub', 'twitter'),
    ('nitter', 'twitter'),
    ('twitter.com', 'twitter'),
    ('x.com', 'twitter'),
]


_FEED_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; PulseBot/1.0)'}


def _fetch_and_parse(feed_url: str):
    """Fetch a feed URL with proper headers and parse it."""
    try:
        resp = _requests.get(feed_url, headers=_FEED_HEADERS, timeout=15)
        resp.raise_for_status()
        return feedparser.parse(resp.text)
    except Exception as e:
        logger.warning("HTTP fetch failed for %s: %s, falling back to feedparser", feed_url, e)
        return _fetch_and_parse(feed_url)


def _detect_platform(feed_url: str) -> str:
    """Guess platform from URL patterns."""
    lower = feed_url.lower()
    for pattern, platform in _PLATFORM_PATTERNS:
        if pattern in lower:
            return platform
    return 'rss'


def _parse_published(entry) -> datetime | None:
    """Convert feedparser entry's published_parsed to a timezone-aware datetime."""
    tp = entry.get('published_parsed')
    if tp is None:
        tp = entry.get('updated_parsed')
    if tp is None:
        return None
    try:
        return datetime.fromtimestamp(timegm(tp), tz=timezone.utc)
    except Exception:
        return None


class SocialService:
    """Manages social channel follows, post fetching, and LLM-based summaries."""

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def follow_channel(self, feed_url: str, name: str | None = None,
                       platform: str | None = None) -> dict:
        """Follow a new RSS/social channel by its feed URL.

        If the channel was previously unfollowed (is_active=False), reactivate it.
        Raises ValueError if the channel is already actively followed.
        """
        platform = platform or _detect_platform(feed_url)

        existing = SocialChannel.query.filter_by(feed_url=feed_url).first()
        if existing:
            if existing.is_active:
                raise ValueError(f"Channel already followed: {existing.name}")
            # Reactivate
            existing.is_active = True
            existing.consecutive_failures = 0
            existing.last_error = None
            db.session.commit()
            logger.info("Reactivated channel id=%s url=%s", existing.id, feed_url)
            return existing.to_dict()

        # Fetch feed metadata
        feed = _fetch_and_parse(feed_url)
        feed_meta = feed.feed if feed.feed else {}

        resolved_name = name or feed_meta.get('title') or feed_url
        avatar_url = None
        image = feed_meta.get('image')
        if image:
            avatar_url = image.get('href') if isinstance(image, dict) else None
        description = feed_meta.get('subtitle', '') or feed_meta.get('description', '') or ''

        channel = SocialChannel(
            name=resolved_name,
            platform=platform,
            feed_url=feed_url,
            avatar_url=avatar_url,
            description=description[:2048] if description else None,
            is_active=True,
        )
        db.session.add(channel)
        db.session.commit()
        logger.info("Followed new channel id=%s name=%s platform=%s",
                     channel.id, channel.name, platform)
        return channel.to_dict()

    def unfollow_channel(self, channel_id: int) -> None:
        """Soft-delete a channel by setting is_active=False."""
        channel = db.session.get(SocialChannel, channel_id)
        if channel is None:
            raise ValueError(f"Channel not found: {channel_id}")
        channel.is_active = False
        db.session.commit()
        logger.info("Unfollowed channel id=%s name=%s", channel.id, channel.name)

    def list_channels(self, active_only: bool = True) -> list[dict]:
        """Return all channels with post_count and last_post_date stats."""
        post_count_sq = (
            db.session.query(
                SocialPost.channel_id,
                func.count(SocialPost.id).label('post_count'),
                func.max(SocialPost.published_at).label('last_post_date'),
            )
            .group_by(SocialPost.channel_id)
            .subquery()
        )

        query = (
            db.session.query(
                SocialChannel,
                func.coalesce(post_count_sq.c.post_count, 0).label('post_count'),
                post_count_sq.c.last_post_date,
            )
            .outerjoin(post_count_sq, SocialChannel.id == post_count_sq.c.channel_id)
        )

        if active_only:
            query = query.filter(SocialChannel.is_active.is_(True))

        query = query.order_by(SocialChannel.name)

        results = []
        for channel, post_count, last_post_date in query.all():
            channel.post_count = post_count
            channel.last_post_date = last_post_date
            results.append(channel.to_dict(include_stats=True))
        return results

    # ------------------------------------------------------------------
    # Post fetching
    # ------------------------------------------------------------------

    def fetch_new_posts(self, channel_id: int | None = None) -> int:
        """Fetch new posts from RSS feeds. Returns total number of new posts inserted."""
        if channel_id:
            channels = SocialChannel.query.filter_by(id=channel_id, is_active=True).all()
        else:
            channels = SocialChannel.query.filter_by(is_active=True).all()

        total_new = 0
        for channel in channels:
            try:
                new_count = self._fetch_channel_posts(channel)
                channel.last_fetched_at = datetime.now(timezone.utc)
                channel.last_success_at = datetime.now(timezone.utc)
                channel.consecutive_failures = 0
                channel.last_error = None
                total_new += new_count
            except Exception as exc:
                logger.error("Failed to fetch channel id=%s name=%s: %s",
                             channel.id, channel.name, exc)
                channel.last_fetched_at = datetime.now(timezone.utc)
                channel.consecutive_failures = (channel.consecutive_failures or 0) + 1
                channel.last_error = str(exc)[:512]
            db.session.commit()

        logger.info("fetch_new_posts: %d new posts across %d channels",
                     total_new, len(channels))
        return total_new

    def _fetch_channel_posts(self, channel: SocialChannel) -> int:
        """Parse a single channel's feed and insert new posts. Returns count of new posts."""
        feed = feedparser.parse(channel.feed_url)
        if feed.bozo and not feed.entries:
            raise RuntimeError(f"Feed parse error: {feed.bozo_exception}")

        # Collect existing URLs for this channel to deduplicate
        existing_urls: set[str] = set(
            url for (url,) in
            db.session.query(SocialPost.url)
            .filter(SocialPost.channel_id == channel.id)
            .all()
        )

        new_count = 0
        for entry in feed.entries:
            url = entry.get('link')
            if not url or url in existing_urls:
                continue

            post = SocialPost(
                channel_id=channel.id,
                url=url,
                title=entry.get('title', '')[:1024],
                author=entry.get('author', channel.name),
                published_at=_parse_published(entry),
                content_hint=(entry.get('summary') or '')[:4096],
            )
            db.session.add(post)
            existing_urls.add(url)
            new_count += 1

        return new_count

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def generate_post_summary(self, post_id: int) -> bool:
        """Generate an LLM summary for a single post. Returns True if generated."""
        post = db.session.get(SocialPost, post_id)
        if post is None:
            raise ValueError(f"Post not found: {post_id}")
        if post.summary_md:
            return False  # already summarised

        platform_label = post.channel.platform if post.channel else 'rss'
        user_content = (
            f"Platform: {platform_label}\n"
            f"Channel: {post.channel.name if post.channel else 'Unknown'}\n"
            f"Title: {post.title}\n"
            f"Description:\n{post.content_hint or '(no description)'}"
        )

        try:
            llm = LLMGateway()
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': SUMMARY_SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_content},
                ],
                purpose='social_summary',
                section='social',
                provider='openai',
                model='gpt-4.1-nano',
                max_tokens=2000,
            )
        except BudgetExhaustedError:
            logger.warning("Budget exhausted while summarising post id=%s", post_id)
            return False
        except Exception:
            logger.exception("LLM call failed for post id=%s", post_id)
            return False

        post.summary_md = result['content']
        post.summary_generated_at = datetime.now(timezone.utc)
        post.summary_cost_usd = result.get('cost_usd')
        post.summary_tokens = result.get('total_tokens')
        db.session.commit()
        logger.info("Generated summary for post id=%s tokens=%s cost=$%s",
                     post_id, post.summary_tokens, post.summary_cost_usd)
        return True

    def generate_pending_summaries(self, limit: int = 30) -> int:
        """Generate summaries for posts that don't have one yet. Returns count generated."""
        pending_posts = (
            SocialPost.query
            .filter(SocialPost.summary_md.is_(None))
            .order_by(SocialPost.published_at.desc())
            .limit(limit)
            .all()
        )

        generated = 0
        for post in pending_posts:
            if self.generate_post_summary(post.id):
                generated += 1
        logger.info("generate_pending_summaries: %d/%d generated", generated, len(pending_posts))
        return generated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_week_summary(self) -> dict:
        """Return a summary of posts from the last 7 days grouped by channel."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        today = datetime.now(timezone.utc)

        rows = (
            db.session.query(
                SocialChannel.id,
                SocialChannel.name,
                SocialChannel.platform,
                func.count(SocialPost.id).label('post_count'),
            )
            .join(SocialPost, SocialPost.channel_id == SocialChannel.id)
            .filter(SocialChannel.is_active.is_(True))
            .filter(SocialPost.published_at >= cutoff)
            .group_by(SocialChannel.id, SocialChannel.name, SocialChannel.platform)
            .order_by(func.count(SocialPost.id).desc())
            .all()
        )

        channels = []
        total_posts = 0
        for cid, cname, cplatform, pcount in rows:
            channels.append({
                'id': cid,
                'name': cname,
                'platform': cplatform,
                'post_count': pcount,
            })
            total_posts += pcount

        date_range = f"{cutoff.strftime('%b %d')} - {today.strftime('%b %d, %Y')}"

        return {
            'channels': channels,
            'total_posts': total_posts,
            'date_range': date_range,
        }

    def get_channel_posts(self, channel_id: int, limit: int = 50,
                          offset: int = 0) -> list[dict]:
        """Return paginated posts for a single channel."""
        posts = (
            SocialPost.query
            .filter(SocialPost.channel_id == channel_id)
            .order_by(SocialPost.published_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [p.to_dict() for p in posts]

    def get_all_posts(self, limit: int = 50, offset: int = 0,
                      days: int = 7) -> list[dict]:
        """Return all posts across channels from the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        posts = (
            SocialPost.query
            .filter(SocialPost.published_at >= cutoff)
            .order_by(SocialPost.published_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [p.to_dict() for p in posts]
