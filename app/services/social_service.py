"""Service for managing social media channel follows, fetching posts, and generating summaries."""

import json
import logging
import re
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

YOUTUBE_SUMMARY_PROMPT = """\
You are a content analyst creating rich, visual markdown summaries of YouTube videos.
You are given the video's title, channel name, and its transcript.

## Output Format
- **TL;DR**: One bold sentence summary
- **Key Points**: 5-10 numbered bullets capturing the main ideas and arguments
- Include ONE of these visual elements where appropriate:
  - A markdown comparison table
  - A mermaid flowchart (use ```mermaid code blocks)
  - An ASCII diagram showing architecture or process
- **Timestamps & Topics**: List 3-5 key timestamps with topics (estimate from transcript position)
- **Why It Matters**: 1-2 sentences on significance

Keep under 800 words. Use rich markdown formatting throughout."""

TWITTER_FETCH_PROMPT = """\
Search for the latest tweets/posts from @{handle} on X (Twitter) from the last {hours} hours.

Return a JSON array of tweets. Each tweet should have these fields:
- "text": the full tweet text
- "date": ISO date string (YYYY-MM-DD)
- "urls": array of any URLs mentioned in the tweet
- "is_thread": boolean, true if part of a thread
- "topic": a short 3-5 word topic label

Return ONLY valid JSON array, no markdown or explanation. If no tweets found, return [].
Example: [{"text": "Just published...", "date": "2026-03-20", "urls": ["https://..."], "is_thread": false, "topic": "new paper release"}]"""

_FEED_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; PulseBot/1.0)'}


def _fetch_and_parse(feed_url: str):
    """Fetch a feed URL with proper headers and parse it."""
    try:
        resp = _requests.get(feed_url, headers=_FEED_HEADERS, timeout=15)
        resp.raise_for_status()
        return feedparser.parse(resp.text)
    except Exception as e:
        logger.warning("HTTP fetch failed for %s: %s, falling back to feedparser direct", feed_url, e)
        return feedparser.parse(feed_url)


def _detect_platform(feed_url: str) -> str:
    """Guess platform from URL patterns."""
    lower = feed_url.lower()
    for pattern, platform in [
        ('youtube.com', 'youtube'), ('youtu.be', 'youtube'),
        ('substack.com', 'substack'),
        ('rsshub', 'twitter'), ('nitter', 'twitter'),
        ('twitter.com', 'twitter'), ('x.com', 'twitter'),
        ('grok://twitter/', 'twitter'),
    ]:
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


# ── YouTube helpers ──────────────────────────────────────────────

def _resolve_youtube_input(input_str: str) -> tuple[str, str | None]:
    """Resolve a YouTube URL/handle/ID into (feed_url, channel_name).

    Accepts:
    - https://www.youtube.com/feeds/videos.xml?channel_id=UCxxx (pass through)
    - https://www.youtube.com/channel/UCxxx
    - https://www.youtube.com/@handle
    - @handle or just handle
    - Raw channel ID like UCxxx
    """
    input_str = input_str.strip()

    # Already a feed URL
    if 'feeds/videos.xml' in input_str:
        m = re.search(r'channel_id=([A-Za-z0-9_-]+)', input_str)
        return input_str, None

    # youtube.com/channel/UCxxx
    m = re.search(r'youtube\.com/channel/([A-Za-z0-9_-]+)', input_str)
    if m:
        cid = m.group(1)
        feed_url = f'https://www.youtube.com/feeds/videos.xml?channel_id={cid}'
        return feed_url, None

    # youtube.com/@handle — need to fetch page to resolve channel ID
    m = re.search(r'youtube\.com/@([A-Za-z0-9_.-]+)', input_str)
    if m:
        handle = m.group(1)
        return _resolve_youtube_handle(handle)

    # Bare @handle
    if input_str.startswith('@'):
        return _resolve_youtube_handle(input_str[1:])

    # Raw channel ID (starts with UC)
    if input_str.startswith('UC') and len(input_str) > 10:
        feed_url = f'https://www.youtube.com/feeds/videos.xml?channel_id={input_str}'
        return feed_url, None

    # Treat as handle
    return _resolve_youtube_handle(input_str)


def _resolve_youtube_handle(handle: str) -> tuple[str, str | None]:
    """Fetch youtube.com/@handle page and extract channelId from meta tags."""
    url = f'https://www.youtube.com/@{handle}'
    try:
        resp = _requests.get(url, headers=_FEED_HEADERS, timeout=15)
        resp.raise_for_status()
        # Look for channelId in the HTML
        m = re.search(r'"channelId"\s*:\s*"([A-Za-z0-9_-]+)"', resp.text)
        if not m:
            m = re.search(r'channel_id=([A-Za-z0-9_-]+)', resp.text)
        if not m:
            raise ValueError(f"Could not find channel ID for @{handle}")
        cid = m.group(1)
        # Extract channel name from og:title or <title> tag
        name = None
        nm = re.search(r'<meta\s+property="og:title"\s+content="([^"]+)"', resp.text)
        if nm:
            name = nm.group(1).replace(' - YouTube', '').strip()
        if not name:
            nm = re.search(r'<title>([^<]+)</title>', resp.text)
            if nm:
                name = nm.group(1).replace(' - YouTube', '').strip()
        feed_url = f'https://www.youtube.com/feeds/videos.xml?channel_id={cid}'
        return feed_url, name
    except _requests.RequestException as e:
        raise ValueError(f"Failed to resolve YouTube handle @{handle}: {e}")


def _fetch_youtube_transcript(video_url: str) -> str | None:
    """Fetch auto-generated transcript for a YouTube video. Returns text or None."""
    # Extract video ID
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', video_url)
    if not m:
        return None
    video_id = m.group(1)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manually created, fall back to auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except Exception:
            transcript = transcript_list.find_generated_transcript(['en'])
        segments = transcript.fetch()
        text = ' '.join(seg.get('text', '') if isinstance(seg, dict) else str(seg) for seg in segments)
        # Truncate to ~4000 chars for LLM context
        if len(text) > 4000:
            text = text[:4000] + '...'
        return text
    except Exception as e:
        logger.warning("Could not fetch transcript for %s: %s", video_id, e)
        return None


# ── Twitter/X helpers ────────────────────────────────────────────

def _resolve_twitter_input(input_str: str) -> tuple[str, str]:
    """Resolve Twitter input to (sentinel_feed_url, handle).

    Accepts: @handle, handle, https://x.com/handle, https://twitter.com/handle
    """
    input_str = input_str.strip()
    # Extract handle from URL
    m = re.search(r'(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)', input_str)
    if m:
        handle = m.group(1)
    elif input_str.startswith('@'):
        handle = input_str[1:]
    else:
        handle = input_str.strip('/')

    feed_url = f'grok://twitter/{handle}'
    return feed_url, handle


def _resolve_substack_input(input_str: str) -> str:
    """Resolve Substack input to RSS feed URL.

    Accepts: blog-name, blog-name.substack.com, https://blog-name.substack.com/feed
    """
    input_str = input_str.strip().rstrip('/')
    if input_str.endswith('/feed'):
        return input_str
    m = re.search(r'([a-zA-Z0-9-]+)\.substack\.com', input_str)
    if m:
        return f'https://{m.group(1)}.substack.com/feed'
    # Bare name
    name = input_str.replace('https://', '').replace('http://', '').split('.')[0]
    return f'https://{name}.substack.com/feed'


class SocialService:
    """Manages social channel follows, post fetching, and LLM-based summaries."""

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def follow_channel(self, input_value: str, platform: str,
                       name: str | None = None) -> dict:
        """Follow a new channel. Resolves user-friendly input per platform.

        platform: 'youtube', 'twitter', 'substack', 'rss'
        input_value: channel URL/handle/name depending on platform
        """
        if platform == 'youtube':
            feed_url, resolved_name = _resolve_youtube_input(input_value)
            name = name or resolved_name
        elif platform == 'twitter':
            feed_url, handle = _resolve_twitter_input(input_value)
        elif platform == 'substack':
            feed_url = _resolve_substack_input(input_value)
        else:  # rss
            feed_url = input_value.strip()
            platform = platform or _detect_platform(feed_url)

        # Check existing
        existing = SocialChannel.query.filter_by(feed_url=feed_url).first()
        if existing:
            if existing.is_active:
                raise ValueError(f"Channel already followed: {existing.name}")
            existing.is_active = True
            existing.consecutive_failures = 0
            existing.last_error = None
            db.session.commit()
            logger.info("Reactivated channel id=%s url=%s", existing.id, feed_url)
            return existing.to_dict()

        # For non-Twitter, fetch feed metadata for name
        if platform != 'twitter':
            feed = _fetch_and_parse(feed_url)
            feed_meta = feed.feed if feed.feed else {}
            name = name or feed_meta.get('title') or input_value
            avatar_url = None
            image = feed_meta.get('image')
            if image:
                avatar_url = image.get('href') if isinstance(image, dict) else None
            description = feed_meta.get('subtitle', '') or feed_meta.get('description', '') or ''
        else:
            handle = feed_url.replace('grok://twitter/', '')
            name = name or f'@{handle}'
            avatar_url = None
            description = f'Twitter/X posts from @{handle}'

        channel = SocialChannel(
            name=name,
            handle=handle if platform == 'twitter' else None,
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
        """Fetch new posts from all active channels. Returns total new posts.
        Respects per-channel refresh_interval_hours — skips channels fetched too recently.
        """
        if channel_id:
            channels = SocialChannel.query.filter_by(id=channel_id, is_active=True).all()
        else:
            channels = SocialChannel.query.filter_by(is_active=True).all()

        now = datetime.now(timezone.utc)
        total_new = 0
        for channel in channels:
            # Skip if fetched more recently than the channel's refresh interval
            if not channel_id and channel.last_fetched_at:
                from datetime import timedelta
                interval = timedelta(hours=channel.refresh_interval_hours or 4)
                if (now - channel.last_fetched_at) < interval:
                    logger.debug("Skipping channel id=%s (last fetched %s ago, interval %sh)",
                                 channel.id, now - channel.last_fetched_at, channel.refresh_interval_hours)
                    continue
            try:
                if channel.platform == 'twitter':
                    new_count = self._fetch_twitter_posts(channel)
                else:
                    new_count = self._fetch_rss_posts(channel)
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

    def _fetch_rss_posts(self, channel: SocialChannel) -> int:
        """Parse a single channel's RSS feed and insert new posts."""
        feed = _fetch_and_parse(channel.feed_url)
        if feed.bozo and not feed.entries:
            raise RuntimeError(f"Feed parse error: {feed.bozo_exception}")

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

    def _fetch_twitter_posts(self, channel: SocialChannel, hours: int = 6) -> int:
        """Use Grok API with web search to fetch recent tweets."""
        handle = channel.handle or channel.feed_url.replace('grok://twitter/', '')

        prompt = TWITTER_FETCH_PROMPT.format(handle=handle, hours=hours)

        try:
            llm = LLMGateway()
            result = llm.call(
                messages=[
                    {'role': 'user', 'content': prompt},
                ],
                purpose=f'social_twitter_fetch.{handle}',
                section='social',
                provider='xai',
                search=True,
                max_tokens=3000,
            )
        except BudgetExhaustedError:
            logger.warning("Budget exhausted fetching tweets for @%s", handle)
            return 0

        # Parse the Grok response as JSON
        content = result.get('content', '')
        logger.info("Grok response for @%s (len=%d): %s", handle, len(content), content[:500])
        tweets = self._parse_grok_tweets(content)

        existing_urls: set[str] = set(
            url for (url,) in
            db.session.query(SocialPost.url)
            .filter(SocialPost.channel_id == channel.id)
            .all()
        )

        new_count = 0
        for i, tweet in enumerate(tweets):
            # Create a unique URL for each tweet
            tweet_text = tweet.get('text', '')[:200]
            tweet_date = tweet.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
            # Use content hash for dedup since we don't have actual tweet IDs
            import hashlib
            text_hash = hashlib.md5(tweet_text.encode()).hexdigest()[:10]
            tweet_url = f"https://x.com/{handle}/status/grok-{tweet_date}-{text_hash}"

            if tweet_url in existing_urls:
                continue

            try:
                pub_date = datetime.strptime(tweet_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pub_date = datetime.now(timezone.utc)

            post = SocialPost(
                channel_id=channel.id,
                url=tweet_url,
                title=tweet.get('topic', tweet_text[:80]) or tweet_text[:80],
                author=f'@{handle}',
                published_at=pub_date,
                content_hint=tweet_text[:4096],
            )
            db.session.add(post)
            existing_urls.add(tweet_url)
            new_count += 1

        return new_count

    def _parse_grok_tweets(self, content: str) -> list[dict]:
        """Parse Grok's response into a list of tweet dicts."""
        # Try to extract JSON from the response
        content = content.strip()
        # Remove markdown code fences if present
        if content.startswith('```'):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        try:
            tweets = json.loads(content)
            if isinstance(tweets, list):
                return tweets
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the text
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            try:
                tweets = json.loads(m.group())
                if isinstance(tweets, list):
                    return tweets
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse Grok tweet response as JSON")
        return []

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def generate_post_summary(self, post_id: int) -> bool:
        """Generate an LLM summary for a single post. Returns True if generated."""
        post = db.session.get(SocialPost, post_id)
        if post is None:
            raise ValueError(f"Post not found: {post_id}")
        if post.summary_md:
            return False

        platform = post.channel.platform if post.channel else 'rss'

        # For YouTube: try to fetch transcript for richer summaries
        if platform == 'youtube':
            transcript = _fetch_youtube_transcript(post.url)
            if transcript:
                user_content = (
                    f"Channel: {post.channel.name if post.channel else 'Unknown'}\n"
                    f"Title: {post.title}\n\n"
                    f"Transcript:\n{transcript}"
                )
                system_prompt = YOUTUBE_SUMMARY_PROMPT
            else:
                user_content = (
                    f"Platform: youtube\n"
                    f"Channel: {post.channel.name if post.channel else 'Unknown'}\n"
                    f"Title: {post.title}\n"
                    f"Description:\n{post.content_hint or '(no description)'}"
                )
                system_prompt = SUMMARY_SYSTEM_PROMPT
        else:
            user_content = (
                f"Platform: {platform}\n"
                f"Channel: {post.channel.name if post.channel else 'Unknown'}\n"
                f"Title: {post.title}\n"
                f"Description:\n{post.content_hint or '(no description)'}"
            )
            system_prompt = SUMMARY_SYSTEM_PROMPT

        try:
            llm = LLMGateway()
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
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
        """Generate summaries for posts that don't have one yet."""
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
    # YouTube backfill
    # ------------------------------------------------------------------

    def backfill_youtube_channel(self, channel_id: int, max_videos: int = 5):
        """Generator that fetches recent videos, transcripts, and summaries.

        Yields progress dicts: {'step': str, 'video': int, 'total': int, 'title': str, 'status': str}
        """
        channel = db.session.get(SocialChannel, channel_id)
        if channel is None or channel.platform != 'youtube':
            yield {'step': 'error', 'message': 'Channel not found or not YouTube'}
            return

        # Step 1: Fetch RSS feed
        yield {'step': 'fetch_feed', 'video': 0, 'total': 0, 'title': '', 'status': 'Fetching RSS feed...'}

        feed = _fetch_and_parse(channel.feed_url)
        entries = feed.entries[:max_videos] if feed.entries else []

        if not entries:
            yield {'step': 'done', 'video': 0, 'total': 0, 'title': '', 'status': 'No videos found in feed.'}
            return

        total = len(entries)
        yield {'step': 'feed_loaded', 'video': 0, 'total': total, 'title': '', 'status': f'Found {total} videos'}

        # Step 2: Create posts, fetch transcripts, summarize
        existing_urls = set(
            url for (url,) in
            db.session.query(SocialPost.url)
            .filter(SocialPost.channel_id == channel.id)
            .all()
        )

        for i, entry in enumerate(entries, 1):
            url = entry.get('link')
            title = entry.get('title', 'Untitled')[:1024]

            if not url:
                continue

            # Create post if not exists
            if url not in existing_urls:
                post = SocialPost(
                    channel_id=channel.id,
                    url=url,
                    title=title,
                    author=entry.get('author', channel.name),
                    published_at=_parse_published(entry),
                    content_hint=(entry.get('summary') or '')[:4096],
                )
                db.session.add(post)
                db.session.commit()
                existing_urls.add(url)
            else:
                post = SocialPost.query.filter_by(url=url).first()

            if post is None:
                continue

            # Fetch transcript
            yield {'step': 'transcript', 'video': i, 'total': total, 'title': title, 'status': f'Fetching transcript ({i}/{total})...'}

            transcript = _fetch_youtube_transcript(url)
            transcript_status = 'transcript found' if transcript else 'no transcript available'

            # Generate summary
            yield {'step': 'summarize', 'video': i, 'total': total, 'title': title, 'status': f'Summarizing ({i}/{total})...'}

            if not post.summary_md:
                self._generate_summary_for_post(post, transcript)

            yield {'step': 'video_done', 'video': i, 'total': total, 'title': title, 'status': f'Done ({i}/{total}) — {transcript_status}'}

        # Update channel fetch time
        channel.last_fetched_at = datetime.now(timezone.utc)
        channel.last_success_at = datetime.now(timezone.utc)
        db.session.commit()

        yield {'step': 'done', 'video': total, 'total': total, 'title': '', 'status': f'Backfill complete — {total} videos processed'}

    def backfill_twitter_channel(self, channel_id: int):
        """Generator that fetches last 48h of tweets and summarizes them.

        Yields progress dicts similar to YouTube backfill.
        """
        channel = db.session.get(SocialChannel, channel_id)
        if channel is None or channel.platform != 'twitter':
            yield {'step': 'error', 'video': 0, 'total': 0, 'title': '', 'status': 'Channel not found or not Twitter'}
            return

        handle = channel.handle or channel.feed_url.replace('grok://twitter/', '')

        yield {'step': 'fetch_feed', 'video': 0, 'total': 0, 'title': '', 'status': f'Searching for @{handle} tweets (last 48h)...'}

        # Fetch tweets from last 48 hours
        try:
            new_count = self._fetch_twitter_posts(channel, hours=48)
            db.session.commit()
        except Exception as e:
            yield {'step': 'error', 'video': 0, 'total': 0, 'title': '', 'status': f'Failed to fetch tweets: {e}'}
            return

        yield {'step': 'feed_loaded', 'video': 0, 'total': new_count, 'title': '', 'status': f'Found {new_count} tweets'}

        if new_count == 0:
            channel.last_fetched_at = datetime.now(timezone.utc)
            channel.last_success_at = datetime.now(timezone.utc)
            db.session.commit()
            yield {'step': 'done', 'video': 0, 'total': 0, 'title': '', 'status': 'No tweets found in last 48 hours.'}
            return

        # Summarize each unsummarized tweet
        unsummarized = (
            SocialPost.query
            .filter(SocialPost.channel_id == channel.id, SocialPost.summary_md.is_(None))
            .order_by(SocialPost.published_at.desc())
            .all()
        )
        total = len(unsummarized)

        for i, post in enumerate(unsummarized, 1):
            yield {'step': 'summarize', 'video': i, 'total': total, 'title': post.title[:80], 'status': f'Summarizing tweet ({i}/{total})...'}
            self._generate_summary_for_post(post)
            yield {'step': 'video_done', 'video': i, 'total': total, 'title': post.title[:80], 'status': f'Done ({i}/{total})'}

        channel.last_fetched_at = datetime.now(timezone.utc)
        channel.last_success_at = datetime.now(timezone.utc)
        db.session.commit()

        yield {'step': 'done', 'video': total, 'total': total, 'title': '', 'status': f'Backfill complete — {new_count} tweets fetched, {total} summarized'}

    def _generate_summary_for_post(self, post: SocialPost, transcript: str | None = None):
        """Generate summary for a post with optional pre-fetched transcript."""
        platform = post.channel.platform if post.channel else 'rss'

        if platform == 'youtube' and transcript:
            user_content = (
                f"Channel: {post.channel.name if post.channel else 'Unknown'}\n"
                f"Title: {post.title}\n\n"
                f"Transcript:\n{transcript}"
            )
            system_prompt = YOUTUBE_SUMMARY_PROMPT
        else:
            user_content = (
                f"Platform: {platform}\n"
                f"Channel: {post.channel.name if post.channel else 'Unknown'}\n"
                f"Title: {post.title}\n"
                f"Description:\n{post.content_hint or '(no description)'}"
            )
            system_prompt = SUMMARY_SYSTEM_PROMPT

        try:
            llm = LLMGateway()
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_content},
                ],
                purpose='social_summary',
                section='social',
                provider='openai',
                model='gpt-4.1-nano',
                max_tokens=2000,
            )
        except (BudgetExhaustedError, Exception) as e:
            logger.warning("Summary generation failed for post id=%s: %s", post.id, e)
            return

        post.summary_md = result['content']
        post.summary_generated_at = datetime.now(timezone.utc)
        post.summary_cost_usd = result.get('cost_usd')
        post.summary_tokens = result.get('total_tokens')
        db.session.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_week_summary(self) -> dict:
        """Return a summary of posts from the last 7 days grouped by channel."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        today = datetime.now(timezone.utc)
        rows = (
            db.session.query(
                SocialChannel.id, SocialChannel.name, SocialChannel.platform,
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
            channels.append({'id': cid, 'name': cname, 'platform': cplatform, 'post_count': pcount})
            total_posts += pcount
        date_range = f"{cutoff.strftime('%b %d')} - {today.strftime('%b %d, %Y')}"
        return {'channels': channels, 'total_posts': total_posts, 'date_range': date_range}

    def get_channel_posts(self, channel_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
        posts = (
            SocialPost.query
            .filter(SocialPost.channel_id == channel_id)
            .order_by(SocialPost.published_at.desc())
            .offset(offset).limit(limit).all()
        )
        return [p.to_dict() for p in posts]

    def get_all_posts(self, limit: int = 50, offset: int = 0, days: int = 7) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        posts = (
            SocialPost.query
            .filter(SocialPost.published_at >= cutoff)
            .order_by(SocialPost.published_at.desc())
            .offset(offset).limit(limit).all()
        )
        return [p.to_dict() for p in posts]
