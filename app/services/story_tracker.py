import logging
import re
from datetime import datetime, timezone
from app.extensions import db
from app.models.topic import TrackedTopic, Story, Event
from app import feature_flags

logger = logging.getLogger(__name__)

# Common words to ignore when matching topic terms to cluster text
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'its', 'as', 'are', 'was',
    'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did',
    'not', 'no', 'nor', 'so', 'if', 'up', 'out', 'about', 'into',
    'than', 'then', 'that', 'this', 'these', 'those', 'what', 'which',
    'who', 'how', 'will', 'can', 'may', 'vs', 'new', 'says', 'said',
    'get', 'got', 'set', 'now', 'just', 'also', 'here', 'more', 'most',
    'after', 'before', 'over', 'under', 'between', 'through', 'during',
    'while', 'where', 'when', 'why', 'all', 'any', 'each', 'every',
    'other', 'some', 'such', 'only', 'own', 'same', 'our', 'your',
    'his', 'her', 'their', 'my', 'we', 'you', 'he', 'she', 'they',
}


def _extract_words(text):
    """Extract lowercase words from text as a set."""
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))


def _significant_terms(terms):
    """Filter topic terms to only significant words (not stopwords, len > 1)."""
    return [t for t in terms if t.lower() not in STOPWORDS and len(t) > 1]


def _fuzzy_match(term, word_set):
    """Check if a term matches any word in the set.
    Exact match, or prefix match for words >= 5 chars (handles plurals like
    model/models, benchmark/benchmarks, but NOT short words like tech/technology).
    """
    if term in word_set:
        return True
    if len(term) >= 5:
        for w in word_set:
            if len(w) >= 5 and (w.startswith(term) or term.startswith(w)):
                return True
    return False


def _build_cluster_words(cluster, articles):
    """Build word set from cluster label AND all article titles."""
    texts = [cluster.label or '']
    for a in articles:
        if a.title:
            texts.append(a.title)
    return _extract_words(' '.join(texts))


class StoryTracker:
    """Track topics -> stories -> events over time. Gated by FF_STORY_TRACKING."""

    def is_enabled(self):
        return feature_flags.is_enabled('story_tracking')

    def link_cluster_to_story(self, cluster, articles):
        """
        Match cluster to a tracked topic using topic name keywords.
        Matches against cluster label + all article titles for broader coverage.

        Matching rules (topic name terms only, no description):
        - 1-2 significant terms → ALL must match
        - 3+ significant terms → at least 2 must match
        """
        if not self.is_enabled():
            return None

        cluster_text = (cluster.label or '').lower()
        if not cluster_text:
            return None

        # Build words from cluster label + ALL article titles
        cluster_words = _build_cluster_words(cluster, articles)

        # Check active topics
        topics = TrackedTopic.query.filter_by(is_active=True).all()
        for topic in topics:
            # Extract terms from topic name (handles hyphens: "US-China" → {us, china})
            topic_terms = _significant_terms(list(_extract_words(topic.name)))
            if not topic_terms:
                continue

            # Count matching terms
            matching = sum(1 for t in topic_terms if _fuzzy_match(t, cluster_words))

            # Threshold: 1-2 terms → require all; 3+ terms → require at least 2
            required = len(topic_terms) if len(topic_terms) <= 2 else 2

            if matching >= required:
                story = self._find_or_create_story(topic, cluster, cluster_words)
                if story:
                    self._add_event(story, cluster, articles)
                    logger.info(
                        f"[StoryTracker] Linked '{cluster.label[:60]}' → "
                        f"topic '{topic.name}' ({matching}/{len(topic_terms)} terms)"
                    )
                    return story

        return None

    def _find_or_create_story(self, topic, cluster, cluster_words):
        """Find existing developing story or create new one."""
        stories = Story.query.filter(
            Story.topic_id == topic.id,
            Story.status.in_(['developing', 'ongoing']),
        ).all()

        for story in stories:
            story_terms = _significant_terms(list(_extract_words(story.title)))
            # Require at least 2 significant story terms to fuzzy-match
            matching = sum(1 for term in story_terms if _fuzzy_match(term, cluster_words))
            if story_terms and matching >= min(2, len(story_terms)):
                story.last_updated = datetime.now(timezone.utc)
                ids = story.cluster_ids_json or []
                ids.append(cluster.id)
                story.cluster_ids_json = ids
                return story

        # Create new story
        story = Story(
            topic_id=topic.id,
            title=cluster.label or f"Story in {topic.name}",
            status='developing',
            cluster_ids_json=[cluster.id],
            last_updated=datetime.now(timezone.utc),
        )
        db.session.add(story)
        db.session.flush()  # Assign story.id before creating events
        return story

    def _add_event(self, story, cluster, articles):
        """Add an event to a story from a cluster."""
        source_urls = [a.url for a in articles[:5]]
        event = Event(
            story_id=story.id,
            cluster_id=cluster.id,
            description=cluster.label or 'New development',
            event_date=datetime.now(timezone.utc),
            source_urls_json=source_urls,
        )
        db.session.add(event)

    def update_story_status(self, story_id, status):
        """Update story status (developing, ongoing, resolved, stale)."""
        story = Story.query.get(story_id)
        if story:
            story.status = status
            story.last_updated = datetime.now(timezone.utc)
            db.session.commit()
        return story
