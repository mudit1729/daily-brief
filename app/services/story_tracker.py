import logging
from datetime import datetime, timezone
from app.extensions import db
from app.models.topic import TrackedTopic, Story, Event
from app import feature_flags

logger = logging.getLogger(__name__)


class StoryTracker:
    """Track topics -> stories -> events over time. Gated by FF_STORY_TRACKING."""

    def is_enabled(self):
        return feature_flags.is_enabled('story_tracking')

    def link_cluster_to_story(self, cluster, articles):
        """
        Try to match a cluster to an existing story by title/topic overlap.
        If no match found, optionally create a new story if cluster is significant.
        """
        if not self.is_enabled():
            return None

        cluster_text = (cluster.label or '').lower()
        if not cluster_text:
            return None

        # Check active topics
        topics = TrackedTopic.query.filter_by(is_active=True).all()
        for topic in topics:
            topic_terms = topic.name.lower().split()
            if any(term in cluster_text for term in topic_terms):
                # Check existing stories for this topic
                story = self._find_or_create_story(topic, cluster)
                if story:
                    self._add_event(story, cluster, articles)
                    return story

        return None

    def _find_or_create_story(self, topic, cluster):
        """Find existing developing story or create new one."""
        stories = Story.query.filter(
            Story.topic_id == topic.id,
            Story.status.in_(['developing', 'ongoing']),
        ).all()

        cluster_text = (cluster.label or '').lower()
        for story in stories:
            story_terms = story.title.lower().split()
            if any(term in cluster_text for term in story_terms if len(term) > 3):
                story.last_updated = datetime.now(timezone.utc)
                # Append cluster ID to tracking
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
