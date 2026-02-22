"""
Progress tracking — patched for library use.
Replaces rich/colorama terminal output with Python logging.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, List

logger = logging.getLogger(__name__)


class AgentProgress:
    """Manages progress tracking for multiple agents (logging-only version)."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, str]] = {}
        self.update_handlers: List[Callable] = []
        self.started = False

    def register_handler(self, handler):
        self.update_handlers.append(handler)
        return handler

    def unregister_handler(self, handler):
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        if not self.started:
            self.started = True
            logger.debug("Agent progress tracking started")

    def stop(self):
        if self.started:
            self.started = False
            logger.debug("Agent progress tracking stopped")

    def update_status(self, agent_name: str, ticker: Optional[str] = None,
                      status: str = "", analysis: Optional[str] = None):
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = {"status": "", "ticker": None}

        if ticker:
            self.agent_status[agent_name]["ticker"] = ticker
        if status:
            self.agent_status[agent_name]["status"] = status
        if analysis:
            self.agent_status[agent_name]["analysis"] = analysis

        timestamp = datetime.now(timezone.utc).isoformat()
        self.agent_status[agent_name]["timestamp"] = timestamp

        for handler in self.update_handlers:
            handler(agent_name, ticker, status, analysis, timestamp)

        logger.debug(f"Agent {agent_name} [{ticker or '-'}]: {status}")

    def get_all_status(self):
        return {
            agent_name: {
                "ticker": info["ticker"],
                "status": info["status"],
                "display_name": self._get_display_name(agent_name),
            }
            for agent_name, info in self.agent_status.items()
        }

    def _get_display_name(self, agent_name: str) -> str:
        return agent_name.replace("_agent", "").replace("_", " ").title()


# Global instance
progress = AgentProgress()
