import logging
import time
from datetime import date, datetime, timezone
from flask import current_app
from app.extensions import db
from app.models.cost import LLMCallLog

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
    'gpt-5.2': {'input': 1.75, 'output': 14.00},
}


class BudgetExhaustedError(Exception):
    def __init__(self, section=None):
        self.section = section
        super().__init__(f"Token budget exhausted for section: {section or 'daily'}")


class LLMGateway:
    def __init__(self, app_config=None):
        config = app_config or current_app.config
        self.model = config.get('LLM_MODEL', 'gpt-5.2')
        self.daily_budget_tokens = config.get('LLM_DAILY_TOKEN_BUDGET', 100_000)
        self.daily_budget_usd = config.get('LLM_DAILY_BUDGET_USD', 1.00)
        self.section_budgets = config.get('LLM_SECTION_BUDGETS', {})
        self.api_key = config.get('OPENAI_API_KEY')

    def call(self, messages, purpose, section=None, brief_id=None, max_tokens=None):
        """
        Central LLM call. Checks budget, makes call, logs cost.
        Returns: {content, prompt_tokens, completion_tokens, total_tokens, cost_usd}
        """
        remaining = self._get_remaining_budget(section)
        if remaining <= 0:
            raise BudgetExhaustedError(section)

        effective_max = min(max_tokens or 2000, max(remaining, 100))

        import openai
        client = openai.OpenAI(api_key=self.api_key)

        start_ms = int(time.time() * 1000)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=effective_max,
            )
        except Exception as e:
            logger.error(f"LLM call failed ({purpose}): {e}")
            raise

        latency_ms = int(time.time() * 1000) - start_ms
        usage = response.usage
        cost = self._compute_cost(usage.prompt_tokens, usage.completion_tokens)

        self._log_call(purpose, section, brief_id, usage, cost, latency_ms)

        return {
            'content': response.choices[0].message.content,
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens,
            'cost_usd': cost,
        }

    def _compute_cost(self, prompt_tokens, completion_tokens):
        """Compute USD cost based on model pricing."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING['gpt-5.2'])
        input_cost = (prompt_tokens / 1_000_000) * pricing['input']
        output_cost = (completion_tokens / 1_000_000) * pricing['output']
        return round(input_cost + output_cost, 6)

    def _get_remaining_budget(self, section=None):
        """Get remaining token budget for today (overall or per-section)."""
        today = date.today()
        today_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)

        query = db.session.query(
            db.func.coalesce(db.func.sum(LLMCallLog.total_tokens), 0)
        ).filter(LLMCallLog.created_at >= today_start)

        if section:
            query = query.filter(LLMCallLog.section == section)
            section_fraction = self.section_budgets.get(section, 0.1)
            budget = int(self.daily_budget_tokens * section_fraction)
        else:
            budget = self.daily_budget_tokens

        used = query.scalar()
        return budget - used

    def _log_call(self, purpose, section, brief_id, usage, cost, latency_ms):
        """Insert LLMCallLog row."""
        log = LLMCallLog(
            call_purpose=purpose,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            section=section,
            brief_id=brief_id,
        )
        db.session.add(log)
        db.session.commit()
        logger.info(
            f"LLM call: {purpose} | {usage.total_tokens} tokens | ${cost:.4f} | {latency_ms}ms"
        )

    def determine_degradation_level(self, section):
        """Determine degradation level based on remaining section budget."""
        remaining = self._get_remaining_budget(section)
        section_fraction = self.section_budgets.get(section, 0.1)
        budget = int(self.daily_budget_tokens * section_fraction)

        if budget == 0:
            return 4

        pct = remaining / budget
        if pct > 0.60:
            return 0  # Full synthesis
        elif pct > 0.30:
            return 1  # Skip claims/framing
        elif pct > 0.15:
            return 2  # Shortened summaries
        elif pct > 0.05:
            return 3  # Fewer clusters
        else:
            return 4  # Extractive only
