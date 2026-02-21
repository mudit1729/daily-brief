import logging
from datetime import date
from app.extensions import db
from app.models.investment import InvestmentThesis
from app.integrations.market_data import MarketDataService
from app import feature_flags

logger = logging.getLogger(__name__)


class InvestmentService:
    def __init__(self):
        self.market_service = MarketDataService()

    def generate_thesis(self, target_date, brief_id, snapshots, top_clusters, llm_gateway):
        """
        Generate investment thesis if momentum x value gate passes.
        Otherwise returns "No thesis today".
        """
        if not feature_flags.is_enabled('investment_thesis'):
            return None

        gate_passed, signals = self.market_service.check_momentum_value_gate(snapshots)

        thesis = InvestmentThesis(
            date=target_date,
            brief_id=brief_id,
            gate_passed=gate_passed,
            momentum_signal={
                'count': signals['momentum_count'],
                'pass': signals['momentum_pass'],
            },
            value_signal={
                'gold_change_pct': signals['gold_change_pct'],
                'pass': signals['value_pass'],
            },
        )

        if not gate_passed:
            thesis.thesis_text = "No thesis today. Gate conditions not met."
            thesis.supporting_clusters_json = []
        else:
            thesis_text = self._generate_with_llm(
                snapshots, top_clusters, signals, llm_gateway, brief_id
            )
            thesis.thesis_text = thesis_text
            thesis.supporting_clusters_json = [c.id for c in top_clusters[:5]]

        db.session.add(thesis)
        db.session.commit()
        return thesis

    def _generate_with_llm(self, snapshots, clusters, signals, llm_gateway, brief_id):
        """Generate thesis text using LLM."""
        market_summary = '\n'.join(
            f"- {s['name']} ({s['symbol']}): ${s['price']:.2f} ({s['change_pct']:+.2f}%)"
            for s in snapshots
        )

        cluster_summaries = '\n'.join(
            f"- {c.label}: {c.summary or 'No summary'}"
            for c in clusters[:5]
            if c.label
        )

        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a concise investment analyst. Generate a brief investment thesis '
                    '(3-5 sentences) based on current market data and top news clusters. '
                    'Focus on actionable insights. Be specific about which sectors or trends '
                    'to watch. Include risk factors.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f"Market Data:\n{market_summary}\n\n"
                    f"Momentum Signal: {signals['momentum_count']}/3 US indices up >0.3%\n"
                    f"Gold Change: {signals['gold_change_pct']:+.2f}%\n\n"
                    f"Top News Clusters:\n{cluster_summaries}\n\n"
                    "Generate a brief investment thesis for today."
                ),
            },
        ]

        try:
            result = llm_gateway.call(
                messages=messages,
                purpose='investment_thesis',
                section='investment_thesis',
                brief_id=brief_id,
                max_tokens=300,
            )
            return result['content']
        except Exception as e:
            logger.error(f"Failed to generate investment thesis: {e}")
            return f"Thesis generation failed: {str(e)}"
