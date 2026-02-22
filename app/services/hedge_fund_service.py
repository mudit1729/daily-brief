"""
HedgeFundService — wraps the vendored ai-hedge-fund multi-agent
analysis and integrates it with Signal Brief's budget tracking,
feature flags, and data storage.
"""
import logging
from datetime import date, timedelta

from flask import current_app
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.extensions import db
from app.models.hedge_fund import HedgeFundAnalysis
from app import feature_flags

logger = logging.getLogger(__name__)

# Analyst key mapping — maps our config names to ai-hedge-fund analyst keys
ANALYST_KEY_MAP = {
    'technicals': 'technical_analyst',
    'valuation': 'valuation_analyst',
    'sentiment': 'sentiment_analyst',
    'fundamentals': 'fundamentals_analyst',
    'growth': 'growth_analyst',
    'news_sentiment': 'news_sentiment_analyst',
    'warren_buffett': 'warren_buffett',
    'charlie_munger': 'charlie_munger',
    'ben_graham': 'ben_graham',
    'peter_lynch': 'peter_lynch',
    'cathie_wood': 'cathie_wood',
    'michael_burry': 'michael_burry',
    'aswath_damodaran': 'aswath_damodaran',
    'stanley_druckenmiller': 'stanley_druckenmiller',
    'bill_ackman': 'bill_ackman',
    'phil_fisher': 'phil_fisher',
    'rakesh_jhunjhunwala': 'rakesh_jhunjhunwala',
    'mohnish_pabrai': 'mohnish_pabrai',
}


class HedgeFundService:
    """Runs ai-hedge-fund analysis and stores structured results."""

    DEFAULT_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']
    DEFAULT_ANALYSTS = ['technicals', 'valuation', 'sentiment', 'warren_buffett']

    def __init__(self, app_config=None):
        config = app_config or current_app.config
        raw_tickers = config.get('HEDGE_FUND_TICKERS', self.DEFAULT_TICKERS)
        if isinstance(raw_tickers, str):
            self.tickers = [t.strip() for t in raw_tickers.split(',') if t.strip()]
        else:
            self.tickers = list(raw_tickers)

        raw_analysts = config.get('HEDGE_FUND_ANALYSTS', self.DEFAULT_ANALYSTS)
        if isinstance(raw_analysts, str):
            self.analysts = [a.strip() for a in raw_analysts.split(',') if a.strip()]
        else:
            self.analysts = list(raw_analysts)

        self.model_name = config.get('LLM_MODEL', 'gpt-4.1')
        self.model_provider = config.get('HEDGE_FUND_MODEL_PROVIDER', 'OpenAI')

    def run_analysis(self, target_date, brief_id=None):
        """
        Run the multi-agent hedge fund analysis for all configured tickers.

        Returns:
            tuple: (list[HedgeFundAnalysis], dict with usage info)
        """
        if not feature_flags.is_enabled('hedge_fund_analysis'):
            logger.info("Hedge fund analysis skipped: feature flag disabled")
            return [], {'total_tokens': 0, 'cost_usd': 0.0}

        # Map our analyst names to ai-hedge-fund keys
        selected_analysts = []
        for a in self.analysts:
            key = ANALYST_KEY_MAP.get(a, a)
            selected_analysts.append(key)

        # Build a stub portfolio (we use this for signals, not actual trading)
        portfolio = {
            'cash': 100_000,
            'margin_requirement': 0.0,
            'margin_used': 0.0,
            'positions': {
                ticker: {
                    'long': 0, 'short': 0,
                    'long_cost_basis': 0.0, 'short_cost_basis': 0.0,
                    'short_margin_used': 0.0,
                }
                for ticker in self.tickers
            },
            'realized_gains': {
                ticker: {'long': 0.0, 'short': 0.0}
                for ticker in self.tickers
            },
        }

        # Date range: 90 days of history for technicals
        end_date = target_date.isoformat()
        start_date = (target_date - timedelta(days=90)).isoformat()

        logger.info(
            f"[HedgeFund] Running analysis: tickers={self.tickers}, "
            f"analysts={self.analysts}, range={start_date}..{end_date}"
        )

        try:
            from vendor.ai_hedge_fund.main import run_hedge_fund

            result = run_hedge_fund(
                tickers=self.tickers,
                start_date=start_date,
                end_date=end_date,
                portfolio=portfolio,
                show_reasoning=True,
                selected_analysts=selected_analysts,
                model_name=self.model_name,
                model_provider=self.model_provider,
            )
        except Exception as e:
            logger.error(f"[HedgeFund] Analysis failed: {e}")
            return [], {'total_tokens': 0, 'cost_usd': 0.0}

        # Parse results into per-ticker HedgeFundAnalysis records
        analyses = self._store_results(result, target_date, brief_id)

        logger.info(f"[HedgeFund] Complete: {len(analyses)} tickers analyzed")
        return analyses, {'total_tokens': 0, 'cost_usd': 0.0}

    def _store_results(self, result, target_date, brief_id):
        """Parse ai-hedge-fund result dict and upsert HedgeFundAnalysis rows."""
        decisions = result.get('decisions') or {}
        analyst_signals = result.get('analyst_signals') or {}

        analyses = []
        for ticker in self.tickers:
            # Build per-ticker analyst signals dict
            ticker_signals = {}
            for agent_key, agent_data in analyst_signals.items():
                if isinstance(agent_data, dict) and ticker in agent_data:
                    clean_key = agent_key.replace('_agent', '')
                    ticker_signals[clean_key] = agent_data[ticker]

            # Compute consensus from signals
            consensus_signal, consensus_confidence = self._compute_consensus(ticker_signals)

            # Get portfolio manager decision for this ticker
            decision = decisions.get(ticker)

            # Upsert
            analysis = HedgeFundAnalysis.query.filter_by(
                date=target_date, ticker=ticker
            ).first()
            if not analysis:
                analysis = HedgeFundAnalysis(date=target_date, ticker=ticker)
                db.session.add(analysis)

            analysis.brief_id = brief_id
            analysis.analyst_signals_json = ticker_signals
            analysis.consensus_signal = consensus_signal
            analysis.consensus_confidence = consensus_confidence
            analysis.analysts_used = self.analysts
            analysis.decision_json = decision
            analysis.total_cost_usd = 0.0

            analyses.append(analysis)

        db.session.commit()
        return analyses

    def _compute_consensus(self, ticker_signals):
        """Derive consensus signal and confidence from individual analyst signals."""
        if not ticker_signals:
            return 'neutral', 0.0

        bullish = 0
        bearish = 0
        neutral = 0
        total_confidence = 0.0
        count = 0

        for agent, sig in ticker_signals.items():
            if not isinstance(sig, dict):
                continue
            signal = (sig.get('signal') or '').lower()
            confidence = sig.get('confidence', 50)

            if signal == 'bullish':
                bullish += 1
            elif signal == 'bearish':
                bearish += 1
            else:
                neutral += 1

            total_confidence += confidence
            count += 1

        if count == 0:
            return 'neutral', 0.0

        avg_confidence = total_confidence / count

        if bullish > bearish and bullish > neutral:
            return 'bullish', round(avg_confidence, 1)
        elif bearish > bullish and bearish > neutral:
            return 'bearish', round(avg_confidence, 1)
        else:
            return 'neutral', round(avg_confidence, 1)
