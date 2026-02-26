"""
Stock research service — orchestrates in-depth stock analysis
for the Telegram /research command. Sends progressive updates.
"""
import logging
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def _pct(val):
    """Format a decimal ratio as percentage string."""
    if val is None:
        return 'N/A'
    return f'{val * 100:+.1f}%' if val > 0 else f'{val * 100:.1f}%'


def _fmt(val, prefix='$', decimals=2):
    """Format a number with optional prefix."""
    if val is None:
        return 'N/A'
    abs_val = abs(val)
    sign = '-' if val < 0 else ''
    if abs_val >= 1e12:
        return f'{sign}{prefix}{abs_val/1e12:.1f}T'
    if abs_val >= 1e9:
        return f'{sign}{prefix}{abs_val/1e9:.2f}B'
    if abs_val >= 1e6:
        return f'{sign}{prefix}{abs_val/1e6:.0f}M'
    if abs_val >= 1e3:
        return f'{sign}{prefix}{abs_val/1e3:.1f}K'
    return f'{sign}{prefix}{abs_val:.{decimals}f}'


def _safe(val, fmt='.2f'):
    if val is None:
        return 'N/A'
    return f'{val:{fmt}}'


class StockResearchService:
    """Run in-depth stock research and send results via Telegram."""

    def __init__(self, bot, app):
        self.bot = bot
        self.app = app

    def run_research(self, ticker, chat_id):
        """
        Main entry point. Runs in a background thread with app context.
        Sends progressive Telegram messages as each section completes.
        """
        ticker = ticker.upper().strip()
        self.bot.send_message(chat_id, f'*{ticker} Research* — fetching data...')

        try:
            data = self._fetch_data(ticker)
        except Exception as e:
            logger.error(f"[Research] Data fetch failed for {ticker}: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'Error fetching data for {ticker}: {e}')
            return

        # Phase 1: Company overview
        self._send_overview(chat_id, ticker, data)

        # Phase 2: Fundamentals
        self._send_fundamentals(chat_id, ticker, data)

        # Phase 3: Value assessment
        self._send_value(chat_id, ticker, data)

        # Phase 4: Momentum
        self._send_momentum(chat_id, ticker, data)

        # Phase 5: Financial statement summary
        self._send_financials(chat_id, ticker, data)

        # Phase 6: Chart
        self._send_chart(chat_id, ticker, data)

        # Phase 7: AI Analysis
        self._send_ai_analysis(chat_id, ticker)

    def _fetch_data(self, ticker):
        """Fetch all data from financialdatasets.ai API."""
        from vendor.ai_hedge_fund.tools.api import (
            get_prices, get_financial_metrics, search_line_items,
            get_market_cap, prices_to_df,
        )
        from vendor.ai_hedge_fund.data.models import CompanyFacts, CompanyFactsResponse

        end = date.today().isoformat()
        start_1y = (date.today() - timedelta(days=365)).isoformat()

        # Company facts
        company_facts = None
        try:
            import requests as req
            api_key = os.environ.get('FINANCIAL_DATASETS_API_KEY')
            headers = {'X-API-KEY': api_key} if api_key else {}
            resp = req.get(
                f'https://api.financialdatasets.ai/company/facts/?ticker={ticker}',
                headers=headers, timeout=15,
            )
            if resp.status_code == 200:
                cf_resp = CompanyFactsResponse(**resp.json())
                company_facts = cf_resp.company_facts
        except Exception as e:
            logger.warning(f"[Research] Company facts failed: {e}")

        # Prices (1 year)
        prices = get_prices(ticker, start_1y, end)
        prices_df = prices_to_df(prices) if prices else None

        # Financial metrics (8 quarters)
        metrics = get_financial_metrics(ticker, end, period='quarterly', limit=8)

        # Line items (8 quarters)
        line_items = search_line_items(
            ticker,
            line_items=[
                'revenue', 'net_income', 'earnings_per_share',
                'total_assets', 'total_liabilities', 'total_debt',
                'free_cash_flow', 'operating_income',
            ],
            end_date=end, period='quarterly', limit=8,
        )

        # Market cap
        market_cap = get_market_cap(ticker, end)

        return {
            'company': company_facts,
            'prices': prices,
            'prices_df': prices_df,
            'metrics': metrics,
            'line_items': line_items,
            'market_cap': market_cap,
        }

    def _send_overview(self, chat_id, ticker, data):
        co = data['company']
        mc = data['market_cap']
        lines = [f'*{ticker} — Company Overview*']
        if co:
            if co.name:
                lines.append(f'Name: {co.name}')
            if co.sector:
                lines.append(f'Sector: {co.sector}')
            if co.industry:
                lines.append(f'Industry: {co.industry}')
            if co.number_of_employees:
                lines.append(f'Employees: {co.number_of_employees:,}')
            if co.exchange:
                lines.append(f'Exchange: {co.exchange}')
        if mc:
            lines.append(f'Market Cap: {_fmt(mc)}')
        if not co and not mc:
            lines.append('No company data available')
        self.bot.send_message(chat_id, '\n'.join(lines))

    def _send_fundamentals(self, chat_id, ticker, data):
        metrics = data['metrics']
        if not metrics:
            self.bot.send_message(chat_id, f'*{ticker} — Fundamentals*\nNo metrics data available')
            return

        m = metrics[0]  # Latest quarter
        lines = [
            f'*{ticker} — Fundamentals* ({m.report_period})',
            '',
            '*Valuation*',
            f'P/E: {_safe(m.price_to_earnings_ratio)}',
            f'P/B: {_safe(m.price_to_book_ratio)}',
            f'P/S: {_safe(m.price_to_sales_ratio)}',
            f'EV/EBITDA: {_safe(m.enterprise_value_to_ebitda_ratio)}',
            f'PEG: {_safe(m.peg_ratio)}',
            f'FCF Yield: {_pct(m.free_cash_flow_yield)}',
            '',
            '*Profitability*',
            f'Gross Margin: {_pct(m.gross_margin)}',
            f'Operating Margin: {_pct(m.operating_margin)}',
            f'Net Margin: {_pct(m.net_margin)}',
            f'ROE: {_pct(m.return_on_equity)}',
            f'ROA: {_pct(m.return_on_assets)}',
            f'ROIC: {_pct(m.return_on_invested_capital)}',
            '',
            '*Growth*',
            f'Revenue Growth: {_pct(m.revenue_growth)}',
            f'EPS Growth: {_pct(m.earnings_per_share_growth)}',
            f'FCF Growth: {_pct(m.free_cash_flow_growth)}',
            '',
            '*Health*',
            f'Current Ratio: {_safe(m.current_ratio)}',
            f'D/E: {_safe(m.debt_to_equity)}',
            f'Interest Coverage: {_safe(m.interest_coverage, ".1f")}',
        ]
        self.bot.send_message(chat_id, '\n'.join(lines))

    def _send_value(self, chat_id, ticker, data):
        metrics = data['metrics']
        if not metrics:
            return

        m = metrics[0]
        lines = [f'*{ticker} — Value Assessment*', '']

        pe = m.price_to_earnings_ratio
        pb = m.price_to_book_ratio
        fcf_yield = m.free_cash_flow_yield
        peg = m.peg_ratio
        ev_rev = m.enterprise_value_to_revenue_ratio

        lines.append(f'P/E: {_safe(pe)}')
        lines.append(f'P/B: {_safe(pb)}')
        lines.append(f'EV/Revenue: {_safe(ev_rev)}')
        lines.append(f'PEG Ratio: {_safe(peg)}')
        lines.append(f'FCF Yield: {_pct(fcf_yield)}')
        lines.append(f'Book Value/Share: {_fmt(m.book_value_per_share)}')
        lines.append('')

        # Simple rule-based assessment
        signals = []
        if pe is not None:
            if pe < 15:
                signals.append('low P/E')
            elif pe > 30:
                signals.append('high P/E')
        if peg is not None:
            if peg < 1:
                signals.append('PEG < 1 (growth undervalued)')
            elif peg > 2:
                signals.append('PEG > 2 (growth expensive)')
        if fcf_yield is not None:
            if fcf_yield > 0.05:
                signals.append('strong FCF yield')
            elif fcf_yield < 0.02:
                signals.append('weak FCF yield')

        if signals:
            lines.append('Signals: ' + ', '.join(signals))
        self.bot.send_message(chat_id, '\n'.join(lines))

    def _send_momentum(self, chat_id, ticker, data):
        prices_df = data['prices_df']
        metrics = data['metrics']
        lines = [f'*{ticker} — Momentum*', '']

        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            current = close.iloc[-1]

            # Price changes
            def _change(days):
                if len(close) > days:
                    old = close.iloc[-days - 1]
                    return (current - old) / old
                return None

            lines.append('*Price Performance*')
            for label, days in [('1M', 21), ('3M', 63), ('6M', 126), ('1Y', 252)]:
                chg = _change(days)
                if chg is not None:
                    emoji = '🟢' if chg >= 0 else '🔴'
                    lines.append(f'{emoji} {label}: {_pct(chg)}')

            # SMAs
            lines.append('')
            if len(close) >= 50:
                sma50 = close.rolling(50).mean().iloc[-1]
                pos = 'above' if current > sma50 else 'below'
                lines.append(f'50 SMA: ${sma50:.2f} (price {pos})')
            if len(close) >= 200:
                sma200 = close.rolling(200).mean().iloc[-1]
                pos = 'above' if current > sma200 else 'below'
                lines.append(f'200 SMA: ${sma200:.2f} (price {pos})')

            # RSI-14
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                label = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
                lines.append(f'RSI-14: {rsi:.1f} ({label})')

            # Volume trend
            if len(prices_df) >= 60:
                vol = prices_df['volume']
                avg_20 = vol.iloc[-20:].mean()
                avg_60 = vol.iloc[-60:].mean()
                ratio = avg_20 / avg_60 if avg_60 > 0 else 1
                trend = 'increasing' if ratio > 1.1 else 'decreasing' if ratio < 0.9 else 'stable'
                lines.append(f'Volume trend: {trend} (20d/60d = {ratio:.2f}x)')
        else:
            lines.append('No price data available')

        # EPS trajectory
        if metrics and len(metrics) >= 4:
            lines.append('')
            lines.append('*EPS Trajectory (last 4Q)*')
            for m in reversed(metrics[:4]):
                if m.earnings_per_share is not None:
                    lines.append(f'  {m.report_period[:7]}: ${m.earnings_per_share:.2f}')

        self.bot.send_message(chat_id, '\n'.join(lines))

    def _send_financials(self, chat_id, ticker, data):
        line_items = data['line_items']
        if not line_items:
            self.bot.send_message(chat_id, f'*{ticker} — Financials*\nNo financial statement data')
            return

        latest = line_items[0]
        lines = [
            f'*{ticker} — Financial Statement* ({latest.report_period})',
            '',
        ]

        rev = getattr(latest, 'revenue', None)
        ni = getattr(latest, 'net_income', None)
        eps = getattr(latest, 'earnings_per_share', None)
        ta = getattr(latest, 'total_assets', None)
        tl = getattr(latest, 'total_liabilities', None)
        td = getattr(latest, 'total_debt', None)
        fcf = getattr(latest, 'free_cash_flow', None)
        oi = getattr(latest, 'operating_income', None)

        lines.append(f'Revenue: {_fmt(rev)}')
        lines.append(f'Net Income: {_fmt(ni)}')
        lines.append(f'EPS: {_fmt(eps, "$", 2)}')
        lines.append(f'Operating Income: {_fmt(oi)}')
        lines.append(f'Free Cash Flow: {_fmt(fcf)}')
        lines.append('')
        lines.append(f'Total Assets: {_fmt(ta)}')
        lines.append(f'Total Liabilities: {_fmt(tl)}')
        lines.append(f'Total Debt: {_fmt(td)}')

        # QoQ comparison
        if len(line_items) >= 2:
            prev = line_items[1]
            lines.append('')
            lines.append(f'*vs Prior Quarter* ({prev.report_period})')
            prev_rev = getattr(prev, 'revenue', None)
            prev_ni = getattr(prev, 'net_income', None)
            if rev and prev_rev and prev_rev != 0:
                lines.append(f'Revenue QoQ: {_pct((rev - prev_rev) / abs(prev_rev))}')
            if ni and prev_ni and prev_ni != 0:
                lines.append(f'Net Income QoQ: {_pct((ni - prev_ni) / abs(prev_ni))}')

        # YoY comparison
        if len(line_items) >= 5:
            yoy = line_items[4]
            yoy_rev = getattr(yoy, 'revenue', None)
            yoy_ni = getattr(yoy, 'net_income', None)
            lines.append('')
            lines.append(f'*vs Year Ago* ({yoy.report_period})')
            if rev and yoy_rev and yoy_rev != 0:
                lines.append(f'Revenue YoY: {_pct((rev - yoy_rev) / abs(yoy_rev))}')
            if ni and yoy_ni and yoy_ni != 0:
                lines.append(f'Net Income YoY: {_pct((ni - yoy_ni) / abs(yoy_ni))}')

        self.bot.send_message(chat_id, '\n'.join(lines))

    def _send_chart(self, chat_id, ticker, data):
        try:
            from app.services.stock_chart_service import generate_research_chart
            chart_path = generate_research_chart(
                ticker, data['prices_df'], data['metrics'], data['line_items'],
            )
            self.bot.send_photo(chat_id, chart_path, caption=f'{ticker} Research Chart')
            os.unlink(chart_path)
        except Exception as e:
            logger.error(f"[Research] Chart generation failed: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'Chart generation failed: {e}')

    def _send_ai_analysis(self, chat_id, ticker):
        """Run hedge fund multi-agent analysis and send results."""
        self.bot.send_message(chat_id, f'*{ticker} — AI Analysis*\nRunning analyst agents...')

        try:
            from app.services.hedge_fund_service import HedgeFundService
            from app import feature_flags

            with self.app.app_context():
                # Temporarily enable the flag — /research is an explicit user request
                was_enabled = feature_flags.is_enabled('hedge_fund_analysis')
                if not was_enabled:
                    feature_flags.set_flag('hedge_fund_analysis', True)

                try:
                    service = HedgeFundService()
                    service.tickers = [ticker]
                    service.analysts = ['fundamentals', 'technicals', 'valuation', 'sentiment']

                    analyses, usage = service.run_analysis(date.today())
                finally:
                    if not was_enabled:
                        feature_flags.set_flag('hedge_fund_analysis', False)

                if not analyses:
                    self.bot.send_message(chat_id, 'AI analysis returned no results.')
                    return

                analysis = analyses[0]
                signals = analysis.analyst_signals_json or {}

                for agent, sig in signals.items():
                    if not isinstance(sig, dict):
                        continue
                    signal = sig.get('signal', 'N/A')
                    confidence = sig.get('confidence', 'N/A')
                    reasoning = sig.get('reasoning', '')

                    emoji = '🟢' if signal == 'bullish' else '🔴' if signal == 'bearish' else '🟡'
                    name = agent.replace('_', ' ').title()

                    msg = f'{emoji} *{name}*: {signal} ({confidence}% confidence)'
                    if reasoning:
                        if isinstance(reasoning, dict):
                            # Extract key points from reasoning dict
                            points = []
                            for k, v in reasoning.items():
                                if isinstance(v, str) and v:
                                    points.append(f'• {v[:200]}')
                            if points:
                                msg += '\n' + '\n'.join(points[:5])
                        elif isinstance(reasoning, str):
                            msg += f'\n{reasoning[:500]}'
                    self.bot.send_message(chat_id, msg)

                # Consensus
                consensus = analysis.consensus_signal or 'neutral'
                conf = analysis.consensus_confidence or 0
                emoji = '🟢' if consensus == 'bullish' else '🔴' if consensus == 'bearish' else '🟡'
                self.bot.send_message(
                    chat_id,
                    f'\n*{ticker} — Consensus*\n'
                    f'{emoji} Signal: *{consensus.upper()}* ({conf}% confidence)',
                )

        except Exception as e:
            logger.error(f"[Research] AI analysis failed for {ticker}: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'AI analysis failed: {e}')
