"""
Stock research service — orchestrates in-depth stock analysis
for the Telegram /research command. Sends progressive updates
with verbose data, numbers, and source citations.
"""
import logging
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)

DATA_SOURCE = 'financialdatasets.ai'
DATA_CITE = f'_Source: {DATA_SOURCE}_'
SEC_URL = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=include&count=10'


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
        return f'{sign}{prefix}{abs_val/1e12:.2f}T'
    if abs_val >= 1e9:
        return f'{sign}{prefix}{abs_val/1e9:.2f}B'
    if abs_val >= 1e6:
        return f'{sign}{prefix}{abs_val/1e6:.1f}M'
    if abs_val >= 1e3:
        return f'{sign}{prefix}{abs_val/1e3:.1f}K'
    return f'{sign}{prefix}{abs_val:.{decimals}f}'


def _safe(val, fmt='.2f'):
    if val is None:
        return 'N/A'
    return f'{val:{fmt}}'


def _ratio_context(name, val, low_thresh, high_thresh, low_label, high_label):
    """Return a value with contextual label."""
    if val is None:
        return f'{name}: N/A'
    note = ''
    if val < low_thresh:
        note = f' ({low_label})'
    elif val > high_thresh:
        note = f' ({high_label})'
    return f'{name}: {val:.2f}{note}'


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
        self.bot.send_message(
            chat_id,
            f'📊 *{ticker} Deep Research Report*\n'
            f'Fetching company data, financials, prices, insider trades, and news...',
        )

        try:
            data = self._fetch_data(ticker)
        except Exception as e:
            logger.error(f"[Research] Data fetch failed for {ticker}: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'❌ Error fetching data for {ticker}: {e}')
            return

        # Phase 1: Company overview
        self._send_overview(chat_id, ticker, data)

        # Phase 2: Fundamentals (exhaustive metrics)
        self._send_fundamentals(chat_id, ticker, data)

        # Phase 3: Value assessment
        self._send_value(chat_id, ticker, data)

        # Phase 4: Momentum & technicals
        self._send_momentum(chat_id, ticker, data)

        # Phase 5: Financial statement detail + 8-quarter trends
        self._send_financials(chat_id, ticker, data)

        # Phase 6: Insider trading activity
        self._send_insider_trades(chat_id, ticker, data)

        # Phase 7: Recent news with citation URLs
        self._send_news(chat_id, ticker, data)

        # Phase 8: Chart
        self._send_chart(chat_id, ticker, data)

        # Phase 9: AI Multi-Agent Analysis
        self._send_ai_analysis(chat_id, ticker)

        # Phase 10: LLM Investment Summary
        self._send_llm_summary(chat_id, ticker, data)

    # ──────────────────────────────────────────────
    # Data fetching
    # ──────────────────────────────────────────────

    def _fetch_data(self, ticker):
        """Fetch all data from financialdatasets.ai API."""
        from vendor.ai_hedge_fund.tools.api import (
            get_prices, get_financial_metrics, search_line_items,
            get_market_cap, prices_to_df, get_insider_trades,
            get_company_news,
        )
        from vendor.ai_hedge_fund.data.models import CompanyFactsResponse

        end = date.today().isoformat()
        start_1y = (date.today() - timedelta(days=365)).isoformat()
        start_90d = (date.today() - timedelta(days=90)).isoformat()

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

        # Line items (8 quarters) — expanded list
        line_items = search_line_items(
            ticker,
            line_items=[
                'revenue', 'net_income', 'earnings_per_share',
                'total_assets', 'total_liabilities', 'total_debt',
                'free_cash_flow', 'operating_income', 'gross_profit',
                'research_and_development', 'outstanding_shares',
                'capital_expenditure', 'dividends_and_other_cash_distributions',
                'selling_general_and_administrative', 'total_equity',
                'cash_and_equivalents', 'operating_cash_flow',
            ],
            end_date=end, period='quarterly', limit=8,
        )

        # Market cap
        market_cap = get_market_cap(ticker, end)

        # Insider trades (last 90 days)
        insider_trades = []
        try:
            insider_trades = get_insider_trades(ticker, end, start_date=start_90d, limit=50)
        except Exception as e:
            logger.warning(f"[Research] Insider trades failed: {e}")

        # Company news (last 30 days)
        news = []
        try:
            start_30d = (date.today() - timedelta(days=30)).isoformat()
            news = get_company_news(ticker, end, start_date=start_30d, limit=10)
        except Exception as e:
            logger.warning(f"[Research] Company news failed: {e}")

        return {
            'company': company_facts,
            'prices': prices,
            'prices_df': prices_df,
            'metrics': metrics,
            'line_items': line_items,
            'market_cap': market_cap,
            'insider_trades': insider_trades,
            'news': news,
        }

    # ──────────────────────────────────────────────
    # Phase 1: Company Overview
    # ──────────────────────────────────────────────

    def _send_overview(self, chat_id, ticker, data):
        co = data['company']
        mc = data['market_cap']
        prices_df = data['prices_df']
        metrics = data['metrics']

        lines = [f'🏢 *{ticker} — Company Overview*', '']

        if co:
            if co.name:
                lines.append(f'*Company:* {co.name}')
            if co.sector:
                lines.append(f'*Sector:* {co.sector}')
            if co.industry:
                lines.append(f'*Industry:* {co.industry}')
            if co.sic_industry and co.sic_industry != co.industry:
                lines.append(f'*SIC Industry:* {co.sic_industry}')
            if co.exchange:
                lines.append(f'*Exchange:* {co.exchange}')
            if co.location:
                lines.append(f'*HQ:* {co.location}')
            if co.number_of_employees:
                lines.append(f'*Employees:* {co.number_of_employees:,}')
            if co.listing_date:
                lines.append(f'*Listed:* {co.listing_date}')
            if co.website_url:
                lines.append(f'*Website:* {co.website_url}')
            if co.cik:
                lines.append(f'*CIK:* {co.cik}')

        lines.append('')

        # Market cap & enterprise value
        if mc:
            lines.append(f'*Market Cap:* {_fmt(mc)}')
        if metrics and metrics[0].enterprise_value:
            lines.append(f'*Enterprise Value:* {_fmt(metrics[0].enterprise_value)}')
        if metrics and metrics[0].market_cap and mc:
            # Show EV/MC ratio
            ev = metrics[0].enterprise_value
            if ev and mc:
                lines.append(f'*EV/Market Cap:* {ev / mc:.2f}x')

        # Current price & 52-week range from price data
        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            high_col = prices_df['high']
            low_col = prices_df['low']
            current = close.iloc[-1]
            prev_close = close.iloc[-2] if len(close) >= 2 else current
            day_change = (current - prev_close) / prev_close if prev_close else 0
            day_emoji = '🟢' if day_change >= 0 else '🔴'

            high_52w = high_col.max()
            low_52w = low_col.min()
            pct_from_high = (current - high_52w) / high_52w
            pct_from_low = (current - low_52w) / low_52w if low_52w else 0

            latest_vol = prices_df['volume'].iloc[-1]
            avg_vol_30d = prices_df['volume'].iloc[-21:].mean() if len(prices_df) >= 21 else latest_vol

            lines.append('')
            lines.append(f'{day_emoji} *Current Price:* ${current:.2f} ({_pct(day_change)} today)')
            lines.append(f'*52-Week High:* ${high_52w:.2f} ({_pct(pct_from_high)} from high)')
            lines.append(f'*52-Week Low:* ${low_52w:.2f} ({_pct(pct_from_low)} from low)')
            lines.append(f'*Today Volume:* {_fmt(latest_vol, "", 0)}')
            lines.append(f'*Avg Volume (30d):* {_fmt(avg_vol_30d, "", 0)}')

            # Shares outstanding from line items
            if data['line_items']:
                shares = getattr(data['line_items'][0], 'outstanding_shares', None)
                if shares:
                    lines.append(f'*Shares Outstanding:* {_fmt(shares, "", 0)}')

        if not co and not mc:
            lines.append('No company data available.')

        lines.append('')
        lines.append(DATA_CITE)
        if co and co.sec_filings_url:
            lines.append(f'_SEC Filings: {co.sec_filings_url}_')

        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 2: Fundamentals (exhaustive)
    # ──────────────────────────────────────────────

    def _send_fundamentals(self, chat_id, ticker, data):
        metrics = data['metrics']
        if not metrics:
            self.bot.send_message(chat_id, f'📉 *{ticker} — Fundamentals*\nNo metrics data available.')
            return

        m = metrics[0]  # Latest quarter
        lines = [
            f'📊 *{ticker} — Fundamentals* ({m.report_period})',
            '',
            '*── Valuation Multiples ──*',
            _ratio_context('P/E Ratio', m.price_to_earnings_ratio, 15, 30, 'cheap', 'expensive'),
            _ratio_context('P/B Ratio', m.price_to_book_ratio, 1.0, 5.0, 'below book', 'premium'),
            _ratio_context('P/S Ratio', m.price_to_sales_ratio, 1.0, 10.0, 'low', 'high'),
            _ratio_context('EV/EBITDA', m.enterprise_value_to_ebitda_ratio, 10, 25, 'cheap', 'expensive'),
            f'EV/Revenue: {_safe(m.enterprise_value_to_revenue_ratio)}',
            _ratio_context('PEG Ratio', m.peg_ratio, 1.0, 2.0, 'growth undervalued', 'growth pricey'),
            f'FCF Yield: {_pct(m.free_cash_flow_yield)}',
            f'Earnings Yield: {_safe(1/m.price_to_earnings_ratio * 100 if m.price_to_earnings_ratio else None, ".2f")}%' if m.price_to_earnings_ratio else 'Earnings Yield: N/A',
            '',
            '*── Profitability ──*',
            f'Gross Margin: {_pct(m.gross_margin)}',
            f'Operating Margin: {_pct(m.operating_margin)}',
            f'Net Margin: {_pct(m.net_margin)}',
            f'ROE: {_pct(m.return_on_equity)}',
            f'ROA: {_pct(m.return_on_assets)}',
            f'ROIC: {_pct(m.return_on_invested_capital)}',
            f'Asset Turnover: {_safe(m.asset_turnover)}',
            '',
            '*── Growth (QoQ) ──*',
            f'Revenue Growth: {_pct(m.revenue_growth)}',
            f'Earnings Growth: {_pct(m.earnings_growth)}',
            f'EPS Growth: {_pct(m.earnings_per_share_growth)}',
            f'FCF Growth: {_pct(m.free_cash_flow_growth)}',
            f'Operating Income Growth: {_pct(m.operating_income_growth)}',
            f'EBITDA Growth: {_pct(m.ebitda_growth)}',
            f'Book Value Growth: {_pct(m.book_value_growth)}',
            '',
            '*── Per-Share Data ──*',
            f'EPS: ${_safe(m.earnings_per_share)}',
            f'Book Value/Share: {_fmt(m.book_value_per_share)}',
            f'FCF/Share: {_fmt(m.free_cash_flow_per_share)}',
            f'Payout Ratio: {_pct(m.payout_ratio)}',
            '',
            '*── Balance Sheet Health ──*',
            _ratio_context('Current Ratio', m.current_ratio, 1.0, 3.0, 'tight liquidity', 'strong liquidity'),
            _ratio_context('Quick Ratio', m.quick_ratio, 0.8, 2.0, 'low', 'strong'),
            f'Cash Ratio: {_safe(m.cash_ratio)}',
            f'Operating CF Ratio: {_safe(m.operating_cash_flow_ratio)}',
            _ratio_context('Debt/Equity', m.debt_to_equity, 0.3, 2.0, 'low leverage', 'high leverage'),
            f'Debt/Assets: {_safe(m.debt_to_assets)}',
            f'Interest Coverage: {_safe(m.interest_coverage, ".1f")}x' if m.interest_coverage else 'Interest Coverage: N/A',
            '',
            '*── Efficiency ──*',
            f'Inventory Turnover: {_safe(m.inventory_turnover, ".1f")}' if m.inventory_turnover else 'Inventory Turnover: N/A',
            f'Receivables Turnover: {_safe(m.receivables_turnover, ".1f")}' if m.receivables_turnover else 'Receivables Turnover: N/A',
            f'Days Sales Outstanding: {_safe(m.days_sales_outstanding, ".0f")} days' if m.days_sales_outstanding else 'Days Sales Outstanding: N/A',
            f'Operating Cycle: {_safe(m.operating_cycle, ".0f")} days' if m.operating_cycle else 'Operating Cycle: N/A',
            f'Working Capital Turnover: {_safe(m.working_capital_turnover, ".1f")}' if m.working_capital_turnover else 'Working Capital Turnover: N/A',
        ]

        # Historical comparison — show prior quarter + YoY
        if len(metrics) >= 2:
            pm = metrics[1]
            lines.append('')
            lines.append(f'*── vs Prior Quarter ({pm.report_period}) ──*')
            if m.price_to_earnings_ratio and pm.price_to_earnings_ratio:
                lines.append(f'P/E: {pm.price_to_earnings_ratio:.2f} → {m.price_to_earnings_ratio:.2f}')
            if m.net_margin is not None and pm.net_margin is not None:
                lines.append(f'Net Margin: {_pct(pm.net_margin)} → {_pct(m.net_margin)}')
            if m.return_on_equity is not None and pm.return_on_equity is not None:
                lines.append(f'ROE: {_pct(pm.return_on_equity)} → {_pct(m.return_on_equity)}')

        if len(metrics) >= 5:
            ym = metrics[4]
            lines.append('')
            lines.append(f'*── vs Year Ago ({ym.report_period}) ──*')
            if m.price_to_earnings_ratio and ym.price_to_earnings_ratio:
                lines.append(f'P/E: {ym.price_to_earnings_ratio:.2f} → {m.price_to_earnings_ratio:.2f}')
            if m.net_margin is not None and ym.net_margin is not None:
                lines.append(f'Net Margin: {_pct(ym.net_margin)} → {_pct(m.net_margin)}')
            if m.return_on_equity is not None and ym.return_on_equity is not None:
                lines.append(f'ROE: {_pct(ym.return_on_equity)} → {_pct(m.return_on_equity)}')
            if m.revenue_growth is not None and ym.revenue_growth is not None:
                lines.append(f'Rev Growth: {_pct(ym.revenue_growth)} → {_pct(m.revenue_growth)}')

        lines.append('')
        lines.append(DATA_CITE)
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 3: Value Assessment
    # ──────────────────────────────────────────────

    def _send_value(self, chat_id, ticker, data):
        metrics = data['metrics']
        if not metrics:
            return

        m = metrics[0]
        lines = [f'💰 *{ticker} — Value Assessment* ({m.report_period})', '']

        pe = m.price_to_earnings_ratio
        pb = m.price_to_book_ratio
        ps = m.price_to_sales_ratio
        fcf_yield = m.free_cash_flow_yield
        peg = m.peg_ratio
        ev_rev = m.enterprise_value_to_revenue_ratio
        ev_ebitda = m.enterprise_value_to_ebitda_ratio

        lines.append('*Key Valuation Ratios:*')
        lines.append(f'  P/E: {_safe(pe)} (S&P 500 avg ~22)')
        lines.append(f'  P/B: {_safe(pb)} (below 1.0 = trading under book value)')
        lines.append(f'  P/S: {_safe(ps)}')
        lines.append(f'  EV/Revenue: {_safe(ev_rev)}')
        lines.append(f'  EV/EBITDA: {_safe(ev_ebitda)} (below 10 = potentially cheap)')
        lines.append(f'  PEG Ratio: {_safe(peg)} (below 1.0 = growth undervalued)')
        lines.append(f'  FCF Yield: {_pct(fcf_yield)} (above 5% = strong cash generation)')
        lines.append(f'  Book Value/Share: {_fmt(m.book_value_per_share)}')
        lines.append(f'  FCF/Share: {_fmt(m.free_cash_flow_per_share)}')
        if pe:
            earnings_yield = 1.0 / pe * 100
            lines.append(f'  Earnings Yield: {earnings_yield:.2f}% (inverse P/E; compare to bond yields)')
        lines.append('')

        # Rule-based valuation signals
        bull_signals = []
        bear_signals = []

        if pe is not None:
            if pe < 15:
                bull_signals.append(f'P/E of {pe:.1f} is below 15 — historically cheap territory')
            elif pe < 22:
                bull_signals.append(f'P/E of {pe:.1f} is below S&P 500 average (~22)')
            elif pe > 35:
                bear_signals.append(f'P/E of {pe:.1f} is well above market average — priced for perfection')
            elif pe > 25:
                bear_signals.append(f'P/E of {pe:.1f} is above market average of ~22')

        if peg is not None:
            if peg < 0.8:
                bull_signals.append(f'PEG of {peg:.2f} suggests growth is significantly undervalued')
            elif peg < 1.0:
                bull_signals.append(f'PEG of {peg:.2f} < 1.0 — growth undervalued relative to earnings expansion')
            elif peg > 2.5:
                bear_signals.append(f'PEG of {peg:.2f} > 2.5 — paying a steep premium for growth')

        if fcf_yield is not None:
            if fcf_yield > 0.08:
                bull_signals.append(f'FCF yield of {fcf_yield*100:.1f}% is exceptional — strong cash machine')
            elif fcf_yield > 0.05:
                bull_signals.append(f'FCF yield of {fcf_yield*100:.1f}% is healthy — company generates solid free cash')
            elif fcf_yield < 0.01:
                bear_signals.append(f'FCF yield of {fcf_yield*100:.1f}% is very weak — limited free cash generation')

        if pb is not None:
            if pb < 1.0:
                bull_signals.append(f'P/B of {pb:.2f} — stock trading below book value (potential deep value)')
            elif pb > 10:
                bear_signals.append(f'P/B of {pb:.2f} — extreme premium to tangible assets')

        if ev_ebitda is not None:
            if ev_ebitda < 8:
                bull_signals.append(f'EV/EBITDA of {ev_ebitda:.1f} is low — potentially undervalued')
            elif ev_ebitda > 30:
                bear_signals.append(f'EV/EBITDA of {ev_ebitda:.1f} is very high — rich valuation')

        if m.debt_to_equity is not None and m.debt_to_equity > 2.0:
            bear_signals.append(f'D/E of {m.debt_to_equity:.2f} — high leverage increases risk')

        if bull_signals:
            lines.append('🟢 *Bullish Value Signals:*')
            for s in bull_signals:
                lines.append(f'  • {s}')
        if bear_signals:
            lines.append('🔴 *Bearish Value Signals:*')
            for s in bear_signals:
                lines.append(f'  • {s}')

        if not bull_signals and not bear_signals:
            lines.append('🟡 Valuation appears neutral relative to common benchmarks.')

        # Overall verdict
        lines.append('')
        bull_count = len(bull_signals)
        bear_count = len(bear_signals)
        if bull_count > bear_count + 1:
            lines.append('*Verdict:* 🟢 Stock appears undervalued based on multiple metrics')
        elif bear_count > bull_count + 1:
            lines.append('*Verdict:* 🔴 Stock appears overvalued — caution warranted')
        else:
            lines.append('*Verdict:* 🟡 Mixed signals — valuation is roughly fair')

        lines.append('')
        lines.append(DATA_CITE)
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 4: Momentum & Technicals
    # ──────────────────────────────────────────────

    def _send_momentum(self, chat_id, ticker, data):
        prices_df = data['prices_df']
        metrics = data['metrics']
        lines = [f'📈 *{ticker} — Momentum & Technicals*', '']

        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            high_col = prices_df['high']
            low_col = prices_df['low']
            current = close.iloc[-1]

            lines.append(f'*Current Price:* ${current:.2f}')
            lines.append(f'*Last Close Date:* {prices_df.index[-1].strftime("%Y-%m-%d")}')
            lines.append('')

            # Price changes
            def _change(days):
                if len(close) > days:
                    old = close.iloc[-days - 1]
                    return (current - old) / old, old
                return None, None

            lines.append('*Price Performance:*')
            for label, days in [('1W', 5), ('1M', 21), ('3M', 63), ('6M', 126), ('1Y', 252)]:
                chg, old = _change(days)
                if chg is not None:
                    emoji = '🟢' if chg >= 0 else '🔴'
                    lines.append(f'  {emoji} {label}: {_pct(chg)} (${old:.2f} → ${current:.2f})')

            lines.append('')

            # 52-week statistics
            high_52w = high_col.max()
            low_52w = low_col.min()
            range_52w = high_52w - low_52w
            position_in_range = (current - low_52w) / range_52w if range_52w > 0 else 0.5
            lines.append('*52-Week Statistics:*')
            lines.append(f'  High: ${high_52w:.2f}')
            lines.append(f'  Low: ${low_52w:.2f}')
            lines.append(f'  Range: ${range_52w:.2f}')
            lines.append(f'  Position in range: {position_in_range*100:.0f}% (0%=at low, 100%=at high)')
            lines.append('')

            # Moving averages
            lines.append('*Moving Averages:*')
            for window, name in [(10, '10-day SMA'), (20, '20-day SMA'), (50, '50-day SMA'), (200, '200-day SMA')]:
                if len(close) >= window:
                    sma = close.rolling(window).mean().iloc[-1]
                    pct_diff = (current - sma) / sma
                    pos = '↑ above' if current > sma else '↓ below'
                    lines.append(f'  {name}: ${sma:.2f} — price {pos} by {abs(pct_diff)*100:.1f}%')

            # Golden/death cross check
            if len(close) >= 200:
                sma50 = close.rolling(50).mean().iloc[-1]
                sma200 = close.rolling(200).mean().iloc[-1]
                if sma50 > sma200:
                    lines.append('  ✅ *Golden Cross* — 50 SMA above 200 SMA (bullish)')
                else:
                    lines.append('  ⚠️ *Death Cross* — 50 SMA below 200 SMA (bearish)')
            lines.append('')

            # RSI-14
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                if rsi > 70:
                    rsi_label = '⚠️ OVERBOUGHT — potential pullback risk'
                elif rsi > 60:
                    rsi_label = 'bullish momentum'
                elif rsi < 30:
                    rsi_label = '⚠️ OVERSOLD — potential bounce candidate'
                elif rsi < 40:
                    rsi_label = 'bearish momentum'
                else:
                    rsi_label = 'neutral zone'
                lines.append(f'*RSI-14:* {rsi:.1f} — {rsi_label}')

            # MACD approximation (12/26 EMA)
            if len(close) >= 26:
                ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
                ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
                macd_line = ema12 - ema26
                macd_signal = 'bullish' if macd_line > 0 else 'bearish'
                lines.append(f'*MACD:* {macd_line:.2f} ({macd_signal} — EMA12-EMA26)')

            lines.append('')

            # Volume analysis
            lines.append('*Volume Analysis:*')
            vol = prices_df['volume']
            today_vol = vol.iloc[-1]
            lines.append(f'  Today: {_fmt(today_vol, "", 0)}')
            for window, name in [(5, '5-day avg'), (20, '20-day avg'), (60, '60-day avg')]:
                if len(vol) >= window:
                    avg = vol.iloc[-window:].mean()
                    ratio = today_vol / avg if avg > 0 else 1
                    lines.append(f'  {name}: {_fmt(avg, "", 0)} (today {ratio:.1f}x)')

            if len(vol) >= 60:
                avg_20 = vol.iloc[-20:].mean()
                avg_60 = vol.iloc[-60:].mean()
                ratio = avg_20 / avg_60 if avg_60 > 0 else 1
                if ratio > 1.3:
                    lines.append('  📊 Volume surging — 20d avg 30%+ above 60d avg')
                elif ratio > 1.1:
                    lines.append('  📊 Volume increasing — accumulation signal')
                elif ratio < 0.7:
                    lines.append('  📊 Volume declining — low interest / distribution')

            # Volatility
            if len(close) >= 21:
                daily_returns = close.pct_change().dropna()
                vol_20d = daily_returns.iloc[-20:].std() * (252 ** 0.5)
                lines.append(f'\n*Annualized Volatility (20d):* {vol_20d*100:.1f}%')
                if vol_20d > 0.6:
                    lines.append('  ⚠️ Very high volatility')
                elif vol_20d > 0.35:
                    lines.append('  ⚠️ Elevated volatility')

        else:
            lines.append('No price data available.')

        # EPS trajectory
        if metrics and len(metrics) >= 4:
            lines.append('')
            lines.append('*EPS Trajectory (last 8Q):*')
            for m in reversed(metrics[:min(8, len(metrics))]):
                if m.earnings_per_share is not None:
                    growth_note = ''
                    if m.earnings_per_share_growth is not None:
                        growth_note = f' ({_pct(m.earnings_per_share_growth)} growth)'
                    lines.append(f'  {m.report_period}: ${m.earnings_per_share:.2f}{growth_note}')

        lines.append('')
        lines.append(DATA_CITE)
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 5: Financial Statements (verbose)
    # ──────────────────────────────────────────────

    def _send_financials(self, chat_id, ticker, data):
        line_items = data['line_items']
        if not line_items:
            self.bot.send_message(chat_id, f'📋 *{ticker} — Financials*\nNo financial statement data.')
            return

        latest = line_items[0]

        # Helper to safely get line item attrs
        def _g(item, attr):
            return getattr(item, attr, None)

        rev = _g(latest, 'revenue')
        ni = _g(latest, 'net_income')
        eps = _g(latest, 'earnings_per_share')
        ta = _g(latest, 'total_assets')
        tl = _g(latest, 'total_liabilities')
        td = _g(latest, 'total_debt')
        fcf = _g(latest, 'free_cash_flow')
        oi = _g(latest, 'operating_income')
        gp = _g(latest, 'gross_profit')
        rnd = _g(latest, 'research_and_development')
        sga = _g(latest, 'selling_general_and_administrative')
        capex = _g(latest, 'capital_expenditure')
        te = _g(latest, 'total_equity')
        cash = _g(latest, 'cash_and_equivalents')
        ocf = _g(latest, 'operating_cash_flow')
        divs = _g(latest, 'dividends_and_other_cash_distributions')
        shares = _g(latest, 'outstanding_shares')

        lines = [
            f'📋 *{ticker} — Financial Statements* ({latest.report_period})',
            '',
            '*── Income Statement ──*',
            f'Revenue: {_fmt(rev)}',
            f'Gross Profit: {_fmt(gp)}',
        ]

        if rev and gp:
            lines.append(f'Gross Margin: {gp/rev*100:.1f}%')
        lines.append(f'Operating Income: {_fmt(oi)}')
        if rev and oi:
            lines.append(f'Operating Margin: {oi/rev*100:.1f}%')
        lines.append(f'Net Income: {_fmt(ni)}')
        if rev and ni:
            lines.append(f'Net Margin: {ni/rev*100:.1f}%')
        lines.append(f'EPS: {_fmt(eps, "$", 2)}')

        lines.append('')
        lines.append('*── Cost Breakdown ──*')
        if rev and gp:
            cogs = rev - gp
            lines.append(f'COGS: {_fmt(cogs)} ({cogs/rev*100:.1f}% of revenue)')
        if rnd:
            rnd_pct = f' ({rnd/rev*100:.1f}% of revenue)' if rev else ''
            lines.append(f'R&D: {_fmt(rnd)}{rnd_pct}')
        if sga:
            sga_pct = f' ({sga/rev*100:.1f}% of revenue)' if rev else ''
            lines.append(f'SG&A: {_fmt(sga)}{sga_pct}')

        lines.append('')
        lines.append('*── Balance Sheet ──*')
        lines.append(f'Total Assets: {_fmt(ta)}')
        lines.append(f'Total Liabilities: {_fmt(tl)}')
        lines.append(f'Total Equity: {_fmt(te)}')
        lines.append(f'Total Debt: {_fmt(td)}')
        lines.append(f'Cash & Equivalents: {_fmt(cash)}')
        if shares:
            lines.append(f'Shares Outstanding: {_fmt(shares, "", 0)}')
        if ta and tl:
            lines.append(f'Debt/Assets: {tl/ta*100:.1f}%')
        if td and te and te > 0:
            lines.append(f'Debt/Equity: {td/te:.2f}x')
        if cash and td:
            net_debt = td - cash
            lines.append(f'Net Debt: {_fmt(net_debt)} {"(net cash position)" if net_debt < 0 else ""}')

        lines.append('')
        lines.append('*── Cash Flow ──*')
        lines.append(f'Operating Cash Flow: {_fmt(ocf)}')
        lines.append(f'Capital Expenditure: {_fmt(capex)}')
        lines.append(f'Free Cash Flow: {_fmt(fcf)}')
        if divs:
            lines.append(f'Dividends Paid: {_fmt(divs)}')
        if ocf and capex:
            lines.append(f'CapEx/OCF: {abs(capex)/ocf*100:.1f}%' if ocf > 0 else 'CapEx/OCF: N/A')
        if fcf and rev:
            lines.append(f'FCF Margin: {fcf/rev*100:.1f}%')

        # QoQ comparison
        if len(line_items) >= 2:
            prev = line_items[1]
            lines.append('')
            lines.append(f'*── vs Prior Quarter ({prev.report_period}) ──*')
            prev_rev = _g(prev, 'revenue')
            prev_ni = _g(prev, 'net_income')
            prev_eps = _g(prev, 'earnings_per_share')
            prev_fcf = _g(prev, 'free_cash_flow')
            prev_oi = _g(prev, 'operating_income')
            if rev and prev_rev and prev_rev != 0:
                lines.append(f'Revenue: {_fmt(prev_rev)} → {_fmt(rev)} ({_pct((rev - prev_rev) / abs(prev_rev))})')
            if ni and prev_ni and prev_ni != 0:
                lines.append(f'Net Income: {_fmt(prev_ni)} → {_fmt(ni)} ({_pct((ni - prev_ni) / abs(prev_ni))})')
            if eps and prev_eps and prev_eps != 0:
                lines.append(f'EPS: ${prev_eps:.2f} → ${eps:.2f} ({_pct((eps - prev_eps) / abs(prev_eps))})')
            if oi and prev_oi and prev_oi != 0:
                lines.append(f'Op Income: {_fmt(prev_oi)} → {_fmt(oi)} ({_pct((oi - prev_oi) / abs(prev_oi))})')
            if fcf and prev_fcf and prev_fcf != 0:
                lines.append(f'FCF: {_fmt(prev_fcf)} → {_fmt(fcf)} ({_pct((fcf - prev_fcf) / abs(prev_fcf))})')

        # YoY comparison
        if len(line_items) >= 5:
            yoy = line_items[4]
            yoy_rev = _g(yoy, 'revenue')
            yoy_ni = _g(yoy, 'net_income')
            yoy_eps = _g(yoy, 'earnings_per_share')
            yoy_fcf = _g(yoy, 'free_cash_flow')
            lines.append('')
            lines.append(f'*── vs Year Ago ({yoy.report_period}) ──*')
            if rev and yoy_rev and yoy_rev != 0:
                lines.append(f'Revenue: {_fmt(yoy_rev)} → {_fmt(rev)} ({_pct((rev - yoy_rev) / abs(yoy_rev))})')
            if ni and yoy_ni and yoy_ni != 0:
                lines.append(f'Net Income: {_fmt(yoy_ni)} → {_fmt(ni)} ({_pct((ni - yoy_ni) / abs(yoy_ni))})')
            if eps and yoy_eps and yoy_eps != 0:
                lines.append(f'EPS: ${yoy_eps:.2f} → ${eps:.2f} ({_pct((eps - yoy_eps) / abs(yoy_eps))})')
            if fcf and yoy_fcf and yoy_fcf != 0:
                lines.append(f'FCF: {_fmt(yoy_fcf)} → {_fmt(fcf)} ({_pct((fcf - yoy_fcf) / abs(yoy_fcf))})')

        lines.append('')
        lines.append(DATA_CITE)
        self.bot.send_message(chat_id, '\n'.join(lines))

        # Send 8-quarter trends as a separate message (can be long)
        self._send_quarterly_trends(chat_id, ticker, line_items)

    def _send_quarterly_trends(self, chat_id, ticker, line_items):
        """Send 8-quarter revenue, EPS, and FCF trends."""
        if len(line_items) < 3:
            return

        lines = [f'📅 *{ticker} — Quarterly Trends*', '']

        # Revenue trend
        lines.append('*Revenue by Quarter:*')
        for item in reversed(line_items):
            rev = getattr(item, 'revenue', None)
            if rev is not None:
                lines.append(f'  {item.report_period}: {_fmt(rev)}')

        # EPS trend
        lines.append('')
        lines.append('*EPS by Quarter:*')
        for item in reversed(line_items):
            eps = getattr(item, 'earnings_per_share', None)
            if eps is not None:
                lines.append(f'  {item.report_period}: ${eps:.2f}')

        # FCF trend
        lines.append('')
        lines.append('*Free Cash Flow by Quarter:*')
        for item in reversed(line_items):
            fcf = getattr(item, 'free_cash_flow', None)
            if fcf is not None:
                lines.append(f'  {item.report_period}: {_fmt(fcf)}')

        # Net income trend
        lines.append('')
        lines.append('*Net Income by Quarter:*')
        for item in reversed(line_items):
            ni = getattr(item, 'net_income', None)
            if ni is not None:
                lines.append(f'  {item.report_period}: {_fmt(ni)}')

        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 6: Insider Trading
    # ──────────────────────────────────────────────

    def _send_insider_trades(self, chat_id, ticker, data):
        trades = data.get('insider_trades', [])
        if not trades:
            self.bot.send_message(
                chat_id,
                f'👔 *{ticker} — Insider Trading (90d)*\nNo insider trades found in the last 90 days.',
            )
            return

        # Aggregate stats
        total_buys = 0
        total_sells = 0
        buy_value = 0.0
        sell_value = 0.0
        buy_shares = 0
        sell_shares = 0

        for t in trades:
            shares = abs(t.transaction_shares or 0)
            value = abs(t.transaction_value or 0)
            if t.transaction_shares and t.transaction_shares > 0:
                total_buys += 1
                buy_value += value
                buy_shares += shares
            elif t.transaction_shares and t.transaction_shares < 0:
                total_sells += 1
                sell_value += value
                sell_shares += shares

        lines = [
            f'👔 *{ticker} — Insider Trading (Last 90 Days)*',
            '',
            f'*Total Transactions:* {len(trades)}',
            f'*Buys:* {total_buys} transactions ({_fmt(buy_value)} total, {_fmt(buy_shares, "", 0)} shares)',
            f'*Sells:* {total_sells} transactions ({_fmt(sell_value)} total, {_fmt(sell_shares, "", 0)} shares)',
            f'*Net:* {"🟢 Net buying" if buy_value > sell_value else "🔴 Net selling" if sell_value > buy_value else "🟡 Balanced"}',
            '',
        ]

        if buy_value + sell_value > 0:
            buy_pct = buy_value / (buy_value + sell_value) * 100
            lines.append(f'Buy/Sell ratio by value: {buy_pct:.0f}%/{100-buy_pct:.0f}%')
            lines.append('')

        # Show most recent 8 trades with details
        lines.append('*Recent Transactions:*')
        for t in trades[:8]:
            name = t.name or 'Unknown'
            title = f' ({t.title})' if t.title else ''
            director = ' [Board]' if t.is_board_director else ''
            shares = t.transaction_shares or 0
            action = '🟢 BUY' if shares > 0 else '🔴 SELL'
            price = f' @ ${t.transaction_price_per_share:.2f}' if t.transaction_price_per_share else ''
            value = f' (${_fmt(abs(t.transaction_value))})' if t.transaction_value else ''

            lines.append(f'  {action} {name}{title}{director}')
            lines.append(f'    {abs(shares):,.0f} shares{price}{value}')
            lines.append(f'    Filed: {t.filing_date[:10]}')
            if t.shares_owned_after_transaction:
                lines.append(f'    Remaining: {t.shares_owned_after_transaction:,.0f} shares')
            lines.append('')

        lines.append(DATA_CITE)
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 7: Recent News
    # ──────────────────────────────────────────────

    def _send_news(self, chat_id, ticker, data):
        news = data.get('news', [])
        if not news:
            self.bot.send_message(
                chat_id,
                f'📰 *{ticker} — Recent News*\nNo news articles found in the last 30 days.',
            )
            return

        # Sentiment summary
        sentiments = [n.sentiment for n in news if n.sentiment]
        bullish_count = sum(1 for s in sentiments if s.lower() in ('positive', 'bullish'))
        bearish_count = sum(1 for s in sentiments if s.lower() in ('negative', 'bearish'))
        neutral_count = len(sentiments) - bullish_count - bearish_count

        lines = [
            f'📰 *{ticker} — Recent News (Last 30 Days)*',
            '',
            f'*Articles Found:* {len(news)}',
        ]
        if sentiments:
            lines.append(f'*Sentiment:* 🟢 {bullish_count} positive | 🔴 {bearish_count} negative | 🟡 {neutral_count} neutral')
        lines.append('')

        # Show each article
        for i, article in enumerate(news[:10], 1):
            sentiment_emoji = '🟢' if article.sentiment and article.sentiment.lower() in ('positive', 'bullish') \
                else '🔴' if article.sentiment and article.sentiment.lower() in ('negative', 'bearish') \
                else '🟡'
            date_str = article.date[:10] if article.date else ''
            source = article.source or ''

            lines.append(f'{i}. {sentiment_emoji} *{article.title}*')
            lines.append(f'   {source} | {date_str}')
            if article.url:
                lines.append(f'   🔗 {article.url}')
            lines.append('')

        lines.append(DATA_CITE)
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 8: Chart
    # ──────────────────────────────────────────────

    def _send_chart(self, chat_id, ticker, data):
        try:
            from app.services.stock_chart_service import generate_research_chart
            chart_path = generate_research_chart(
                ticker, data['prices_df'], data['metrics'], data['line_items'],
            )
            self.bot.send_photo(chat_id, chart_path, caption=f'{ticker} — Research Chart (1Y price, revenue, EPS, P/E)')
            os.unlink(chart_path)
        except Exception as e:
            logger.error(f"[Research] Chart generation failed: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'⚠️ Chart generation failed: {e}')

    # ──────────────────────────────────────────────
    # Phase 9: AI Multi-Agent Analysis
    # ──────────────────────────────────────────────

    def _send_ai_analysis(self, chat_id, ticker):
        """Run hedge fund multi-agent analysis and send results."""
        self.bot.send_message(chat_id, f'🤖 *{ticker} — AI Analysis*\nRunning 4 analyst agents (fundamentals, technicals, valuation, sentiment)...')

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
                    self.bot.send_message(chat_id, '⚠️ AI analysis returned no results.')
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
                            points = []
                            for k, v in reasoning.items():
                                if isinstance(v, str) and v:
                                    points.append(f'• _{k}_: {v[:300]}')
                                elif isinstance(v, (int, float)):
                                    points.append(f'• _{k}_: {v}')
                                elif isinstance(v, list) and v:
                                    points.append(f'• _{k}_: {", ".join(str(x)[:100] for x in v[:5])}')
                            if points:
                                msg += '\n' + '\n'.join(points[:8])
                        elif isinstance(reasoning, str):
                            msg += f'\n{reasoning[:800]}'
                    self.bot.send_message(chat_id, msg)

                # Consensus
                consensus = analysis.consensus_signal or 'neutral'
                conf = analysis.consensus_confidence or 0
                emoji = '🟢' if consensus == 'bullish' else '🔴' if consensus == 'bearish' else '🟡'
                self.bot.send_message(
                    chat_id,
                    f'\n*{ticker} — AI Consensus*\n'
                    f'{emoji} Signal: *{consensus.upper()}* ({conf}% confidence)\n\n'
                    f'_Based on 4 independent AI analyst agents analyzing fundamentals, technicals, valuation, and sentiment._',
                )

        except Exception as e:
            logger.error(f"[Research] AI analysis failed for {ticker}: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'⚠️ AI analysis failed: {e}')

    # ──────────────────────────────────────────────
    # Phase 10: LLM Investment Summary
    # ──────────────────────────────────────────────

    def _send_llm_summary(self, chat_id, ticker, data):
        """Generate a comprehensive LLM-powered investment summary."""
        self.bot.send_message(chat_id, f'🧠 *{ticker} — Generating AI Investment Summary...*')

        try:
            from app.integrations.llm_gateway import LLMGateway

            with self.app.app_context():
                llm = LLMGateway()

                # Build a data package for the LLM
                data_summary = self._build_data_summary(ticker, data)

                system_prompt = """You are a senior equity research analyst writing an investment brief.
You produce concise, data-driven analysis. Always cite specific numbers.
Use Telegram Markdown formatting (*bold*, _italic_).
Structure your response with clear sections."""

                user_prompt = f"""Write a comprehensive investment summary for {ticker} based on this data:

{data_summary}

Structure your response as:

1. *Executive Summary* (3-4 sentences with key numbers — price, market cap, P/E, revenue growth, margin trend)

2. *Bull Case* (3-4 bullet points with specific numbers supporting a buy thesis)

3. *Bear Case* (3-4 bullet points with specific numbers supporting caution)

4. *Future Outlook* (3-4 sentences on what to expect next quarter and next 12 months based on current trends — revenue trajectory, margin expansion/compression, catalysts, headwinds)

5. *Key Metrics to Watch* (3-4 metrics that will determine near-term direction, with current values and thresholds)

6. *Verdict* (1-2 sentences with clear directional bias and confidence level)

Be specific with numbers throughout. Do not hedge excessively — take a clear analytical stance.
Keep the total response under 2500 characters for Telegram readability."""

                result = llm.call(
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    purpose=f'research.summary.{ticker}',
                    section='research',
                    max_tokens=1500,
                )

                summary = result['content'].strip()
                model = result.get('model', 'unknown')
                cost = result.get('cost_usd', 0)

                msg = f'🧠 *{ticker} — AI Investment Summary*\n\n{summary}'
                msg += f'\n\n_Generated by {model} | Cost: ${cost:.4f}_'
                msg += f'\n_Data source: {DATA_SOURCE}_'

                self.bot.send_message(chat_id, msg)

        except Exception as e:
            logger.error(f"[Research] LLM summary failed for {ticker}: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'⚠️ LLM summary generation failed: {e}')

    def _build_data_summary(self, ticker, data):
        """Build a compact text summary of all fetched data for LLM consumption."""
        parts = []

        # Company info
        co = data.get('company')
        if co:
            parts.append(f"Company: {co.name or ticker}")
            parts.append(f"Sector: {co.sector or 'N/A'} | Industry: {co.industry or 'N/A'}")
            if co.number_of_employees:
                parts.append(f"Employees: {co.number_of_employees:,}")

        mc = data.get('market_cap')
        if mc:
            parts.append(f"Market Cap: {_fmt(mc)}")

        # Price data
        prices_df = data.get('prices_df')
        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            current = close.iloc[-1]
            high_52w = prices_df['high'].max()
            low_52w = prices_df['low'].min()
            parts.append(f"\nPrice: ${current:.2f} | 52w High: ${high_52w:.2f} | 52w Low: ${low_52w:.2f}")

            # Returns
            for label, days in [('1M', 21), ('3M', 63), ('6M', 126), ('1Y', 252)]:
                if len(close) > days:
                    old = close.iloc[-days - 1]
                    chg = (current - old) / old
                    parts.append(f"{label} return: {_pct(chg)}")

            # SMAs
            if len(close) >= 50:
                sma50 = close.rolling(50).mean().iloc[-1]
                parts.append(f"50 SMA: ${sma50:.2f} ({'above' if current > sma50 else 'below'})")
            if len(close) >= 200:
                sma200 = close.rolling(200).mean().iloc[-1]
                parts.append(f"200 SMA: ${sma200:.2f} ({'above' if current > sma200 else 'below'})")

            # RSI
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
                rsi = 100 - (100 / (1 + rs))
                parts.append(f"RSI-14: {rsi:.1f}")

        # Metrics
        metrics = data.get('metrics', [])
        if metrics:
            m = metrics[0]
            parts.append(f"\nFundamentals ({m.report_period}):")
            for attr, label in [
                ('price_to_earnings_ratio', 'P/E'),
                ('price_to_book_ratio', 'P/B'),
                ('price_to_sales_ratio', 'P/S'),
                ('enterprise_value_to_ebitda_ratio', 'EV/EBITDA'),
                ('peg_ratio', 'PEG'),
                ('free_cash_flow_yield', 'FCF Yield'),
                ('gross_margin', 'Gross Margin'),
                ('operating_margin', 'Op Margin'),
                ('net_margin', 'Net Margin'),
                ('return_on_equity', 'ROE'),
                ('return_on_assets', 'ROA'),
                ('return_on_invested_capital', 'ROIC'),
                ('revenue_growth', 'Revenue Growth'),
                ('earnings_per_share_growth', 'EPS Growth'),
                ('free_cash_flow_growth', 'FCF Growth'),
                ('current_ratio', 'Current Ratio'),
                ('debt_to_equity', 'D/E'),
                ('interest_coverage', 'Interest Coverage'),
                ('earnings_per_share', 'EPS'),
                ('book_value_per_share', 'BVPS'),
                ('free_cash_flow_per_share', 'FCFPS'),
            ]:
                val = getattr(m, attr, None)
                if val is not None:
                    if 'margin' in attr or 'growth' in attr or 'yield' in attr or 'return' in attr:
                        parts.append(f"  {label}: {val*100:.1f}%")
                    else:
                        parts.append(f"  {label}: {val:.2f}")

        # Line items
        line_items = data.get('line_items', [])
        if line_items:
            li = line_items[0]
            parts.append(f"\nFinancials ({li.report_period}):")
            for attr, label in [
                ('revenue', 'Revenue'),
                ('net_income', 'Net Income'),
                ('operating_income', 'Op Income'),
                ('gross_profit', 'Gross Profit'),
                ('free_cash_flow', 'FCF'),
                ('total_assets', 'Total Assets'),
                ('total_liabilities', 'Total Liabilities'),
                ('total_debt', 'Total Debt'),
                ('cash_and_equivalents', 'Cash'),
            ]:
                val = getattr(li, attr, None)
                if val is not None:
                    parts.append(f"  {label}: {_fmt(val)}")

        # Insider trades summary
        trades = data.get('insider_trades', [])
        if trades:
            buys = sum(1 for t in trades if t.transaction_shares and t.transaction_shares > 0)
            sells = sum(1 for t in trades if t.transaction_shares and t.transaction_shares < 0)
            parts.append(f"\nInsider Trading (90d): {buys} buys, {sells} sells")

        # News sentiment
        news = data.get('news', [])
        if news:
            sentiments = [n.sentiment for n in news if n.sentiment]
            pos = sum(1 for s in sentiments if s.lower() in ('positive', 'bullish'))
            neg = sum(1 for s in sentiments if s.lower() in ('negative', 'bearish'))
            parts.append(f"\nNews (30d): {len(news)} articles, {pos} positive, {neg} negative")
            # Include top 3 headlines
            for n in news[:3]:
                parts.append(f"  - {n.title} ({n.source}, {n.date[:10]})")

        return '\n'.join(parts)
