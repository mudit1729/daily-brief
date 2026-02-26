"""
Stock research service — orchestrates in-depth stock analysis
for the Telegram /research command. Sends progressive updates
with verbose data, numbers, and source citations.

Supports two data modes:
  - API mode: fetches live data from financialdatasets.ai
  - LLM mode: uses a single gpt-4.1-nano call when API is unavailable
"""
import json
import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

DATA_SOURCE_API = 'financialdatasets.ai'
DATA_SOURCE_LLM = 'LLM-estimated (gpt-4.1-nano — not real-time)'
LLM_DATA_MODEL = 'gpt-4.1-nano'


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

    def _data_cite(self, data):
        if data.get('_source') == 'llm':
            return f'_Source: {DATA_SOURCE_LLM}_'
        return f'_Source: {DATA_SOURCE_API}_'

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

        # Notify if using LLM-estimated data
        if data.get('_source') == 'llm':
            self.bot.send_message(
                chat_id,
                f'⚠️ *Note:* Live market data API unavailable. '
                f'Using LLM-estimated data ({LLM_DATA_MODEL}). '
                f'Figures are approximate and based on training data, not real-time.',
            )

        # Phase 1: Company overview
        self._send_overview(chat_id, ticker, data)

        # Phase 2: Fundamentals (exhaustive metrics)
        self._send_fundamentals(chat_id, ticker, data)

        # Phase 3: Value assessment
        self._send_value(chat_id, ticker, data)

        # Phase 4: Momentum & technicals
        self._send_momentum(chat_id, ticker, data)

        # Phase 5: Financial statement detail + quarterly trends
        self._send_financials(chat_id, ticker, data)

        # Phase 6: Insider trading activity
        self._send_insider_trades(chat_id, ticker, data)

        # Phase 7: Recent news with citation URLs
        self._send_news(chat_id, ticker, data)

        # Phase 8: Chart
        self._send_chart(chat_id, ticker, data)

        # Phase 9: AI Multi-Agent Analysis (skip in LLM mode — agents need live API)
        if data.get('_source') == 'llm':
            self.bot.send_message(
                chat_id,
                f'🤖 *{ticker} — AI Multi-Agent Analysis*\n'
                f'_Skipped: requires live market data API. '
                f'Multi-perspective analysis included in AI Summary below._',
            )
        else:
            self._send_ai_analysis(chat_id, ticker)

        # Phase 10: LLM Investment Summary (enhanced in LLM mode)
        self._send_llm_summary(chat_id, ticker, data)

    # ──────────────────────────────────────────────
    # Data fetching — with LLM fallback
    # ──────────────────────────────────────────────

    def _fetch_data(self, ticker):
        """Fetch data from API, falling back to LLM if API returns nothing."""
        with self.app.app_context():
            use_llm = self.app.config.get('RESEARCH_USE_LLM_DATA', False)

        if use_llm:
            logger.info(f"[Research] RESEARCH_USE_LLM_DATA=true, using LLM for {ticker}")
            return self._fetch_data_via_llm(ticker)

        # Try API first
        data = self._fetch_data_from_api(ticker)

        # If API returned nothing useful, fall back to LLM
        if not data['company'] and not data['metrics'] and not data['line_items']:
            logger.warning(f"[Research] API returned no data for {ticker}, falling back to LLM")
            return self._fetch_data_via_llm(ticker)

        data['_source'] = 'api'
        return data

    def _fetch_data_from_api(self, ticker):
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

        prices = get_prices(ticker, start_1y, end)
        prices_df = prices_to_df(prices) if prices else None
        metrics = get_financial_metrics(ticker, end, period='quarterly', limit=8)
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
        market_cap = get_market_cap(ticker, end)

        insider_trades = []
        try:
            insider_trades = get_insider_trades(ticker, end, start_date=start_90d, limit=50)
        except Exception as e:
            logger.warning(f"[Research] Insider trades failed: {e}")

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
    # LLM-based data fetching (single cheap call)
    # ──────────────────────────────────────────────

    def _fetch_data_via_llm(self, ticker):
        """Fetch financial data via a single gpt-4.1-nano LLM call."""
        from app.integrations.llm_gateway import LLMGateway
        from vendor.ai_hedge_fund.data.models import (
            CompanyFacts, FinancialMetrics, LineItem, InsiderTrade, CompanyNews,
        )

        with self.app.app_context():
            llm = LLMGateway()

            system_prompt = (
                "You are a financial data provider. Return ONLY valid JSON matching "
                "the exact schema requested. Use null for unknown values. "
                "All financial figures should be in USD. Percentages should be "
                "expressed as decimals (e.g. 0.25 for 25%). "
                "Provide the most recent data you know. Output ONLY the JSON object, "
                "no markdown, no explanation."
            )

            user_prompt = f"""Provide comprehensive financial data for {ticker} as a JSON object:

{{
  "company": {{
    "name": "string", "ticker": "{ticker}", "sector": "string", "industry": "string",
    "exchange": "string", "location": "string or null", "employees": number_or_null,
    "website_url": "string or null", "cik": "string or null",
    "market_cap": number_or_null, "listing_date": "YYYY-MM-DD or null",
    "sic_industry": "string or null", "sec_filings_url": "string or null"
  }},
  "current_price": number,
  "week_52_high": number,
  "week_52_low": number,
  "monthly_closes": [
    {{"date": "YYYY-MM-01", "close": number}}
  ],
  "metrics": [
    {{
      "report_period": "YYYY-MM-DD", "period": "quarterly", "currency": "USD",
      "market_cap": number_or_null, "enterprise_value": number_or_null,
      "pe": number_or_null, "pb": number_or_null, "ps": number_or_null,
      "ev_ebitda": number_or_null, "ev_revenue": number_or_null,
      "peg": number_or_null, "fcf_yield": number_or_null,
      "gross_margin": number_or_null, "operating_margin": number_or_null,
      "net_margin": number_or_null, "roe": number_or_null, "roa": number_or_null,
      "roic": number_or_null, "revenue_growth": number_or_null,
      "earnings_growth": number_or_null, "eps_growth": number_or_null,
      "fcf_growth": number_or_null, "operating_income_growth": number_or_null,
      "ebitda_growth": number_or_null, "book_value_growth": number_or_null,
      "current_ratio": number_or_null, "quick_ratio": number_or_null,
      "debt_to_equity": number_or_null, "debt_to_assets": number_or_null,
      "interest_coverage": number_or_null,
      "eps": number_or_null, "bvps": number_or_null, "fcfps": number_or_null,
      "payout_ratio": number_or_null, "asset_turnover": number_or_null
    }}
  ],
  "financials": [
    {{
      "report_period": "YYYY-MM-DD", "period": "quarterly", "currency": "USD",
      "revenue": number_or_null, "net_income": number_or_null,
      "operating_income": number_or_null, "gross_profit": number_or_null,
      "free_cash_flow": number_or_null, "total_assets": number_or_null,
      "total_liabilities": number_or_null, "total_debt": number_or_null,
      "total_equity": number_or_null, "cash": number_or_null,
      "operating_cash_flow": number_or_null, "eps": number_or_null,
      "outstanding_shares": number_or_null, "capex": number_or_null,
      "rnd": number_or_null, "sga": number_or_null, "dividends": number_or_null
    }}
  ],
  "insider_trades": [
    {{
      "name": "string", "title": "string or null", "is_director": boolean,
      "shares": number, "price": number_or_null, "value": number_or_null,
      "filing_date": "YYYY-MM-DD"
    }}
  ],
  "news": [
    {{"title": "string", "source": "string or null", "date": "YYYY-MM-DD",
      "sentiment": "positive|negative|neutral"}}
  ]
}}

Requirements:
- monthly_closes: 12 entries for the last 12 months
- metrics: 4 entries for the last 4 quarters (newest first)
- financials: 4 entries for the last 4 quarters (newest first)
- insider_trades: up to 5 notable recent insider trades
- news: up to 5 major recent news items/themes
- Use null (not "N/A") for unknown numeric values"""

            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                purpose=f'research.data.{ticker}',
                section='research',
                max_tokens=4000,
                model=LLM_DATA_MODEL,
            )

            raw = result['content'].strip()
            # Strip markdown code fences if present
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
                if raw.endswith('```'):
                    raw = raw[:-3]
                raw = raw.strip()

            try:
                j = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(f"[Research] LLM returned invalid JSON for {ticker}: {raw[:200]}")
                raise ValueError(f"LLM returned invalid JSON for {ticker}")

        # Convert JSON → Pydantic models
        company = self._parse_llm_company(j.get('company', {}), ticker)
        metrics = self._parse_llm_metrics(j.get('metrics', []), ticker)
        line_items = self._parse_llm_line_items(j.get('financials', []), ticker)
        insider_trades = self._parse_llm_insider_trades(j.get('insider_trades', []), ticker)
        news = self._parse_llm_news(j.get('news', []), ticker)
        prices_df = self._parse_llm_prices(
            j.get('monthly_closes', []),
            j.get('current_price'),
            j.get('week_52_high'),
            j.get('week_52_low'),
        )
        market_cap = None
        if company:
            market_cap = company.market_cap
        if not market_cap and metrics:
            market_cap = metrics[0].market_cap

        return {
            'company': company,
            'prices': [],
            'prices_df': prices_df,
            'metrics': metrics,
            'line_items': line_items,
            'market_cap': market_cap,
            'insider_trades': insider_trades,
            'news': news,
            '_source': 'llm',
        }

    # ── LLM JSON → Pydantic converters ──

    def _parse_llm_company(self, j, ticker):
        from vendor.ai_hedge_fund.data.models import CompanyFacts
        if not j:
            return None
        return CompanyFacts(
            ticker=ticker,
            name=j.get('name', ticker),
            sector=j.get('sector'),
            industry=j.get('industry'),
            exchange=j.get('exchange'),
            location=j.get('location'),
            number_of_employees=j.get('employees'),
            website_url=j.get('website_url'),
            cik=j.get('cik'),
            market_cap=j.get('market_cap'),
            listing_date=j.get('listing_date'),
            sic_industry=j.get('sic_industry'),
            sec_filings_url=j.get('sec_filings_url'),
        )

    def _parse_llm_metrics(self, items, ticker):
        from vendor.ai_hedge_fund.data.models import FinancialMetrics
        results = []
        field_map = {
            'pe': 'price_to_earnings_ratio', 'pb': 'price_to_book_ratio',
            'ps': 'price_to_sales_ratio', 'ev_ebitda': 'enterprise_value_to_ebitda_ratio',
            'ev_revenue': 'enterprise_value_to_revenue_ratio',
            'peg': 'peg_ratio', 'fcf_yield': 'free_cash_flow_yield',
            'gross_margin': 'gross_margin', 'operating_margin': 'operating_margin',
            'net_margin': 'net_margin', 'roe': 'return_on_equity',
            'roa': 'return_on_assets', 'roic': 'return_on_invested_capital',
            'revenue_growth': 'revenue_growth', 'earnings_growth': 'earnings_growth',
            'eps_growth': 'earnings_per_share_growth',
            'fcf_growth': 'free_cash_flow_growth',
            'operating_income_growth': 'operating_income_growth',
            'ebitda_growth': 'ebitda_growth', 'book_value_growth': 'book_value_growth',
            'current_ratio': 'current_ratio', 'quick_ratio': 'quick_ratio',
            'debt_to_equity': 'debt_to_equity', 'debt_to_assets': 'debt_to_assets',
            'interest_coverage': 'interest_coverage',
            'eps': 'earnings_per_share', 'bvps': 'book_value_per_share',
            'fcfps': 'free_cash_flow_per_share', 'payout_ratio': 'payout_ratio',
            'asset_turnover': 'asset_turnover',
            'market_cap': 'market_cap', 'enterprise_value': 'enterprise_value',
        }
        for item in items:
            kwargs = {
                'ticker': ticker,
                'report_period': item.get('report_period', ''),
                'period': item.get('period', 'quarterly'),
                'currency': item.get('currency', 'USD'),
            }
            for src_key, dst_key in field_map.items():
                val = item.get(src_key)
                kwargs[dst_key] = val
            try:
                results.append(FinancialMetrics(**kwargs))
            except Exception as e:
                logger.warning(f"[Research] Failed to parse LLM metric: {e}")
        return results

    def _parse_llm_line_items(self, items, ticker):
        from vendor.ai_hedge_fund.data.models import LineItem
        field_map = {
            'revenue': 'revenue', 'net_income': 'net_income',
            'operating_income': 'operating_income', 'gross_profit': 'gross_profit',
            'free_cash_flow': 'free_cash_flow', 'total_assets': 'total_assets',
            'total_liabilities': 'total_liabilities', 'total_debt': 'total_debt',
            'total_equity': 'total_equity', 'cash': 'cash_and_equivalents',
            'operating_cash_flow': 'operating_cash_flow',
            'eps': 'earnings_per_share', 'outstanding_shares': 'outstanding_shares',
            'capex': 'capital_expenditure', 'rnd': 'research_and_development',
            'sga': 'selling_general_and_administrative',
            'dividends': 'dividends_and_other_cash_distributions',
        }
        results = []
        for item in items:
            kwargs = {
                'ticker': ticker,
                'report_period': item.get('report_period', ''),
                'period': item.get('period', 'quarterly'),
                'currency': item.get('currency', 'USD'),
            }
            for src_key, dst_key in field_map.items():
                val = item.get(src_key)
                if val is not None:
                    kwargs[dst_key] = val
            try:
                results.append(LineItem(**kwargs))
            except Exception as e:
                logger.warning(f"[Research] Failed to parse LLM line item: {e}")
        return results

    def _parse_llm_insider_trades(self, items, ticker):
        from vendor.ai_hedge_fund.data.models import InsiderTrade
        results = []
        for item in items:
            try:
                results.append(InsiderTrade(
                    ticker=ticker,
                    issuer=None,
                    name=item.get('name', 'Unknown'),
                    title=item.get('title'),
                    is_board_director=item.get('is_director', False),
                    transaction_date=item.get('filing_date'),
                    transaction_shares=item.get('shares'),
                    transaction_price_per_share=item.get('price'),
                    transaction_value=item.get('value'),
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=None,
                    security_title='Common Stock',
                    filing_date=item.get('filing_date', date.today().isoformat()),
                ))
            except Exception as e:
                logger.warning(f"[Research] Failed to parse LLM insider trade: {e}")
        return results

    def _parse_llm_news(self, items, ticker):
        from vendor.ai_hedge_fund.data.models import CompanyNews
        results = []
        for item in items:
            try:
                results.append(CompanyNews(
                    ticker=ticker,
                    title=item.get('title', ''),
                    author='LLM-estimated',
                    source=item.get('source', ''),
                    date=item.get('date', date.today().isoformat()),
                    url='',
                    sentiment=item.get('sentiment'),
                ))
            except Exception as e:
                logger.warning(f"[Research] Failed to parse LLM news: {e}")
        return results

    def _parse_llm_prices(self, monthly_closes, current_price, high_52w, low_52w):
        """Build a synthetic prices DataFrame from monthly close data."""
        if not monthly_closes:
            return None

        rows = []
        for entry in monthly_closes:
            close_val = entry.get('close')
            date_str = entry.get('date')
            if close_val is not None and date_str:
                rows.append({
                    'Date': pd.Timestamp(date_str),
                    'open': close_val,
                    'close': close_val,
                    'high': close_val,
                    'low': close_val,
                    'volume': 0,
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Adjust high/low of first and last rows with 52w data
        if high_52w is not None:
            df['high'] = df['high'].clip(upper=high_52w)
            idx_max = df['close'].idxmax()
            df.loc[idx_max, 'high'] = high_52w
        if low_52w is not None:
            df['low'] = df['low'].clip(lower=low_52w)
            idx_min = df['close'].idxmin()
            df.loc[idx_min, 'low'] = low_52w

        return df

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

        if mc:
            lines.append(f'*Market Cap:* {_fmt(mc)}')
        if metrics and metrics[0].enterprise_value:
            lines.append(f'*Enterprise Value:* {_fmt(metrics[0].enterprise_value)}')
            if mc:
                ev = metrics[0].enterprise_value
                if ev and mc:
                    lines.append(f'*EV/Market Cap:* {ev / mc:.2f}x')

        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            current = close.iloc[-1]

            high_52w = prices_df['high'].max()
            low_52w = prices_df['low'].min()
            pct_from_high = (current - high_52w) / high_52w if high_52w else 0
            pct_from_low = (current - low_52w) / low_52w if low_52w else 0

            lines.append('')
            lines.append(f'*Current Price:* ${current:.2f}')
            lines.append(f'*52-Week High:* ${high_52w:.2f} ({_pct(pct_from_high)} from high)')
            lines.append(f'*52-Week Low:* ${low_52w:.2f} ({_pct(pct_from_low)} from low)')

            # Volume only if we have real daily data (not LLM monthly)
            if data.get('_source') != 'llm':
                prev_close = close.iloc[-2] if len(close) >= 2 else current
                day_change = (current - prev_close) / prev_close if prev_close else 0
                latest_vol = prices_df['volume'].iloc[-1]
                avg_vol_30d = prices_df['volume'].iloc[-21:].mean() if len(prices_df) >= 21 else latest_vol
                lines.append(f'*Today Volume:* {_fmt(latest_vol, "", 0)}')
                lines.append(f'*Avg Volume (30d):* {_fmt(avg_vol_30d, "", 0)}')

            if data['line_items']:
                shares = getattr(data['line_items'][0], 'outstanding_shares', None)
                if shares:
                    lines.append(f'*Shares Outstanding:* {_fmt(shares, "", 0)}')

        if not co and not mc:
            lines.append('No company data available.')

        lines.append('')
        lines.append(self._data_cite(data))
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

        m = metrics[0]
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
            _ratio_context('Debt/Equity', m.debt_to_equity, 0.3, 2.0, 'low leverage', 'high leverage'),
            f'Debt/Assets: {_safe(m.debt_to_assets)}',
            f'Interest Coverage: {_safe(m.interest_coverage, ".1f")}x' if m.interest_coverage else 'Interest Coverage: N/A',
        ]

        # Historical comparison
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

        lines.append('')
        lines.append(self._data_cite(data))
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
            lines.append(f'  Earnings Yield: {earnings_yield:.2f}% (inverse P/E)')
        lines.append('')

        # Rule-based valuation signals
        bull_signals = []
        bear_signals = []

        if pe is not None:
            if pe < 15:
                bull_signals.append(f'P/E of {pe:.1f} is below 15 — cheap territory')
            elif pe < 22:
                bull_signals.append(f'P/E of {pe:.1f} is below S&P 500 average (~22)')
            elif pe > 35:
                bear_signals.append(f'P/E of {pe:.1f} is well above market average')
            elif pe > 25:
                bear_signals.append(f'P/E of {pe:.1f} is above market average of ~22')

        if peg is not None:
            if peg < 1.0:
                bull_signals.append(f'PEG of {peg:.2f} < 1.0 — growth undervalued')
            elif peg > 2.5:
                bear_signals.append(f'PEG of {peg:.2f} > 2.5 — steep premium for growth')

        if fcf_yield is not None:
            if fcf_yield > 0.05:
                bull_signals.append(f'FCF yield of {fcf_yield*100:.1f}% — strong cash generation')
            elif fcf_yield < 0.01:
                bear_signals.append(f'FCF yield of {fcf_yield*100:.1f}% — weak free cash')

        if pb is not None and pb < 1.0:
            bull_signals.append(f'P/B of {pb:.2f} — trading below book value')
        if ev_ebitda is not None and ev_ebitda < 8:
            bull_signals.append(f'EV/EBITDA of {ev_ebitda:.1f} is low')
        if m.debt_to_equity is not None and m.debt_to_equity > 2.0:
            bear_signals.append(f'D/E of {m.debt_to_equity:.2f} — high leverage')

        if bull_signals:
            lines.append('🟢 *Bullish Signals:*')
            for s in bull_signals:
                lines.append(f'  • {s}')
        if bear_signals:
            lines.append('🔴 *Bearish Signals:*')
            for s in bear_signals:
                lines.append(f'  • {s}')
        if not bull_signals and not bear_signals:
            lines.append('🟡 Valuation appears neutral.')

        bull_count = len(bull_signals)
        bear_count = len(bear_signals)
        lines.append('')
        if bull_count > bear_count + 1:
            lines.append('*Verdict:* 🟢 Stock appears undervalued')
        elif bear_count > bull_count + 1:
            lines.append('*Verdict:* 🔴 Stock appears overvalued')
        else:
            lines.append('*Verdict:* 🟡 Mixed signals — roughly fair')

        lines.append('')
        lines.append(self._data_cite(data))
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 4: Momentum & Technicals
    # ──────────────────────────────────────────────

    def _send_momentum(self, chat_id, ticker, data):
        prices_df = data['prices_df']
        metrics = data['metrics']
        is_llm = data.get('_source') == 'llm'
        lines = [f'📈 *{ticker} — Momentum & Technicals*', '']

        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            current = close.iloc[-1]
            lines.append(f'*Current Price:* ${current:.2f}')
            lines.append('')

            if is_llm:
                # LLM mode: monthly data — show approximate returns
                lines.append('*Approximate Price Performance (monthly data):*')
                for label, months in [('1M', 1), ('3M', 3), ('6M', 6), ('1Y', 12)]:
                    if len(close) > months:
                        old = close.iloc[-months - 1]
                        chg = (current - old) / old
                        emoji = '🟢' if chg >= 0 else '🔴'
                        lines.append(f'  {emoji} {label}: {_pct(chg)} (${old:.2f} → ${current:.2f})')

                lines.append('')
                high_52w = prices_df['high'].max()
                low_52w = prices_df['low'].min()
                range_52w = high_52w - low_52w
                pos_in_range = (current - low_52w) / range_52w if range_52w > 0 else 0.5
                lines.append('*52-Week Statistics:*')
                lines.append(f'  High: ${high_52w:.2f}')
                lines.append(f'  Low: ${low_52w:.2f}')
                lines.append(f'  Range: ${range_52w:.2f}')
                lines.append(f'  Position in range: {pos_in_range*100:.0f}%')
                lines.append('')
                lines.append('_Detailed technical indicators (SMAs, RSI, MACD, volume) require daily price data from live API._')

            else:
                # API mode: full daily data
                high_col = prices_df['high']
                low_col = prices_df['low']

                def _change(days):
                    if len(close) > days:
                        old = close.iloc[-days - 1]
                        return (current - old) / old, old
                    return None, None

                lines.append(f'*Last Close Date:* {prices_df.index[-1].strftime("%Y-%m-%d")}')
                lines.append('')
                lines.append('*Price Performance:*')
                for label, days in [('1W', 5), ('1M', 21), ('3M', 63), ('6M', 126), ('1Y', 252)]:
                    chg, old = _change(days)
                    if chg is not None:
                        emoji = '🟢' if chg >= 0 else '🔴'
                        lines.append(f'  {emoji} {label}: {_pct(chg)} (${old:.2f} → ${current:.2f})')
                lines.append('')

                high_52w = high_col.max()
                low_52w = low_col.min()
                range_52w = high_52w - low_52w
                pos_in_range = (current - low_52w) / range_52w if range_52w > 0 else 0.5
                lines.append('*52-Week Statistics:*')
                lines.append(f'  High: ${high_52w:.2f}')
                lines.append(f'  Low: ${low_52w:.2f}')
                lines.append(f'  Range: ${range_52w:.2f}')
                lines.append(f'  Position in range: {pos_in_range*100:.0f}%')
                lines.append('')

                lines.append('*Moving Averages:*')
                for window, name in [(10, '10d SMA'), (20, '20d SMA'), (50, '50d SMA'), (200, '200d SMA')]:
                    if len(close) >= window:
                        sma = close.rolling(window).mean().iloc[-1]
                        pct_diff = (current - sma) / sma
                        pos = '↑ above' if current > sma else '↓ below'
                        lines.append(f'  {name}: ${sma:.2f} — price {pos} by {abs(pct_diff)*100:.1f}%')

                if len(close) >= 200:
                    sma50 = close.rolling(50).mean().iloc[-1]
                    sma200 = close.rolling(200).mean().iloc[-1]
                    if sma50 > sma200:
                        lines.append('  ✅ *Golden Cross* — 50 SMA > 200 SMA (bullish)')
                    else:
                        lines.append('  ⚠️ *Death Cross* — 50 SMA < 200 SMA (bearish)')
                lines.append('')

                if len(close) >= 15:
                    delta = close.diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
                    rsi = 100 - (100 / (1 + rs))
                    rsi_label = 'OVERBOUGHT' if rsi > 70 else 'OVERSOLD' if rsi < 30 else 'neutral'
                    lines.append(f'*RSI-14:* {rsi:.1f} — {rsi_label}')

                if len(close) >= 26:
                    ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
                    ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
                    macd_line = ema12 - ema26
                    lines.append(f'*MACD:* {macd_line:.2f} ({"bullish" if macd_line > 0 else "bearish"})')
                lines.append('')

                vol = prices_df['volume']
                today_vol = vol.iloc[-1]
                lines.append('*Volume Analysis:*')
                lines.append(f'  Today: {_fmt(today_vol, "", 0)}')
                for window, name in [(20, '20d avg'), (60, '60d avg')]:
                    if len(vol) >= window:
                        avg = vol.iloc[-window:].mean()
                        ratio = today_vol / avg if avg > 0 else 1
                        lines.append(f'  {name}: {_fmt(avg, "", 0)} (today {ratio:.1f}x)')

                if len(close) >= 21:
                    daily_returns = close.pct_change().dropna()
                    vol_20d = daily_returns.iloc[-20:].std() * (252 ** 0.5)
                    lines.append(f'\n*Annualized Volatility (20d):* {vol_20d*100:.1f}%')
        else:
            lines.append('No price data available.')

        # EPS trajectory
        if metrics and len(metrics) >= 2:
            lines.append('')
            lines.append('*EPS Trajectory:*')
            for m in reversed(metrics[:min(8, len(metrics))]):
                if m.earnings_per_share is not None:
                    growth_note = ''
                    if m.earnings_per_share_growth is not None:
                        growth_note = f' ({_pct(m.earnings_per_share_growth)} growth)'
                    lines.append(f'  {m.report_period}: ${m.earnings_per_share:.2f}{growth_note}')

        lines.append('')
        lines.append(self._data_cite(data))
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 5: Financial Statements
    # ──────────────────────────────────────────────

    def _send_financials(self, chat_id, ticker, data):
        line_items = data['line_items']
        if not line_items:
            self.bot.send_message(chat_id, f'📋 *{ticker} — Financials*\nNo financial statement data.')
            return

        latest = line_items[0]

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
            rnd_pct = f' ({rnd/rev*100:.1f}% of rev)' if rev else ''
            lines.append(f'R&D: {_fmt(rnd)}{rnd_pct}')
        if sga:
            sga_pct = f' ({sga/rev*100:.1f}% of rev)' if rev else ''
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
        if td and te and te > 0:
            lines.append(f'Debt/Equity: {td/te:.2f}x')
        if cash and td:
            net_debt = td - cash
            lines.append(f'Net Debt: {_fmt(net_debt)} {"(net cash)" if net_debt < 0 else ""}')

        lines.append('')
        lines.append('*── Cash Flow ──*')
        lines.append(f'Operating Cash Flow: {_fmt(ocf)}')
        lines.append(f'Capital Expenditure: {_fmt(capex)}')
        lines.append(f'Free Cash Flow: {_fmt(fcf)}')
        if divs:
            lines.append(f'Dividends Paid: {_fmt(divs)}')
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
            if rev and prev_rev and prev_rev != 0:
                lines.append(f'Revenue: {_fmt(prev_rev)} → {_fmt(rev)} ({_pct((rev - prev_rev) / abs(prev_rev))})')
            if ni and prev_ni and prev_ni != 0:
                lines.append(f'Net Income: {_fmt(prev_ni)} → {_fmt(ni)} ({_pct((ni - prev_ni) / abs(prev_ni))})')
            if eps and prev_eps and prev_eps != 0:
                lines.append(f'EPS: ${prev_eps:.2f} → ${eps:.2f} ({_pct((eps - prev_eps) / abs(prev_eps))})')

        lines.append('')
        lines.append(self._data_cite(data))
        self.bot.send_message(chat_id, '\n'.join(lines))

        # Quarterly trends
        self._send_quarterly_trends(chat_id, ticker, line_items)

    def _send_quarterly_trends(self, chat_id, ticker, line_items):
        if len(line_items) < 3:
            return
        lines = [f'📅 *{ticker} — Quarterly Trends*', '']
        lines.append('*Revenue by Quarter:*')
        for item in reversed(line_items):
            rev = getattr(item, 'revenue', None)
            if rev is not None:
                lines.append(f'  {item.report_period}: {_fmt(rev)}')
        lines.append('')
        lines.append('*EPS by Quarter:*')
        for item in reversed(line_items):
            eps = getattr(item, 'earnings_per_share', None)
            if eps is not None:
                lines.append(f'  {item.report_period}: ${eps:.2f}')
        lines.append('')
        lines.append('*Free Cash Flow by Quarter:*')
        for item in reversed(line_items):
            fcf = getattr(item, 'free_cash_flow', None)
            if fcf is not None:
                lines.append(f'  {item.report_period}: {_fmt(fcf)}')
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 6: Insider Trading
    # ──────────────────────────────────────────────

    def _send_insider_trades(self, chat_id, ticker, data):
        trades = data.get('insider_trades', [])
        if not trades:
            self.bot.send_message(
                chat_id,
                f'👔 *{ticker} — Insider Trading*\nNo insider trades found.',
            )
            return

        total_buys = 0
        total_sells = 0
        buy_value = 0.0
        sell_value = 0.0

        for t in trades:
            value = abs(t.transaction_value or 0)
            if t.transaction_shares and t.transaction_shares > 0:
                total_buys += 1
                buy_value += value
            elif t.transaction_shares and t.transaction_shares < 0:
                total_sells += 1
                sell_value += value

        lines = [
            f'👔 *{ticker} — Insider Trading*',
            '',
            f'*Total Transactions:* {len(trades)}',
            f'*Buys:* {total_buys} ({_fmt(buy_value)})',
            f'*Sells:* {total_sells} ({_fmt(sell_value)})',
            f'*Net:* {"🟢 Net buying" if buy_value > sell_value else "🔴 Net selling" if sell_value > buy_value else "🟡 Balanced"}',
            '',
        ]

        lines.append('*Recent Transactions:*')
        for t in trades[:6]:
            name = t.name or 'Unknown'
            title = f' ({t.title})' if t.title else ''
            shares = t.transaction_shares or 0
            action = '🟢 BUY' if shares > 0 else '🔴 SELL'
            price = f' @ ${t.transaction_price_per_share:.2f}' if t.transaction_price_per_share else ''
            value_str = f' ({_fmt(abs(t.transaction_value))})' if t.transaction_value else ''
            lines.append(f'  {action} {name}{title}')
            lines.append(f'    {abs(shares):,.0f} shares{price}{value_str}')
            lines.append(f'    Filed: {t.filing_date[:10]}')
            lines.append('')

        lines.append(self._data_cite(data))
        self.bot.send_message(chat_id, '\n'.join(lines))

    # ──────────────────────────────────────────────
    # Phase 7: Recent News
    # ──────────────────────────────────────────────

    def _send_news(self, chat_id, ticker, data):
        news = data.get('news', [])
        if not news:
            self.bot.send_message(
                chat_id,
                f'📰 *{ticker} — Recent News*\nNo news articles found.',
            )
            return

        sentiments = [n.sentiment for n in news if n.sentiment]
        bullish_count = sum(1 for s in sentiments if s.lower() in ('positive', 'bullish'))
        bearish_count = sum(1 for s in sentiments if s.lower() in ('negative', 'bearish'))
        neutral_count = len(sentiments) - bullish_count - bearish_count

        lines = [
            f'📰 *{ticker} — Recent News*',
            '',
            f'*Articles:* {len(news)}',
        ]
        if sentiments:
            lines.append(f'*Sentiment:* 🟢 {bullish_count} positive | 🔴 {bearish_count} negative | 🟡 {neutral_count} neutral')
        lines.append('')

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

        lines.append(self._data_cite(data))
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
            is_llm = data.get('_source') == 'llm'
            caption = f'{ticker} — Research Chart (monthly, LLM-estimated)' if is_llm \
                else f'{ticker} — Research Chart (1Y price, revenue, EPS, P/E)'
            self.bot.send_photo(chat_id, chart_path, caption=caption)
            os.unlink(chart_path)
        except Exception as e:
            logger.error(f"[Research] Chart generation failed: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'⚠️ Chart generation failed: {e}')

    # ──────────────────────────────────────────────
    # Phase 9: AI Multi-Agent Analysis
    # ──────────────────────────────────────────────

    def _send_ai_analysis(self, chat_id, ticker):
        """Run hedge fund multi-agent analysis and send results."""
        self.bot.send_message(chat_id, f'🤖 *{ticker} — AI Analysis*\nRunning analyst agents...')

        try:
            from app.services.hedge_fund_service import HedgeFundService
            from app import feature_flags

            with self.app.app_context():
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
                            if points:
                                msg += '\n' + '\n'.join(points[:8])
                        elif isinstance(reasoning, str):
                            msg += f'\n{reasoning[:800]}'
                    self.bot.send_message(chat_id, msg)

                consensus = analysis.consensus_signal or 'neutral'
                conf = analysis.consensus_confidence or 0
                emoji = '🟢' if consensus == 'bullish' else '🔴' if consensus == 'bearish' else '🟡'
                self.bot.send_message(
                    chat_id,
                    f'\n*{ticker} — AI Consensus*\n'
                    f'{emoji} Signal: *{consensus.upper()}* ({conf}% confidence)',
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

        is_llm = data.get('_source') == 'llm'

        try:
            from app.integrations.llm_gateway import LLMGateway

            with self.app.app_context():
                llm = LLMGateway()
                data_summary = self._build_data_summary(ticker, data)

                system_prompt = """You are a senior equity research analyst writing an investment brief.
You produce concise, data-driven analysis. Always cite specific numbers.
Use Telegram Markdown formatting (*bold*, _italic_).
Structure your response with clear sections."""

                # In LLM mode, add multi-perspective analysis to compensate for skipped Phase 9
                extra_section = ""
                if is_llm:
                    extra_section = """
7. *Multi-Perspective Analysis* (provide these since live AI agents were unavailable)
   - Fundamentals Analyst: signal (bullish/bearish/neutral) + 1-sentence reasoning with numbers
   - Technical Analyst: signal + 1-sentence reasoning
   - Valuation Analyst: signal + 1-sentence reasoning
   - Sentiment Analyst: signal + 1-sentence reasoning
   - Overall Consensus: signal + confidence percentage
"""

                user_prompt = f"""Write a comprehensive investment summary for {ticker} based on this data:

{data_summary}

Structure your response as:

1. *Executive Summary* (3-4 sentences with key numbers — price, market cap, P/E, revenue growth, margin trend)

2. *Bull Case* (3-4 bullet points with specific numbers supporting a buy thesis)

3. *Bear Case* (3-4 bullet points with specific numbers supporting caution)

4. *Future Outlook* (3-4 sentences on what to expect next quarter and next 12 months)

5. *Key Metrics to Watch* (3-4 metrics with current values and thresholds)

6. *Verdict* (1-2 sentences with clear directional bias and confidence level)
{extra_section}
Be specific with numbers throughout. Take a clear analytical stance.
Keep the total response under {"3000" if is_llm else "2500"} characters for Telegram readability."""

                result = llm.call(
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    purpose=f'research.summary.{ticker}',
                    section='research',
                    max_tokens=2000 if is_llm else 1500,
                )

                summary = result['content'].strip()
                model = result.get('model', 'unknown')
                cost = result.get('cost_usd', 0)

                msg = f'🧠 *{ticker} — AI Investment Summary*\n\n{summary}'
                msg += f'\n\n_Generated by {model} | Cost: ${cost:.4f}_'
                msg += f'\n{self._data_cite(data)}'

                self.bot.send_message(chat_id, msg)

        except Exception as e:
            logger.error(f"[Research] LLM summary failed for {ticker}: {e}", exc_info=True)
            self.bot.send_message(chat_id, f'⚠️ LLM summary generation failed: {e}')

    def _build_data_summary(self, ticker, data):
        """Build a compact text summary of all fetched data for LLM consumption."""
        parts = []

        co = data.get('company')
        if co:
            parts.append(f"Company: {co.name or ticker}")
            parts.append(f"Sector: {co.sector or 'N/A'} | Industry: {co.industry or 'N/A'}")
            if co.number_of_employees:
                parts.append(f"Employees: {co.number_of_employees:,}")

        mc = data.get('market_cap')
        if mc:
            parts.append(f"Market Cap: {_fmt(mc)}")

        prices_df = data.get('prices_df')
        if prices_df is not None and not prices_df.empty:
            close = prices_df['close']
            current = close.iloc[-1]
            high_52w = prices_df['high'].max()
            low_52w = prices_df['low'].min()
            parts.append(f"\nPrice: ${current:.2f} | 52w High: ${high_52w:.2f} | 52w Low: ${low_52w:.2f}")

            for label, months in [('1M', 1), ('3M', 3), ('6M', 6), ('1Y', 12)]:
                if len(close) > months:
                    old = close.iloc[-months - 1]
                    chg = (current - old) / old
                    parts.append(f"{label} return: {_pct(chg)}")

        metrics = data.get('metrics', [])
        if metrics:
            m = metrics[0]
            parts.append(f"\nFundamentals ({m.report_period}):")
            for attr, label in [
                ('price_to_earnings_ratio', 'P/E'), ('price_to_book_ratio', 'P/B'),
                ('price_to_sales_ratio', 'P/S'), ('enterprise_value_to_ebitda_ratio', 'EV/EBITDA'),
                ('peg_ratio', 'PEG'), ('free_cash_flow_yield', 'FCF Yield'),
                ('gross_margin', 'Gross Margin'), ('operating_margin', 'Op Margin'),
                ('net_margin', 'Net Margin'), ('return_on_equity', 'ROE'),
                ('revenue_growth', 'Revenue Growth'), ('earnings_per_share_growth', 'EPS Growth'),
                ('current_ratio', 'Current Ratio'), ('debt_to_equity', 'D/E'),
                ('earnings_per_share', 'EPS'),
            ]:
                val = getattr(m, attr, None)
                if val is not None:
                    if 'margin' in attr or 'growth' in attr or 'yield' in attr or 'return' in attr:
                        parts.append(f"  {label}: {val*100:.1f}%")
                    else:
                        parts.append(f"  {label}: {val:.2f}")

        line_items = data.get('line_items', [])
        if line_items:
            li = line_items[0]
            parts.append(f"\nFinancials ({li.report_period}):")
            for attr, label in [
                ('revenue', 'Revenue'), ('net_income', 'Net Income'),
                ('free_cash_flow', 'FCF'), ('total_debt', 'Total Debt'),
                ('cash_and_equivalents', 'Cash'),
            ]:
                val = getattr(li, attr, None)
                if val is not None:
                    parts.append(f"  {label}: {_fmt(val)}")

        trades = data.get('insider_trades', [])
        if trades:
            buys = sum(1 for t in trades if t.transaction_shares and t.transaction_shares > 0)
            sells = sum(1 for t in trades if t.transaction_shares and t.transaction_shares < 0)
            parts.append(f"\nInsider Trading: {buys} buys, {sells} sells")

        news = data.get('news', [])
        if news:
            sentiments = [n.sentiment for n in news if n.sentiment]
            pos = sum(1 for s in sentiments if s.lower() in ('positive', 'bullish'))
            neg = sum(1 for s in sentiments if s.lower() in ('negative', 'bearish'))
            parts.append(f"\nNews: {len(news)} articles, {pos} positive, {neg} negative")
            for n in news[:3]:
                parts.append(f"  - {n.title} ({n.source}, {n.date[:10]})")

        return '\n'.join(parts)
