import logging
from datetime import date

logger = logging.getLogger(__name__)

TRACKED_SYMBOLS = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'NASDAQ Composite',
    '^NSEI': 'NIFTY 50',
    '^BSESN': 'SENSEX',
    '^KS11': 'KOSPI',
    '000001.SS': 'Shanghai Composite',
    'QQQ': 'QQQ (Nasdaq 100)',
    'GC=F': 'Gold Futures',
    'BTC-USD': 'Bitcoin',
}


class MarketDataService:
    def fetch_snapshots(self, target_date=None):
        """Fetch latest price data with multi-period performance for all tracked symbols."""
        import yfinance as yf

        target_date = target_date or date.today()
        snapshots = []

        for symbol, name in TRACKED_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                # Fetch 13 months to reliably compute 1-year change
                hist = ticker.history(period='13mo')
                if len(hist) < 1:
                    logger.warning(f"No data for {symbol}")
                    continue

                latest = hist.iloc[-1]
                close_now = float(latest['Close'])

                # 1-day change
                prev = hist.iloc[-2] if len(hist) >= 2 else latest
                day_change = ((close_now - float(prev['Close'])) / float(prev['Close'])) * 100 if float(prev['Close']) else 0

                # 1-month change (~21 trading days)
                month_idx = max(0, len(hist) - 22)
                close_1m = float(hist.iloc[month_idx]['Close'])
                month_change = ((close_now - close_1m) / close_1m) * 100 if close_1m else 0

                # 3-month change (~63 trading days)
                quarter_idx = max(0, len(hist) - 64)
                close_3m = float(hist.iloc[quarter_idx]['Close'])
                quarter_change = ((close_now - close_3m) / close_3m) * 100 if close_3m else 0

                # 1-year change (~252 trading days)
                year_idx = max(0, len(hist) - 253)
                close_1y = float(hist.iloc[year_idx]['Close'])
                year_change = ((close_now - close_1y) / close_1y) * 100 if close_1y else 0

                snapshots.append({
                    'symbol': symbol,
                    'name': name,
                    'price': round(close_now, 2),
                    'change_pct': round(float(day_change), 2),
                    'change_abs': round(close_now - float(prev['Close']), 2),
                    'change_1m_pct': round(float(month_change), 2),
                    'change_3m_pct': round(float(quarter_change), 2),
                    'change_1y_pct': round(float(year_change), 2),
                    'volume': int(latest['Volume']) if latest['Volume'] else None,
                    'snapshot_date': target_date,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        return snapshots

    def check_momentum_value_gate(self, snapshots):
        """
        Simple momentum x value gate:
        - Momentum: at least 2 of 3 US indices up > 0.3%
        - Value: gold not up > 2% (no panic flight to safety)
        """
        us_indices = [s for s in snapshots if s['symbol'] in ('^GSPC', '^DJI', '^IXIC')]
        gold = next((s for s in snapshots if s['symbol'] == 'GC=F'), None)

        momentum_count = sum(1 for s in us_indices if s['change_pct'] > 0.3)
        momentum_pass = momentum_count >= 2

        gold_change = gold['change_pct'] if gold else 0
        value_pass = gold_change < 2.0

        gate_passed = momentum_pass and value_pass
        signals = {
            'momentum_count': momentum_count,
            'momentum_pass': momentum_pass,
            'gold_change_pct': gold_change,
            'value_pass': value_pass,
        }
        return gate_passed, signals
