import logging
from datetime import date

logger = logging.getLogger(__name__)

TRACKED_SYMBOLS = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'NASDAQ Composite',
    '^NSEI': 'NIFTY 50',
    '^BSESN': 'SENSEX',
    'GC=F': 'Gold Futures',
}


class MarketDataService:
    def fetch_snapshots(self, target_date=None):
        """Fetch latest price data for all tracked symbols."""
        import yfinance as yf

        target_date = target_date or date.today()
        snapshots = []

        for symbol, name in TRACKED_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if len(hist) < 1:
                    logger.warning(f"No data for {symbol}")
                    continue

                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) >= 2 else latest
                change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100 if prev['Close'] else 0

                snapshots.append({
                    'symbol': symbol,
                    'name': name,
                    'price': round(float(latest['Close']), 2),
                    'change_pct': round(float(change_pct), 2),
                    'change_abs': round(float(latest['Close'] - prev['Close']), 2),
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
