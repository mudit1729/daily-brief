"""
Stock chart generation using matplotlib.
Produces a 4-panel research chart (price+SMAs, revenue, EPS, P/E).
"""
import logging
import tempfile
import matplotlib
matplotlib.use('Agg')  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# Dark theme colors
BG_COLOR = '#1a1a2e'
PANEL_BG = '#16213e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2a4a'
ACCENT_BLUE = '#4fc3f7'
ACCENT_GREEN = '#66bb6a'
ACCENT_RED = '#ef5350'
ACCENT_ORANGE = '#ffa726'
ACCENT_PURPLE = '#ab47bc'
SMA_50_COLOR = '#ffa726'
SMA_200_COLOR = '#ef5350'


def _fmt_large_number(val):
    """Format large numbers: 1.2B, 340M, 12.5K."""
    if val is None:
        return 'N/A'
    abs_val = abs(val)
    sign = '-' if val < 0 else ''
    if abs_val >= 1e12:
        return f'{sign}${abs_val/1e12:.1f}T'
    if abs_val >= 1e9:
        return f'{sign}${abs_val/1e9:.1f}B'
    if abs_val >= 1e6:
        return f'{sign}${abs_val/1e6:.0f}M'
    if abs_val >= 1e3:
        return f'{sign}${abs_val/1e3:.1f}K'
    return f'{sign}${abs_val:.2f}'


def _apply_dark_theme(fig, axes):
    """Apply dark theme to figure and axes."""
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.spines['bottom'].set_color(GRID_COLOR)
        ax.spines['left'].set_color(GRID_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, color=GRID_COLOR, alpha=0.3, linewidth=0.5)


def generate_research_chart(ticker, prices_df, metrics, line_items):
    """
    Generate a 4-panel research chart and return path to temp PNG.

    Args:
        ticker: Stock ticker symbol
        prices_df: DataFrame with Date index + close/volume columns
        metrics: list[FinancialMetrics] — quarterly, newest first
        line_items: list[LineItem] — quarterly financials, newest first

    Returns:
        str: Path to temporary PNG file (caller must delete)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{ticker} — Research Overview', color=TEXT_COLOR,
                 fontsize=14, fontweight='bold', y=0.98)
    _apply_dark_theme(fig, axes.flat)

    # Panel 1: Stock Price + SMAs (top-left)
    _plot_price_panel(axes[0, 0], prices_df, ticker)

    # Panel 2: Quarterly Revenue (top-right)
    _plot_revenue_panel(axes[0, 1], line_items)

    # Panel 3: Quarterly EPS (bottom-left)
    _plot_eps_panel(axes[1, 0], metrics)

    # Panel 4: P/E Ratio (bottom-right)
    _plot_pe_panel(axes[1, 1], metrics)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    logger.info(f"Generated research chart for {ticker}: {tmp.name}")
    return tmp.name


def _plot_price_panel(ax, prices_df, ticker):
    """Stock price with 50/200-day SMAs."""
    if prices_df is None or prices_df.empty:
        ax.text(0.5, 0.5, 'No price data', ha='center', va='center',
                color=TEXT_COLOR, fontsize=11, transform=ax.transAxes)
        ax.set_title('Stock Price')
        return

    close = prices_df['close']
    dates = prices_df.index

    ax.plot(dates, close, color=ACCENT_BLUE, linewidth=1.5, label='Close')

    if len(close) >= 50:
        sma50 = close.rolling(50).mean()
        ax.plot(dates, sma50, color=SMA_50_COLOR, linewidth=1, alpha=0.8,
                label='50 SMA', linestyle='--')
    if len(close) >= 200:
        sma200 = close.rolling(200).mean()
        ax.plot(dates, sma200, color=SMA_200_COLOR, linewidth=1, alpha=0.8,
                label='200 SMA', linestyle='--')

    ax.set_title('Stock Price (1Y)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
    # Rotate date labels
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_fontsize(7)


def _plot_revenue_panel(ax, line_items):
    """Quarterly revenue bar chart."""
    revenues = []
    labels = []
    for item in reversed(line_items):
        rev = getattr(item, 'revenue', None)
        if rev is not None:
            revenues.append(rev)
            labels.append(item.report_period[:7])  # YYYY-MM

    if not revenues:
        ax.text(0.5, 0.5, 'No revenue data', ha='center', va='center',
                color=TEXT_COLOR, fontsize=11, transform=ax.transAxes)
        ax.set_title('Quarterly Revenue')
        return

    x = np.arange(len(revenues))
    colors = [ACCENT_GREEN if r >= 0 else ACCENT_RED for r in revenues]
    bars = ax.bar(x, revenues, color=colors, alpha=0.85, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_title('Quarterly Revenue', fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, pos: _fmt_large_number(val).replace('$', '')))

    # Value labels on top
    for bar, val in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                _fmt_large_number(val), ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=6)


def _plot_eps_panel(ax, metrics):
    """Quarterly EPS bar chart."""
    eps_vals = []
    labels = []
    for m in reversed(metrics):
        if m.earnings_per_share is not None:
            eps_vals.append(m.earnings_per_share)
            labels.append(m.report_period[:7])

    if not eps_vals:
        ax.text(0.5, 0.5, 'No EPS data', ha='center', va='center',
                color=TEXT_COLOR, fontsize=11, transform=ax.transAxes)
        ax.set_title('Quarterly EPS')
        return

    x = np.arange(len(eps_vals))
    colors = [ACCENT_GREEN if e >= 0 else ACCENT_RED for e in eps_vals]
    bars = ax.bar(x, eps_vals, color=colors, alpha=0.85, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_title('Quarterly EPS', fontsize=10, fontweight='bold')

    for bar, val in zip(bars, eps_vals):
        y = bar.get_height() if val >= 0 else bar.get_y()
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f'${val:.2f}', ha='center', va=va,
                color=TEXT_COLOR, fontsize=6)


def _plot_pe_panel(ax, metrics):
    """P/E ratio line chart."""
    pe_vals = []
    labels = []
    for m in reversed(metrics):
        if m.price_to_earnings_ratio is not None:
            pe_vals.append(m.price_to_earnings_ratio)
            labels.append(m.report_period[:7])

    if not pe_vals:
        ax.text(0.5, 0.5, 'No P/E data', ha='center', va='center',
                color=TEXT_COLOR, fontsize=11, transform=ax.transAxes)
        ax.set_title('P/E Ratio')
        return

    x = np.arange(len(pe_vals))
    ax.plot(x, pe_vals, color=ACCENT_PURPLE, linewidth=2, marker='o',
            markersize=5, markerfacecolor=ACCENT_PURPLE)
    ax.fill_between(x, pe_vals, alpha=0.15, color=ACCENT_PURPLE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_title('P/E Ratio', fontsize=10, fontweight='bold')

    for xi, val in zip(x, pe_vals):
        ax.text(xi, val, f'{val:.1f}', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=6)
