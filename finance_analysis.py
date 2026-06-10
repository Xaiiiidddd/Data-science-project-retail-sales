import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json, os

np.random.seed(42)

sns.set_theme(style='dark', palette='deep')
plt.rcParams.update({
    'figure.facecolor': '#0d1420',
    'axes.facecolor':   '#0d1420',
    'axes.edgecolor':   '#1e2f48',
    'axes.labelcolor':  '#7a9bbc',
    'xtick.color':      '#4d6a8a',
    'ytick.color':      '#4d6a8a',
    'text.color':       '#b8d0ea',
    'grid.color':       '#1e2f48',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
})

COLORS = {
    'AAPL':  '#3b82f6',
    'MSFT':  '#22c55e',
    'GOOGL': '#ef4444',
    'AMZN':  '#f59e0b',
    'NVDA':  '#a855f7',
    'SP500': '#64748b',
}

def simulate_gbm(S0: float, annual_mu: float, annual_sigma: float,
                 trading_days: int = 252) -> np.ndarray:
  
    dt          = 1.0 / 252
    daily_mu    = (annual_mu - 0.5 * annual_sigma**2) * dt
    daily_sigma = annual_sigma * np.sqrt(dt)

    log_returns = np.random.normal(daily_mu, daily_sigma, trading_days)

    log_price_path = np.log(S0) + np.concatenate([[0], np.cumsum(log_returns)])
    prices         = np.exp(log_price_path)

    return prices.round(2)

STOCKS_CONFIG = {
    'AAPL': {'S0': 182,  'mu': 0.15, 'sigma': 0.22, 'sector': 'Technology'},
    'MSFT': {'S0': 375,  'mu': 0.18, 'sigma': 0.20, 'sector': 'Technology'},
    'GOOGL':{'S0': 140,  'mu': 0.12, 'sigma': 0.25, 'sector': 'Technology'},
    'AMZN': {'S0': 178,  'mu': 0.20, 'sigma': 0.28, 'sector': 'Consumer'},
    'NVDA': {'S0': 495,  'mu': 0.45, 'sigma': 0.45, 'sector': 'Technology'},
}
WEIGHTS      = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.15, 'AMZN': 0.15, 'NVDA': 0.20}
INITIAL_VALUE = 100_000
TICKERS      = list(STOCKS_CONFIG.keys())
DAYS         = 252  

def generate_stock_data() -> dict:
   
    data = {}
    for ticker, cfg in STOCKS_CONFIG.items():
        prices = simulate_gbm(cfg['S0'], cfg['mu'], cfg['sigma'], DAYS)

        daily_returns = np.diff(prices) / prices[:-1] * 100  # in %
        vol  = daily_returns.std()
        mean = daily_returns.mean()
        sharpe = (mean * 252) / (vol * np.sqrt(252)) if vol > 0 else 0

        data[ticker] = {
            'prices':     prices.tolist(),
            'returns':    daily_returns.tolist(),
            'sector':     cfg['sector'],
            'start':      float(prices[0]),
            'end':        float(prices[-1]),
            'change_pct': round((prices[-1]/prices[0]-1)*100, 2),
            'high':       round(float(prices.max()), 2),
            'low':        round(float(prices.min()), 2),
            'avg':        round(float(prices.mean()), 2),
            'volatility': round(float(vol), 4),
            'sharpe':     round(float(sharpe), 3),
        }
        print(f"  {ticker}: ${prices[0]} → ${prices[-1]:.2f}  "
              f"({data[ticker]['change_pct']:+.1f}%)  Sharpe: {sharpe:.3f}")
    return data

def compute_portfolio_value(stock_data: dict) -> np.ndarray:
    n = DAYS
    values = np.zeros(n)
    for ticker, weight in WEIGHTS.items():
        prices = np.array(stock_data[ticker]['prices'][:n])
        values += weight * INITIAL_VALUE * (prices / prices[0])
    return values.round(2)


def compute_risk_metrics(stock_data: dict, sp500_returns: np.ndarray) -> dict:
   
    metrics = {}
    for ticker in TICKERS:
        r = np.array(stock_data[ticker]['returns'])

        var_95 = float(np.percentile(r, 5))

        tail_returns = r[r <= var_95]
        cvar_95 = float(tail_returns.mean()) if len(tail_returns) > 0 else var_95

        prices = np.array(stock_data[ticker]['prices'])
        peak, max_dd = prices[0], 0.0
        for p in prices:
            if p > peak: peak = p
            dd = (p - peak) / peak * 100
            if dd < max_dd: max_dd = dd

        min_len = min(len(r), len(sp500_returns))
        cov_mat = np.cov(r[:min_len], sp500_returns[:min_len])
        beta    = float(cov_mat[0, 1] / np.var(sp500_returns[:min_len])) if np.var(sp500_returns[:min_len]) > 0 else 0

        metrics[ticker] = {
            'var_95':        round(var_95, 4),
            'cvar_95':       round(cvar_95, 4),
            'max_drawdown':  round(max_dd, 2),
            'beta':          round(beta, 3),
        }
    return metrics


def compute_correlation_matrix(stock_data: dict) -> pd.DataFrame:
  
    rets_df = pd.DataFrame({t: stock_data[t]['returns'] for t in TICKERS})
    corr    = rets_df.corr(method='pearson').round(3)
    print("\nCorrelation Matrix:")
    print(corr.to_string())
    return corr


def compute_monthly_returns(stock_data: dict) -> dict:
    step = DAYS // 12
    monthly = {}
    for ticker in TICKERS:
        prices = stock_data[ticker]['prices']
        monthly[ticker] = []
        for m in range(12):
            s = m * step
            e = min((m+1)*step, DAYS-1)
            ret = round((prices[e] / prices[s] - 1) * 100, 2)
            monthly[ticker].append(ret)
    return monthly


def plot_portfolio_vs_sp500(portfolio: np.ndarray, sp500: np.ndarray,
                            output_dir: str = 'charts') -> None:

    fig, ax = plt.subplots(figsize=(14, 6))
    days = range(len(portfolio))

    ax.plot(days, portfolio, color=COLORS['AAPL'], linewidth=2.5, label='Portfolio ($100K)', zorder=3)
    ax.fill_between(days, portfolio, INITIAL_VALUE, alpha=0.12, color=COLORS['AAPL'])
    ax.plot(days, sp500, color=COLORS['SP500'], linewidth=1.5, linestyle='--', label='S&P 500 Equivalent', zorder=2)
    ax.axhline(INITIAL_VALUE, color='#1e2f48', linewidth=1, linestyle=':')

    port_ret = (portfolio[-1]/INITIAL_VALUE - 1)*100
    sp_ret   = (sp500[-1]/INITIAL_VALUE - 1)*100
    alpha    = port_ret - sp_ret

    ax.set_title(f'Portfolio vs S&P 500  |  Portfolio: {port_ret:+.1f}%  S&P: {sp_ret:+.1f}%  Alpha: {alpha:+.1f}%',
                 fontsize=13, pad=14)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'${v/1000:.0f}K'))
    ax.legend(framealpha=0, labelcolor='#7a9bbc')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'portfolio_vs_sp500.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: portfolio_vs_sp500.png")


def plot_individual_stocks(stock_data: dict, output_dir: str = 'charts') -> None:
   
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, ticker in enumerate(TICKERS):
        ax   = axes[i]
        s    = stock_data[ticker]
        pr   = s['prices']
        col  = COLORS[ticker]
        days = range(len(pr))

        ax.plot(days, pr, color=col, linewidth=1.8, zorder=2)
        ax.fill_between(days, pr, min(pr), alpha=0.15, color=col)
        ax.axhline(s['start'], color='#3a5070', linewidth=0.8, linestyle=':')

        change_str = f"{s['change_pct']:+.1f}%"
        ax.set_title(f"{ticker}  {change_str}  |  Sharpe: {s['sharpe']:.2f}", fontsize=11, color=col)
        ax.set_ylabel('Price ($)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'${v:.0f}'))
        ax.grid(True, alpha=0.25)

        hi_idx = np.argmax(pr)
        lo_idx = np.argmin(pr)
        ax.annotate(f'${s["high"]:.0f}', xy=(hi_idx, s['high']), color='#22c55e', fontsize=8,
                    xytext=(5,5), textcoords='offset points')
        ax.annotate(f'${s["low"]:.0f}', xy=(lo_idx, s['low']), color='#ef4444', fontsize=8,
                    xytext=(5,-12), textcoords='offset points')

    axes[-1].axis('off')
    fig.suptitle('Individual Stock Price Series (252 Trading Days)', fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'individual_stocks.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: individual_stocks.png")


def plot_correlation_heatmap(corr: pd.DataFrame, output_dir: str = 'charts') -> None:
   
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=2, linecolor='#0d1420',
                cbar_kws={'shrink':0.75}, ax=ax,
                annot_kws={'size':12, 'weight':'bold', 'family':'monospace'})
    ax.set_title('Return Correlation Matrix\n(Pearson — Daily Returns)', fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: correlation_matrix.png")


def plot_risk_metrics(risk_metrics: dict, output_dir: str = 'charts') -> None:
   
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(TICKERS))
    w = 0.35
    var_vals  = [abs(risk_metrics[t]['var_95'])   for t in TICKERS]
    cvar_vals = [abs(risk_metrics[t]['cvar_95'])  for t in TICKERS]
    axes[0].bar(x - w/2, var_vals,  w, label='VaR 95%',  color=[COLORS[t] for t in TICKERS], alpha=0.85, edgecolor='none')
    axes[0].bar(x + w/2, cvar_vals, w, label='CVaR 95%', color=[COLORS[t] for t in TICKERS], alpha=0.45, edgecolor='none')
    axes[0].set_xticks(x); axes[0].set_xticklabels(TICKERS)
    axes[0].set_title('VaR & CVaR (95%, daily %)', fontsize=11)
    axes[0].set_ylabel('Risk (%)')
    axes[0].legend(framealpha=0, labelcolor='#7a9bbc')

    dd_vals = [risk_metrics[t]['max_drawdown'] for t in TICKERS]
    bars = axes[1].barh(TICKERS, dd_vals, color=[COLORS[t] for t in TICKERS], alpha=0.8, edgecolor='none')
    axes[1].set_title('Max Drawdown (%)', fontsize=11)
    axes[1].set_xlabel('Drawdown (%)')
    for bar, v in zip(bars, dd_vals):
        axes[1].text(v - 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=9, ha='right')

    beta_vals = [risk_metrics[t]['beta'] for t in TICKERS]
    colors_b  = ['#22c55e' if b >= 0 else '#ef4444' for b in beta_vals]
    axes[2].bar(TICKERS, beta_vals, color=colors_b, alpha=0.8, edgecolor='none')
    axes[2].axhline(0, color='#4d6a8a', linewidth=0.8, linestyle='--')
    axes[2].axhline(1, color='#3a5070', linewidth=0.8, linestyle=':')
    axes[2].set_title('Beta vs S&P 500', fontsize=11)
    axes[2].set_ylabel('Beta')

    for ax in axes:
        ax.grid(True, alpha=0.25, axis='both')

    fig.suptitle('Portfolio Risk Metrics', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'risk_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: risk_metrics.png")


def plot_monthly_heatmap(monthly: dict, output_dir: str = 'charts') -> None:
    
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df = pd.DataFrame(monthly, index=months).T  # rows=tickers, cols=months
    df.columns = months

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                linewidths=2, linecolor='#0d1420', ax=ax,
                cbar_kws={'shrink':0.8, 'label':'Return (%)'},
                annot_kws={'size':9})
    ax.set_title('Monthly Return Heatmap (%)  —  Green = Positive  |  Red = Negative', fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'monthly_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: monthly_heatmap.png")


def plot_risk_return_scatter(stock_data: dict, output_dir: str = 'charts') -> None:
   
    fig, ax = plt.subplots(figsize=(9, 6))

    for ticker in TICKERS:
        s = stock_data[ticker]
        ax.scatter(s['volatility'], s['change_pct'],
                   color=COLORS[ticker], s=150, zorder=3)
        ax.annotate(ticker,
                    xy=(s['volatility'], s['change_pct']),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=11, fontweight='bold', color=COLORS[ticker])

    ax.axhline(0, color='#3a5070', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Daily Volatility (%)  [Risk]')
    ax.set_ylabel('Total Return (%)  [Reward]')
    ax.set_title('Risk vs Return Scatter\n(Lower-Left = Efficient  |  Upper-Right = High Risk/Reward)', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'risk_return_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: risk_return_scatter.png")


def plot_sharpe_comparison(stock_data: dict, output_dir: str = 'charts') -> None:
  
    sharpes = {t: stock_data[t]['sharpe'] for t in TICKERS}
    sorted_tickers = sorted(sharpes, key=sharpes.get, reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals   = [sharpes[t] for t in sorted_tickers]
    colors = [COLORS[t] if sharpes[t] > 0 else '#ef4444' for t in sorted_tickers]

    bars = ax.barh(sorted_tickers, vals, color=colors, alpha=0.8, edgecolor='none')
    ax.axvline(0, color='#4d6a8a', linewidth=1, linestyle='-')
    ax.axvline(1, color='#22c55e', linewidth=0.8, linestyle='--', label='Sharpe=1.0 threshold')

    for bar, v, t in zip(bars, vals, sorted_tickers):
        ax.text(v + 0.02 * np.sign(v), bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=10, ha='left' if v >= 0 else 'right')

    ax.set_title('Sharpe Ratio Ranking\n(Higher = Better Risk-Adjusted Return)', fontsize=11)
    ax.set_xlabel('Sharpe Ratio')
    ax.legend(framealpha=0, labelcolor='#22c55e')
    ax.grid(True, alpha=0.25, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sharpe_ratios.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: sharpe_ratios.png")

def save_json(stock_data: dict, portfolio: np.ndarray, sp500: np.ndarray,
              risk_metrics: dict, corr: pd.DataFrame, monthly: dict,
              output_path: str = 'finance_data.json') -> None:
    """Saves all computed data to JSON for the HTML dashboard."""
    output = {
        'stocks':           stock_data,
        'tickers':          TICKERS,
        'weights':          WEIGHTS,
        'portfolio_values': portfolio.tolist(),
        'sp500':            sp500.tolist(),
        'monthly_returns':  monthly,
        'months':           ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        'corr_matrix':      corr.to_dict(),
        'risk_metrics':     risk_metrics,
        'sectors':          {'Technology': 85.0, 'Consumer': 15.0},
        'portfolio_summary': {
            'initial':       INITIAL_VALUE,
            'final':         round(float(portfolio[-1]), 2),
            'total_return':  round((portfolio[-1]/INITIAL_VALUE - 1)*100, 2),
            'sp500_return':  round((sp500[-1]/INITIAL_VALUE - 1)*100, 2),
            'alpha':         round(((portfolio[-1]-sp500[-1])/INITIAL_VALUE)*100, 2),
            'best_stock':    max(TICKERS, key=lambda t: stock_data[t]['change_pct']),
            'worst_stock':   min(TICKERS, key=lambda t: stock_data[t]['change_pct']),
        }
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ JSON saved to: {output_path}")

s
def main():
    print("=" * 60)
    print("  FINANCE DATA SCIENCE PROJECT — PORTFOLIO ANALYTICS")
    print("=" * 60)

    print("\n[1/6] Simulating stock prices (GBM)...")
    stock_data = generate_stock_data()

    print("\n[2/6] Computing portfolio value...")
    portfolio = compute_portfolio_value(stock_data)
    sp500     = simulate_gbm(4800, 0.12, 0.15, DAYS)
    sp500_norm = INITIAL_VALUE * sp500 / sp500[0]
    port_ret  = (portfolio[-1]/INITIAL_VALUE - 1)*100
    sp_ret    = (sp500_norm[-1]/INITIAL_VALUE - 1)*100
    print(f"  Portfolio: ${INITIAL_VALUE:,} → ${portfolio[-1]:,.2f}  ({port_ret:+.1f}%)")
    print(f"  S&P 500:   ${INITIAL_VALUE:,} → ${sp500_norm[-1]:,.2f}  ({sp_ret:+.1f}%)")
    print(f"  Alpha:     {port_ret-sp_ret:+.1f}%")

    print("\n[3/6] Computing risk metrics...")
    sp500_rets   = np.diff(sp500) / sp500[:-1] * 100
    risk_metrics = compute_risk_metrics(stock_data, sp500_rets)

    print("\n[4/6] Computing correlation matrix...")
    corr    = compute_correlation_matrix(stock_data)
    monthly = compute_monthly_returns(stock_data)

    print("\n[5/6] Generating charts...")
    charts_dir = 'charts'
    plot_portfolio_vs_sp500(portfolio, sp500_norm, charts_dir)
    plot_individual_stocks(stock_data, charts_dir)
    plot_correlation_heatmap(corr, charts_dir)
    plot_risk_metrics(risk_metrics, charts_dir)
    plot_monthly_heatmap(monthly, charts_dir)
    plot_risk_return_scatter(stock_data, charts_dir)
    plot_sharpe_comparison(stock_data, charts_dir)

    print("\n[6/6] Saving JSON for dashboard...")
    save_json(stock_data, portfolio, sp500_norm, risk_metrics, corr, monthly)

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("  ├── finance_data.json   → open index.html in browser")
    print("  └── charts/            → 7 static PNG charts")
    print("=" * 60)


if __name__ == '__main__':
    main()
