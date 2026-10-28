"""
Financial Data Scraper & Macro Analyzer
Author: Nicholas Giustinelli
Description: Extracts financial data (via yfinance or synthetic fallback),
calculates Z-Scores, rolling metrics, intermarket correlation matrices,
and generates both static charts (PNG) and interactive web dashboards (HTML/Chart.js).
"""

import sys
import argparse
import math
import json
from datetime import datetime, timedelta

# --- GRACEFUL IMPORTS ---
HAS_PANDAS = False
HAS_NUMPY = False
HAS_PLT = False
HAS_YFINANCE = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception as e:
    print(f"[WARNING] Pandas import issue ({e}). Fallback to Pure Python mode.")

try:
    import numpy as np
    HAS_NUMPY = True
except Exception as e:
    print(f"[WARNING] NumPy import issue ({e}). Fallback to Pure Python mode.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLT = True
except Exception as e:
    print(f"[WARNING] Matplotlib/Seaborn import issue ({e}). Using HTML Dashboard renderer.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception as e:
    print(f"[WARNING] yfinance not found. Will use synthetic data generator if offline.")

# --- PURE PYTHON MATH FALLBACKS ---
def pure_mean(data):
    return sum(data) / len(data) if data else 0.0

def pure_std(data):
    if len(data) < 2:
        return 0.0
    m = pure_mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def pure_correlation(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = pure_mean(x), pure_mean(y)
    std_x, std_y = pure_std(x), pure_std(y)
    if std_x == 0 or std_y == 0:
        return 0.0
    covariance = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - 1)
    return covariance / (std_x * std_y)

def generate_synthetic_series(tickers, years=3):
    print("[INFO] Generating synthetic financial data for testing...")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=365 * years)
    dates = []
    curr = start_dt
    while curr <= end_dt:
        if curr.weekday() < 5:
            dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    initial_prices = {"SPY": 400.0, "TLT": 100.0, "GLD": 180.0}
    volatilities = {"SPY": 0.012, "TLT": 0.008, "GLD": 0.010}
    drifts = {"SPY": 0.0004, "TLT": -0.0001, "GLD": 0.0002}
    data = {t: [] for t in tickers}
    import random
    random.seed(42)
    for t in tickers:
        p = initial_prices.get(t, 100.0)
        vol = volatilities.get(t, 0.01)
        d = drifts.get(t, 0.0001)
        for _ in dates:
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2.0 * math.pi * u2)
            ret = d + vol * z
            p = p * (1.0 + ret)
            data[t].append(round(p, 2))
    return dates, data

def fetch_data(tickers, years=3, use_synthetic=False):
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    if not use_synthetic and HAS_YFINANCE and HAS_PANDAS:
        print(f"[INFO] Fetching live data from Yahoo Finance for: {tickers}")
        try:
            raw = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                if "Adj Close" in raw.columns.levels[0]:
                    df = raw["Adj Close"]
                elif "Close" in raw.columns.levels[0]:
                    df = raw["Close"]
                else:
                    df = raw.iloc[:, :len(tickers)]
            elif "Adj Close" in raw:
                df = raw["Adj Close"]
            else:
                df = raw
            df = df.ffill().dropna()
            dates = df.index.strftime("%Y-%m-%d").tolist()
            data_dict = {col: df[col].tolist() for col in df.columns}
            return dates, data_dict
        except Exception as e:
            print(f"[WARNING] Live fetch failed ({e}). Falling back to synthetic data.")
    return generate_synthetic_series(tickers, years=years)

def create_html_dashboard(dates, data, corr_matrix, filename="intermarket_dashboard.html"):
    norm_data = {}
    for ticker, prices in data.items():
        base = prices[0] if prices[0] != 0 else 1.0
        norm_data[ticker] = [round((p / base) * 100.0, 2) for p in prices]
    
    tickers = list(data.keys())
    
    table_rows = ""
    for t1 in tickers:
        table_rows += f"<tr><th>{t1}</th>"
        for t2 in tickers:
            val = corr_matrix[t1][t2]
            cls = "high-corr" if abs(val) > 0.5 and t1 != t2 else "neutral-corr"
            table_rows += f'<td class="{cls}">{val:.2f}</td>'
        table_rows += "</tr>\n"
        
    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>Intermarket Analysis Dashboard | Nicholas Giustinelli</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: system-ui, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 20px 0; border-bottom: 1px solid #1e293b; margin-bottom: 30px; }}
        .header h1 {{ color: #38bdf8; margin: 0 0 10px 0; }}
        .header p {{ color: #94a3b8; margin: 0; }}
        .card {{ background-color: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 30px; }}
        .card h2 {{ color: #f1f5f9; margin-top: 0; font-size: 18px; border-left: 4px solid #38bdf8; padding-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: center; border-bottom: 1px solid #334155; }}
        th {{ background-color: #0f172a; color: #38bdf8; }}
        .high-corr {{ color: #ef4444; font-weight: bold; }}
        .neutral-corr {{ color: #10b981; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Intermarket Analysis & Relative Performance</h1>
            <p>Sviluppato da Nicholas Giustinelli | Portfolio Quantitative Analytics</p>
        </div>
        <div class="card">
            <h2>Intermarket Relative Performance (Base 100)</h2>
            <canvas id="performanceChart" height="100"></canvas>
        </div>
        <div class="card">
            <h2>Cross-Asset Correlation Matrix (Daily Returns)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Asset</th>
                        {''.join([f'<th>{t}</th>' for t in tickers])}
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>
    <script>
        const dates = {json.dumps(dates)};
        const normData = {json.dumps(norm_data)};
        const datasets = [
            {{ label: "SPY (S&P 500)", data: normData["SPY"], borderColor: "#38bdf8", borderWidth: 2, pointRadius: 0 }},
            {{ label: "TLT (Treasuries)", data: normData["TLT"], borderColor: "#f97316", borderWidth: 2, pointRadius: 0 }},
            {{ label: "GLD (Gold)", data: normData["GLD"], borderColor: "#eab308", borderWidth: 2, pointRadius: 0 }}
        ];
        const ctx = document.getElementById("performanceChart").getContext("2d");
        new Chart(ctx, {{
            type: "line",
            data: {{ labels: dates, datasets: datasets }},
            options: {{
                responsive: true,
                scales: {{ x: {{ ticks: {{ color: "#94a3b8" }} }}, y: {{ ticks: {{ color: "#94a3b8" }} }} }},
                plugins: {{ legend: {{ labels: {{ color: "#f8fafc" }} }} }}
            }}
        }});
    </script>
</body>
</html>"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SUCCESS] Interactive Dashboard saved to '{filename}'. Open this file in any web browser!")

def main(use_synthetic=False):
    tickers = ["SPY", "TLT", "GLD"]
    dates, data = fetch_data(tickers, years=3, use_synthetic=use_synthetic)
    daily_returns = {t: [] for t in tickers}
    for t in tickers:
        prices = data[t]
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            daily_returns[t].append(ret)
    
    corr_matrix = {t1: {t2: 0.0 for t2 in tickers} for t1 in tickers}
    for t1 in tickers:
        for t2 in tickers:
            corr_matrix[t1][t2] = pure_correlation(daily_returns[t1], daily_returns[t2])
            
    create_html_dashboard(dates, data, corr_matrix)
    
    if HAS_PANDAS and HAS_PLT and HAS_NUMPY:
        try:
            print("[INFO] Generating Matplotlib PNG chart...")
            df = pd.DataFrame(data, index=pd.to_datetime(dates))
            ret_df = df.pct_change().dropna()
            sns.set_theme(style="darkgrid")
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            norm_df = (df / df.iloc[0]) * 100
            norm_df.plot(ax=axes[0], linewidth=1.8)
            axes[0].set_title("Intermarket Relative Performance (Base 100)", fontsize=14, fontweight="bold")
            axes[0].set_ylabel("Normalized Price")
            sns.heatmap(ret_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1], vmin=-1, vmax=1)
            axes[1].set_title("Cross-Asset Correlation Matrix (Daily Returns)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig("intermarket_analysis.png", dpi=300)
            plt.close()
            print("[SUCCESS] PNG chart saved to 'intermarket_analysis.png'.")
        except Exception as e:
            print(f"[NOTE] Matplotlib PNG skipped ({e}). HTML dashboard is ready and complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Data Scraper & Intermarket Analyzer")
    parser.add_argument("--test", action="store_true", help="Run using synthetic data generator")
    args = parser.parse_args()
    main(use_synthetic=args.test)