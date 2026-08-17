"""
Quantum Portfolio Optimization - Top Selected Stocks Visual Dashboard
Visualizes Expected Returns, Risk (Variance), QAOA Selected Portfolio, 
Historical Trends, and Train/Test Metrics specifically for the Top Selected Stocks.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import streamlit as st

# Default sample data corresponding exactly to the API response from the screenshot
DEFAULT_DATA = {
    "selected_stocks": ["AMZN", "META"],
    "individual_expected_returns": {
        "AAPL": 0.5093280509863953,
        "MSFT": 0.5362141861904132,
        "GOOGL": 0.6686371880979166,
        "AMZN": 0.8169691192251061,
        "META": 1.3892890837744931
    },
    "individual_risks_variance": {
        "AAPL": 0.045854012520229526,
        "MSFT": 0.07247276864429127,
        "GOOGL": 0.09735240018974581,
        "AMZN": 0.11670846727142573,
        "META": 0.19538792152403536
    },
    "train_return": 1.1031291014997997,
    "train_risk": 0.11984487827941605,
    "train_sharpe": 3.186530296786172,
    "test_return": 0.3709302705788555,
    "test_risk": 0.07040198077689616,
    "test_sharpe": 1.3979763912169838
}

def fetch_historical_prices(tickers, start_date="2023-01-01", end_date="2024-01-01"):
    """Fetch historical daily close prices for specified tickers."""
    data = pd.DataFrame()
    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                data[t] = df[('Close', t)]
            else:
                data[t] = df['Close']
        except Exception as e:
            print(f"Warning: Could not fetch data for {t}: {e}")
    return data

def generate_matplotlib_dashboard(data=DEFAULT_DATA, save_path="top5_stocks_analysis.png", top_only=True):
    """Generate high-quality static Matplotlib visualization dashboard focused on Top Selected Stocks."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), facecolor='#0E1117')
    
    selected_stocks = data["selected_stocks"]
    returns_all = data["individual_expected_returns"]
    risks_all = data["individual_risks_variance"]
    
    # Filter for graphs: if top_only is True, plot ONLY selected top stocks
    target_tickers = selected_stocks if top_only else list(returns_all.keys())
    returns = {t: returns_all[t] for t in target_tickers if t in returns_all}
    risks = {t: risks_all[t] for t in target_tickers if t in risks_all}
    
    # 1. Risk vs. Expected Return Scatter Plot (Top Selected Stocks)
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('#161B22')
    
    x_risks = [risks[t] for t in target_tickers]
    y_returns = [returns[t] for t in target_tickers]
    
    palette = ['#00E676', '#FFD600', '#29B6F6', '#FF5252', '#AB47BC']
    
    for i, t in enumerate(target_tickers):
        color = palette[i % len(palette)]
        ax1.scatter(risks[t], returns[t], color=color, s=300, marker='*', 
                    edgecolor='#FFFFFF', linewidth=1.5, zorder=4)
        
        ax1.annotate(f"TOP #{i+1}: {t}\nReturn: {returns[t]*100:.1f}%\nRisk: {risks[t]:.3f}",
                     (risks[t], returns[t]),
                     textcoords="offset points", xytext=(12, -12),
                     fontsize=10, fontweight='bold', color=color,
                     bbox=dict(boxstyle="round,pad=0.4", fc="#102A1C", ec=color, alpha=0.9))

    # Quantum Portfolio Combined Point
    ax1.scatter(data["train_risk"], data["train_return"], color='#FF9100', s=350, marker='D',
                edgecolor='#FFD180', linewidth=2, zorder=5, label='Combined Quantum Portfolio')
    ax1.annotate(f"Optimal Portfolio (Top {len(selected_stocks)})\nSharpe: {data['train_sharpe']:.2f}",
                 (data["train_risk"], data["train_return"]),
                 textcoords="offset points", xytext=(-90, 22),
                 fontsize=10, fontweight='bold', color='#FF9100',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#3E2723", ec="#FF9100", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#FF9100"))

    ax1.set_title(f"Top {len(selected_stocks)} Selected Stocks: Risk vs. Return Spectrum", fontsize=13, fontweight='bold', color='#E6EDF3', pad=12)
    ax1.set_xlabel("Risk (Annual Variance)", fontsize=10, color='#8B949E')
    ax1.set_ylabel("Expected Annual Return", fontsize=10, color='#8B949E')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#8B949E')
    if x_risks and y_returns:
        ax1.set_xlim(min(x_risks)*0.75, max(x_risks)*1.3)
        ax1.set_ylim(min(y_returns)*0.75, max(y_returns)*1.2)

    # 2. Individual Expected Returns & Risk Bar Charts (Top Selected Stocks)
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('#161B22')
    
    x = np.arange(len(target_tickers))
    width = 0.35
    
    ret_values = [returns[t] * 100 for t in target_tickers]
    risk_values = [risks[t] * 100 for t in target_tickers]
    
    rects1 = ax2.bar(x - width/2, ret_values, width, label='Expected Return (%)', color='#00E676', alpha=0.9)
    rects2 = ax2.bar(x + width/2, risk_values, width, label='Variance Risk (x100)', color='#FF5252', alpha=0.75)
    
    ax2.set_title(f"Top {len(selected_stocks)} Selected Stocks: Expected Return vs. Risk", fontsize=13, fontweight='bold', color='#E6EDF3', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"#{i+1} {t}" for i, t in enumerate(target_tickers)], fontsize=11, fontweight='bold', color='#E6EDF3')
    ax2.set_ylabel("Percentage (%)", fontsize=10, color='#8B949E')
    ax2.legend(loc='upper left', facecolor='#21262D', edgecolor='#30363D')
    ax2.grid(True, linestyle='--', alpha=0.15, axis='y')

    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#00E676', fontweight='bold')
                     
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#FF5252', fontweight='bold')

    # 3. Train vs. Test Portfolio Performance Metrics
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('#161B22')
    
    metrics = ['Return', 'Risk (Variance)', 'Sharpe Ratio']
    train_vals = [data['train_return'], data['train_risk'], data['train_sharpe']]
    test_vals = [data['test_return'], data['test_risk'], data['test_sharpe']]
    
    x_m = np.arange(len(metrics))
    rects_tr = ax3.bar(x_m - width/2, train_vals, width, label='Train Set (In-Sample)', color='#7C4DFF', alpha=0.9)
    rects_te = ax3.bar(x_m + width/2, test_vals, width, label='Test Set (Out-of-Sample)', color='#00BCD4', alpha=0.9)
    
    ax3.set_title(f"Top {len(selected_stocks)} Portfolio Metrics: Train vs. Test", fontsize=13, fontweight='bold', color='#E6EDF3', pad=12)
    ax3.set_xticks(x_m)
    ax3.set_xticklabels(metrics, fontsize=10, fontweight='bold', color='#E6EDF3')
    ax3.legend(loc='upper right', facecolor='#21262D', edgecolor='#30363D')
    ax3.grid(True, linestyle='--', alpha=0.15, axis='y')

    for rect in rects_tr:
        h = rect.get_height()
        ax3.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#B388FF', fontweight='bold')
    for rect in rects_te:
        h = rect.get_height()
        ax3.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#80DEEA', fontweight='bold')

    # 4. Historical Normalized Performance Line Chart (Top Selected Stocks Only)
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('#161B22')
    
    hist_data = fetch_historical_prices(target_tickers)
    if not hist_data.empty:
        norm_data = (hist_data / hist_data.iloc[0]) * 100
        for i, t in enumerate(target_tickers):
            color = palette[i % len(palette)]
            ax4.plot(norm_data.index, norm_data[t], label=f"#{i+1} {t}", linewidth=2.8, color=color)
            
        ax4.set_title(f"Top {len(selected_stocks)} Historical Growth Trends (Base=100: 2023-2024)", fontsize=13, fontweight='bold', color='#E6EDF3', pad=12)
        ax4.set_ylabel("Normalized Growth", fontsize=10, color='#8B949E')
        ax4.legend(loc='upper left', facecolor='#21262D', edgecolor='#30363D', fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.15)
        plt.xticks(rotation=20, fontsize=8)
    else:
        ax4.text(0.5, 0.5, "Historical Data Unavailable", ha='center', va='center', color='#8B949E')

    plt.suptitle(f"Quantum Portfolio Optimization (QAOA) - Top {len(selected_stocks)} Selected Stocks ({', '.join(selected_stocks)})", 
                 fontsize=16, fontweight='bold', color='#FFFFFF', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Saved top-{len(selected_stocks)} dashboard figure to {save_path}")
    
    return fig

# Streamlit UI
def run_streamlit_dashboard():
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.set_page_config(
        page_title="Quantum Portfolio Optimization",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
                color: #E6EDF3;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚛️ QAOA Quantum Portfolio Optimization Dashboard")
    st.caption("Risk-Return Analysis & Visual Graphs Focused on Top Selected Stocks")
    
    # Sidebar Controls
    st.sidebar.header("⚙️ Optimization Parameters")
    tickers_input = st.sidebar.text_input("Input Tickers (5 Stocks)", "AAPL, MSFT, GOOGL, AMZN, META")
    tickers_all = [t.strip().upper() for t in tickers_input.split(",")]
    
    col_sb1, col_sb2 = st.sidebar.columns(2)
    start_date = col_sb1.text_input("Start Date", "2023-01-01")
    end_date = col_sb2.text_input("End Date", "2024-01-01")
    
    risk_factor = st.sidebar.slider("Risk Factor (q)", 0.0, 1.0, 0.5, 0.05)
    budget = st.sidebar.slider("Output Budget (Top K Stocks)", min_value=1, max_value=len(tickers_all), value=2)
    
    # Display Filter Option: Default to Top Selected Stocks Only
    graph_filter = st.sidebar.radio(
        "📊 Graph Filter Mode",
        [f"Top {budget} Selected Stocks Only", "All Input Stocks"],
        index=0
    )
    top_only = (graph_filter == f"Top {budget} Selected Stocks Only")
    
    api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000/optimize")
    
    data = DEFAULT_DATA
    
    if st.sidebar.button("🚀 Run Live QAOA Optimization", type="primary"):
        with st.spinner(f"Optimizing for Top {budget} Selected Output..."):
            try:
                payload = {
                    "tickers": tickers_all,
                    "start_date": start_date,
                    "end_date": end_date,
                    "risk_factor": risk_factor,
                    "budget": int(budget),
                    "use_ibm_simulator": False
                }
                res = requests.post(api_url, json=payload, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"QAOA Optimization Complete! Top {len(data['selected_stocks'])} Stocks Selected.")
                else:
                    st.error(f"API Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.warning(f"Could not connect to live API ({e}). Using default screenshot data.")

    selected_stocks = data["selected_stocks"]
    returns_all = data["individual_expected_returns"]
    risks_all = data["individual_risks_variance"]
    
    # Target tickers for graphs based on filter mode
    graph_tickers = selected_stocks if top_only else list(returns_all.keys())
    returns = {t: returns_all[t] for t in graph_tickers if t in returns_all}
    risks = {t: risks_all[t] for t in graph_tickers if t in risks_all}
    
    # Info Banner
    st.info(f"🏆 **Top {len(selected_stocks)} Selected Stocks Output:** `{', '.join(selected_stocks)}` &nbsp;&nbsp;|&nbsp;&nbsp; **Graph Display:** `{graph_filter}`")

    # Metric Banner Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Top Selected Stocks", ", ".join(selected_stocks))
    m2.metric("Train Sharpe Ratio", f"{data['train_sharpe']:.2f}")
    m3.metric("Train Return", f"{data['train_return']*100:.1f}%")
    m4.metric("Test Sharpe Ratio", f"{data['test_sharpe']:.2f}")
    m5.metric("Test Return", f"{data['test_return']*100:.1f}%")
    
    st.divider()

    # SECTION: Ranked Leaderboard Table for 5 Inputs -> Top Output Selection
    st.subheader(f"🏆 Ranked Leaderboard: Input Stocks ➔ Top {len(selected_stocks)} Selected Output")
    
    leaderboard_data = []
    for t in tickers_all:
        ret = returns_all.get(t, 0.0)
        risk = risks_all.get(t, 0.0)
        sharpe = ret / np.sqrt(risk) if risk > 0 else 0.0
        is_sel = t in selected_stocks
        leaderboard_data.append({
            "Ticker": t,
            "Expected Return": f"{ret*100:.2f}%",
            "Variance Risk": f"{risk:.4f}",
            "Individual Sharpe": f"{sharpe:.2f}",
            "QAOA Status": f"✅ SELECTED (Top {len(selected_stocks)})" if is_sel else f"❌ Unselected (Budget limit K={budget})"
        })
    
    df_leaderboard = pd.DataFrame(leaderboard_data)
    df_leaderboard = df_leaderboard.sort_values(by="Expected Return", ascending=False).reset_index(drop=True)
    df_leaderboard.index += 1
    df_leaderboard.index.name = "Rank"
    
    st.table(df_leaderboard)
    
    st.divider()

    col_chart1, col_chart2 = st.columns(2)
    
    # Plot 1: Risk vs Return Scatter Plot (Top Selected Stocks)
    with col_chart1:
        st.subheader(f"📊 Risk vs. Expected Return ({'Top ' + str(len(selected_stocks)) + ' Selected' if top_only else 'All Stocks'})")
        scatter_df = pd.DataFrame({
            "Ticker": list(returns.keys()),
            "Expected Return": [returns[t] for t in returns],
            "Risk (Variance)": [risks[t] for t in returns],
            "Status": [f"Top #{i+1} Selected" if t in selected_stocks else "Unselected" for i, t in enumerate(returns)]
        })
        
        fig_scatter = px.scatter(
            scatter_df, x="Risk (Variance)", y="Expected Return", text="Ticker",
            color="Status",
            size=[26 if t in selected_stocks else 14 for t in scatter_df["Ticker"]],
            title=f"Risk vs. Return Spectrum: Top {len(selected_stocks)} Selected Stocks" if top_only else "Risk vs Return Spectrum"
        )
        
        fig_scatter.add_trace(go.Scatter(
            x=[data["train_risk"]], y=[data["train_return"]],
            mode="markers+text", name="Combined Portfolio",
            text=[f"Quantum Portfolio (Top {len(selected_stocks)})"], textposition="top center",
            marker=dict(size=22, symbol="diamond", color="#FF9100")
        ))
        
        fig_scatter.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Plot 2: Returns & Risk Bar Chart (Top Selected Stocks)
    with col_chart2:
        st.subheader(f"📈 Expected Return vs Risk ({'Top ' + str(len(selected_stocks)) + ' Selected' if top_only else 'All Stocks'})")
        bar_df = pd.DataFrame({
            "Ticker": list(returns.keys()),
            "Return (%)": [returns[t]*100 for t in returns],
            "Risk (x100)": [risks[t]*100 for t in returns]
        })
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=bar_df["Ticker"], y=bar_df["Return (%)"], name="Expected Return (%)",
            marker_color="#00E676"
        ))
        fig_bar.add_trace(go.Bar(
            x=bar_df["Ticker"], y=bar_df["Risk (x100)"], name="Variance Risk (x100)",
            marker_color="#FF5252"
        ))
        fig_bar.update_layout(barmode="group", template="plotly_dark", height=450, title=f"Top {len(selected_stocks)} Selected Metrics")
        st.plotly_chart(fig_bar, use_container_width=True)

    col_chart3, col_chart4 = st.columns(2)
    
    # Plot 3: Historical Trend Line Chart (Top Selected Stocks Only)
    with col_chart3:
        st.subheader(f"📉 Historical Trends ({'Top ' + str(len(selected_stocks)) + ' Selected' if top_only else 'All Stocks'})")
        hist_df = fetch_historical_prices(list(returns.keys()))
        if not hist_df.empty:
            norm_df = (hist_df / hist_df.iloc[0]) * 100
            fig_line = px.line(norm_df, title=f"Top {len(selected_stocks)} Selected Normalized Price Growth")
            fig_line.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Fetching yfinance live price trends...")

    # Plot 4: Top 2 Portfolio Weight Distribution / Pie Chart
    with col_chart4:
        st.subheader(f"🍩 Portfolio Allocation (Top {len(selected_stocks)} Stocks)")
        pie_df = pd.DataFrame({
            "Ticker": selected_stocks,
            "Weight": [100.0 / len(selected_stocks)] * len(selected_stocks)
        })
        fig_pie = px.pie(pie_df, names="Ticker", values="Weight", title=f"Top {len(selected_stocks)} Asset Weights", hole=0.4,
                         color_discrete_sequence=["#00E676", "#FFD600", "#29B6F6", "#FF5252"])
        fig_pie.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

# Main Execution Switcher
if st.runtime.exists():
    run_streamlit_dashboard()
else:
    print("Generating static dashboard figure for Top 2 stocks...")
    generate_matplotlib_dashboard(DEFAULT_DATA, "top5_stocks_analysis.png", top_only=True)
    print("Done! To run the interactive web app, use: python -m streamlit run dashboard.py")
