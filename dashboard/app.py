"""Streamlit dashboard for stock metrics visualization."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.stock_analyzer import StockAnalyzer


# Page configuration
st.set_page_config(
    page_title="Stock Metrics Dashboard",
    page_icon="📈",
    layout="wide",
)


def format_large_number(num: float) -> str:
    """Format large numbers with B/M/K suffixes."""
    if num is None:
        return "N/A"
    if num >= 1e12:
        return f"${num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"${num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"${num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"${num / 1e3:.2f}K"
    return f"${num:.2f}"


def format_volume(num: int) -> str:
    """Format volume numbers."""
    if num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str) -> dict:
    """Fetch and cache stock data."""
    analyzer = StockAnalyzer(ticker, period)
    if not analyzer.fetch_data():
        return None

    # Collect all metrics
    current_price = analyzer.get_current_price()
    dollar_change, percent_change = analyzer.get_price_change()
    low_52, high_52 = analyzer.get_52_week_range()
    current_vol, avg_vol = analyzer.get_volume_info()
    volatility = analyzer.get_volatility()
    sharpe = analyzer.get_sharpe_ratio()
    max_dd = analyzer.get_max_drawdown()
    total_return = analyzer.get_total_return()
    rolling_vol = analyzer.get_rolling_volatility()

    return {
        "history": analyzer.history.copy(),
        "company_name": analyzer.get_company_name(),
        "sector": analyzer.get_sector(),
        "industry": analyzer.get_industry(),
        "market_cap": analyzer.get_market_cap(),
        "current_price": current_price,
        "dollar_change": dollar_change,
        "percent_change": percent_change,
        "low_52": low_52,
        "high_52": high_52,
        "current_vol": current_vol,
        "avg_vol": avg_vol,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "rolling_vol": rolling_vol,
    }


def create_candlestick_chart(history: pd.DataFrame, ticker: str) -> go.Figure:
    """Create a candlestick chart with volume."""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=history.index,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            name="Price",
        )
    )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
    )

    return fig


def create_rolling_volatility_chart(
    history: pd.DataFrame, rolling_vol: dict
) -> go.Figure:
    """Create rolling volatility chart."""
    fig = go.Figure()

    colors = {20: "#00CC96", 50: "#FFA15A", 100: "#EF553B"}

    for window, vol_array in rolling_vol.items():
        # Align dates with volatility values
        dates = history.index[window:]
        if len(dates) == len(vol_array):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=vol_array * 100,
                    mode="lines",
                    name=f"{window}-day",
                    line=dict(color=colors.get(window, "#636EFA")),
                )
            )

    fig.update_layout(
        title="Rolling Volatility",
        yaxis_title="Volatility (%)",
        xaxis_title="Date",
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    st.title("Stock Metrics Dashboard")
    st.markdown("Enter a stock ticker to view key metrics and analytics.")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
        period = st.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
        )
        st.markdown("---")
        show_options = st.checkbox("Show Options Analysis", value=False)

        if show_options:
            st.subheader("Options Parameters")
            strike_pct = st.slider(
                "Strike (% of spot)", min_value=80, max_value=120, value=100
            )
            days_to_expiry = st.slider(
                "Days to Expiry", min_value=1, max_value=365, value=30
            )
            option_type = st.selectbox("Option Type", options=["call", "put"])
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0
            ) / 100

    # Main content
    if ticker:
        with st.spinner(f"Loading data for {ticker}..."):
            data = fetch_stock_data(ticker, period)

        if data is None:
            st.error(
                f"Could not fetch data for '{ticker}'. Please check the ticker symbol."
            )
            return

        # Company header
        st.header(f"{data['company_name']} ({ticker})")
        if data["sector"] or data["industry"]:
            st.caption(
                f"{data['sector'] or 'N/A'} | {data['industry'] or 'N/A'} | Market Cap: {format_large_number(data['market_cap'])}"
            )

        # Price metrics row
        st.subheader("Price & Volume")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            price = data["current_price"]
            change = data["dollar_change"]
            pct = data["percent_change"]
            delta_str = f"${change:+.2f} ({pct:+.2f}%)" if change else None
            st.metric("Current Price", f"${price:.2f}" if price else "N/A", delta_str)

        with col2:
            low_52, high_52 = data["low_52"], data["high_52"]
            if low_52 and high_52:
                st.metric("52-Week Low", f"${low_52:.2f}")
                st.metric("52-Week High", f"${high_52:.2f}")
            else:
                st.metric("52-Week Range", "N/A")

        with col3:
            st.metric("Volume", format_volume(data["current_vol"]))

        with col4:
            st.metric("Avg Volume", format_volume(data["avg_vol"]))

        # Price chart
        st.plotly_chart(
            create_candlestick_chart(data["history"], ticker), use_container_width=True
        )

        # Risk metrics
        st.subheader("Risk Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            vol = data["volatility"]
            st.metric(
                "Historical Volatility", f"{vol * 100:.1f}%" if vol else "N/A"
            )

        with col2:
            sharpe = data["sharpe"]
            st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")

        with col3:
            max_dd = data["max_dd"]
            st.metric(
                "Max Drawdown",
                f"-{max_dd[0] * 100:.1f}%" if max_dd else "N/A",
            )

        with col4:
            total_ret = data["total_return"]
            st.metric(
                "Total Return",
                f"{total_ret * 100:+.1f}%" if total_ret else "N/A",
            )

        with col5:
            # Annualized return approximation
            if total_ret and period:
                period_years = {
                    "1mo": 1 / 12,
                    "3mo": 0.25,
                    "6mo": 0.5,
                    "1y": 1,
                    "2y": 2,
                    "5y": 5,
                }.get(period, 1)
                ann_ret = (1 + total_ret) ** (1 / period_years) - 1
                st.metric("Annualized Return", f"{ann_ret * 100:+.1f}%")
            else:
                st.metric("Annualized Return", "N/A")

        # Rolling volatility chart
        if data["rolling_vol"]:
            st.plotly_chart(
                create_rolling_volatility_chart(data["history"], data["rolling_vol"]),
                use_container_width=True,
            )

        # Options analysis section
        if show_options:
            st.markdown("---")
            st.subheader("Options Analysis (Black-Scholes)")

            # Create a fresh analyzer for options calculations
            analyzer = StockAnalyzer(ticker, period)
            analyzer.fetch_data()

            current_price = analyzer.get_current_price()
            if current_price:
                strike = current_price * (strike_pct / 100)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Option Parameters**")
                    st.write(f"- Spot Price: ${current_price:.2f}")
                    st.write(f"- Strike Price: ${strike:.2f}")
                    st.write(f"- Days to Expiry: {days_to_expiry}")
                    st.write(f"- Option Type: {option_type.upper()}")
                    st.write(f"- Risk-Free Rate: {risk_free_rate * 100:.1f}%")
                    st.write(
                        f"- Implied Volatility: {analyzer.get_volatility() * 100:.1f}%"
                    )

                with col2:
                    option_price = analyzer.get_option_price(
                        strike=strike,
                        days_to_expiry=days_to_expiry,
                        option_type=option_type,
                        risk_free_rate=risk_free_rate,
                    )
                    st.metric(
                        "Theoretical Option Price",
                        f"${option_price:.2f}" if option_price else "N/A",
                    )

                # Greeks table
                greeks = analyzer.get_option_greeks(
                    strike=strike,
                    days_to_expiry=days_to_expiry,
                    option_type=option_type,
                    risk_free_rate=risk_free_rate,
                )

                if greeks:
                    st.markdown("**Option Greeks**")
                    greeks_df = pd.DataFrame(
                        {
                            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                            "Value": [
                                f"{greeks['delta']:.4f}",
                                f"{greeks['gamma']:.4f}",
                                f"{greeks['vega']:.4f}",
                                f"{greeks['theta']:.4f}",
                                f"{greeks['rho']:.4f}",
                            ],
                            "Description": [
                                "Price sensitivity to underlying",
                                "Delta sensitivity to underlying",
                                "Sensitivity to volatility (per 1%)",
                                "Daily time decay",
                                "Sensitivity to interest rate (per 1%)",
                            ],
                        }
                    )
                    st.table(greeks_df)

        # Footer
        st.markdown("---")
        st.caption(
            f"Data provided by Yahoo Finance. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


if __name__ == "__main__":
    main()
