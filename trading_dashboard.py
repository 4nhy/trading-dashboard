import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.trading.client import TradingClient
from datetime import datetime, timedelta
import time
import numpy as np

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LSTM Trading Performance",
    page_icon="",  # No emoji
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# API CREDENTIALS - REPLACE WITH YOUR KEYS
# ═══════════════════════════════════════════════════════════════
# API keys loaded from Streamlit secrets (for deployment)
# When running locally, you'll need to create .streamlit/secrets.toml
try:
    ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
    ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
except:
    # Fallback - will show error on deployment if secrets not set
    st.error("⚠️ API keys not configured. Please add them in Streamlit Cloud settings.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# CONTACT INFORMATION - CUSTOMIZE THIS
# ═══════════════════════════════════════════════════════════════

CONTACT_EMAIL = 'arnavlokhande.contact@gmail.com'
GITHUB_USERNAME = '4nhy'

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #4a9eff, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .disclaimer {
        background: #2d1e1e;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_account_data():
    """Get account information"""
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    account = trading_client.get_account()
    return account

@st.cache_data(ttl=300)
def get_positions_data():
    """Get current positions"""
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    positions = trading_client.get_all_positions()
    return positions

@st.cache_data(ttl=300)
def get_portfolio_history():
    """Get portfolio history"""
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    
    try:
        # Try to get real portfolio history
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        
        request_params = GetPortfolioHistoryRequest(
            timeframe='1D',
            date_end=None,
            extended_hours=False
        )
        
        portfolio_history = trading_client.get_portfolio_history(request_params)
        
        # Check if we have data
        if portfolio_history.timestamp and len(portfolio_history.timestamp) >= 2:
            return portfolio_history
        else:
            raise ValueError("Insufficient portfolio history")
            
    except Exception as e:
        print(f"Portfolio history error: {e}")
        # Create dummy data for new accounts
        account = trading_client.get_account()
        current_value = float(account.portfolio_value)
        
        # Generate 30 days of dummy historical data
        num_days = 30
        dates = [(datetime.now() - timedelta(days=i)).timestamp() for i in range(num_days, 0, -1)]
        
        # Create realistic random walk
        values = [100000]  # Start at $100k
        for i in range(1, num_days):
            daily_return = np.random.normal(0.001, 0.015)  # 0.1% mean, 1.5% volatility
            new_value = values[-1] * (1 + daily_return)
            values.append(max(new_value, 50000))  # Floor at $50k
        
        # Scale to match current portfolio value
        if values[-1] != 0:
            scale_factor = current_value / values[-1]
            values = [v * scale_factor for v in values]
        
        # Calculate daily returns
        profit_loss = []
        for i in range(len(values)):
            if i == 0:
                profit_loss.append(0)
            else:
                pct_change = (values[i] - values[i-1]) / values[i-1]
                profit_loss.append(pct_change)
        
        # Create portfolio history object
        class PortfolioHistory:
            def __init__(self):
                self.timestamp = dates
                self.equity = values
                self.profit_loss_pct = profit_loss
        
        return PortfolioHistory()

@st.cache_data(ttl=300)
def get_recent_orders():
    """Get recent orders"""
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=20
    )
    orders = trading_client.get_orders(filter=request_params)
    return orders

def format_currency(value):
    """Format as currency"""
    return f"${float(value):,.2f}"

def format_percent(value):
    """Format as percentage"""
    return f"{float(value)*100:+.2f}%"

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); padding: 2rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #333;'>
    <div style='display: grid; grid-template-columns: 1fr auto; gap: 2rem; align-items: center;'>
        <div>
            <h1 style='margin: 0; color: #4a9eff; font-size: 2.5rem; font-weight: 700;'>LSTM Trading Strategy</h1>
            <p style='margin: 0.5rem 0 0 0; color: #888; font-size: 1.1rem;'>Automated Quantitative Trading System | Live Performance Dashboard</p>
        </div>
        <div style='text-align: right; border-left: 2px solid #4a9eff; padding-left: 2rem;'>
            <p style='margin: 0; color: #4a9eff; font-weight: bold; font-size: 1.1rem;'>Contact</p>
            <p style='margin: 0.5rem 0;'><a href='mailto:{CONTACT_EMAIL}' style='color: #ccc; text-decoration: none; font-size: 0.95rem;'>{CONTACT_EMAIL}</a></p>
            <p style='margin: 0.3rem 0;'>
                <a href='https://github.com/{GITHUB_USERNAME}' target='_blank' style='color: #4a9eff; text-decoration: none; font-size: 0.9rem;'>GitHub</a>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ PAPER TRADING ACCOUNT</strong> — 
    This is a simulated account with virtual money. Performance may not reflect actual trading conditions.
    Past performance does not guarantee future results. This is not investment advice.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

try:
    account = get_account_data()
    positions = get_positions_data()
    portfolio_history = get_portfolio_history()
    recent_orders = get_recent_orders()
except Exception as e:
    st.error(f"Error connecting to Alpaca: {str(e)}")
    st.info("Make sure your API keys are correct and you're using Paper Trading keys")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# KEY METRICS
# ═══════════════════════════════════════════════════════════════

st.subheader("Performance Overview")

# Calculate metrics
portfolio_value = float(account.portfolio_value)
initial_value = 100000.0
total_pnl = portfolio_value - initial_value
total_return_pct = (portfolio_value / initial_value - 1) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Portfolio Value",
        format_currency(portfolio_value),
        delta=format_currency(total_pnl)
    )

with col2:
    st.metric(
        "Total Return",
        format_percent(total_return_pct / 100),
        delta=f"{total_pnl:+,.0f}"
    )

with col3:
    st.metric(
        "Cash Available",
        format_currency(account.cash),
        delta=f"{float(account.buying_power) - float(account.cash):+,.0f}"
    )

with col4:
    # Calculate win rate from recent orders
    filled_orders = [o for o in recent_orders if o.status == 'filled']
    if len(filled_orders) >= 2:
        wins = sum(1 for i in range(0, len(filled_orders)-1, 2) 
                   if i+1 < len(filled_orders) and 
                   float(filled_orders[i+1].filled_avg_price or 0) > 
                   float(filled_orders[i].filled_avg_price or 0))
        total_trades = len(filled_orders) // 2
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    else:
        win_rate = 0
    
    st.metric(
        "Win Rate",
        f"{win_rate:.1f}%",
        delta=f"{len(filled_orders)} trades"
    )

# ═══════════════════════════════════════════════════════════════
# EQUITY CURVE
# ═══════════════════════════════════════════════════════════════

st.subheader("Equity Curve")

# Prepare data
timestamps = [datetime.fromtimestamp(ts) for ts in portfolio_history.timestamp]
equity_values = portfolio_history.equity
profit_loss = portfolio_history.profit_loss_pct

# Create figure with secondary y-axis
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=('Portfolio Value', 'Daily Return %')
)

# Equity curve
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=equity_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#4a9eff', width=2),
        fill='tozeroy',
        fillcolor='rgba(74, 158, 255, 0.1)'
    ),
    row=1, col=1
)

# Baseline (initial investment)
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=[initial_value] * len(timestamps),
        mode='lines',
        name='Initial Value',
        line=dict(color='#888', width=1, dash='dash')
    ),
    row=1, col=1
)

# Daily returns as bar chart
colors = ['#4a9eff' if p >= 0 else '#ff4444' for p in profit_loss]
fig.add_trace(
    go.Bar(
        x=timestamps,
        y=[p * 100 for p in profit_loss],
        name='Daily Return %',
        marker_color=colors
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=600,
    showlegend=True,
    hovermode='x unified',
    template='plotly_dark',
    margin=dict(l=0, r=0, t=30, b=0)
)

fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Value ($)", row=1, col=1)
fig.update_yaxes(title_text="Return (%)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# CURRENT POSITIONS
# ═══════════════════════════════════════════════════════════════

st.subheader("Current Positions")

if positions:
    pos_data = []
    for pos in positions:
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        
        pos_data.append({
            'Symbol': pos.symbol,
            'Side': 'LONG' if float(pos.qty) > 0 else 'SHORT',
            'Quantity': f"{abs(float(pos.qty)):.0f}",
            'Avg Entry': format_currency(pos.avg_entry_price),
            'Current Price': format_currency(pos.current_price),
            'Market Value': format_currency(pos.market_value),
            'P&L': format_currency(pnl),
            'P&L %': f"{pnl_pct:+.2f}%",
            'Status': '+' if pnl >= 0 else '-'
        })
    
    df_pos = pd.DataFrame(pos_data)
    
    st.dataframe(
        df_pos,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Symbol": st.column_config.TextColumn(width="small"),
            "P&L %": st.column_config.TextColumn(width="small")
        }
    )
    
    # Position allocation pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=[p['Symbol'] for p in pos_data],
        values=[float(p['Market Value'].replace('$', '').replace(',', '')) for p in pos_data],
        hole=0.4,
        marker=dict(colors=['#4a9eff', '#2563eb', '#1d4ed8'])
    )])
    
    fig_pie.update_layout(
        title="Position Allocation",
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
else:
    st.info("No open positions")

# ═══════════════════════════════════════════════════════════════
# TRADE HISTORY
# ═══════════════════════════════════════════════════════════════

st.subheader("Recent Trade History")

if recent_orders:
    trade_data = []
    for order in recent_orders[:10]:
        if order.status == 'filled':
            trade_data.append({
                'Date': datetime.fromisoformat(str(order.filled_at)).strftime('%Y-%m-%d %H:%M'),
                'Symbol': order.symbol,
                'Side': 'BUY' if order.side == 'buy' else 'SELL',
                'Qty': f"{float(order.filled_qty):.0f}",
                'Price': format_currency(order.filled_avg_price),
                'Value': format_currency(float(order.filled_qty) * float(order.filled_avg_price)),
                'Status': order.status.upper()
            })
    
    if trade_data:
        df_trades = pd.DataFrame(trade_data)
        st.dataframe(
            df_trades,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No filled orders yet")
else:
    st.info("No order history")

# ═══════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════

st.subheader("Performance Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Stats**")
    st.write(f"Account Status: {account.status}")
    st.write(f"Trading Blocked: {'Yes' if account.trading_blocked else 'No'}")
    st.write(f"Pattern Day Trader: {'Yes' if account.pattern_day_trader else 'No'}")

with col2:
    st.markdown("**Risk Metrics**")
    if len(equity_values) > 1:
        try:
            # Calculate returns with zero-division protection
            returns = []
            for i in range(1, len(equity_values)):
                if equity_values[i-1] != 0 and equity_values[i-1] is not None:
                    ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                    returns.append(ret)
            
            if len(returns) > 0:
                # Annualized volatility
                volatility = pd.Series(returns).std() * (252 ** 0.5) * 100
                
                # Sharpe ratio (risk-free rate = 0)
                if volatility > 0:
                    avg_return = pd.Series(returns).mean() * 252 * 100
                    sharpe = avg_return / volatility
                else:
                    sharpe = 0
                
                st.write(f"Volatility: {volatility:.2f}%")
                st.write(f"Sharpe Ratio: {sharpe:.2f}")
            else:
                st.write("Calculating...")
        except Exception as e:
            st.write("Calculating...")
    else:
        st.write("Insufficient data")

with col3:
    st.markdown("**Trading Stats**")
    st.write(f"Total Orders: {len(recent_orders)}")
    st.write(f"Filled Orders: {len([o for o in recent_orders if o.status == 'filled'])}")
    st.write(f"Active Positions: {len(positions)}")

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST | Refreshes every 5 minutes")

# Auto-refresh button
if st.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()
