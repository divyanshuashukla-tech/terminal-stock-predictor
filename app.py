import time
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================
# Configure the page title, icon, and layout
st.set_page_config(page_title="Terminal Stock Predictor", page_icon="💻", layout="wide")

# ==========================================
# 2. HACKER TERMINAL THEME (CSS INJECTION)
# ==========================================
# We inject custom CSS to override Streamlit defaults and create a glowing hacker terminal aesthetic.
st.markdown(
    """
<style>
    /* Import monospace fonts */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');

    /* Global application background and text colors */
    .stApp {
        background-color: #0d0d0d;
        color: #00ff66;
        font-family: 'Fira Code', 'Courier New', monospace;
    }

    /* Override standard headings to Neon Green with glow */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #00ff66 !important;
        font-family: 'Fira Code', 'Courier New', monospace !important;
    }
    
    h1 {
        text-shadow: 0 0 10px #00ff66;
        border-bottom: 1px solid #00ff66;
        padding-bottom: 10px;
    }

    /* Style the text input box */
    .stTextInput input {
        background-color: #1a1a1a !important;
        color: #00ff66 !important;
        border: 1px solid #00ff66 !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 16px;
    }

    /* Style the Predict Button */
    .stButton>button {
        background-color: #00ff66 !important;
        color: #0d0d0d !important;
        font-family: 'Fira Code', monospace !important;
        font-weight: bold;
        border: 1px solid #00ff66 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #00cccc !important;
        border-color: #00cccc !important;
        box-shadow: 0 0 15px #00cccc;
    }

    /* Custom Terminal Log Box */
    .terminal-box {
        background-color: #050505;
        border: 1px solid #00ff66;
        padding: 15px;
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
        color: #00ff66;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 255, 102, 0.2);
    }

    /* Blinking Cursor Animation */
    .blink {
        animation: blink-animation 1s steps(5, start) infinite;
    }
    @keyframes blink-animation {
        to {
            visibility: hidden;
        }
    }

    /* Green Bordered Metric/Prediction Cards */
    .card {
        background-color: #111111;
        border: 1px solid #00ff66;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 255, 102, 0.15);
        margin-bottom: 10px;
    }
    
    .card-title {
        font-size: 14px;
        color: #00cccc !important;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    
    .card-value {
        font-size: 28px;
        font-weight: bold;
        color: #00ff66 !important;
        text-shadow: 0 0 5px #00ff66;
    }

    /* Cyan Recommendation Highlights */
    .rec-buy {
        color: #00ff66 !important;
        font-size: 24px;
        font-weight: bold;
        text-shadow: 0 0 10px #00ff66;
    }
    .rec-sell {
        color: #ff3333 !important;
        font-size: 24px;
        font-weight: bold;
        text-shadow: 0 0 10px #ff3333;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. TITLE & DESCRIPTION
# ==========================================
st.markdown(
    "<h1>> TERMINAL STOCK PREDICTOR <span class='blink'>_</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "`[SYSTEM READY] Machine Learning prediction engine initialized. Enter ticker to begin.`"
)

# ==========================================
# 4. TICKER INPUT SECTION
# ==========================================
col_input, col_button = st.columns([3, 1])

with col_input:
    # Let the user type any Yahoo Finance ticker symbol
    ticker = st.text_input("ENTER TICKER SYMBOL:", value="AAPL").upper()

with col_button:
    st.write("")  # Spacing alignment
    st.write("")
    predict_clicked = st.button("EXECUTE PREDICTION")

# ==========================================
# 5. CORE PREDICTION PIPELINE
# ==========================================
if predict_clicked and ticker:

    # --- Step 5a: Create Fake Hacker Terminal Output ---
    st.markdown("### `> SYSTEM LOGS`")
    log_placeholder = st.empty()

    # Define realistic terminal sequence messages
    logs = [
        f"> Connecting to Yahoo Finance [{ticker}]...",
        "> Downloading 2-year market data...",
        "> Building feature matrix (Open, High, Low, Close, Volume)...",
        "> Training RandomForestRegressor (n_estimators=100)...",
        "> Predicting tomorrow's close price...",
        "> Prediction Complete. Rendering output...",
    ]

    # Display logs step-by-step with slight delays for effect
    terminal_text = ""
    for log in logs:
        terminal_text += f"{log}<br>"
        log_placeholder.markdown(
            f"<div class='terminal-box'>{terminal_text}<span class='blink'>█</span></div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.3)  # Brief delay to simulate processing

    # --- Step 5b: Download Historical Stock Data ---
    # Fetch 2 years of daily stock prices
    stock_data = yf.download(ticker, period="2y", progress=False)

    # Check if data was found
    if stock_data.empty:
        st.error(
            f"Error: Could not retrieve data for ticker '{ticker}'. Check symbol and try again."
        )
        st.stop()

    # Flatten multi-level column names if returned by yfinance v0.2+
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # --- Step 5c: Prepare Machine Learning Features & Target ---
    df = stock_data.copy()

    # Select our core features
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[feature_cols].dropna()

    # Create the Target: Tomorrow's Close Price (shifted 1 day into the future)
    df["Target"] = df["Close"].shift(-1)

    # Split data into training features (X) and training target (y)
    # We drop the last row because its 'Target' is NaN (tomorrow hasn't happened yet!)
    train_df = df.dropna()
    X = train_df[feature_cols]
    y = train_df["Target"]

    # --- Step 5d: Train Machine Learning Model ---
    # We use a Random Forest because it handles tabular financial data well without complex tuning
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- Step 5e: Predict Tomorrow's Price ---
    # To predict tomorrow, we feed today's latest features into the trained model
    latest_features = df[feature_cols].iloc[[-1]]
    predicted_price = float(model.predict(latest_features)[0])

    # Get today's actual closing price
    current_price = float(df["Close"].iloc[-1])

    # Calculate price difference and percentage change
    price_diff = predicted_price - current_price
    pct_change = (price_diff / current_price) * 100

    # ==========================================
    # 6. PREDICTION CARDS & RECOMMENDATION
    # ==========================================
    st.markdown("### `> ANALYSIS OUTPUT`")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class='card'>
            <div class='card-title'>Current Close Price</div>
            <div class='card-value'>${current_price:,.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class='card'>
            <div class='card-title'>Predicted Price (Tomorrow)</div>
            <div class='card-value' style='color: #00cccc !important;'>${predicted_price:,.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        sign = "+" if price_diff >= 0 else ""
        st.markdown(
            f"""
        <div class='card'>
            <div class='card-title'>Expected Difference</div>
            <div class='card-value'>{sign}${price_diff:,.2f} ({sign}{pct_change:.2f}%)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Simple logic: Buy if model predicts an upward movement, Sell otherwise
    st.write("")  # Spacer
    if predicted_price > current_price:
        rec_html = "<span class='rec-buy'>[ RECOMMENDATION: BUY ] (PREDICTION > CURRENT)</span>"
    else:
        rec_html = "<span class='rec-sell'>[ RECOMMENDATION: SELL ] (PREDICTION < CURRENT)</span>"

    st.markdown(f"<div class='card'>{rec_html}</div>", unsafe_allow_html=True)

    # ==========================================
    # 7. HISTORICAL CHART WITH PLOTLY
    # ==========================================
    st.markdown("### `> HISTORICAL PRICE TRAJECTORY & PROJECTION`")

    # Create an interactive chart using Plotly
    fig = go.Figure()

    # 1. Historical Closing Price Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#00ff66", width=2),
        )
    )

    # 2. Highlight Today's Price (Last known actual point)
    fig.add_trace(
        go.Scatter(
            x=[df.index[-1]],
            y=[current_price],
            mode="markers",
            name="Today's Close",
            marker=dict(color="#00ff66", size=10, symbol="circle"),
        )
    )

    # 3. Highlight Tomorrow's Predicted Price
    # Create a synthetic date for tomorrow by adding 1 business day
    next_day = df.index[-1] + pd.Timedelta(days=1)

    fig.add_trace(
        go.Scatter(
            x=[df.index[-1], next_day],
            y=[current_price, predicted_price],
            mode="lines+markers",
            name="Predicted Target",
            line=dict(color="#00cccc", width=2, dash="dot"),
            marker=dict(color="#00cccc", size=10, symbol="diamond"),
        )
    )

    # Apply terminal aesthetic styling to the Plotly chart
    fig.update_layout(
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#050505",
        font=dict(family="Fira Code, monospace", color="#00ff66"),
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", title="Price (USD)"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Render the chart inside Streamlit
    st.plotly_chart(fig, use_container_width=True)
