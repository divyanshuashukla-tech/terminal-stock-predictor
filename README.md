# > Stock Predictor 💻

A beginner-friendly Machine Learning stock forecasting tool wrapped in a responsive cyberpunk hacker terminal UI. Built for hackathons to showcase simple, effective regression modeling without deployment overhead.

---

## What It Does

1. **Fetches Data:** Downloads the last 2 years of daily market prices for any Yahoo Finance ticker.
2. **Trains on the Fly:** Prepares historical OHLCV features (`Open`, `High`, `Low`, `Close`, `Volume`) and trains a `RandomForestRegressor`.
3. **Predicts Tomorrow:** Projects the next trading day's closing price.
4. **Recommends Action:** Issues an instant **BUY** signal if the predicted price exceeds today's close, or a **SELL** signal otherwise.

---

## Features

* **Cyberpunk Hacker Theme:** Complete with custom dark styling, monospace typography, neon glowing borders, and blinking terminal cursor animations.
* **Fake Processing Logs:** Simulates real-time system execution in the browser.
* **Interactive Charting:** Fully customizable Plotly line graphs highlighting historical trajectories and tomorrow's prediction target.
* **No Complex Architecture:** No APIs, databases, Docker containers, or heavy neural networks required.

---

## How to Run Locally

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Setup a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
