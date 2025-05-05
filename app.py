import os
import tempfile
import time
import random
import pandas as pd
import streamlit as st
from datetime import datetime
import pytz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

# List of user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
]

# Define constants for URLs
NSE_HOME_URL = "https://www.nseindia.com/"
NSE_OC_URL = "https://www.nseindia.com/option-chain"

def get_random_user_agent():
    return random.choice(USER_AGENTS)

# Function to simulate human-like delays
def human_like_delay():
    time.sleep(random.uniform(1.5, 4.5))  # Random delay between 1.5 and 4.5 seconds


def get_selenium_driver():
    # Set Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Force ChromeDriver version 114.0.5735.90 to match your Chrome version
    driver_path = ChromeDriverManager(version="114.0.5735.90").install()  # Specify ChromeDriver version 114
    
    service = Service(driver_path)  # Create a Service with the correct driver

    # Create the WebDriver with the correct service and options
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    return driver



def fetch_option_chain(symbol):
    driver = get_selenium_driver()
    try:
        driver.get(NSE_HOME_URL)
        human_like_delay()  # Simulate human behavior with a delay

        # Navigate to the API endpoint for the symbol's option chain
        driver.get(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}")
        human_like_delay()  # Wait to ensure the data is fully loaded

        # Wait for the element containing the data (JSON) to be available
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "pre")))

        # Extract the JSON data from the 'pre' tag
        pre = driver.find_element(By.TAG_NAME, "pre").text
        data = json.loads(pre)
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        driver.quit()  # Ensure driver is closed
        return None, None, None

    driver.quit()  # Close the browser after fetching data

    # Extract relevant details from the response data
    records = data.get("records", {})
    option_chain = records.get("data", [])
    expiry_dates = records.get("expiryDates", [])
    underlying_value = records.get("underlyingValue", None)

    if underlying_value is None:
        logging.error("Data format error: Missing underlying value.")
        return None, None, None

    # Prepare rows for the DataFrame
    rows = []
    for entry in option_chain:
        try:
            row = {
                "Expiry Date": entry.get("expiryDate", "N/A"),
                "Strike Price": entry["strikePrice"],
                "Underlying Value": underlying_value
            }
            # Process Call (CE) and Put (PE) data
            if "CE" in entry:
                ce = entry["CE"]
                row.update({
                    "CE OI": ce.get("openInterest", 0),
                    "CE Chng in OI": ce.get("changeinOpenInterest", 0),
                    "CE Volume": ce.get("totalTradedVolume", 0),
                    "CE IV": ce.get("impliedVolatility", "-"),
                    "CE LTP": ce.get("lastPrice", 0),
                    "CE Chng": ce.get("change", 0),
                    "CE Bid Qty": ce.get("bidQty", 0),
                    "CE Bid": ce.get("bidprice", 0),
                    "CE Ask": ce.get("askPrice", 0),
                    "CE Ask Qty": ce.get("askQty", 0)
                })
            if "PE" in entry:
                pe = entry["PE"]
                row.update({
                    "PE OI": pe.get("openInterest", 0),
                    "PE Chng in OI": pe.get("changeinOpenInterest", 0),
                    "PE Volume": pe.get("totalTradedVolume", 0),
                    "PE IV": pe.get("impliedVolatility", "-"),
                    "PE LTP": pe.get("lastPrice", 0),
                    "PE Chng": pe.get("change", 0),
                    "PE Bid Qty": pe.get("bidQty", 0),
                    "PE Bid": pe.get("bidprice", 0),
                    "PE Ask": pe.get("askPrice", 0),
                    "PE Ask Qty": pe.get("askQty", 0),
                })
            rows.append(row)
        except Exception as ex:
            logging.error(f"Error processing entry: {entry} - {ex}")

    # Create DataFrame from the list of rows
    df = pd.DataFrame(rows)

    # Compute intrinsic values for CE and PE
    df["CE Intrinsic Value"] = (df["Underlying Value"] - df["Strike Price"]).clip(lower=0)
    df["PE Intrinsic Value"] = (df["Strike Price"] - df["Underlying Value"]).clip(lower=0)

    # Reorganize columns for clarity
    ce_cols = sorted([col for col in df.columns if col.startswith("CE ")])
    pe_cols = sorted([col for col in df.columns if col.startswith("PE ")])
    center_cols = ["Expiry Date", "Strike Price", "Underlying Value"]
    ordered_columns = ce_cols + center_cols + pe_cols + ["CE Intrinsic Value", "PE Intrinsic Value"]
    df = df[ordered_columns]

    # Identify the ATM strike price
    atm_strike = df["Strike Price"].iloc[
        (df["Strike Price"] - underlying_value).abs().idxmin()] if not df.empty else None

    return df, expiry_dates, atm_strike

def display_time():
    utc_time = datetime.now(pytz.utc)
    ist_time = utc_time.astimezone(pytz.timezone("Asia/Kolkata"))
    st.write(f"Last Updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")

def setup_autorefresh():
    from streamlit_autorefresh import st_autorefresh
    # Random interval between 1 and 3 minutes (60000 to 180000 ms)
    refresh_interval_ms = random.randint(60000, 180000)
    st_autorefresh(refresh_interval_ms, key="data_refresh")
    st.cache_data.clear()

# --- Streamlit Interface ---
st.set_page_config(page_title="NSE Option Chain", layout="wide")
st.sidebar.title("📊 NSE Option Chain Analyzer")
page = st.sidebar.radio("Navigation", ["Option Chain", "Buy/Sell Analysis", "Positional Bets"])

symbol_list = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
selected_symbol = st.selectbox("Select Symbol", symbol_list, key="symbol_common")

if page == "Option Chain":
    st.title("📈 NSE Option Chain Analyzer")
    auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    if auto_refresh:
        setup_autorefresh()
    display_time()

    option_chain_df, expiry_dates, atm_strike = fetch_option_chain(selected_symbol)
    if option_chain_df is not None:
        selected_expiry = st.selectbox("Select Expiry Date", expiry_dates)
        filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]
        # Ensure unique index and columns
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]

        def highlight_atm(row):
            return ['background-color: yellow; color: black' if row['Strike Price'] == atm_strike else '' for _ in row]

        st.subheader(f"📅 Option Chain for {selected_symbol} - {selected_expiry}")
        if atm_strike:
            styled_df = filtered_df.style.apply(highlight_atm, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            st.markdown(f"**ATM Strike Price**: {atm_strike} (Highlighted in yellow)")
        else:
            st.dataframe(filtered_df, use_container_width=True)
        st.markdown(f"**Current Underlying Value**: {filtered_df['Underlying Value'].iloc[0]:.2f}")
    else:
        st.warning("⚠️ Unable to fetch data. Please try again later.")

elif page in ["Buy/Sell Analysis", "Positional Bets"]:
    title = "📊 Market Trend Analysis" if page == "Buy/Sell Analysis" else "📊 Positional Bets Analysis"
    st.title(title)
    auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    if auto_refresh:
        setup_autorefresh()
    display_time()

    option_chain_df, expiry_dates, atm_strike = fetch_option_chain(selected_symbol)
    if option_chain_df is not None:
        if page == "Buy/Sell Analysis":
            selected_expiry = st.selectbox("Select Expiry Date", expiry_dates, key="expiry_analysis")
            filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]
        else:
            filtered_df = option_chain_df

        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]

        # Analyze Put-Call Ratio (PCR)
        total_call_oi = filtered_df["CE OI"].sum()
        total_put_oi = filtered_df["PE OI"].sum()
        pcr_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        if pcr_ratio > 1.2:
            pcr_trend = "BUY 🟢 (Bullish based on PCR)"
        elif pcr_ratio < 0.8:
            pcr_trend = "SELL 🔴 (Bearish based on PCR)"
        else:
            pcr_trend = "Neutral ⚪"

        st.write(f"**PCR Ratio**: {pcr_ratio:.2f} - Trend: {pcr_trend}")
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("⚠️ Unable to fetch data. Please try again later.")
