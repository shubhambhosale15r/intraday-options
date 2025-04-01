import os
import tempfile
import time
import random
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
import pytz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import logging
import chromedriver_autoinstaller

# Configure logging
logging.basicConfig(level=logging.INFO)

# List of user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
]

# Optional: List of proxies (format: "http://ip:port")
PROXIES = []  # e.g., ["http://123.456.78.90:8080", "http://98.76.54.32:3128"]

# NSE API URL and Home URL
NSE_API_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={}"
NSE_HOME_URL = "https://www.nseindia.com/"

# Configuration flag for headless mode (set to False for local development)
HEADLESS = True

def get_random_user_agent():
    return random.choice(USER_AGENTS)

# Create a writable directory for the ChromeDriver
CHROMEDRIVER_ROOT = os.path.join(tempfile.gettempdir(), "chromedriver_autoinstaller")
os.makedirs(CHROMEDRIVER_ROOT, exist_ok=True)
os.environ["CHROMEDRIVER_AUTOINSTALLER_ROOT"] = CHROMEDRIVER_ROOT

# Now import the module
import chromedriver_autoinstaller

def get_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    if HEADLESS:
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("window-size=1920,1080")
    else:
        chrome_options.add_argument("start-maximized")

    # Set a random user agent
    ua = get_random_user_agent()
    chrome_options.add_argument(f"user-agent={ua}")

    # Install ChromeDriver into a specific directory
    chromedriver_autoinstaller.install(path=CHROMEDRIVER_ROOT)

    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to initialize WebDriver: {str(e)}")

def create_session():
    # Use Selenium to retrieve cookies from the NSE home page
    driver = get_selenium_driver()
    driver.get(NSE_HOME_URL)
    # Instead of fixed sleep, consider explicit waits if needed
    time.sleep(random.uniform(4, 6))
    selenium_cookies = driver.get_cookies()
    driver.quit()

    session = requests.Session()
    # Rotate the user agent in session headers
    ua = get_random_user_agent()
    session.headers.update({
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    })

    # Optional: Set a random proxy from the list if available
    if PROXIES:
        proxy = random.choice(PROXIES)
        session.proxies.update({"http": proxy, "https": proxy})
        logging.info(f"Using proxy: {proxy}")

    # Transfer cookies from Selenium to the requests session
    for cookie in selenium_cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    return session

@st.cache_data(ttl=180)
def fetch_option_chain(symbol):
    session = create_session()
    url = NSE_API_URL.format(symbol)
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        logging.error(f"Error fetching data from {url}: {e}")
        return None, None, None

    records = data.get("records", {})
    option_chain = records.get("data", [])
    expiry_dates = records.get("expiryDates", [])
    underlying_value = records.get("underlyingValue", None)

    if underlying_value is None:
        st.error("Data format error: Missing underlying value.")
        logging.error("Missing underlying value in API response.")
        return None, None, None

    rows = []
    for entry in option_chain:
        try:
            row = {
                "Expiry Date": entry.get("expiryDate", "N/A"),
                "Strike Price": entry["strikePrice"],
                "Underlying Value": underlying_value
            }
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

    df = pd.DataFrame(rows)

    # Compute intrinsic values using vectorized operations
    df["CE Intrinsic Value"] = (df["Underlying Value"] - df["Strike Price"]).clip(lower=0)
    df["PE Intrinsic Value"] = (df["Strike Price"] - df["Underlying Value"]).clip(lower=0)

    # Order columns: calls, center, then puts, and computed columns at the end
    ce_cols = sorted([col for col in df.columns if col.startswith("CE ")] )
    pe_cols = sorted([col for col in df.columns if col.startswith("PE ")])
    center_cols = ["Expiry Date", "Strike Price", "Underlying Value"]
    ordered_columns = ce_cols + center_cols + pe_cols + ["CE Intrinsic Value", "PE Intrinsic Value"]
    df = df[ordered_columns]

    # Determine ATM strike price using absolute difference
    atm_strike = df["Strike Price"].iloc[
        (df["Strike Price"] - underlying_value).abs().idxmin()] if not df.empty else None

    return df, expiry_dates, atm_strike

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
