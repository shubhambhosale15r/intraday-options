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
from webdriver_manager.chrome import ChromeDriverManager
import logging
import webdriver_manager
import chromedriver_autoinstaller

print("webdriver_manager version:", webdriver_manager.__version__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# List of user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
]

# Optional: List of proxies (format: "http://ip:port")
PROXIES = []

# NSE API URL and Home URL
NSE_API_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={}"
NSE_HOME_URL = "https://www.nseindia.com/"

# Configuration flag for headless mode
HEADLESS = True


def get_random_user_agent():
    return random.choice(USER_AGENTS)


# Set up ChromeDriver installation directory
CHROMEDRIVER_ROOT = os.path.join(tempfile.gettempdir(), "chromedriver_autoinstaller")
os.makedirs(CHROMEDRIVER_ROOT, exist_ok=True)
os.environ["CHROMEDRIVER_AUTOINSTALLER_ROOT"] = CHROMEDRIVER_ROOT


def get_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    if HEADLESS:
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("window-size=1920,1080")

    ua = get_random_user_agent()
    chrome_options.add_argument(f"user-agent={ua}")

    # Install ChromeDriver & ensure correct version
    chrome_version = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
    chromedriver_path = chromedriver_autoinstaller.install(path=CHROMEDRIVER_ROOT)

    if not chromedriver_path:
        raise RuntimeError("Failed to install ChromeDriver.")

    service = Service(chromedriver_path)
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to initialize WebDriver: {str(e)}")


def create_session():
    driver = get_selenium_driver()
    driver.get(NSE_HOME_URL)
    time.sleep(random.uniform(4, 6))
    selenium_cookies = driver.get_cookies()
    driver.quit()

    session = requests.Session()
    ua = get_random_user_agent()
    session.headers.update({
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    })

    if PROXIES:
        proxy = random.choice(PROXIES)
        session.proxies.update({"http": proxy, "https": proxy})
        logging.info(f"Using proxy: {proxy}")

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
                row.update({
                    "CE OI": entry["CE"].get("openInterest", 0),
                    "CE Chng": entry["CE"].get("change", 0),
                })
            if "PE" in entry:
                row.update({
                    "PE OI": entry["PE"].get("openInterest", 0),
                    "PE Chng": entry["PE"].get("change", 0),
                })
            rows.append(row)
        except Exception as ex:
            logging.error(f"Error processing entry: {entry} - {ex}")

    df = pd.DataFrame(rows)
    return df, expiry_dates, None


def display_time():
    utc_time = datetime.now(pytz.utc)
    ist_time = utc_time.astimezone(pytz.timezone("Asia/Kolkata"))
    st.write(f"Last Updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")


# --- Streamlit UI ---
st.set_page_config(page_title="NSE Option Chain", layout="wide")
st.sidebar.title("📊 NSE Option Chain Analyzer")
page = st.sidebar.radio("Navigation", ["Option Chain", "Analysis"])

symbol_list = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
selected_symbol = st.selectbox("Select Symbol", symbol_list)

if page == "Option Chain":
    st.title("📈 NSE Option Chain Analyzer")
    display_time()

    option_chain_df, expiry_dates, _ = fetch_option_chain(selected_symbol)
    if option_chain_df is not None:
        selected_expiry = st.selectbox("Select Expiry Date", expiry_dates)
        filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("⚠️ Unable to fetch data.")

elif page == "Analysis":
    st.title("📊 Market Trend Analysis")
    display_time()

    option_chain_df, expiry_dates, _ = fetch_option_chain(selected_symbol)
    if option_chain_df is not None:
        st.subheader("🔍 Price & OI Analysis")
        total_call_oi = option_chain_df["CE OI"].sum()
        total_put_oi = option_chain_df["PE OI"].sum()
        pcr_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        if pcr_ratio > 1.2:
            trend = "BUY 🟢 (Bullish)"
        elif pcr_ratio < 0.8:
            trend = "SELL 🔴 (Bearish)"
        else:
            trend = "SIDEWAYS 🔄 (Neutral)"

        st.markdown(f"**Put-Call Ratio (PCR):** {pcr_ratio:.2f} | **Trend:** {trend}")
    else:
        st.warning("⚠️ Unable to fetch data.")
