from datetime import datetime
import streamlit as st
import requests
import pandas as pd
from time import sleep, strftime, gmtime
from streamlit_autorefresh import st_autorefresh
import pytz
# --- NSE API Base URL ---
NSE_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={}"

# Headers to mimic browser behavior
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en",
    "Referer": "https://www.nseindia.com/"
}


@st.cache_data(ttl=300)
def fetch_option_chain(symbol):
    """Fetch option chain data from NSE for the given symbol using session and cookies."""
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=HEADERS)
        sleep(1)
        response = session.get(NSE_URL.format(symbol), headers=HEADERS, cookies=session.cookies)
        response.raise_for_status()
        data = response.json()

        if "records" not in data or "data" not in data["records"]:
            return None, None, None

        option_chain = data["records"]["data"]
        expiry_dates = data["records"]["expiryDates"]
        underlying_value = data["records"]["underlyingValue"]

        rows = []
        for entry in option_chain:
            row = {
                # Corrected key from 'expiryDates' to 'expiryDate'
                "Expiry Date": entry.get("expiryDate", "N/A"),
                "Strike Price": entry["strikePrice"],
                "Underlying Value": underlying_value
            }
            if "CE" in entry:
                row.update({
                    "CE OI": entry["CE"].get("openInterest", 0),
                    "CE Chng in OI": entry["CE"].get("changeinOpenInterest", 0),
                    "CE Volume": entry["CE"].get("totalTradedVolume", 0),
                    "CE IV": entry["CE"].get("impliedVolatility", "-"),
                    "CE LTP": entry["CE"].get("lastPrice", 0),
                    "CE Chng": entry["CE"].get("change", 0),
                    "CE Bid Qty": entry["CE"].get("bidQty", 0),
                    "CE Bid": entry["CE"].get("bidprice", 0),
                    "CE Ask": entry["CE"].get("askPrice", 0),
                    "CE Ask Qty": entry["CE"].get("askQty", 0)
                })
            if "PE" in entry:
                row.update({
                    "PE Bid Qty": entry["PE"].get("bidQty", 0),
                    "PE Bid": entry["PE"].get("bidprice", 0),
                    "PE Ask": entry["PE"].get("askPrice", 0),
                    "PE Ask Qty": entry["PE"].get("askQty", 0),
                    "PE Chng": entry["PE"].get("change", 0),
                    "PE LTP": entry["PE"].get("lastPrice", 0),
                    "PE IV": entry["PE"].get("impliedVolatility", "-"),
                    "PE Volume": entry["PE"].get("totalTradedVolume", 0),
                    "PE Chng in OI": entry["PE"].get("changeinOpenInterest", 0),
                    "PE OI": entry["PE"].get("openInterest", 0)
                })
            rows.append(row)

        df = pd.DataFrame(rows)

        # Organizing Columns with Expiry Date included
        ce_columns = [col for col in df.columns if col.startswith("CE ")]
        pe_columns = [col for col in df.columns if col.startswith("PE ")]
        center_columns = ["Expiry Date", "Strike Price", "Underlying Value"]  # Fixed here

        ordered_columns = ce_columns + center_columns + pe_columns
        df = df[ordered_columns]

        # Calculate ATM strike
        atm_strike = min(df["Strike Price"], key=lambda x: abs(x - underlying_value)) if not df.empty else None

        return df, expiry_dates, atm_strike
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None


# --- Streamlit Interface ---
st.set_page_config(page_title="NSE Option Chain", layout="wide")
st.sidebar.title("📊 NSE Option Chain Analyzer")
page = st.sidebar.radio("Navigation", ["Option Chain", "Buy/Sell Analysis"])

if page == "Option Chain":
    st.title("📈 NSE Option Chain Analyzer")

    auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    if auto_refresh:
        st_autorefresh(interval=60000, key="data_refresh")

    # Get the current UTC time and convert it to IST
    utc_time = datetime.now(pytz.utc)

    # Convert to IST (Indian Standard Time)
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_time.astimezone(ist_timezone)

    # Display the formatted IST time in the Streamlit app
    st.write(f"Last Updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")

    symbol_list = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    selected_symbol = st.selectbox("Select Symbol", symbol_list)

    option_chain_df, expiry_dates, atm_strike = fetch_option_chain(selected_symbol)

    if option_chain_df is not None:
        selected_expiry = st.selectbox("Select Expiry Date", expiry_dates)
        filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]


        # Define highlighting function
        def highlight_atm(row):
            return ['background-color: yellow; color: black' if row['Strike Price'] == atm_strike else ''
                    for _ in row]


        st.subheader(f"📅 Option Chain for {selected_symbol} - {selected_expiry}")

        if atm_strike:
            # Apply styling and show ATM information
            styled_df = filtered_df.style.apply(highlight_atm, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            st.markdown(f"**ATM Strike Price**: {atm_strike} (Highlighted in yellow 🟡)")
        else:
            st.dataframe(filtered_df, use_container_width=True)

        # Display underlying value
        st.markdown(f"**Current Underlying Value**: {filtered_df['Underlying Value'].iloc[0]:.2f}")

    else:
        st.warning("⚠️ Unable to fetch data. Please try again later.")

elif page == "Buy/Sell Analysis":

    st.title("📊 Market Trend Analysis")

    auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    if auto_refresh:
        st_autorefresh(interval=60000, key="data_refresh")

    # Get the current UTC time and convert it to IST
    utc_time = datetime.now(pytz.utc)

    # Convert to IST (Indian Standard Time)
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_time.astimezone(ist_timezone)

    # Display the formatted IST time in the Streamlit app
    st.write(f"Last Updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")

    symbol_list = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    selected_symbol = st.selectbox("Select Symbol", symbol_list, key="symbol_analysis")

    option_chain_df, expiry_dates, atm_strike = fetch_option_chain(selected_symbol)

    if option_chain_df is not None:
        selected_expiry = st.selectbox("Select Expiry Date", expiry_dates, key="expiry_analysis")
        filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]

        # Corrected column names from 'lastPrice' to 'LTP'
        call_itm_df = filtered_df[filtered_df["Strike Price"] < atm_strike].nlargest(5, "Strike Price")
        put_itm_df = filtered_df[filtered_df["Strike Price"] > atm_strike].nsmallest(5, "Strike Price")

        call_total = call_itm_df["CE LTP"].sum()
        put_total = put_itm_df["PE LTP"].sum()

        if call_total > 1.25 * put_total:
            market_trend = "BUY 🟢 (Bullish Market)"
        elif put_total > 1.25 * call_total:
            market_trend = "SELL 🔴 (Bearish Market)"
        else:
            market_trend = "SIDEWAYS 🔄 (Neutral Market)"

        st.write(f"Market Trend: **{market_trend}**")
        st.markdown(f"""
            - **Call Total**: {call_total}
            - **Put Total**: {put_total}
            """)
        st.subheader("Decision Logic:")
        st.markdown("""
            - **BUY 🟢**: ITM Call LTP > 1.25 × ITM Put LTP
            - **SELL 🔴**: ITM Put LTP > 1.25 × ITM Call LTP
            - **SIDEWAYS 🔄**: Ratio between 0.8 - 1.25
            """)
    else:
        st.warning("⚠️ Unable to fetch data. Please try again later.")

        # ... your existing code for market trend ...

        st.subheader("Volatility Skew Analysis")
    if option_chain_df is not None:
            selected_expiry = st.selectbox("Select Expiry Date", expiry_dates, key="iv_analysis")
            filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]

            # Convert IV columns to numeric (handle '-' values)
            filtered_df["CE IV"] = pd.to_numeric(filtered_df["CE IV"], errors='coerce')
            filtered_df["PE IV"] = pd.to_numeric(filtered_df["PE IV"], errors='coerce')

            # Calculate IV Skew
            otm_put_iv = filtered_df[filtered_df["Strike Price"] > atm_strike]["PE IV"].mean()
            otm_call_iv = filtered_df[filtered_df["Strike Price"] < atm_strike]["CE IV"].mean()

            if not pd.isna(otm_put_iv) and not pd.isna(otm_call_iv):
                iv_skew = otm_put_iv - otm_call_iv
                interpretation = ("🟢 Bullish (Calls IV > Puts IV)" if iv_skew < 0
                                  else "🔴 Bearish (Puts IV > Calls IV)")

                st.markdown(f"""
                - **Average OTM Put IV**: {otm_put_iv:.2f}
                - **Average OTM Call IV**: {otm_call_iv:.2f}
                - **IV Skew**: {iv_skew:.2f} ({interpretation})
                """)

                st.caption("""
                **Interpretation Guide**:
                - Positive Skew: Traders pricing in downside protection (Bearish)
                - Negative Skew: Traders expecting upside potential (Bullish)
                """)
            else:
                st.warning("Could not calculate IV Skew - missing volatility data")


