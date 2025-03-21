from datetime import datetime
import streamlit as st
import requests
import pandas as pd
from time import sleep
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

@st.cache_data(ttl=180)
def fetch_option_chain(symbol):
    """Fetch option chain data from NSE for the given symbol using session and cookies."""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
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

        # Add Intrinsic Value Columns
        df["CE Intrinsic Value"] = df.apply(lambda row: max(row["Underlying Value"] - row["Strike Price"], 0), axis=1)
        df["PE Intrinsic Value"] = df.apply(lambda row: max(row["Strike Price"] - row["Underlying Value"], 0), axis=1)

        # Organizing Columns with Expiry Date included
        ce_columns = [col for col in df.columns if col.startswith("CE ") and col not in ["CE Intrinsic Value"]]
        pe_columns = [col for col in df.columns if col.startswith("PE ") and col not in ["PE Intrinsic Value"]]
        center_columns = ["Expiry Date", "Strike Price", "Underlying Value", "CE Intrinsic Value", "PE Intrinsic Value"]

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
page = st.sidebar.radio("Navigation", ["Option Chain", "Buy/Sell Analysis","Positional Bets"])

if page == "Option Chain":
    st.title("📈 NSE Option Chain Analyzer")

    auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    if auto_refresh:
        st_autorefresh(interval=180_000, key="data_refresh")
        st.cache_data.clear()  # Clears cached data so fresh data is fetched

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

if page in ["Buy/Sell Analysis", "Positional Bets"]:
    st.title("📊 Market Trend Analysis" if page == "Buy/Sell Analysis" else "📊 Positional Bets Analysis")

    auto_refresh = st.checkbox("Auto Refresh Data", value=False)
    if auto_refresh:
        st_autorefresh(interval=180_000, key="data_refresh")
        st.cache_data.clear()  # Clears cached data so fresh data is fetched

    utc_time = datetime.now(pytz.utc)
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_time.astimezone(ist_timezone)
    st.write(f"Last Updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")

    symbol_list = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    selected_symbol = st.selectbox("Select Symbol", symbol_list, key="symbol_analysis")

    option_chain_df, expiry_dates, atm_strike = fetch_option_chain(selected_symbol)

    if option_chain_df is not None:
        if page == "Buy/Sell Analysis":
            selected_expiry = st.selectbox("Select Expiry Date", expiry_dates, key="expiry_analysis")
            filtered_df = option_chain_df[option_chain_df["Expiry Date"] == selected_expiry]
        else:
            filtered_df = option_chain_df  # Use all expiry dates for "Positional Bets"

        # --- PCR-Based Market Trend Analysis ---
        total_call_oi = filtered_df["CE OI"].sum()
        total_put_oi = filtered_df["PE OI"].sum()
        pcr_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        if pcr_ratio > 1.2:
            pcr_trend = "BUY 🟢 (Bullish based on PCR)"
        elif pcr_ratio < 0.8:
            pcr_trend = "SELL 🔴 (Bearish based on PCR)"
        else:
            pcr_trend = "SIDEWAYS 🔄 (Neutral based on PCR)"

        st.markdown(f"""
                        - **Put-Call Ratio (PCR):** {pcr_ratio:.2f}
                        - **PCR Trend :** {pcr_trend} """)

        st.caption("""
                                **Interpretation Guide**:
                                - PCR < 0.8: Traders pricing in downside protection (Bearish)
                                - PCR > 1.2: Traders expecting upside potential (Bullish)
                                """)

        # --- Intrinsic Value Strategy (Considering 5 ITM Strikes) ---
        itm_calls = filtered_df[filtered_df["Strike Price"] < atm_strike].nlargest(5, "Strike Price")
        itm_puts = filtered_df[filtered_df["Strike Price"] > atm_strike].nsmallest(5, "Strike Price")

        if not itm_calls.empty and not itm_puts.empty:
            avg_ce_ltp = itm_calls["CE LTP"].mean()
            avg_ce_intrinsic = itm_calls["CE Intrinsic Value"].mean()
            avg_pe_ltp = itm_puts["PE LTP"].mean()
            avg_pe_intrinsic = itm_puts["PE Intrinsic Value"].mean()

            if avg_ce_ltp > avg_ce_intrinsic and avg_pe_ltp < avg_pe_intrinsic:
                intrinsic_strategy = "BUY 🟢 (Call premium above intrinsic & Put premium below intrinsic)"
            elif avg_pe_ltp > avg_pe_intrinsic and avg_ce_ltp < avg_ce_intrinsic:
                intrinsic_strategy = "SELL 🔴 (Put premium above intrinsic & Call premium below intrinsic)"
            else:
                intrinsic_strategy = "SIDEWAYS 🔄 (No clear intrinsic bias)"
        else:
            intrinsic_strategy = "Data insufficient for intrinsic strategy"

        st.markdown(f"""
                        - **Intrinsic Strategy:** {intrinsic_strategy}
                        """)

        # --- Combine Conclusions ---
        conclusion_data = {
            'Market Trend PCR': [pcr_trend],
            'Intrinsic Strategy': [intrinsic_strategy]
        }
        conclusion_data_df = pd.DataFrame(conclusion_data)
        st.subheader("📌 Conclusion")
        st.table(conclusion_data_df)

    else:
        st.warning("⚠️ Unable to fetch data. Please try again later.")
